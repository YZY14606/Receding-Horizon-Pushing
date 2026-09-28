from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import trimesh
import yaml
from chamferdist import ChamferDistance
from transforms3d.euler import quat2euler
from transforms3d.quaternions import quat2mat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_POSE_RECORD_BASE = PROJECT_ROOT / "pose_record"
DEFAULT_PUSH_RECORD_BASE = PROJECT_ROOT / "push_record"
SUCCESS_THRESHOLD = 0.015
SURFACE_SAMPLE_BATCHES = 30
SURFACE_POINTS_PER_BATCH = 1024

OBJECT_DIR_PATTERN = re.compile(r"(train|test)_(\d+)")
TRAJECTORY_DIR_PATTERN = re.compile(r"traj_(\d+)")


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Structured evaluation result for one train/test object."""

    traj_number: int
    success_count: int
    success_rate: float
    success_ci_lower: float
    success_ci_upper: float
    average_steps: Optional[float]
    steps_ci_lower: Optional[float]
    steps_ci_upper: Optional[float]
    mean_rooted_cdis: Optional[float]
    average_position_error: Optional[float]
    average_orientation_error: Optional[float]

    @property
    def success_ci_lower_error(self) -> float:
        return self.success_rate - self.success_ci_lower

    @property
    def success_ci_upper_error(self) -> float:
        return self.success_ci_upper - self.success_rate

    @property
    def steps_ci_lower_error(self) -> Optional[float]:
        if self.average_steps is None or self.steps_ci_lower is None:
            return None
        return self.average_steps - self.steps_ci_lower

    @property
    def steps_ci_upper_error(self) -> Optional[float]:
        if self.average_steps is None or self.steps_ci_upper is None:
            return None
        return self.steps_ci_upper - self.average_steps


@dataclass(frozen=True)
class TrajectoryRecord:
    """A pose trajectory directory and its corresponding push log."""

    relative_dir: Path
    pose_dir: Path
    log_path: Path


def _resolve_base_path(path: Union[str, Path]) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def normalize_object_pose_dir(
    object_pose_dir: Union[str, Path],
    pose_record_base: Union[str, Path] = DEFAULT_POSE_RECORD_BASE,
) -> Tuple[Path, Path]:
    """Resolve an object directory while retaining the old relative-path API."""

    pose_record_base = _resolve_base_path(pose_record_base)
    object_pose_dir = Path(object_pose_dir).expanduser()
    if object_pose_dir.is_absolute():
        resolved_object_dir = object_pose_dir.resolve()
    elif object_pose_dir.parts and object_pose_dir.parts[0] == "pose_record":
        resolved_object_dir = (PROJECT_ROOT / object_pose_dir).resolve()
    else:
        resolved_object_dir = (pose_record_base / object_pose_dir).resolve()

    try:
        resolved_object_dir.relative_to(pose_record_base)
    except ValueError as exc:
        raise ValueError(
            f"Object pose directory must be inside {pose_record_base}: "
            f"{resolved_object_dir}"
        ) from exc

    return resolved_object_dir, pose_record_base


def _parse_object_dir_name(object_pose_dir: Path) -> Tuple[str, int]:
    match = OBJECT_DIR_PATTERN.fullmatch(object_pose_dir.name)
    if match is None:
        raise ValueError(
            "Object directory name must match train_N or test_N: "
            f"{object_pose_dir}"
        )
    return match.group(1), int(match.group(2))


def _trajectory_sort_key(relative_dir: Path) -> Tuple[Tuple[str, ...], int]:
    match = TRAJECTORY_DIR_PATTERN.fullmatch(relative_dir.name)
    if match is None:
        raise ValueError(f"Invalid trajectory directory name: {relative_dir}")
    return relative_dir.parent.parts, int(match.group(1))


def collect_trajectory_records(
    object_pose_dir: Union[str, Path],
    pose_record_base: Union[str, Path] = DEFAULT_POSE_RECORD_BASE,
    push_record_base: Union[str, Path] = DEFAULT_PUSH_RECORD_BASE,
) -> Tuple[TrajectoryRecord, ...]:
    """Validate and pair every pose trajectory with its push log."""

    object_pose_dir, pose_record_base = normalize_object_pose_dir(
        object_pose_dir, pose_record_base
    )
    push_record_base = _resolve_base_path(push_record_base)

    if not object_pose_dir.is_dir():
        raise FileNotFoundError(
            f"Object pose directory does not exist: {object_pose_dir}"
        )

    relative_object_dir = object_pose_dir.relative_to(pose_record_base)
    object_push_dir = push_record_base / relative_object_dir
    if not object_push_dir.is_dir():
        raise FileNotFoundError(
            f"Object push directory does not exist: {object_push_dir}"
        )

    pose_trajectory_dirs = {
        path.relative_to(object_pose_dir): path
        for path in object_pose_dir.rglob("traj_*")
        if path.is_dir()
    }
    push_trajectory_dirs = {
        path.relative_to(object_push_dir): path
        for path in object_push_dir.rglob("traj_*")
        if path.is_dir()
    }

    if not pose_trajectory_dirs:
        raise ValueError(f"No trajectory directories found in {object_pose_dir}")

    pose_keys = set(pose_trajectory_dirs)
    push_keys = set(push_trajectory_dirs)
    if pose_keys != push_keys:
        missing_pose = sorted(str(path) for path in push_keys - pose_keys)
        missing_push = sorted(str(path) for path in pose_keys - push_keys)
        raise ValueError(
            f"Pose/push trajectory directories do not match for {object_pose_dir}. "
            f"Missing pose directories: {missing_pose}; "
            f"missing push directories: {missing_push}"
        )

    records: List[TrajectoryRecord] = []
    for relative_dir in sorted(pose_keys, key=_trajectory_sort_key):
        pose_dir = pose_trajectory_dirs[relative_dir]
        two_pose_path = pose_dir / "two_pose.npy"
        log_path = push_trajectory_dirs[relative_dir] / "log.json"
        if not two_pose_path.is_file():
            raise FileNotFoundError(f"Missing trajectory pose file: {two_pose_path}")
        if not log_path.is_file():
            raise FileNotFoundError(f"Missing trajectory log file: {log_path}")
        _load_pose(two_pose_path)
        _load_push_steps(log_path)
        records.append(
            TrajectoryRecord(
                relative_dir=relative_dir,
                pose_dir=pose_dir,
                log_path=log_path,
            )
        )

    return tuple(records)


def calculate_theta_error(
    pushed_pose: Sequence[float], future_pose: Sequence[float]
) -> Tuple[np.ndarray, bool]:
    """Calculate the shortest yaw error and whether the object has fallen."""

    euler_object = float(quat2euler(pushed_pose[3:7])[2])
    target_euler = float(quat2euler(future_pose[3:7])[2])
    if euler_object < 0:
        euler_object += math.tau
    if target_euler < 0:
        target_euler += math.tau

    theta_diff = abs(euler_object - target_euler)
    processed_diff = min(theta_diff, math.tau - theta_diff)

    euler_now = [abs(value) for value in quat2euler(pushed_pose[3:7])]
    fall = euler_now[0] > math.pi / 6 or euler_now[1] > math.pi / 6
    return np.asarray(processed_diff), fall


def calculate_error(all_pose: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Calculate planar position error and yaw error for two poses."""

    pushed_pose = all_pose[0]
    future_pose = all_pose[1]
    distance = np.linalg.norm(pushed_pose[:2] - future_pose[:2])
    theta_diff, fall = calculate_theta_error(pushed_pose, future_pose)
    return np.asarray([distance, theta_diff], dtype=float), fall


def _sample_surface_batches(
    mesh: trimesh.Trimesh,
    rng: Optional[np.random.Generator],
) -> np.ndarray:
    samples = []
    for _ in range(SURFACE_SAMPLE_BATCHES):
        sample_seed = None
        if rng is not None:
            sample_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        points, _ = trimesh.sample.sample_surface(
            mesh,
            SURFACE_POINTS_PER_BATCH,
            seed=sample_seed,
        )
        samples.append(points)
    return np.stack(samples, axis=0)


def calculate_chamfer_distance(
    args: Dict[str, object], seed: Optional[int] = None
) -> float:
    """Calculate the existing rooted Chamfer metric with optional fixed sampling."""

    config_path = _resolve_base_path(Path(str(args["obj_config_path"])))
    if not config_path.is_file():
        raise FileNotFoundError(f"Object config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        object_data = yaml.safe_load(file)

    object_index = f"{int(args['object_index'])}_object"
    if not isinstance(object_data, dict) or object_index not in object_data:
        raise KeyError(f"Object {object_index} is missing from {config_path}")

    object_entry = object_data[object_index]
    object_path = (REPOSITORY_ROOT / object_entry["relative_path"]).resolve()
    if not object_path.is_file():
        raise FileNotFoundError(f"Object mesh does not exist: {object_path}")

    object_scale = np.asarray(object_entry["test_info"]["test_scale"])
    start_pose = np.asarray(args["start_pose"], dtype=float)
    end_pose = np.asarray(args["end_pose"], dtype=float)

    mesh = trimesh.load(object_path)
    mesh.apply_scale(object_scale)
    mesh_start = mesh.copy()

    transformation = np.eye(4)
    transformation[:3, 3] = start_pose[:3]
    transformation[:3, :3] = quat2mat(start_pose[3:7])
    mesh_start.apply_transform(transformation)

    rng = None if seed is None else np.random.default_rng(seed)
    ori_points = _sample_surface_batches(mesh_start, rng)

    transformation = np.eye(4)
    transformation[:3, 3] = end_pose[:3]
    transformation[:3, :3] = quat2mat(end_pose[3:7])
    mesh.apply_transform(transformation)
    future_points = _sample_surface_batches(mesh, rng)

    cd_loss = ChamferDistance()
    x = torch.tensor(ori_points, dtype=torch.float32)
    y = torch.tensor(future_points, dtype=torch.float32)
    distance = cd_loss(
        x,
        y,
        batch_reduction="mean",
        bidirectional=True,
        point_reduction="mean",
    ).item()
    return float(np.sqrt(distance / 2))


def wilson_ci(
    success_count: int,
    trajectory_count: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Return the Wilson score confidence interval for a proportion."""

    if trajectory_count <= 0:
        raise ValueError("trajectory_count must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if not 0 <= success_count <= trajectory_count:
        raise ValueError("success_count must be between 0 and trajectory_count")

    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    probability = success_count / trajectory_count
    denominator = 1 + z**2 / trajectory_count
    center = (
        probability + z**2 / (2 * trajectory_count)
    ) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1 - probability) / trajectory_count
            + z**2 / (4 * trajectory_count**2)
        )
        / denominator
    )
    return center - margin, center + margin


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = 0,
) -> Tuple[float, float]:
    """Return the percentile bootstrap CI used for per-object step counts."""

    values_array = np.asarray(values, dtype=float)
    if len(values_array) == 0:
        raise ValueError("values must contain at least one number")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if len(values_array) == 1:
        value = float(values_array[0])
        return value, value

    rng = np.random.default_rng(seed)
    boot = rng.choice(
        values_array,
        size=(n_boot, len(values_array)),
        replace=True,
    ).mean(axis=1)
    alpha = 1 - confidence
    lower, upper = np.percentile(
        boot, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    return float(lower), float(upper)


def _load_pose(two_pose_path: Path) -> np.ndarray:
    try:
        all_pose = np.load(two_pose_path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Unable to load trajectory pose: {two_pose_path}") from exc

    if all_pose.shape != (2, 7):
        raise ValueError(
            f"Expected pose shape (2, 7), got {all_pose.shape}: {two_pose_path}"
        )
    if not np.isfinite(all_pose).all():
        raise ValueError(f"Pose contains NaN or infinity: {two_pose_path}")

    all_pose = all_pose.copy()
    all_pose[1, 2] = all_pose[0, 2]
    return all_pose


def _load_push_steps(log_path: Path) -> int:
    try:
        with log_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read trajectory log: {log_path}") from exc

    if not isinstance(data, dict) or "push_steps" not in data:
        raise ValueError(f"Missing push_steps in trajectory log: {log_path}")
    try:
        push_steps = int(data["push_steps"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid push_steps in trajectory log: {log_path}") from exc
    if push_steps < 0:
        raise ValueError(f"push_steps must not be negative: {log_path}")
    return push_steps


def _derive_trajectory_seed(
    base_seed: Optional[int], relative_dir: Path
) -> Optional[int]:
    if base_seed is None:
        return None
    seed_material = f"{base_seed}:{relative_dir.as_posix()}".encode("utf-8")
    digest = hashlib.blake2b(seed_material, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def evaluate_object(
    object_pose_dir: Union[str, Path],
    pose_record_base: Union[str, Path] = DEFAULT_POSE_RECORD_BASE,
    push_record_base: Union[str, Path] = DEFAULT_PUSH_RECORD_BASE,
    success_threshold: float = SUCCESS_THRESHOLD,
    sample_seed: Optional[int] = 0,
) -> TrajectoryMetrics:
    """Evaluate all trajectories for one train/test object."""

    if success_threshold <= 0:
        raise ValueError("success_threshold must be positive")

    resolved_object_dir, resolved_pose_record_base = normalize_object_pose_dir(
        object_pose_dir, pose_record_base
    )
    split, object_index = _parse_object_dir_name(resolved_object_dir)
    records = collect_trajectory_records(
        resolved_object_dir,
        pose_record_base=pose_record_base,
        push_record_base=push_record_base,
    )

    scene_name = resolved_object_dir.parent.name
    config_name = (
        f"{scene_name}.yaml"
        if split == "train"
        else "test_object_config.yaml"
    )
    config_path = PROJECT_ROOT / "config" / config_name


    if not config_path.is_file():
        raise FileNotFoundError(f"Object config does not exist: {config_path}")


    success_count = 0
    successful_steps: List[int] = []
    successful_chamfer_distances: List[float] = []
    successful_position_errors: List[float] = []
    successful_orientation_errors: List[float] = []

    for record in records:
        all_pose = _load_pose(record.pose_dir / "two_pose.npy")
        args: Dict[str, object] = {
            "object_index": object_index,
            "start_pose": all_pose[0].copy(),
            "end_pose": all_pose[1].copy(),
            "obj_config_path": config_path,
        }
        trajectory_seed = _derive_trajectory_seed(
            sample_seed,
            resolved_object_dir.relative_to(resolved_pose_record_base)
            / record.relative_dir,
        )
        mean_euclidean = calculate_chamfer_distance(args, seed=trajectory_seed)
        if not math.isfinite(mean_euclidean):
            raise ValueError(
                f"Chamfer result is not finite: {record.pose_dir}"
            )

        if mean_euclidean >= success_threshold:
            continue

        pose_error, _ = calculate_error(all_pose)
        if pose_error.shape != (2,) or not np.isfinite(pose_error).all():
            raise ValueError(f"Invalid pose error result: {record.pose_dir}")

        success_count += 1
        successful_chamfer_distances.append(float(mean_euclidean))
        successful_position_errors.append(float(pose_error[0]))
        successful_orientation_errors.append(float(pose_error[1]))
        successful_steps.append(_load_push_steps(record.log_path))

    trajectory_count = len(records)
    success_rate = success_count / trajectory_count
    success_ci_lower, success_ci_upper = wilson_ci(
        success_count, trajectory_count
    )

    if success_count == 0:
        return TrajectoryMetrics(
            traj_number=trajectory_count,
            success_count=0,
            success_rate=success_rate,
            success_ci_lower=success_ci_lower,
            success_ci_upper=success_ci_upper,
            average_steps=None,
            steps_ci_lower=None,
            steps_ci_upper=None,
            mean_rooted_cdis=None,
            average_position_error=None,
            average_orientation_error=None,
        )

    average_steps = float(np.mean(successful_steps))
    steps_ci_lower, steps_ci_upper = bootstrap_mean_ci(successful_steps)
    return TrajectoryMetrics(
        traj_number=trajectory_count,
        success_count=success_count,
        success_rate=success_rate,
        success_ci_lower=success_ci_lower,
        success_ci_upper=success_ci_upper,
        average_steps=average_steps,
        steps_ci_lower=steps_ci_lower,
        steps_ci_upper=steps_ci_upper,
        mean_rooted_cdis=float(np.mean(successful_chamfer_distances)),
        average_position_error=float(np.mean(successful_position_errors)),
        average_orientation_error=float(np.mean(successful_orientation_errors)),
    )


def print_trajectory_metrics(
    object_label: Union[str, Path], metrics: TrajectoryMetrics
) -> None:
    """Print a result in the same human-readable form as the old script."""

    print(f"This is {object_label} object.")
    print(
        f"The trajectory number is {metrics.traj_number}. "
        f"Success rate is {metrics.success_rate:.4f}, "
        "95% Wilson CI errors "
        f"[{metrics.success_ci_lower_error:.3f}, "
        f"{metrics.success_ci_upper_error:.3f}]."
    )
    if metrics.success_count == 0:
        print("No successful trajectory; success-only metrics are unavailable.")
        print("-" * 80)
        return

    print(
        "The average steps of the success trajectory is "
        f"{metrics.average_steps:.2f}, 95% Bootstrap CI errors "
        f"[{metrics.steps_ci_lower_error:.2f}, "
        f"{metrics.steps_ci_upper_error:.2f}]."
    )
    print(
        "The average mean rooted chamfer distance of the success trajectory "
        f"is {metrics.mean_rooted_cdis}."
    )
    print(
        "The average position error of the success trajectory is "
        f"{metrics.average_position_error}."
    )
    print(
        "The average orientation error of the success trajectory is "
        f"{metrics.average_orientation_error}."
    )
    print("-" * 80)


def main(dir_path: Union[str, Path]) -> TrajectoryMetrics:
    """Backward-compatible console entry point that now also returns metrics."""

    metrics = evaluate_object(dir_path)
    print_trajectory_metrics(dir_path, metrics)
    return metrics


if __name__ == "__main__":
    pose_record_root = Path("pose_record/20260831_all_step_all_without_pose_failure/easy_2")
    resolved_pose_record_root = _resolve_base_path(pose_record_root)
    for split_name in ["test"]:
        split_dirs = sorted(
            resolved_pose_record_root.glob(f"{split_name}_*"),
            key=lambda path: int(path.name.rsplit("_", 1)[-1]),
        )
        for object_dir in split_dirs:
            main(object_dir)
