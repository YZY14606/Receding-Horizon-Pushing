"""Path layout and worker assignment for evaluation artifacts."""

from dataclasses import dataclass, replace
from pathlib import Path
import re
import shutil
from typing import Optional, Tuple


VALID_DIFFICULTIES = frozenset(
    {"scene_01","scene_02", 
    "scene_03", "scene_04",
    "scene_05", "scene_06"}
)


VALID_OBJECT_TYPES = frozenset({"train", "test"})
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TRAJ_ID = re.compile(r"^traj_[0-9]+$")


def _validate_component(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{field_name} must match {_SAFE_COMPONENT.pattern!r}; got {value!r}"
        )
    if value in {".", ".."}:
        raise ValueError(f"{field_name} cannot be {value!r}")
    return value


def _validate_index(value: int, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return value


def _validate_positive_int(value: int, field_name: str) -> int:
    return _validate_index(value, field_name, minimum=1)


@dataclass(frozen=True)
class WorkerAssignment:
    """A worker's trajectory quota and non-overlapping seed sequence."""

    worker_id: int
    num_traj: int
    seed_start: int
    seed_stride: int

    def __post_init__(self) -> None:
        _validate_index(self.worker_id, "worker_id")
        _validate_positive_int(self.num_traj, "num_traj")
        _validate_index(self.seed_start, "seed_start")
        _validate_positive_int(self.seed_stride, "seed_stride")

    @property
    def worker_dir_name(self) -> str:
        return f"worker_{self.worker_id:03d}"


def build_worker_assignments(
    num_traj: int,
    num_procs: int,
) -> Tuple[WorkerAssignment, ...]:
    """Evenly divide trajectories and give each worker a disjoint seed stream.

    The actual worker count is ``min(num_traj, num_procs)`` so an empty worker
    is never started.  Worker ``i`` consumes seeds ``i, i + W, i + 2W, ...``.
    """
    _validate_positive_int(num_traj, "num_traj")
    _validate_positive_int(num_procs, "num_procs")

    worker_count = min(num_traj, num_procs)
    quotient, remainder = divmod(num_traj, worker_count)
    return tuple(
        WorkerAssignment(
            worker_id=worker_id,
            num_traj=quotient + (worker_id < remainder),
            seed_start=worker_id,
            seed_stride=worker_count,
        )
        for worker_id in range(worker_count)
    )


@dataclass(frozen=True)
class ArtifactLayout:
    """Derive task-level or worker-bound paths from one shared identity.

    Construct an unbound instance to perform the task collision check, then
    call :meth:`for_worker` before resolving any writable artifact path.
    """

    run_id: str
    difficulty: str
    object_type: str
    object_idx: int
    demos_root: Path = Path("demos")
    visualizations_root: Path = Path("visualizations")
    pose_record_root: Path = Path("pose_record")
    push_record_root: Path = Path("push_record")
    worker_id: Optional[int] = None

    def __post_init__(self) -> None:
        _validate_component(self.run_id, "run_id")
        if self.difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                "difficulty must be one of "
                f"{sorted(VALID_DIFFICULTIES)}; got {self.difficulty!r}"
            )
        if self.object_type not in VALID_OBJECT_TYPES:
            raise ValueError(
                "object_type must be one of "
                f"{sorted(VALID_OBJECT_TYPES)}; got {self.object_type!r}"
            )
        _validate_index(self.object_idx, "object_idx", minimum=1)
        if self.worker_id is not None:
            _validate_index(self.worker_id, "worker_id")

        for field_name in (
            "demos_root",
            "visualizations_root",
            "pose_record_root",
            "push_record_root",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))

    @property
    def object_dir_name(self) -> str:
        return f"{self.object_type}_{self.object_idx:02d}"

    @property
    def task_relative_dir(self) -> Path:
        return Path(self.run_id) / self.difficulty / self.object_dir_name

    @property
    def worker_dir_name(self) -> str:
        return f"worker_{self._required_worker_id():03d}"

    @property
    def worker_relative_dir(self) -> Path:
        return self.task_relative_dir / self.worker_dir_name

    def for_worker(self, worker_id: int) -> "ArtifactLayout":
        """Return an immutable layout bound to one worker directory."""
        _validate_index(worker_id, "worker_id")
        return replace(self, worker_id=worker_id)

    @property
    def demos_task_dir(self) -> Path:
        return self.demos_root / self.task_relative_dir

    @property
    def demos_worker_dir(self) -> Path:
        return self.demos_root / self.worker_relative_dir

    @property
    def video_dir(self) -> Path:
        return self.demos_worker_dir / "video"

    def visualization_traj_dir(self, traj_id: str) -> Path:
        return (
            self.visualizations_root
            / self.worker_relative_dir
            / self._validated_traj_id(traj_id)
        )

    def visualization_attempt_dir(
        self,
        traj_id: str,
        push_index: int,
        attempt_index: int,
    ) -> Path:
        _validate_index(push_index, "push_index")
        _validate_index(attempt_index, "attempt_index")
        return (
            self.visualization_traj_dir(traj_id)
            / f"push_{push_index:03d}"
            / f"attempt_{attempt_index:02d}"
        )

    def pose_traj_dir(self, traj_id: str) -> Path:
        return (
            self.pose_record_root
            / self.worker_relative_dir
            / self._validated_traj_id(traj_id)
        )

    def push_traj_dir(self, traj_id: str) -> Path:
        return (
            self.push_record_root
            / self.worker_relative_dir
            / self._validated_traj_id(traj_id)
        )

    def task_dirs(self) -> Tuple[Path, Path, Path, Path]:
        return (
            self.demos_task_dir,
            self.visualizations_root / self.task_relative_dir,
            self.pose_record_root / self.task_relative_dir,
            self.push_record_root / self.task_relative_dir,
        )

    def ensure_task_is_new(self) -> None:
        if self.worker_id is not None:
            raise RuntimeError(
                "ensure_task_is_new() must be called on the unbound task layout"
            )
        existing = [path for path in self.task_dirs() if path.exists()]
        if existing:
            paths = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                "Refusing to overwrite an existing artifact task: " + paths
            )

    def discard_episode_artifacts(self, traj_id: str) -> None:
        """Remove non-HDF5 outputs for an episode rejected by ``run.py``."""
        paths = (
            self.visualization_traj_dir(traj_id),
            self.pose_traj_dir(traj_id),
            self.push_traj_dir(traj_id),
        )
        for path in paths:
            if path.exists():
                shutil.rmtree(path)

    def _required_worker_id(self) -> int:
        if self.worker_id is None:
            raise RuntimeError(
                "A writable artifact path requires layout.for_worker(worker_id)"
            )
        return self.worker_id

    @staticmethod
    def _validated_traj_id(traj_id: str) -> str:
        if not isinstance(traj_id, str) or not _TRAJ_ID.fullmatch(traj_id):
            raise ValueError(f"traj_id must look like 'traj_N'; got {traj_id!r}")
        return traj_id
