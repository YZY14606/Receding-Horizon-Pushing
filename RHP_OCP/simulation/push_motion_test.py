"""Test a sequence of push actions along a planned object path."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sapien
import torch
from sim_utils import (
    convert_action_use,
    generate_action,
    get_current_pc_future_pose,
    get_future_pose,
    get_pose2_wld_frame_points,
    is_path_point_reached,
)


PlanResult = Dict[str, Any]
@dataclass
class PushStepTestResult:
    """Generated action and planning result for one virtual path step.

    ``path_point_index`` is the waypoint targeted by this specific action; it
    is not a request to advance the real-world path state.
    """

    step_index: int
    path_point_index: int
    future_pose: torch.Tensor
    local_future_pose: np.ndarray
    ps_pose: np.ndarray
    pe_pose: np.ndarray
    reach_pose: sapien.Pose
    start_pose: sapien.Pose
    goal_pose: sapien.Pose
    plan_results: List[PlanResult]
    failed_stage: Optional[str]

    @property
    def success(self) -> bool:
        return self.failed_stage is None


@dataclass
class PushPathTestResult:
    """Result of testing consecutive actions with a virtual path cursor.

    ``final_path_point_index`` is the active target held by the committed
    virtual state when rollout stops.  It must never replace the caller's
    real-world waypoint index.
    """

    success: bool
    termination_reason: str
    steps: List[PushStepTestResult]
    final_robot_qpos: np.ndarray
    final_object_pose: torch.Tensor
    final_object_pointcloud: np.ndarray
    final_path_point_index: int

    @property
    def failed_step(self) -> Optional[PushStepTestResult]:
        if self.termination_reason != "motion_plan_failed" or not self.steps:
            return None
        return self.steps[-1]


def _validated_pointcloud_copy(
    points: np.ndarray,
    name: str,
    *,
    allow_empty: bool,
) -> np.ndarray:
    """Return an owned finite ``(N, 3)`` float64 point-cloud copy."""
    try:
        pointcloud = np.array(
            points,
            dtype=np.float64,
            order="C",
            copy=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to float64") from exc

    if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not allow_empty and len(pointcloud) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(pointcloud)):
        raise ValueError(f"{name} must contain only finite values")
    pointcloud.setflags(write=True)
    return pointcloud


def _compose_motion_planner_pointcloud(
    object_collision_pointcloud: np.ndarray,
    fixed_obstacle_pointcloud: np.ndarray,
) -> np.ndarray:
    """Combine one virtual object cloud with the fixed world geometry."""
    pointcloud = np.ascontiguousarray(
        np.vstack(
            (
                object_collision_pointcloud,
                fixed_obstacle_pointcloud,
            )
        ),
        dtype=np.float64,
    )
    pointcloud.setflags(write=True)
    return pointcloud


def _make_robot_poses(
    ps_pose: np.ndarray,
    pe_pose: np.ndarray,
    preparation_height: float,
) -> Tuple[sapien.Pose, sapien.Pose, sapien.Pose]:
    """Build the three robot poses used by the five-stage motion test."""
    ps_pose = np.asarray(ps_pose, dtype=float)
    pe_pose = np.asarray(pe_pose, dtype=float)
    if ps_pose.shape != (7,) or pe_pose.shape != (7,):
        raise ValueError("ps_pose and pe_pose must both have shape (7,)")

    reach_pose_array = ps_pose.copy()
    reach_pose_array[2] = preparation_height

    reach_pose = sapien.Pose(
        p=reach_pose_array[:3],
        q=reach_pose_array[3:7],
    )
    start_pose = sapien.Pose(p=ps_pose[:3], q=ps_pose[3:7])
    goal_pose = sapien.Pose(p=pe_pose[:3], q=pe_pose[3:7])
    return reach_pose, start_pose, goal_pose


def _test_action(
    planner: Any,
    future_pose: torch.Tensor,
    initial_robot_qpos: np.ndarray,
    reach_pose: sapien.Pose,
    start_pose: sapien.Pose,
    goal_pose: sapien.Pose,
) -> Tuple[np.ndarray, List[PlanResult], Optional[str]]:
    """Test one push action and return its final virtual robot state."""
    stages = (
        (
            "approach_above_start",
            planner.test_move_to_pose_with_RRTConnect,
            reach_pose,
        ),
        (
            "descend_to_start",
            planner.test_move_to_pose_with_RRTConnect,
            start_pose,
        ),
        ("push_to_end", planner.test_move_to_pose_with_screw, goal_pose),
        ("return_to_start", planner.test_move_to_pose_with_screw, start_pose),
        (
            "retreat_above_start",
            planner.test_move_to_pose_with_RRTConnect,
            reach_pose,
        ),
    )

    robot_qpos = np.asarray(initial_robot_qpos).copy()
    plan_results: List[PlanResult] = []

    for stage_name, plan_function, target_pose in stages:

        next_robot_qpos, plan_result = plan_function(
            target_pose,
            future_pose,
            robot_qpos=robot_qpos,
        )
        plan_results.append(plan_result)

        if np.array_equal(next_robot_qpos, -1):
            return robot_qpos, plan_results, stage_name

        robot_qpos = next_robot_qpos

    return robot_qpos, plan_results, None


def test_push_path(
    *,
    planner: Any,
    point_predictor: Any,
    path_points: np.ndarray,
    start_path_point_index: int,
    test_count: int,
    stop_at_path_point_index: Optional[int] = None,
    initial_robot_qpos: np.ndarray,
    initial_object_pose: torch.Tensor,
    initial_object_pointcloud: np.ndarray,
    error_radius: float,
    push_step_dict: Dict[str, Any],
    robot_theta_y: float,
    visualization_attempt_dir: Path,
    loop_times: int = 1,
    preparation_height: float = 0.4,
    predictor_vis: bool = True,
    initial_object_collision_pointcloud: Optional[np.ndarray] = None,
    fixed_obstacle_pointcloud: Optional[np.ndarray] = None,
    motion_planner_point_radius: float = 0.012,
) -> PushPathTestResult:
    """Generate and test ``test_count`` consecutive actions along a path.

    After a successful action, its final robot configuration and predicted
    object pose become the initial state of the next action. The object point
    cloud is transformed to the predicted pose as well. Reaching an
    intermediate waypoint advances a local virtual path cursor so the
    remaining test budget can validate later waypoints. When
    ``stop_at_path_point_index`` is provided, rollout stops immediately after
    successfully reaching that waypoint and never advances to the following
    waypoint. This cursor is never a real-world path-state update. If any
    action fails, testing stops immediately and ``success`` is set to
    ``False``. Exhausting ``test_count`` successful steps remains a successful
    tested prefix even if the stop target was not reached. When ``loop_times``
    is 3 or 4, the first step uses the two perpendicular prediction poses as a
    paired retry strategy. When collision point-cloud inputs are supplied,
    every virtual step publishes the object cloud at its current virtual pose
    and the real starting scene is restored before this function exits.
    """
    path_points = np.asarray(path_points).copy()
    if path_points.ndim != 2 or path_points.shape[1] != 3:
        raise ValueError("path_points must have shape (N, 3)")
    if test_count < 1:
        raise ValueError("test_count must be at least 1")
    if start_path_point_index < 0:
        raise ValueError("start_path_point_index cannot be negative")

    if start_path_point_index >= len(path_points):
        raise ValueError("start_path_point_index is outside path_points")
    visualization_attempt_dir = Path(visualization_attempt_dir)
    if stop_at_path_point_index is not None:
        if isinstance(stop_at_path_point_index, (bool, np.bool_)) or not (
            isinstance(stop_at_path_point_index, (int, np.integer))
        ):
            raise ValueError("stop_at_path_point_index must be an integer")
        stop_at_path_point_index = int(stop_at_path_point_index)
        if not 0 <= stop_at_path_point_index < len(path_points):
            raise ValueError(
                "stop_at_path_point_index is outside path_points"
            )

    collision_object_provided = (
        initial_object_collision_pointcloud is not None
    )
    fixed_obstacles_provided = fixed_obstacle_pointcloud is not None
    if collision_object_provided != fixed_obstacles_provided:
        raise ValueError(
            "initial_object_collision_pointcloud and "
            "fixed_obstacle_pointcloud must be provided together"
        )
    collision_sync_enabled = (
        collision_object_provided and fixed_obstacles_provided
    )
    if test_count > 1 and not collision_sync_enabled:
        raise ValueError(
            "multi-step push testing requires synchronized object and "
            "fixed-obstacle collision point clouds"
        )

    if isinstance(motion_planner_point_radius, (bool, np.bool_)):
        raise ValueError(
            "motion_planner_point_radius must be finite and positive"
        )
    try:
        motion_planner_point_radius = float(
            motion_planner_point_radius
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "motion_planner_point_radius must be finite and positive"
        ) from exc
    if (
        not np.isfinite(motion_planner_point_radius)
        or motion_planner_point_radius <= 0
    ):
        raise ValueError(
            "motion_planner_point_radius must be finite and positive"
        )

    robot_qpos = np.asarray(initial_robot_qpos).copy()
    object_pose = initial_object_pose.detach().clone().cpu()
    object_pointcloud = np.asarray(initial_object_pointcloud).copy()
    virtual_path_point_index = start_path_point_index
    steps: List[PushStepTestResult] = []
    termination_reason = "test_budget_exhausted"

    virtual_object_collision_pointcloud = None
    fixed_obstacle_pointcloud_copy = None
    real_collision_pointcloud = None
    if collision_sync_enabled:
        virtual_object_collision_pointcloud = (
            _validated_pointcloud_copy(
                initial_object_collision_pointcloud,
                "initial_object_collision_pointcloud",
                allow_empty=False,
            )
        )
        fixed_obstacle_pointcloud_copy = _validated_pointcloud_copy(
            fixed_obstacle_pointcloud,
            "fixed_obstacle_pointcloud",
            allow_empty=True,
        )
        real_collision_pointcloud = (
            _compose_motion_planner_pointcloud(
                virtual_object_collision_pointcloud,
                fixed_obstacle_pointcloud_copy,
            )
        )

    try:
        for step_index in range(test_count):
            step_start_time = time.perf_counter()
            try:
                print(
                    f"Testing virtual push action "
                    f"{step_index + 1}/{test_count}."
                )
                if collision_sync_enabled:
                    virtual_collision_pointcloud = (
                        _compose_motion_planner_pointcloud(
                            virtual_object_collision_pointcloud,
                            fixed_obstacle_pointcloud_copy,
                        )
                    )
                    planner.update_obstacle_pointcloud(
                        virtual_collision_pointcloud,
                        radius=motion_planner_point_radius,
                    )

                step_target_path_point_index = virtual_path_point_index
                future_pose = get_future_pose(
                    path_points=path_points,
                    object_pose=object_pose,
                    path_point_index=step_target_path_point_index,
                    push_step_dict=push_step_dict,
                )
                current_pc, local_future_pose = (
                    get_current_pc_future_pose(
                        object_pointcloud,
                        object_pose,
                        future_pose,
                    )
                )

                point_predictor.vis = predictor_vis
                contact_point, orientation, push_distance = (
                    point_predictor.predict(
                        current_pc,
                        local_future_pose,
                        visualization_dir=(
                            visualization_attempt_dir
                            / f"prediction_{step_index:03d}"
                        ),
                    )
                )


                ps_pose, pe_pose = generate_action(
                    contact_point,
                    orientation,
                    push_distance,
                    object_pose,
                )
                ps_pose, pe_pose = convert_action_use(
                    ps_pose,
                    pe_pose,
                    robot_theta_y,
                )

                reach_pose, start_pose, goal_pose = (
                    _make_robot_poses(
                        ps_pose,
                        pe_pose,
                        preparation_height,
                    )
                )

                next_robot_qpos, plan_results, failed_stage = (
                    _test_action(
                        planner=planner,
                        future_pose=future_pose,
                        initial_robot_qpos=robot_qpos,
                        reach_pose=reach_pose,
                        start_pose=start_pose,
                        goal_pose=goal_pose,
                    )
                )

                elapsed_time_2 = (
                    time.perf_counter() - step_start_time
                )
                print(
                    f"Virtual push {step_index + 1}/{test_count} "
                    f"Test action runtime: {elapsed_time_2:.3f} s"
                )

                step_result = PushStepTestResult(
                    step_index=step_index,
                    path_point_index=step_target_path_point_index,
                    future_pose=future_pose,
                    local_future_pose=local_future_pose,
                    ps_pose=ps_pose,
                    pe_pose=pe_pose,
                    reach_pose=reach_pose,
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    plan_results=plan_results,
                    failed_stage=failed_stage,
                )
                steps.append(step_result)

                if not step_result.success:
                    return PushPathTestResult(
                        success=False,
                        termination_reason="motion_plan_failed",
                        steps=steps,
                        final_robot_qpos=robot_qpos,
                        final_object_pose=object_pose,
                        final_object_pointcloud=object_pointcloud,
                        final_path_point_index=(
                            virtual_path_point_index
                        ),
                    )

                current_pose_array = (
                    object_pose.detach().cpu().numpy().copy()
                )
                future_pose_array = (
                    future_pose.detach().cpu().numpy().copy()
                )
                object_pointcloud = get_pose2_wld_frame_points(
                    object_pointcloud,
                    current_pose_array.copy(),
                    future_pose_array.copy(),
                )
                if collision_sync_enabled:
                    virtual_object_collision_pointcloud = (
                        get_pose2_wld_frame_points(
                            virtual_object_collision_pointcloud,
                            current_pose_array.copy(),
                            future_pose_array.copy(),
                        )
                    )
                    virtual_object_collision_pointcloud = (
                        np.ascontiguousarray(
                            virtual_object_collision_pointcloud,
                            dtype=np.float64,
                        )
                    )

                robot_qpos = next_robot_qpos
                object_pose = future_pose.detach().clone().cpu()
                if is_path_point_reached(
                    path_points=path_points,
                    object_pose=object_pose,
                    path_point_index=(
                        step_target_path_point_index
                    ),
                    error_radius=error_radius,
                ):
                    if (
                        step_target_path_point_index
                        == len(path_points) - 1
                    ):
                        termination_reason = "path_end_reached"
                        break
                    if (
                        step_target_path_point_index
                        == stop_at_path_point_index
                    ):
                        termination_reason = (
                            "continuation_waypoint_reached"
                        )
                        break
                    virtual_path_point_index = (
                        step_target_path_point_index + 1
                    )
            finally:
                elapsed_time = (
                    time.perf_counter() - step_start_time
                )
                print(
                    f"Virtual push {step_index + 1}/{test_count} "
                    f"total runtime: {elapsed_time:.3f} s"
                )

        return PushPathTestResult(
            success=True,
            termination_reason=termination_reason,
            steps=steps,
            final_robot_qpos=robot_qpos,
            final_object_pose=object_pose,
            final_object_pointcloud=object_pointcloud,
            final_path_point_index=virtual_path_point_index,
        )
    finally:
        if collision_sync_enabled:
            try:
                planner.update_obstacle_pointcloud(
                    real_collision_pointcloud,
                    radius=motion_planner_point_radius,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to restore the real motion-planner "
                    "collision pointcloud after virtual push testing."
                ) from exc


__all__ = [
    "PushPathTestResult",
    "PushStepTestResult",
    "test_push_path",
]
