import numpy as np
import object_planner_py as opp
import torch

from mani_skill.envs.tasks import PushCubeEnv
from artifact_layout import ArtifactLayout
from motion_planner_solver import PandaArmMotionPlanningSolver
from push_motion_test import test_push_path
from sim_utils import (
    First_path_plan,
    build_local_frame,
    evalute_object_falled,
    evaluate_complete_action,
    is_path_point_reached,
    pcd_downsample,
    plan_from_waypoint,
    pointcloud_segmentation_fusion,
    record_finishing_state,
    record_pushed_pose_and_future_pose,
)
from predictor.contact_predictor import contact_predictor
from transforms3d.euler import quat2euler
import trimesh 
from pose_estimate.pose_estimator import Pose_Estimator


COLLISION_MARGIN_LIST = (0.05, 0.02, 0.01)
MOTION_PLANNER_OBJECT_POINT_COUNT = 500
MOTION_PLANNER_POINT_RADIUS = 0.012


def object_pose_to_config(object_pose):
    """Convert a finite world-frame object pose to the planner's SE(2) config."""
    if isinstance(object_pose, torch.Tensor):
        pose = object_pose.detach().cpu().numpy().copy()
    else:
        pose = np.asarray(object_pose, dtype=np.float64).copy()

    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(
            "object_pose must be a finite world-frame pose with shape (7,)"
        )

    yaw = quat2euler(pose[3:7])[2] % (2 * np.pi)
    return opp.Config(float(pose[0]), float(pose[1]), float(yaw))


def remember_failed_pose(failed_poses, failed_pose, max_count=5):
    """Remember a distinct failed XY region while bounding context size."""
    if max_count < 1:
        raise ValueError("max_count must be at least 1")

    for known in failed_poses:
        dxy = np.hypot(failed_pose.x - known.x, failed_pose.y - known.y)
        if dxy < 0.015:
            return False

    failed_poses.append(failed_pose)
    if len(failed_poses) > max_count:
        del failed_poses[:-max_count]
    return True


def remember_failed_transition(
    failed_transitions,
    failed_transition,
    max_count=5,
):
    """Remember a distinct ordered XY transition with bounded history."""
    if max_count < 1:
        raise ValueError("max_count must be at least 1")

    start = failed_transition.start
    goal = failed_transition.goal
    if np.hypot(goal.x - start.x, goal.y - start.y) <= 1e-9:
        return False

    for known in failed_transitions:
        start_distance = np.hypot(
            start.x - known.start.x,
            start.y - known.start.y,
        )
        goal_distance = np.hypot(
            goal.x - known.goal.x,
            goal.y - known.goal.y,
        )
        if start_distance < 0.015 and goal_distance < 0.015:
            return False

    failed_transitions.append(failed_transition)
    if len(failed_transitions) > max_count:
        del failed_transitions[:-max_count]
    return True


def solve(
    env: PushCubeEnv,
    seed=None,
    debug=False,
    vis=False,
    artifact_layout: ArtifactLayout = None,
    push_test_count: int = 20,
):
    if artifact_layout is None:
        raise ValueError("artifact_layout is required for test artifact routing")
    env.reset(seed=seed)

    # Motion planner
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=True,
        print_env_info=False,
        joint_vel_limits=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        joint_acc_limits = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    )

    # Sample obstacle pointcloud and add them to motion planner for obstacle avoidance
    obstacle_index = env.unwrapped.obstacle_index
    obstacle_pointcloud = []
    for index in obstacle_index:
        obstacle_mesh = env.unwrapped.obstacle_dic[index].get_collision_meshes()[0]
        pointcloud,_ = trimesh.sample.sample_surface(obstacle_mesh, count = 500)
        obstacle_pointcloud.append(pointcloud)
    # Generate table pointcloud
    x_range = (-0.5, 0.5)
    y_range = (-0.5, 0.5)
    spacing = 0.01
    x_points = np.arange(x_range[0], x_range[1] + spacing/2, spacing)
    y_points = np.arange(y_range[0], y_range[1] + spacing/2, spacing)
    xx, yy = np.meshgrid(x_points, y_points)
    zz = np.zeros_like(xx) - 0.015
    table_pointcloud = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    obstacle_pointcloud.append(table_pointcloud)
    fixed_obstacle_pointcloud = np.ascontiguousarray(
        np.concatenate(obstacle_pointcloud, axis=0),
        dtype=np.float64,
    )

    traj_id = env.traj_id 
    env = env.unwrapped
    
    # Define a contact point predictor
    point_predictor = contact_predictor()

    # Hide the sub-goal and target visuals.
    env.goal_region.hide_visual()
    planner.grasp_pose_visual.hide_visual()
    planner.object_pe_pose_visual.hide_visual()
    # Refresh sensor and human-render camera state.
    env.scene.update_render(update_sensors=True, update_human_render_cameras=True)

    # Close the robotic grippers
    planner.close_gripper()


    # Get the single push parameters
    push_step_dict = env.push_step_dict

    # Get the object pointcloud for the first loop
    camera_name_list = ["base_camera_1","base_camera_2","base_camera_3"]
    object_wrld_frame_pcd = pointcloud_segmentation_fusion(env,camera_name_list)

    # Record the times of pushing
    itr_plan = 0
    # Object-path state is kept for the whole trial.
    path_points = None
    path_planner = None
    path_point_index = 1
    need_object_replan = True
    failed_poses = []
    failed_transitions = []
    pending_waypoint_index = None

    # Stop virtual validation after reaching the continuation waypoint.
    STOP_VIRTUAL_TEST_AT_CONTINUATION_WAYPOINT = True

    MAX_PATH_TEST_RETRY = 4
    MAX_FAILURE_RECORDS = 5
    PREPARATION_HEIGHT = 0.4

    res = -1
    success_judge = torch.tensor([False],dtype=torch.bool)
    path_plan_state = 'Normal'
    fall_state = 'Normal'
    RRT_error_judge = False
    segmentation_error = False
    # Allow at most 25 real pushes to reach the goal.
    for _ in range(25):
        itr_plan = itr_plan +1
        print(f"This is the {itr_plan} push.")
        # Set a list to record the motion-planner state
        planner_state = []

        # Get the original pose for falling evaluation
        original_pose = env.obj.pose.raw_pose[0].clone()

        # Update the motionplanner pointcloud
        initial_object_collision_pointcloud = pcd_downsample(
            points=object_wrld_frame_pcd.copy(),
            num_samples=MOTION_PLANNER_OBJECT_POINT_COUNT,
        )
        real_collision_pointcloud = np.ascontiguousarray(
            np.vstack(
                (
                    initial_object_collision_pointcloud,
                    fixed_obstacle_pointcloud,
                )
            ),
            dtype=np.float64,
        )
        planner.update_obstacle_pointcloud(
            real_collision_pointcloud,
            radius=MOTION_PLANNER_POINT_RADIUS,
        )

        # Construct local frame
        local_frame_pose = build_local_frame(object_wrld_frame_pcd.copy())

        if itr_plan == 1:
            # Define a object pose estimator
            pose_estimator = Pose_Estimator(source_pcd_wld = object_wrld_frame_pcd, ori_pose = original_pose.clone().cpu().numpy(),
                                            original_PCA_frame_pose = local_frame_pose)
            object_pose = original_pose.clone()
            object_pose[2] = 0
        else:
            object_pose = pose_estimator.estimate_object_pose(now_pointcloud = object_wrld_frame_pcd,local_frame_pose = local_frame_pose)
            object_pose = torch.tensor(object_pose)

        # The previous real push is accepted against the first pose estimate
        # built from its post-push observation. Virtual rollout indices never
        # update this real path state.
        if pending_waypoint_index is not None:
            if path_points is None:
                raise RuntimeError(
                    "A pending waypoint requires an existing object path"
                )
            if pending_waypoint_index != path_point_index:
                raise RuntimeError(
                    "Pending waypoint and active waypoint index diverged"
                )

            reached_waypoint = is_path_point_reached(
                path_points=path_points,
                object_pose=object_pose,
                path_point_index=pending_waypoint_index,
                error_radius=env.goal_radius,
            )
            if reached_waypoint:
                print(
                    f"Object reached waypoint {pending_waypoint_index} "
                    "after the previous push. Replanning before the next "
                    "action."
                )
                need_object_replan = True
            else:
                print(
                    f"Object has not reached waypoint "
                    f"{pending_waypoint_index}. Continuing toward it."
                )
            pending_waypoint_index = None

        selected_step = None

        for path_test_attempt in range(MAX_PATH_TEST_RETRY):
            print(
                f"Push path validation attempt "
                f"{path_test_attempt + 1}/{MAX_PATH_TEST_RETRY}."
            )
            object_path_planned_this_attempt = False
            validation_mode = "interpolation_continuation"

            if path_planner is None:
                path_points, path_planner = First_path_plan(
                    env,
                    object_pose.clone(),
                    object_wrld_frame_pcd.copy(),
                    collision_margin_list=COLLISION_MARGIN_LIST,
                    show_failed_configs=True,
                    visulize_result=False,
                )
                object_path_planned_this_attempt = True
                validation_mode = "initial_path"
                path_point_index = 1
                need_object_replan = False
            elif need_object_replan:
                path_planner.set_failure_context(
                    failed_poses=failed_poses,
                    failed_transitions=failed_transitions,
                )
                path_points, path_planner = plan_from_waypoint(
                    env,
                    object_pose.clone(),
                    path_planner,
                    collision_margin_list=COLLISION_MARGIN_LIST,
                    show_failed_configs=True,
                    visulize_result=False,
                )
                object_path_planned_this_attempt = True
                validation_mode = "replanned_path"
                path_point_index = 1
                need_object_replan = False

            # BIT* returns [] when no solution is found. A one-point path has
            # no next waypoint to validate either.
            if path_points is None or len(path_points) < 2:
                need_object_replan = True
                continue

            stop_at_path_point_index = (
                path_point_index
                if (
                    STOP_VIRTUAL_TEST_AT_CONTINUATION_WAYPOINT
                    and not object_path_planned_this_attempt
               )
               else None
            )
            print(
                f"Push validation mode={validation_mode}, "
                f"max_virtual_steps={push_test_count}, "
                f"stop_at_continuation_waypoint="
                f"{STOP_VIRTUAL_TEST_AT_CONTINUATION_WAYPOINT}, "
                f"stop_at_path_point_index={stop_at_path_point_index}, "
                f"path_waypoint_count={len(path_points)}."
            )

            initial_robot_qpos = env.agent.robot.get_qpos().cpu().numpy()[0]
            path_test_result = test_push_path(
                planner=planner,
                point_predictor=point_predictor,
                path_points=path_points,
                start_path_point_index=path_point_index,
                test_count=push_test_count,
                stop_at_path_point_index=stop_at_path_point_index,
                initial_robot_qpos=initial_robot_qpos,
                initial_object_pose=object_pose.clone(),
                initial_object_pointcloud=object_wrld_frame_pcd.copy(),
                initial_object_collision_pointcloud=(
                    initial_object_collision_pointcloud.copy()
                ),
                fixed_obstacle_pointcloud=fixed_obstacle_pointcloud,
                motion_planner_point_radius=MOTION_PLANNER_POINT_RADIUS,
                error_radius=env.goal_radius,
                push_step_dict=push_step_dict,
                robot_theta_y=env.robot_theta_y,
                visualization_attempt_dir=(
                    artifact_layout.visualization_attempt_dir(
                        traj_id=traj_id,
                        push_index=itr_plan,
                        attempt_index=path_test_attempt,
                    )
                ),
                # Always use the path-derived target. The old third/fourth
                # perpendicular-pose fallback is intentionally disabled.
                loop_times=1,
                preparation_height=PREPARATION_HEIGHT,
                predictor_vis=False,
            )
            print(
                f"Virtual validation mode={validation_mode}, "
                f"stop_at={stop_at_path_point_index}, "
                f"tested_steps={len(path_test_result.steps)}, "
                f"max_virtual_steps={push_test_count}, "
                f"reason={path_test_result.termination_reason}."
            )

            if path_test_result.success:
                if not path_test_result.steps:
                    raise RuntimeError(
                        "Successful path test returned no push step"
                    )

                selected_step = path_test_result.steps[0]
                if selected_step.path_point_index != path_point_index:
                    raise RuntimeError(
                        "The first tested push did not target the real "
                        "active waypoint"
                    )
                break

            failed_step = path_test_result.failed_step
            if failed_step is not None:
                # On failure, final_object_pose is q_k: the virtual object's
                # state before the failed q_k -> q_(k+1) transition.
                failed_pose = object_pose_to_config(
                    path_test_result.final_object_pose
                )
                failed_goal = object_pose_to_config(
                    failed_step.future_pose
                )
                failed_transition = opp.FailedTransition(
                    start=failed_pose,
                    goal=failed_goal,
                )
                remembered_new_pose = remember_failed_pose(
                    failed_poses,
                    failed_pose,
                    max_count=MAX_FAILURE_RECORDS,
                )
                remembered_new_transition = remember_failed_transition(
                    failed_transitions,
                    failed_transition,
                    max_count=MAX_FAILURE_RECORDS,
                )
                pose_memory_status = (
                    "remembered new" if remembered_new_pose
                    else "matched existing"
                )
                transition_memory_status = (
                    "remembered new" if remembered_new_transition
                    else "matched existing or zero-length"
                )
                print(
                    f"Path test failed at virtual step "
                    f"{failed_step.step_index}, path point "
                    f"{failed_step.path_point_index}, stage "
                    f"{failed_step.failed_stage}. Failure memory "
                    f"pose={pose_memory_status}, "
                    f"transition={transition_memory_status}. "
                    f"Pose ({failed_pose.x:.4f}, "
                    f"{failed_pose.y:.4f}, "
                    f"{failed_pose.theta:.4f}); transition goal "
                    f"({failed_goal.x:.4f}, {failed_goal.y:.4f}, "
                    f"{failed_goal.theta:.4f})."
                )

            path_planner.set_failure_context(
                failed_poses=failed_poses,
                failed_transitions=failed_transitions,
            )
            need_object_replan = True

        if selected_step is None:
            path_plan_state = 'Error'
            print(
                "No feasible failure-aware object path and push sequence "
                "was found."
            )
            break

        future_pose = selected_step.future_pose
        reach_pose1 = selected_step.reach_pose
        reach_pose2 = selected_step.start_pose
        goal_pose = selected_step.goal_pose
        bi_level_plan_results = selected_step.plan_results
        if len(bi_level_plan_results) != 5:
            raise RuntimeError(
                "A successful push validation must contain five motion plans"
            )
        # Move above ps. The validated plan is a fallback if live replanning
        # from the real robot state fails.
        state = planner.move_to_pose_with_RRTConnect(
            reach_pose1,
            future_pose,
            hide_visual=True,
            bi_level_plan_result=bi_level_plan_results[0],
        )
        planner_state.append(state)


        # Move the robot to the push start pose.
        state = planner.move_to_pose_with_RRTConnect(
            reach_pose2,
            future_pose,
            hide_visual=True,
            bi_level_plan_result=bi_level_plan_results[1],
        )
        planner_state.append(state)
        # Push the object to the end pose.
        result = planner.move_to_pose_with_screw(
            goal_pose,
            future_pose,
            hide_visual=True,
            bi_level_plan_result=bi_level_plan_results[2],
        )
        if isinstance(result, tuple):
            res = result[:-1]
        else:
            res = -1
        planner_state.append(res)
        

        # Hide the sub-goal and target visuals.
        env.goal_region.hide_visual()
        planner.grasp_pose_visual.hide_visual()
        planner.object_pe_pose_visual.hide_visual()
        # Return to the start pose before capturing the next observation.
        state = planner.move_to_pose_with_screw(
            reach_pose2,
            future_pose,
            hide_visual=True,
            bi_level_plan_result=bi_level_plan_results[3],
        )
        planner_state.append(state)
        if not (-1 in planner_state):
            state = planner.move_to_pose_with_RRTConnect(
                reach_pose1,
                future_pose,
                hide_visual=True,
                bi_level_plan_result=bi_level_plan_results[4],
            )
            planner_state.append(state)

        # Record the original object's pointcloud and pose for final evaluation
        if itr_plan == 1:
            source_obj_pcd = object_wrld_frame_pcd.copy()
            source_obj_pose = original_pose.clone().cpu().numpy()

        # Refresh camera state after the push.
        env.scene.update_render(update_sensors=True, update_human_render_cameras=True)
        # Get the object's pointcloud
        camera_name_list = ["base_camera_1","base_camera_2","base_camera_3"]
        object_wrld_frame_pcd = pointcloud_segmentation_fusion(env,camera_name_list)

        # Get the object gt pose for evaluation
        now_pose = env.obj.pose.raw_pose[0].clone()
        fall_state = evalute_object_falled(planner_state,last_pose = original_pose, now_pose = now_pose)

        # Track failures caused while executing an RRT plan.
        if fall_state == 'Conduct RRT planner error':
            RRT_error_judge = True
        if fall_state != 'Normal':
            break

        # Judge the success
        success_judge = evaluate_complete_action(env,source_obj_pcd,source_obj_pose,object_wrld_frame_pcd.copy())
        # Stop when the object reaches the target.
        if success_judge == True:
            print("Success!")
            break

        # Only a normally completed real push creates a waypoint awaiting
        # acceptance from the next post-push pose estimate.
        pending_waypoint_index = selected_step.path_point_index


    print(f"This trial uses {itr_plan} steps.")

    # Record the final object pose and target pose.
    pushed_pose = env.obj.pose.raw_pose[0].clone().cpu().numpy()
    target_pose = env.goal_region.pose.raw_pose.clone().cpu().numpy()[0]
    record_pushed_pose_and_future_pose(
        pushed_pose,
        target_pose,
        output_dir=artifact_layout.pose_traj_dir(traj_id),
    )

    # Record this trajectory finishing status
    record_finishing_state(
        fall_state=fall_state,
        success_state=success_judge,
        itr_plan=itr_plan,
        path_plan_state=path_plan_state,
        segmentation_error=segmentation_error,
        output_dir=artifact_layout.push_traj_dir(traj_id),
    )

    return res, success_judge.item(), itr_plan, RRT_error_judge
