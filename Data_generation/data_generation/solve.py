import numpy as np
import sapien

from mani_skill.envs.tasks import PushCubeEnv
from .motionplanner import PandaArmMotionPlanningSolver
from .utils import (
    get_global_object_pointcloud,
    get_save_object_pose,
    get_save_push_ps_pe,
    pointcloud_get_reach_pe,
)



def solve(env: PushCubeEnv, seed= None, debug = False, vis = False ):
    env.reset(seed=seed)

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    traj_id = env.traj_id 
    time_name = env.time_name
    env = env.unwrapped

    planner.close_gripper(traj_id = traj_id,time_name=time_name)

    # Panda_ee moves to the position above the start pose 0.45m
    reach_pose1 = sapien.Pose(p=env.obj.pose.sp.p + env.all_dic["rp_delta1"], q=env.agent.tcp.pose.sp.q)
    planner.move_to_pose_with_screw(reach_pose1,traj_id = traj_id,time_name=time_name)

    # Update the camera rendering state.
    env.scene.update_render(update_sensors=True, update_human_render_cameras=True)
    # Get and save object pointcloud before push
    camera_name_list = ["base_camera_1","base_camera_2","base_camera_3"]
    get_global_object_pointcloud(env,traj_id ,time_name,camera_name_list,file_name = "obj_pcd_before.npy")

    # Move to the start pose
    start_p = env.obj.pose.sp.p + env.all_dic["rp_delta2"]
    start_p[2] = env.all_dic["rp_delta2"][2] #
    reach_pose2 = sapien.Pose(p=start_p, q=env.agent.tcp.pose.sp.q)
    res_used_for_pcd = planner.move_to_pose_with_screw(reach_pose2 , traj_id = traj_id,time_name=time_name)

    elapsed_steps = res_used_for_pcd[-1]
    # Get and save object pose before push
    get_save_object_pose(scene_dir=env.scene_dir,object_pose = env.obj.pose.raw_pose,traj_id = traj_id, time_name=time_name, file_name = "object_pose_before.npy")

    #Move to the end pose
    reach_2_xyz = pointcloud_get_reach_pe(env.obj.get_collision_meshes()[0],start_p,obj_idx=env.object_index)
    reach_2_xyz[2] = start_p[2]
    goal_pose = sapien.Pose(reach_2_xyz, q=env.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_screw(goal_pose,traj_id = traj_id,time_name=time_name)[:-1]

    #Save the start_pose and end_pose of panda_ee
    get_save_push_ps_pe(scene_dir = env.scene_dir,ps = np.hstack((reach_pose2.p,reach_pose2.q)) , pe = np.hstack((goal_pose.p,goal_pose.q)),
                        time_name = time_name, traj_id = traj_id, itr_simulator = elapsed_steps)

    #Return to pose_2
    planner.move_to_pose_with_screw(reach_pose2,traj_id = traj_id,time_name=time_name)

    #Return to pose_1, wait object stable
    planner.move_to_pose_with_screw(reach_pose1,traj_id = traj_id,time_name=time_name)
    # Get and save object pose after push
    get_save_object_pose(scene_dir=env.scene_dir,object_pose = env.obj.pose.raw_pose,traj_id = traj_id, time_name=time_name, file_name = "object_pose_after.npy")

    return res
