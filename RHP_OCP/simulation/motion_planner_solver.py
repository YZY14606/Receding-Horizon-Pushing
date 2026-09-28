"""Panda motion-planning utilities used by the pushing solver."""

from motion_planner.planner import Planner
import numpy as np
import sapien
import torch
from transforms3d import quaternions

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose
from sim_utils import judge_contact_point

OPEN = 1
CLOSED = -1


class PandaArmMotionPlanningSolver:
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose(base_pose)
        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode
        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = OPEN
        self.grasp_pose_visual = True
        self.object_pe_pose_visual = True

        if self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_panda_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.object_pe_pose_visual = build_object_pe_pose_visual(self.base_env,self.base_env.scene)
            
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp.pose)
            self.object_pe_pose_visual.set_pose(self.base_env.obj.pose.raw_pose[0])
            self.grasp_pose_visual.hide_visual()
            self.object_pe_pose_visual.hide_visual()

        self.elapsed_steps = 0  # Count executed simulation steps.

    def update_obstacle_pointcloud(self,pointcloud,radius = 0.012):
        self.planner.update_point_cloud(pointcloud,radius)

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner

    def follow_path(self, result, contact_judge=False, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])

            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1

            # Judge contact between robot and object when using RRT planner
            if contact_judge:
                truncated = judge_contact_point(env = self.env)
                if truncated and (not torch.is_tensor(truncated)):
                    return obs, reward, terminated, truncated, info ,self.elapsed_steps
            
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info ,self.elapsed_steps

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, Mt_1_pose, hide_visual=True, use_pointcloud=True, dry_run: bool = False, refine_steps: int = 0,
        bi_level_plan_result = None
    ):
        pose = to_sapien_pose(pose)
        # try plan one time before giving up
        if self.grasp_pose_visual is not None and (not hide_visual):
            self.grasp_pose_visual.set_pose(pose)
            self.grasp_pose_visual.show_visual()
            Mt_1_pose[2] = self.base_env.obj.pose.raw_pose[0].clone().cpu()[2]
            Mt_1_pose = to_sapien_pose(Mt_1_pose)
            self.object_pe_pose_visual.set_pose(Mt_1_pose)
            self.object_pe_pose_visual.show_visual()
        pose = sapien.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=use_pointcloud,
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            if bi_level_plan_result != None:
                return self.follow_path(result=bi_level_plan_result, contact_judge=True, refine_steps=refine_steps)
            else:
                return -1
        self.render_wait()
        if dry_run:
            return result
        
        return self.follow_path(result, contact_judge=True, refine_steps=refine_steps)



    def move_to_pose_with_screw(
        self, pose: sapien.Pose, Mt_1_pose, hide_visual=True, dry_run: bool = False, refine_steps: int = 0,
        bi_level_plan_result = None,
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        if self.grasp_pose_visual is not None and (not hide_visual):
            self.grasp_pose_visual.set_pose(pose)
            self.grasp_pose_visual.show_visual()
            Mt_1_pose[2] = self.base_env.obj.pose.raw_pose[0].clone().cpu()[2]
            Mt_1_pose = to_sapien_pose(Mt_1_pose)
            self.object_pe_pose_visual.set_pose(Mt_1_pose)
            self.object_pe_pose_visual.show_visual()
        pose = sapien.Pose(p=pose.p , q=pose.q)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=False,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                use_point_cloud=False,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return self.follow_path(result=bi_level_plan_result, refine_steps=refine_steps)
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def close_gripper(self, t=6, gripper_state=CLOSED):
        self.gripper_state = gripper_state
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for _ in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])

            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1


            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info , self.elapsed_steps

    # Planner for test
    def test_move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose,Mt_1_pose,robot_qpos,use_pointcloud=True,
    ):
        pose = to_sapien_pose(pose)
        Mt_1_pose = to_sapien_pose(Mt_1_pose)
        # try plan one time before giving up
        pose = sapien.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            robot_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=use_pointcloud,
            wrt_world=True,
        )
        # When Test the motion-planner 
        if result["status"] != "Success":
            print('In test, RRTConnect plan cannot find a solution.')
            return -1,result
        else:
            robot_qpos = result["position"][-1,:]
            robot_qpos = np.append(robot_qpos, [0, 0])
            return robot_qpos ,result

    def test_move_to_pose_with_screw(
        self, pose: sapien.Pose, Mt_1_pose,robot_qpos
    ):
        pose = to_sapien_pose(pose)
        Mt_1_pose = to_sapien_pose(Mt_1_pose)
        # try screw two times before giving up
        pose = sapien.Pose(p=pose.p , q=pose.q)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            robot_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=False,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                robot_qpos,
                time_step=self.base_env.control_timestep,
                use_point_cloud=False,
            )

        # When Test the motion-planner 
        if result["status"] != "Success":
            return -1,result
        else:
            robot_qpos = result["position"][-1,:]
            robot_qpos = np.append(robot_qpos, [0, 0])
            return robot_qpos,result

def build_panda_gripper_grasp_pose_visual(scene: ManiSkillScene):
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    builder.initial_pose = sapien.Pose(p=[0, 0, 0])
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")

    return grasp_pose_visual


def build_object_pe_pose_visual(base_env,scene: ManiSkillScene):
    pe_vis = scene.create_actor_builder()
    pe_vis.add_visual_from_file(
            filename = base_env.all_dic["filename"],
            scale=base_env.all_dic["scale"],
            )
    pe_vis.initial_pose = sapien.Pose(p=[0, 0, 0])
    object_pose_pe = pe_vis.build_kinematic(name="pe_vis")

    return object_pose_pe
