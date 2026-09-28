
from typing import Union
import numpy as np
import sapien
import torch
import mani_skill.agents.controllers.utils.kinematics as kinematics


from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig, SceneConfig
from .parameter_config import random_initialize_para
from mani_skill import PACKAGE_ASSET_DIR



@register_env("Push_solve", max_episode_steps=500)
class PushCupEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda"]

    # Specify some supported robot types
    agent: Union[Panda]

    def __init__(self, *args, robot_uids="panda",scene_dir = None, object_index = 1,control_frequency=10, robot_init_qpos_noise=0.02, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("Push_solve) will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # Store the object index.
        self.object_index = object_index
        self.control_frequency = control_frequency
        self.scene_dir = scene_dir
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq = 90,
            control_freq = self.control_frequency,
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            ),
            scene_config = SceneConfig(contact_offset = 0.001)
        )


    @property
    def _default_sensor_configs(self):
        # a smaller sized camera will be lower quality, but render faster
        pose1 = sapien_utils.look_at(eye=[0.5, 0, 0.5], target=[0, 0.001, 0])
        pose2 = sapien_utils.look_at(eye=[-0.5, 0.6, 0.4], target=[0, 0.001, -0.1])
        pose3 = sapien_utils.look_at(eye=[-0.5, -0.6, 0.4], target=[0, 0.001, -0.1])
        return [
            CameraConfig(
                "base_camera_1",
                pose=pose1,
                width=512,
                height=512,
                fov=np.pi / 2,  # 90-degree field of view
                near=0.01,  # Camera near plane
                far=5,  # Camera far plane
            ),

            CameraConfig(
                "base_camera_2",
                pose=pose2,
                width=512,
                height=512,
                fov=np.pi / 2,  # 90-degree field of view
                near=0.01,  # Camera near plane
                far=5,  # Camera far plane
            ),

            CameraConfig(
                "base_camera_3",
                pose=pose3,
                width=512,
                height=512,
                fov=np.pi / 2,  # 90-degree field of view
                near=0.01,  # Camera near plane
                far=5,  # Camera far plane
            )
        ]



    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (1920x1440) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.9, 1.0], [0.0, 0.0, 0.01])
        return CameraConfig(
            "render_camera", pose=pose, width=1920, height=1440, fov=1, near=0.01, far=10, shader_pack="rt-fast",
        )




    def _load_agent(self, options: dict):
        # set a reasonable initial pose for the agent that doesn't intersect other objects
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()  # Create the table and floor.


        self.all_dic = random_initialize_para(self.object_index)  # Load randomized parameters.

        # Create the object to push.
        object_pushed = self.scene.create_actor_builder()
        object_pushed.add_multiple_convex_collisions_from_file(
            decomposition = "coacd",
            filename = self.all_dic["filename"],
            scale=self.all_dic["scale"]  # For example, scale a model to 50% of its size.
            )
        object_pushed.add_visual_from_file(
            filename = self.all_dic["filename"],
            scale=self.all_dic["scale"]  # For example, scale a model to 50% of its size.
            )
        object_pushed.initial_pose = sapien.Pose(p=[0, 0, 0])
        self.obj = object_pushed.build(name="object")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):

            b = len(env_idx)

            self.table_scene.initialize(env_idx)

            # Set the object's initial pose.
            nxyz = torch.zeros((b, 3))
            q = torch.tensor(self.all_dic["pose_q"])
            nxyz[..., 2] = 0.001
            obj_pose = Pose.create_from_pq(p = nxyz, q=q)
            self.obj.set_pose(obj_pose)

            panda_kinematic = kinematics.Kinematics(urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v2.urdf",
                                end_link_name = "panda_hand_tcp",
                                articulation = self.agent.robot,
                                active_joint_indices = torch.tensor([0,1,2,3,4,5,6])
            )
            arm_target_qpos = panda_kinematic.compute_ik(
                target_pose= Pose.create_from_pq(p=np.array([0.4,0,0.5]), q=self.all_dic["q_robot_ee"]),
                q0=self.agent.robot.get_qpos(),
            )

            gipper_qpose = torch.zeros(b, 2) - 1
            new_tensor = torch.cat((arm_target_qpos, gipper_qpose),dim=1)

            self.agent.robot.set_qpos(new_tensor)




    def evaluate(self):

        is_obj_placed = torch.tensor([False])

        return {"success": is_obj_placed}
