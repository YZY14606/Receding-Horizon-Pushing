
from typing import Union
import numpy as np
import sapien
import torch
import mani_skill.agents.controllers.utils.kinematics as kinematics


from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig, SceneConfig
from parameter_confi import random_initialize_para
from mani_skill import PACKAGE_ASSET_DIR



@register_env("Push-v1", max_episode_steps=500)
class PushEnv(BaseEnv):


    SUPPORTED_ROBOTS = ["panda"]

    # Specify some supported robot types
    agent: Union[Panda]

    # Maximum position error used to determine whether the goal was reached.
    goal_radius = 0.02

    def __init__(self, *args, robot_uids="panda", object_index = 1,object_type = 'train',scene_difficulty = 'scene_01', robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # Store the selected object and scene configuration.
        self.object_index = object_index
        self.object_type = object_type
        self.scene_difficulty = scene_difficulty
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            ),
            scene_config = SceneConfig(contact_offset = 0.001)
        )



    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
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
                fov=np.pi / 2,  # 90-degree field of view.
                near=0.01,  # Near clipping plane.
                far=10,  # Far clipping plane.
            ),

            CameraConfig(
                "base_camera_2",
                pose=pose2,
                width=512,
                height=512,
                fov=np.pi / 2,  # 90-degree field of view.
                near=0.01,  # Near clipping plane.
                far=10,  # Far clipping plane.
            ),

            CameraConfig(
                "base_camera_3",
                pose=pose3,
                width=512,
                height=512,
                fov=np.pi / 2,  # 90-degree field of view.
                near=0.01,  # Near clipping plane.
                far=10,  # Far clipping plane.
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.9, 1.0], [0.0, 0.0, 0.01])
        return CameraConfig(
            "render_camera", pose=pose, width=1920, height=1440, fov=1, near=0.01, far=10,
        )
    

    def _load_agent(self, options: dict):
        # set a reasonable initial pose for the agent that doesn't intersect other objects
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()  # Build the table and ground plane.

        self.all_dic = random_initialize_para(object_rank = self.object_index,object_type = self.object_type, scene_difficulty = self.scene_difficulty)  # Load randomized scene parameters.

        self.push_step_dict = self.all_dic['push_step_dict']

        # Build the movable object.
        pushed_object = self.scene.create_actor_builder()
        pushed_object.add_multiple_convex_collisions_from_file(
            decomposition = "coacd",
            filename = self.all_dic["filename"],
            scale=self.all_dic["scale"]
        )
        pushed_object.add_visual_from_file(
            filename = self.all_dic["filename"],
            scale=self.all_dic["scale"]
            )
        pushed_object.initial_pose = sapien.Pose(p=[0, 0, 0])
        self.obj = pushed_object.build(name="object")

        self.obj_scale = self.all_dic["scale"]  # Preserve the imported scale.

        # Build the kinematic target visualization.
        target = self.scene.create_actor_builder()
        target.add_visual_from_file(
            filename = self.all_dic["filename"],
            scale=self.all_dic["scale"]
            )
        target.initial_pose = sapien.Pose(p=[0, 0, 0])
        self.goal_region = target.build_kinematic(name="goal_region")

        # Build static obstacles on the table.
        # Lamp, index 1.
        if 1 in self.all_dic["obstacle_scale"]:
            lamp = self.scene.create_actor_builder()
            lamp.add_multiple_convex_collisions_from_file(
                decomposition = "coacd",
                filename = self.all_dic["obstacle_path"][1],
                scale=self.all_dic["obstacle_scale"][1]
                )
            lamp.add_visual_from_file(
                filename = self.all_dic["obstacle_path"][1],
                scale=self.all_dic["obstacle_scale"][1]
            )
            lamp.initial_pose = sapien.Pose(p=[0, 0, 0])
            self.lamp = lamp.build_static(name="lamp")
        # Wine bottle, index 3.
        if 3 in self.all_dic["obstacle_scale"]:
            wine_bottle = self.scene.create_actor_builder()
            wine_bottle.add_multiple_convex_collisions_from_file(
                decomposition = "coacd",
                filename = self.all_dic["obstacle_path"][3],
                scale=self.all_dic["obstacle_scale"][3]
                )
            wine_bottle.add_visual_from_file(
                filename = self.all_dic["obstacle_path"][3],
                scale=self.all_dic["obstacle_scale"][3]
                )
            wine_bottle.initial_pose = sapien.Pose(p=[0, 0, 0])
            self.wine_bottle = wine_bottle.build_static(name="wine_bottle")
        # Plate, index 4.
        if 4 in self.all_dic["obstacle_scale"]:
            plate = self.scene.create_actor_builder()
            plate.add_multiple_convex_collisions_from_file(
                decomposition = "coacd",
                filename = self.all_dic["obstacle_path"][4],
                scale=self.all_dic["obstacle_scale"][4]
                )
            plate.add_visual_from_file(
                filename = self.all_dic["obstacle_path"][4],
                scale=self.all_dic["obstacle_scale"][4]
                )
            plate.initial_pose = sapien.Pose(p=[0, 0, 0])
            self.plate = plate.build_static(name="plate")
        # Complex shelf, index 5.
        if 5 in self.all_dic["obstacle_scale"]:
            complex_shelf = self.scene.create_actor_builder()
            complex_shelf.add_multiple_convex_collisions_from_file(
                decomposition = "coacd",
                filename = self.all_dic["obstacle_path"][5],
                scale=self.all_dic["obstacle_scale"][5]
                ) 
            complex_shelf.add_visual_from_file(
                filename = self.all_dic["obstacle_path"][5],
                scale=self.all_dic["obstacle_scale"][5]
                )
            complex_shelf.initial_pose = sapien.Pose(p=[0, 0, 0])
            self.complex_shelf = complex_shelf.build_static(name="complex_shelf")
        # toy wardrobe; No.6
        if 6 in self.all_dic["obstacle_scale"]:
            toy_wardrobe = self.scene.create_actor_builder()
            toy_wardrobe.add_multiple_convex_collisions_from_file(
                decomposition = "coacd",
                filename = self.all_dic["obstacle_path"][6],
                scale=self.all_dic["obstacle_scale"][6]
                ) 
            toy_wardrobe.add_visual_from_file(
                filename = self.all_dic["obstacle_path"][6],
                scale=self.all_dic["obstacle_scale"][6]
                )
            toy_wardrobe.initial_pose = sapien.Pose(p=[0, 0, 0])
            self.toy_wardrobe = toy_wardrobe.build_static(name="toy_wardrobe")
        # Clock, index 7.
        if 7 in self.all_dic["obstacle_scale"]:
            clock = self.scene.create_actor_builder()
            clock.add_multiple_convex_collisions_from_file(
                decomposition = "coacd",
                filename = self.all_dic["obstacle_path"][7],
                scale=self.all_dic["obstacle_scale"][7]
                )
            clock.add_visual_from_file(
                filename = self.all_dic["obstacle_path"][7],
                scale=self.all_dic["obstacle_scale"][7]
                )
            clock.initial_pose = sapien.Pose(p=[0, 0, 0])
            self.clock = clock.build_static(name="clock")
        # Toy chair, index 8.
        if 8 in self.all_dic["obstacle_scale"]:
            toy_chair = self.scene.create_actor_builder()
            toy_chair.add_multiple_convex_collisions_from_file(
                decomposition = "coacd",
                filename = self.all_dic["obstacle_path"][8],
                scale=self.all_dic["obstacle_scale"][8]
                )
            toy_chair.add_visual_from_file(
                filename = self.all_dic["obstacle_path"][8],
                scale=self.all_dic["obstacle_scale"][8]
            )
            toy_chair.initial_pose = sapien.Pose(p=[0, 0, 0])
            self.toy_chair = toy_chair.build_static(name="toy_chair")
        # Phone; No.10
        if 10 in self.all_dic["obstacle_scale"]:
            phone = self.scene.create_actor_builder()
            phone.add_multiple_convex_collisions_from_file(
                decomposition = "coacd",
                filename = self.all_dic["obstacle_path"][10],
                scale=self.all_dic["obstacle_scale"][10]
                )
            phone.add_visual_from_file(
                filename = self.all_dic["obstacle_path"][10],
                scale=self.all_dic["obstacle_scale"][10]
            )
            phone.initial_pose = sapien.Pose(p=[0, 0, 0])
            self.phone = phone.build_static(name="phone")
        # Keyboard; No.11
        if 11 in self.all_dic["obstacle_scale"]:
            keyboard = self.scene.create_actor_builder()
            keyboard.add_multiple_convex_collisions_from_file(
                decomposition = "coacd",
                filename = self.all_dic["obstacle_path"][11],
                scale=self.all_dic["obstacle_scale"][11]
                )
            keyboard.add_visual_from_file(
                filename = self.all_dic["obstacle_path"][11],
                scale=self.all_dic["obstacle_scale"][11]
            )
            keyboard.initial_pose = sapien.Pose(p=[0, 0, 0])
            self.keyboard = keyboard.build_static(name="keyboard")


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):

            b = len(env_idx)

            self.table_scene.initialize(env_idx)

            # Initialize the movable object's pose.
            nxyz = torch.zeros((b, 3))
            nxyz = nxyz + self.all_dic["pose_p"]
            nxyz[..., 2] = 0.03
            q = torch.tensor(self.all_dic["pose_q"])
            obj_pose = Pose.create_from_pq(p = nxyz, q=q)
            self.obj.set_pose(obj_pose)
            # Initialize obstacle poses.
            # Lamp, index 1.
            if 1 in self.all_dic["obstacle_scale"]:
                lamp_xyz = self.all_dic["obstacle_pose_p"][1] + torch.tensor([0,0,0.05])
                q = torch.tensor(self.all_dic["obstacle_pose_q"][1])
                lamp_pose = Pose.create_from_pq(p = lamp_xyz, q=q)
                self.lamp.set_pose(lamp_pose)
            # Wine bottle, index 3.
            if 3 in self.all_dic["obstacle_scale"]:
                wine_bottle_xyz = self.all_dic["obstacle_pose_p"][3]
                q = torch.tensor(self.all_dic["obstacle_pose_q"][3])
                wine_bottle_pose = Pose.create_from_pq(p = wine_bottle_xyz, q=q)
                self.wine_bottle.set_pose(wine_bottle_pose)
            # Plate, index 4.
            if 4 in self.all_dic["obstacle_scale"]:
                plate_xyz = self.all_dic["obstacle_pose_p"][4]
                q = torch.tensor(self.all_dic["obstacle_pose_q"][4])
                plate_pose = Pose.create_from_pq(p = plate_xyz, q=q)
                self.plate.set_pose(plate_pose)
            # Complex shelf, index 5.
            if 5 in self.all_dic["obstacle_scale"]:
                complex_shelf_xyz = self.all_dic["obstacle_pose_p"][5]
                q = torch.tensor(self.all_dic["obstacle_pose_q"][5])
                complex_shelf_pose = Pose.create_from_pq(p = complex_shelf_xyz, q=q)
                self.complex_shelf.set_pose(complex_shelf_pose)
            # Set toy wardrobe; No.6
            if 6 in self.all_dic["obstacle_scale"]:
                toy_wardrobe_xyz = self.all_dic["obstacle_pose_p"][6]
                q = torch.tensor(self.all_dic["obstacle_pose_q"][6])
                toy_wardrobe_pose = Pose.create_from_pq(p = toy_wardrobe_xyz, q=q)
                self.toy_wardrobe.set_pose(toy_wardrobe_pose)
            # Clock with tray, index 7.
            if 7 in self.all_dic["obstacle_scale"]:
                clock_xyz = self.all_dic["obstacle_pose_p"][7]
                q = torch.tensor(self.all_dic["obstacle_pose_q"][7])
                clock_pose = Pose.create_from_pq(p = clock_xyz, q=q)
                self.clock.set_pose(clock_pose)
            # Toy chair, index 8.
            if 8 in self.all_dic["obstacle_scale"]:
                toy_chair_xyz = self.all_dic["obstacle_pose_p"][8]
                q = torch.tensor(self.all_dic["obstacle_pose_q"][8])
                toy_chair_pose = Pose.create_from_pq(p = toy_chair_xyz, q=q)
                self.toy_chair.set_pose(toy_chair_pose)
            # Phone; No.10
            if 10 in self.all_dic["obstacle_scale"]:
                phone_xyz = self.all_dic["obstacle_pose_p"][10]
                q = torch.tensor(self.all_dic["obstacle_pose_q"][10])
                phone_pose = Pose.create_from_pq(p = phone_xyz, q=q)
                self.phone.set_pose(phone_pose)
            # Keyboard; No.11
            if 11 in self.all_dic["obstacle_scale"]:
                keyboard_xyz = self.all_dic["obstacle_pose_p"][11]
                q = torch.tensor(self.all_dic["obstacle_pose_q"][11])
                keyboard_pose = Pose.create_from_pq(p = keyboard_xyz, q=q)
                self.keyboard.set_pose(keyboard_pose)


            # # Record the obstacles' indices
            self.obstacle_index = list(self.all_dic["obstacle_scale"].keys())
            # Record the obstacles by a dictionary
            self.obstacle_dic = dict()
            self.obstacle_dic[1] = self.lamp if 1 in self.obstacle_index else None
            self.obstacle_dic[3] = self.wine_bottle if 3 in self.obstacle_index else None
            self.obstacle_dic[4] = self.plate if 4 in self.obstacle_index else None
            self.obstacle_dic[5] = self.complex_shelf if 5 in self.obstacle_index else None
            self.obstacle_dic[6] = self.toy_wardrobe if 6 in self.obstacle_index else None
            self.obstacle_dic[7] = self.clock if 7 in self.obstacle_index else None
            self.obstacle_dic[8] = self.toy_chair if 8 in self.obstacle_index else None
            self.obstacle_dic[10] = self.phone if 10 in self.obstacle_index else None
            self.obstacle_dic[11] = self.keyboard if 11 in self.obstacle_index else None


            # Initialize the target pose.
            goal_pose = self.all_dic["goal_pose"]
            xyz_shape = torch.zeros((b, 3))
            xyz = goal_pose[:3]
            target_region_xyz = xyz_shape + xyz
            traget_q = goal_pose[3:7]

            # Set the robot-ee-theta-y
            self.robot_theta_y = self.all_dic["robot_theta_y"]

            # Set the target position and orientation used for relocation.
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p= target_region_xyz,
                    q= traget_q,
                )
            )
            panda_kinematic = kinematics.Kinematics(urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v2.urdf",
                                end_link_name = "panda_hand_tcp",
                                articulation = self.agent.robot,
                                active_joint_indices = torch.tensor([0,1,2,3,4,5,6])
            )
            arm_target_qpos = panda_kinematic.compute_ik(
                target_pose= Pose.create_from_pq(p=np.array([0.4,0,0.4]), q=self.all_dic["q_robot_ee"]),
                q0=self.agent.robot.get_qpos(),
            )

            gipper_qpose = torch.zeros(b, 2) - 1
            new_tensor = torch.cat((arm_target_qpos, gipper_qpose),dim=1)

            self.agent.robot.set_qpos(new_tensor)




    def evaluate(self):
        is_obj_placed = torch.tensor([False])
        return {"success": is_obj_placed}
