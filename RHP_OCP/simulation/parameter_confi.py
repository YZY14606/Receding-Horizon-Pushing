import random
import numpy as np
import torch
from transforms3d.euler import euler2quat  # (w, x, y, z)
from pathlib import Path
import yaml
import os


# This module centralizes randomized parameters used during simulation.
def sample_position_in_circle(center_xyz, random_radius):
    # The square root produces a uniform distribution over the circle's area.
    distance = random_radius * np.sqrt(random.random())
    angle = random.uniform(0, 2 * np.pi)

    return torch.tensor([
        center_xyz[0] + distance * np.cos(angle),
        center_xyz[1] + distance * np.sin(angle),
        center_xyz[2],  # Preserve the z coordinate from the YAML file.
    ])

def random_initialize_para(object_rank,object_type,scene_difficulty):

    # Configure object randomization.
    # Resolve the repository root from this source file.
    REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

    # Load the object's .yaml
    if object_type == 'train':
        yaml_name = f'{scene_difficulty}.yaml'
    else:
        yaml_name = 'test_object_config.yaml'

    obj_cfg_path = os.path.join('config', yaml_name)

    with open(obj_cfg_path, 'r', encoding='utf-8') as file:
        object_data = yaml.safe_load(file)

    object_index = str(object_rank) + '_object'
    object_relative_path = object_data[object_index]['relative_path']
    object_path = os.path.join(REPOSITORY_ROOT,object_relative_path)

    object_size = object_data[object_index]['size']

    # Get push parameters
    single_push_len = object_data[object_index]['single_step_len']
    single_push_theta = object_data[object_index]['single_step_theta']
    push_step_dict = dict()
    push_step_dict['single_push_len'] = single_push_len
    push_step_dict['single_push_theta'] = single_push_theta

    # Get the object test scale
    object_scale = np.array(object_data[object_index]['test_info']['test_scale'])

    # Object importing path
    object_path_dic = dict([(object_rank, object_path)])
    # Object dimensions [X, Y, Z] in centimeters.
    object_size_dic = dict([(object_rank, object_size)])
    # Scale derived from the object's longest side.
    object_scale_dic = dict([(object_rank, object_scale)])



    # Configure action-sampling randomization.
    # Randomize the Panda end-effector position.
    x = random.uniform(0.2 ,0.3)
    y = random.uniform(-0.2 ,0.3)
    z = 0.4
    robot_ee_p = np.array([x,y,z])
    # Set the end-effector rotation about the z axis.
    robot_ee_q = euler2quat(np.pi ,0,0)

    # Get the theta-y of robot in pushing
    robot_theta_y = object_data[object_index]['robot_theta_y']

    # Configure obstacle parameters.
    # Load the obstacle's .yaml
    obstacle_cfg_path = os.path.join(
        'config',
        f'{scene_difficulty}.yaml'
    )
    with open(obstacle_cfg_path, 'r', encoding='utf-8') as file:
        obstacle_data = yaml.safe_load(file)

    # Set obstacle information
    # 1. Parameter: Obstacles' scale
    obstacle_scale_dic = dict()
    obstacle_pose_p_dic = dict()
    obstacle_pose_q_dic = dict()
    obstacle_path = dict()
    for key in range(1,12):
        obstacle_index = str(key) + '_object'
        if obstacle_data[obstacle_index]['obstacle_info']['is_obstacle']:
            obstacle_scale_dic[key] = list(obstacle_data[obstacle_index]['obstacle_info']['obstacle_scale'])
            # Set obstacle's position
            position = obstacle_data[obstacle_index]['obstacle_info']['position']
            obstacle_size = obstacle_data[obstacle_index]['size']
            if key == 6:
                position[2] = obstacle_scale_dic[key][1] * obstacle_size[1] * 0.01 * 0.5
            else:
                position[2] = obstacle_scale_dic[key][2] * obstacle_size[2] * 0.01 * 0.5       
            obstacle_pose_p_dic[key] = torch.tensor(position)
            # Set obstacle's orientation
            euler = obstacle_data[obstacle_index]['obstacle_info']['orientation']
            obstacle_pose_q_dic[key] = euler2quat(euler[0],euler[1],euler[2])
            # Set the obstacle's path
            obstacle_relative_path = obstacle_data[obstacle_index]['relative_path']
            obstacle_path[key] = os.path.join(REPOSITORY_ROOT,obstacle_relative_path)


    # Read the position-randomization radius.
    random_radius = float(obstacle_data['random_radius'][0])

    # Set the start pose
    # Randomize the initial orientation about the z axis.
    pose_q_dic = object_size_dic.copy()
    for key in object_size_dic.keys():
            theta_z = random.uniform(0 , 2 *np.pi)
            pose_q_dic[key] = euler2quat(0, 0, theta_z)

    # Initialize the object position.
    pose_p_dic = object_size_dic.copy()
    for key in object_size_dic.keys():
        pose_p_dic[key] = sample_position_in_circle(
            obstacle_data['start_pose']['xyz'], random_radius
        )


    # Set the goal pose
    # Sample the goal position around goal_pose.xyz.
    goal_xyz = sample_position_in_circle(obstacle_data['goal_pose']['xyz'], random_radius)

    goal_theta_z = random.uniform(0, 2 * np.pi)
    goal_orientation_quat = torch.tensor(
        euler2quat(0, 0, goal_theta_z)
    )
    goal_pose = torch.cat([goal_xyz, goal_orientation_quat])


    all_dic = dict([("filename", object_path_dic[object_rank]), ("scale", object_scale_dic[object_rank]),("object_size", object_size_dic), ("pose_p", pose_p_dic[object_rank]),
                    ("pose_q",pose_q_dic[object_rank]), ("p_robot_ee", robot_ee_p), ("q_robot_ee", robot_ee_q),
                    ("obstacle_scale",obstacle_scale_dic),("obstacle_pose_p",obstacle_pose_p_dic),("obstacle_pose_q",obstacle_pose_q_dic),
                    ('obstacle_path',obstacle_path),('robot_theta_y',robot_theta_y),('push_step_dict',push_step_dict),
                    ('goal_pose',goal_pose)
                    ])

    return all_dic
