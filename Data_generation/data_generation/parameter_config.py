import os
import random

import numpy as np
import yaml
from transforms3d.euler import euler2quat


def random_initialize_para(object_rank):
    """Sample the object and robot parameters consumed by the push environment."""
    config_path = os.path.join("config", "train_object_config.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        object_data = yaml.safe_load(file)

    object_key = f"{object_rank}_object"
    object_config = object_data[object_key]
    object_path = os.path.join("object_model", object_config["relative_path"])
    object_size = object_config["size"]

    scale_min, scale_max = object_config["scale"]
    object_scale = random.uniform(scale_min, scale_max)
    scale = [object_scale] * 3
    pose_q = euler2quat(0, 0, random.uniform(0, 2 * np.pi))

    max_xy_size = max(object_size[0], object_size[1])
    reach_xy_min = max_xy_size * object_scale * 0.01 / 2 + 0.03
    reach_xy_max = max_xy_size * object_scale * 0.01 / 2 + 0.035
    reach_z_min = 0.01
    reach_z_max = object_size[2] * object_scale * 0.01
    reach_delta = np.array(
        [
            -random.uniform(reach_xy_min, reach_xy_max),
            random.uniform(reach_xy_min, reach_xy_max),
            random.uniform(reach_z_min, reach_z_max),
        ]
    )

    reach_delta_above = reach_delta.copy()
    reach_delta_above[2] = 0.45

    push_angle = np.arctan2(reach_delta[1], reach_delta[0]) % (2 * np.pi)
    robot_ee_q = euler2quat(np.pi, object_config["robot_theta_y"], push_angle)

    return {
        "filename": object_path,
        "scale": scale,
        "pose_q": pose_q,
        "q_robot_ee": robot_ee_q,
        "rp_delta1": reach_delta_above,
        "rp_delta2": reach_delta,
    }
