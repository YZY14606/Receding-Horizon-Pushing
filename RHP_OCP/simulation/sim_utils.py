import numpy as np
import torch
import torch.random
import os
from transforms3d import quaternions
from scipy.spatial.transform import Rotation as R #(x,y,z,w)
import trimesh
import open3d as o3d
import math
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from mani_skill.utils.sapien_utils import get_multiple_pairwise_contacts
from transforms3d.euler import quat2euler,euler2quat
from transforms3d.quaternions import quat2mat, mat2quat

from path_planner.planner import Path_planner
import object_planner_py as opp
from segmentation.segmentation import Segmentation_tool, Points_fusion_tool
from chamferdist import ChamferDistance

def transform_to_world_frame(points, obj_position, obj_quaternion):
    """
    Transform points from object frame to world frame

    Args:
        points: Points in object frame (N, 3)
        obj_position: Object position in world frame [x, y, z]
        obj_quaternion: Object orientation in world frame [w, x, y, z]

    Returns:
        Points in world frame (N, 3)
    """
    # Convert quaternion to rotation matrix
    rotation_matrix = quaternions.quat2mat(obj_quaternion)

    # Construt the matrix, convert the points in object frame to world frame
    T_local_to_world = np.eye(4) 
    T_local_to_world[:3, :3] = rotation_matrix  # Rotation
    T_local_to_world[:3, 3] = obj_position      # Translation

    ones = np.ones((points.shape[0], 1))  # shape (N,1)
    points_homogeneous = np.hstack([points, ones])  # shape (N,4)
    points_wld_frame = points_homogeneous @ T_local_to_world.T # @ is matrix multiple

    return points_wld_frame[:,:3]




def generate_action(contact_point, orientation, push_distance,object_pose, push_gap = 0.05):
    """
    Transform action parameters to a complete action.

    Args:
        contact_point: The position of contact point (3).
        orientation: The direction of push (3).
        push_distance: The push distance (1).

    Returns:
        ps: The start position of ee (3).
        pe: The end position of ee (3).
    """
    # Normalize orientation
    orientation_normalized = orientation / np.linalg.norm(orientation)

    # Calculate the pe_position in object frame
    pe_position = contact_point + orientation_normalized * push_distance.item()
    # Calculate the ps_position in object frame
    ps_position = contact_point - orientation_normalized * push_gap
    points_obj_frame =np.stack([ps_position, pe_position], axis=0)

    points_world = transform_to_world_frame(points = points_obj_frame, obj_position = object_pose[:3], obj_quaternion = object_pose[3:7])

    # Get world frame ps and pe
    ps_position_wld = points_world[0]
    pe_position_wld = points_world[1]

    # Rondom q
    q = euler2quat(np.pi, 0, 0)

    # Combine p and q
    ps_pose = np.concatenate([ps_position_wld, q])
    pe_pose = np.concatenate([pe_position_wld, q])

    return ps_pose, pe_pose

def evalute_object_falled(planner_state,last_pose,now_pose):

    # Check the contact point when conducting RRT planner
    for state in planner_state:
        if isinstance(state, tuple):
            if state[3] and (not torch.is_tensor(state[3])):
                print('Error in conductig RRT planner.')
                return 'Conduct RRT planner error'


    if -1 in planner_state:
        print('Error in motion planning.')
        return 'Motion planner error'

    # Firstly, judge the last object state
    quat_last_np = last_pose[3:7].cpu().numpy()
    euler_last_np = [abs(x) for x in quat2euler(quat_last_np)]
    if euler_last_np[0] > np.pi/6  or euler_last_np[1] > np.pi/6:
        print('Error in object falling judging, the falling in last judging is not detected.')
        return 'Code error'
    
    # Secondly, judge the now object state
    quat_now_np = now_pose[3:7].cpu().numpy()
    euler_now_np = [abs(x) for x in quat2euler(quat_now_np)]
    if euler_now_np[0] > np.pi/6  or euler_now_np[1] > np.pi/6:
        print('Object is pushed over in this step.')
        return 'Fall'
    
    return 'Normal'

def evaluate_complete_action(env,source_obj_pcd,source_obj_pose,object_wrld_frame_pcd):
    # Set chamfer distance evaluation
    target_pose = env.goal_region.pose.raw_pose.clone().cpu().numpy()[0]
    expected_pointcloud = get_pose2_wld_frame_points(points_wld_frame_pose1 = source_obj_pcd, object_pose = source_obj_pose,
                                                      future_pose = target_pose)

    cd_loss = ChamferDistance()

    x = torch.tensor(expected_pointcloud, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(object_wrld_frame_pcd, dtype=torch.float32).unsqueeze(0)
    cm_dist = cd_loss(x, y, bidirectional = True, point_reduction = 'mean').item()
    mean_euclidean = np.sqrt(cm_dist / 2)

    is_obj_placed = (mean_euclidean < 0.015)

    return is_obj_placed

def convert_action_use(ps_pose,pe_pose,robot_theta_y):
    delta_pose = pe_pose - ps_pose

    angle = np.arctan2(delta_pose[1],delta_pose[0])
    theta_z = angle % (2 * np.pi)
    robot_ee_q = euler2quat(np.pi ,-robot_theta_y, theta_z)

    # Normalize orientation
    orientation_normalized = delta_pose[:3] / np.linalg.norm(delta_pose[:3])
    # Calculate the ps_position in object frame
    ps_xyz = ps_pose[:3]

    ps_pose[:3] = ps_xyz
    ps_pose[3:7] = np.array(robot_ee_q)

    # Calculate the pe_xy
    pe_xyz = pe_pose[:3] - orientation_normalized * 0.01
    pe_pose[:3] = pe_xyz
    pe_pose[3:7] = ps_pose[3:7]


    # Clamp both poses above the table to prevent excessive penetration.
    if ps_pose[2]<0.02:
        ps_pose[2] = 0.02
    if pe_pose[2]<0.02:
        pe_pose[2] = 0.02

    return ps_pose ,pe_pose


def get_pose2_wld_frame_points(points_wld_frame_pose1, object_pose, future_pose):
    """
    Get future_pc in world frame.

    Args:
        points_wld_frame_pose1: The current_pc in world frame.
        object_pose: Object pose in world frame (7).
        future_pose: Object pose in world frame next time (7).

    Returns:
        points_world_pose2: Future point cloud in world frame (N, 3).
    """
    future_pose[2] = object_pose[2]

    # Build the current object transform in the world frame.
    Rotation_current_gt = quat2mat(object_pose[3:7])
    T_current_gt = np.eye(4)
    T_current_gt[:3, :3] = Rotation_current_gt
    T_current_gt[:3, 3] = object_pose[:3]

    # Build the future object transform in the world frame.
    Rotation_future_gt = quat2mat(future_pose[3:7])
    T_future_gt = np.eye(4)
    T_future_gt[:3, :3] = Rotation_future_gt
    T_future_gt[:3, 3] = future_pose[:3]

    # Compute the current-to-future transform in the world frame.
    T_base_gt = T_future_gt @ np.linalg.inv(T_current_gt) 

    points_world_pose2 = trimesh.transform_points(points_wld_frame_pose1, T_base_gt)
    
    return points_world_pose2

def build_local_frame(points):
    # Down sample
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)
    o3d_pcd = o3d_pcd.voxel_down_sample(0.05)
    points = np.asarray(o3d_pcd.points)

    # Step 1: Center the observed point cloud.
    centroid = np.mean(points, axis=0)
    centroid[2] = 0
    centered = points - centroid
    
    # Step 2: Align the local z axis with the world z axis.
    z_local = np.array([0, 0, 1])
    
    # Step 3: Project onto the xy plane.
    projected = centered[:, :2]
    
    # Step 4: Run two-dimensional PCA.
    cov_2d = np.cov(projected.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    v1 = eigenvectors[:, sorted_indices[0]]
    
    # Step 5: Construct a right-handed local frame.
    x_candidate = np.append(v1, 0)
    x_local = x_candidate / np.linalg.norm(x_candidate)
    
    # Derive the y axis using a cross product.
    y_local = np.cross(z_local, x_local)
    y_local = y_local / np.linalg.norm(y_local)
    
    # Verify the right-handed coordinate system.
    if not np.allclose(np.cross(x_local, y_local), z_local):
        raise ValueError("The local axes do not form a right-handed frame")
    
    # Step 6: Build the rotation matrix.
    R = np.column_stack([x_local, y_local, z_local])

    local_frame_pose = transform_ori_to_pose(Rot = R, centroid = centroid)
    
    return local_frame_pose 

def transform_to_local(points, R, centroid,num_samples=1024):
    # Translate the point cloud to its centroid.
    centered = points - centroid
    # Apply the inverse rotation, which is the transpose here.
    local_points = (R.T @ centered.T).T
    # Downsample when the point count exceeds the requested size.
    if len(local_points) > num_samples:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(local_points)
        downsampled_pcd = pcd.farthest_point_down_sample(num_samples)
        local_points = np.asarray(downsampled_pcd.points)

    return local_points


def get_target_pose_local_frame(object_pose,future_pose):
    """Return the future object pose expressed in the current local frame."""
    # Build the current object transform in the world frame.
    Rotation_current = quat2mat(object_pose[3:7])
    T_current = np.eye(4)
    T_current[:3, :3] = Rotation_current
    T_current[:3, 3] = object_pose[:3]

    # Build the future object transform in the world frame.
    Rotation_future = quat2mat(future_pose[3:7])
    T_future = np.eye(4)
    T_future[:3, :3] = Rotation_future
    T_future[:3, 3] = future_pose[:3]

    # Express the future pose relative to the current object frame.
    T_self = np.linalg.inv(T_current) @ T_future

    # Convert the relative transform to a seven-element pose.
    T_self_transformation = np.zeros(7)
    T_self_transformation[:3] = T_self[:3,3]
    T_self_transformation[3:7] = mat2quat(T_self[:3,:3])

    return T_self_transformation

def get_current_pc_future_pose(object_pcd_wld_frame, object_pose, future_pose):
    """
    Get current_pc and future_pc in object frame. They are uesd for model input.

    Args:
        object_pose: Object pose in world frame (7).
        future_pose: Object pose in world frame next time (7).
        points_num: Sample number from mesh.

    Returns:
        current_pc: Current point cloud in object frame (N, 3).
        local_frame_future_pose: Future pose of object in local frame (7).
    """
    # Transform current point cloud from world frame to object frame
    Rot = quat2mat(object_pose.clone().cpu().numpy()[3:7])
    centroid = object_pose.clone().cpu().numpy()[:3]
    current_pc = transform_to_local(object_pcd_wld_frame, R = Rot, centroid = centroid)

    self_frame_future_pose = get_target_pose_local_frame(object_pose.clone().cpu().numpy(),future_pose.clone().cpu().numpy())

    return current_pc,self_frame_future_pose

def transform_ori_to_pose(Rot, centroid):
    """Combine a rotation matrix and centroid into a wxyz pose."""
    quat_wxyz = mat2quat(Rot)
    position = np.array(centroid)
    pose_7d = np.concatenate([position, quat_wxyz])
    return pose_7d


def interpolatio_compute_next_obj_pose(object_pose,path_point,push_step_dict):
    # Initialize the next object pose.
    next_pose = torch.zeros( 7, dtype = float)

    # Get single push parameter 
    single_push_len = push_step_dict['single_push_len']
    single_push_theta = push_step_dict['single_push_theta']

    # Interpolate the xy translation toward the next path point.
    distance = torch.norm(path_point[:2] - object_pose[:2])
    delta_x = path_point[0]-object_pose[0]
    delta_y = path_point[1]-object_pose[1]
    if distance > single_push_len:
        # Compute the unit translation direction.
        unit_x = delta_x / distance
        unit_y = delta_y / distance
        # Limit translation to one configured push step.
        step_len = single_push_len
        new_delta_x = unit_x * step_len
        new_delta_y = unit_y * step_len
        # Advance by one translation step.
        next_pose[0] = new_delta_x + object_pose[0]
        next_pose[1] = new_delta_y + object_pose[1]
        next_pose[2] = object_pose[2]
    else:
        # Use the remaining translation when it is shorter than one step.
        next_pose[0] = delta_x + object_pose[0]
        next_pose[1] = delta_y + object_pose[1]
        next_pose[2] = object_pose[2]

    # Interpolate rotation about the z axis.
    euler_object = quat2euler(object_pose[3:7])[2]
    target_eule_z = path_point[2]
    # Normalize both angles to [0, 2*pi).
    if euler_object < 0:
        euler_object = math.pi*2 + euler_object
    if target_eule_z <0:
        target_eule_z = math.pi*2 + target_eule_z

    # Limit rotation to one configured angular step.
    threshold = single_push_theta
    theta_diff = torch.abs(euler_object - target_eule_z)
    # Convert the angular difference to [0, pi].
    processed_diff = min(theta_diff, 2 * np.pi - theta_diff)
    if processed_diff > threshold:
        delta_degree = threshold
    elif processed_diff <= threshold:
        delta_degree = theta_diff
    # A negative delta represents a clockwise rotation.
    if (target_eule_z - euler_object) * (math.pi -theta_diff ) <0:
        delta_degree =  -1 * delta_degree
    
    # Apply the interpolated rotation about z.
    rad_z = delta_degree
    euler_obj = list(quat2euler(object_pose[3:7]))
    euler_obj[2] = rad_z + euler_obj[2]
    next_pose[3:7] = torch.tensor(euler2quat(euler_obj[0],euler_obj[1],euler_obj[2]))

    return next_pose

DEFAULT_COLLISION_MARGIN_LIST = (0.05, 0.02, 0.01)
PATH_PLANNER_OBSTACLE_SAMPLE_SPACING = 0.01
PATH_PLANNER_MIN_SURFACE_SAMPLES = 500
PATH_PLANNER_MAX_SURFACE_SAMPLES = 2000
ROBOT_BASE_LINK_NAME = "panda_link0"


def _as_numpy_pose(object_pose, name="object_pose"):
    if isinstance(object_pose, torch.Tensor):
        pose = object_pose.detach().cpu().numpy()
    else:
        pose = np.asarray(object_pose, dtype=np.float64)

    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"{name} must have shape (7,)")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must contain only finite values")

    quaternion_norm = np.linalg.norm(pose[3:7])
    if quaternion_norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} quaternion must have non-zero norm")

    normalized_pose = pose.copy()
    normalized_pose[3:7] /= quaternion_norm
    return normalized_pose


def _validate_collision_margins(collision_margin_list):
    if isinstance(collision_margin_list, (str, bytes)):
        raise ValueError("collision_margin_list must be a non-empty sequence")

    try:
        raw_margins = tuple(collision_margin_list)
    except TypeError as exc:
        raise ValueError(
            "collision_margin_list must be a non-empty sequence"
        ) from exc
    if not raw_margins:
        raise ValueError("collision_margin_list must not be empty")

    margins = []
    for margin in raw_margins:
        if isinstance(margin, (bool, np.bool_)):
            raise ValueError(
                "collision margins must be finite non-negative numbers"
            )
        try:
            margin = float(margin)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "collision margins must be finite non-negative numbers"
            ) from exc
        if not np.isfinite(margin) or margin < 0:
            raise ValueError(
                "collision margins must be finite non-negative numbers"
            )
        margins.append(margin)

    if any(left < right for left, right in zip(margins, margins[1:])):
        raise ValueError(
            "collision_margin_list must be ordered from strict to relaxed"
        )
    return tuple(margins)


def prepare_object_points_for_path_planner(
    object_world_points,
    object_pose,
    max_points=2048,
):
    """Build the raw object template used by the planner's SE(2) checker."""
    points = np.asarray(object_world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("object_world_points must have shape (N, 3)")
    if len(points) == 0:
        raise ValueError("object_world_points must not be empty")
    if not np.all(np.isfinite(points)):
        raise ValueError("object_world_points must contain only finite values")

    if not isinstance(max_points, (int, np.integer)) or isinstance(
        max_points,
        (bool, np.bool_),
    ):
        raise ValueError("max_points must be a positive integer")
    max_points = int(max_points)
    if max_points < 1:
        raise ValueError("max_points must be a positive integer")

    if len(points) > max_points:
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(points)
        object_pcd = object_pcd.farthest_point_down_sample(max_points)
        points = np.asarray(object_pcd.points)

    pose = _as_numpy_pose(object_pose)
    yaw = quat2euler(pose[3:7])[2]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_from_object_xy = np.array(
        [
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw],
        ],
        dtype=np.float64,
    )

    object_points = np.empty_like(points, dtype=np.float64)
    object_points[:, :2] = (
        points[:, :2] - pose[np.newaxis, :2]
    ) @ world_from_object_xy
    # The C++ Config contains only x, y and yaw, so it never adds object z.
    # Preserve world/table-frame height instead of subtracting pose[2].
    object_points[:, 2] = points[:, 2]
    return np.ascontiguousarray(object_points, dtype=np.float64)


def _get_robot_base_collision_mesh(
    env,
    link_name=ROBOT_BASE_LINK_NAME,
):
    """Return the single-environment Panda base collision mesh in world frame."""
    agent = getattr(env, "agent", None)
    robot = getattr(agent, "robot", None)
    if robot is None:
        raise ValueError("env.agent.robot is required to sample robot base")

    try:
        base_link = robot.find_link_by_name(link_name)
    except KeyError as exc:
        raise ValueError(f"robot link {link_name!r} was not found") from exc
    if base_link is None:
        raise ValueError(f"robot link {link_name!r} was not found")

    # ManiSkill Link currently has no public collision-mesh getter. _objs
    # contains the managed PhysX articulation-link components.
    managed_components = tuple(getattr(base_link, "_objs", ()))
    if len(managed_components) != 1:
        raise ValueError(
            "robot base mesh extraction currently requires num_envs=1; "
            f"got {len(managed_components)} managed components"
        )

    base_mesh = get_component_mesh(
        managed_components[0],
        to_world_frame=True,
    )
    if base_mesh is None:
        raise ValueError(
            f"robot link {link_name!r} does not contain collision geometry"
        )
    return base_mesh


def _sample_raw_obstacle_mesh(
    obstacle_mesh,
    point_spacing,
    min_surface_samples,
    max_surface_samples,
    sample_seed,
):
    if obstacle_mesh is None:
        raise ValueError("obstacle collision mesh must not be None")

    vertices = np.asarray(obstacle_mesh.vertices, dtype=np.float64)
    faces = np.asarray(obstacle_mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("obstacle collision mesh must contain vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("obstacle collision mesh must contain triangle faces")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("obstacle collision mesh vertices must be finite")

    surface_area = float(obstacle_mesh.area)
    if not np.isfinite(surface_area) or surface_area <= 0:
        raise ValueError("obstacle collision mesh must have positive area")

    vertices = np.unique(vertices, axis=0)
    if len(vertices) >= max_surface_samples:
        obstacle_pcd = o3d.geometry.PointCloud()
        obstacle_pcd.points = o3d.utility.Vector3dVector(vertices)
        obstacle_pcd = obstacle_pcd.farthest_point_down_sample(
            max_surface_samples
        )
        obstacle_points = np.asarray(obstacle_pcd.points)
    else:
        sample_count = int(math.ceil(surface_area / (point_spacing ** 2)))
        sample_count = max(min_surface_samples, sample_count)
        sample_count = min(
            max_surface_samples - len(vertices),
            sample_count,
        )
        sampled_points, _ = trimesh.sample.sample_surface(
            obstacle_mesh,
            count=sample_count,
            seed=sample_seed,
        )
        # Exact mesh vertices are retained; only the random surface sample is
        # limited by the remaining per-obstacle point budget.
        obstacle_points = np.vstack((vertices, sampled_points))
    if not np.all(np.isfinite(obstacle_points)):
        raise ValueError("sampled obstacle points must be finite")
    return np.ascontiguousarray(obstacle_points, dtype=np.float64)



def _sample_robot_base_collision_points(
    env,
    point_spacing,
    min_surface_samples,
    max_surface_samples,
    sample_seed,
):
    """Sample the fixed Panda-base collision surface in world frame."""
    return _sample_raw_obstacle_mesh(
        obstacle_mesh=_get_robot_base_collision_mesh(env),
        point_spacing=point_spacing,
        min_surface_samples=min_surface_samples,
        max_surface_samples=max_surface_samples,
        sample_seed=sample_seed,
    )


def prepare_obstacle_points_for_path_planner(
    env,
    surface_sample_spacing=PATH_PLANNER_OBSTACLE_SAMPLE_SPACING,
    min_surface_samples=PATH_PLANNER_MIN_SURFACE_SAMPLES,
    max_surface_samples=PATH_PLANNER_MAX_SURFACE_SAMPLES,
    sample_seed=0,
):
    """Sample and merge static world-frame obstacles once for one trial."""
    if isinstance(surface_sample_spacing, (bool, np.bool_)):
        raise ValueError("surface_sample_spacing must be finite and positive")
    surface_sample_spacing = float(surface_sample_spacing)
    if (
        not np.isfinite(surface_sample_spacing)
        or surface_sample_spacing <= 0
    ):
        raise ValueError("surface_sample_spacing must be finite and positive")
    if (
        not isinstance(min_surface_samples, (int, np.integer))
        or isinstance(min_surface_samples, (bool, np.bool_))
        or int(min_surface_samples) < 1
    ):
        raise ValueError("min_surface_samples must be a positive integer")
    if (
        not isinstance(max_surface_samples, (int, np.integer))
        or isinstance(max_surface_samples, (bool, np.bool_))
        or int(max_surface_samples) < int(min_surface_samples)
    ):
        raise ValueError(
            "max_surface_samples must be an integer no smaller than "
            "min_surface_samples"
        )
    if (
        not isinstance(sample_seed, (int, np.integer))
        or isinstance(sample_seed, (bool, np.bool_))
        or int(sample_seed) < 0
    ):
        raise ValueError("sample_seed must be a non-negative integer")

    min_surface_samples = int(min_surface_samples)
    max_surface_samples = int(max_surface_samples)
    sample_seed = int(sample_seed)
    obstacle_indices = list(getattr(env, "obstacle_index", []))
    obstacle_dictionary = getattr(env, "obstacle_dic", {})
    obstacle_clouds = []

    for obstacle_order, obstacle_index in enumerate(obstacle_indices):
        obstacle_actor = obstacle_dictionary.get(obstacle_index)
        if obstacle_actor is None:
            raise ValueError(
                f"obstacle actor {obstacle_index!r} is missing from obstacle_dic"
            )
        obstacle_mesh = obstacle_actor.get_first_collision_mesh(
            to_world_frame=True
        )
        obstacle_clouds.append(
            _sample_raw_obstacle_mesh(
                obstacle_mesh=obstacle_mesh,
                point_spacing=surface_sample_spacing,
                min_surface_samples=min_surface_samples,
                max_surface_samples=max_surface_samples,
                sample_seed=sample_seed + obstacle_order,
            )
        )

    # panda_link0 is fixed, so cache its world-frame collision surface with
    # the other static obstacle point clouds for this trial.
    obstacle_clouds.append(
        _sample_robot_base_collision_points(
            env=env,
            point_spacing=surface_sample_spacing,
            min_surface_samples=min_surface_samples,
            max_surface_samples=max_surface_samples,
            sample_seed=sample_seed + len(obstacle_indices),
        )
    )

    if not obstacle_clouds:
        return np.empty((0, 3), dtype=np.float64)

    return np.ascontiguousarray(
        np.vstack(obstacle_clouds),
        dtype=np.float64,
    )

def _planner_config_from_pose(pose, name):
    pose_array = _as_numpy_pose(pose, name=name)
    yaw = quat2euler(pose_array[3:7])[2] % (2 * np.pi)
    return opp.Config(
        float(pose_array[0]),
        float(pose_array[1]),
        float(yaw),
    )


def _plan_with_collision_margins(
    path_planner,
    scene_dic,
    start_config,
    target_config,
    collision_margin_list,
):
    margins = _validate_collision_margins(collision_margin_list)
    working_scene = dict(scene_dic)
    path = []

    for attempt_index, margin in enumerate(margins, start=1):
        print(
            f"Object path planning with point_inflation={margin:.3f} "
            f"({attempt_index}/{len(margins)})."
        )
        working_scene["point_inflation"] = margin
        path_planner.set_up_planner(working_scene)
        working_scene = dict(path_planner.scene_dic)

        start_in_collision = path_planner.path_planner.is_config_in_collision(
            start_config
        )
        goal_in_collision = path_planner.path_planner.is_config_in_collision(
            target_config
        )
        if start_in_collision or goal_in_collision:
            print(
                f"Skip point_inflation={margin:.3f}: "
                f"start_collision={start_in_collision}, "
                f"goal_collision={goal_in_collision}."
            )
            continue

        path = path_planner.plan_path(
            start_config=start_config,
            goal_config=target_config,
        )
        if path is None or len(path) <= 1:
            path = path_planner.plan_path(
                start_config=start_config,
                goal_config=target_config,
            )

        if path is not None and len(path) > 1:
            print(
                f"Object path found with point_inflation={margin:.3f}."
            )
            break

        print(f"No object path found with point_inflation={margin:.3f}.")
        path = []

    return path


def _path_to_numpy(path):
    return np.asarray(
        [[config.x, config.y, config.theta] for config in path],
        dtype=np.float64,
    ).reshape(-1, 3)


# Plan a collision-free global object path.
def First_path_plan(
    env,
    object_pose,
    object_wrld_frame_pcd,
    collision_margin_list=DEFAULT_COLLISION_MARGIN_LIST,
    visulize_result=True,
    show_failed_configs=True,
):
    margins = _validate_collision_margins(collision_margin_list)
    start_config = _planner_config_from_pose(object_pose, "object_pose")
    target_config = _planner_config_from_pose(
        env.goal_region.pose.raw_pose[0],
        "target_pose",
    )

    object_points = prepare_object_points_for_path_planner(
        object_wrld_frame_pcd,
        object_pose,
    )
    positive_margins = [margin for margin in margins if margin > 0]
    surface_sample_spacing = PATH_PLANNER_OBSTACLE_SAMPLE_SPACING
    if positive_margins:
        surface_sample_spacing = min(
            surface_sample_spacing,
            min(positive_margins),
        )
    obstacle_points = prepare_obstacle_points_for_path_planner(
        env,
        surface_sample_spacing=surface_sample_spacing,
    )

    planner_params = {
        "max_batches": 20,
        "samples_per_batch": 200,
        "collision_check_resolution": 0.01,
        "point_inflation": margins[0],
        "simplify_path": True,
        "simplify_max_relative_cost_increase": 0.0,
        "simplify_absolute_cost_tolerance": 1e-9,
    }
    path_planner = Path_planner(planner_params)
    scene_dic = {
        "object_points": object_points,
        "all_obstacle_points": obstacle_points,
        "map_x_bounds": (-0.6, 0.6),
        "map_y_bounds": (-1, 1),
        "map_theta_bound": (0, 2 * np.pi),
        "point_inflation": margins[0],
    }
    path = _plan_with_collision_margins(
        path_planner=path_planner,
        scene_dic=scene_dic,
        start_config=start_config,
        target_config=target_config,
        collision_margin_list=margins,
    )

    if visulize_result:
        path_planner.visualize_path(
            object_points=path_planner.object_points,
            obstacle_points=path_planner.obstacle_points,
            path=path,
            start_config=start_config,
            goal_config=target_config,
            map_bounds_x=(-0.6, 0.6),
            map_bounds_y=(-1, 1),
            show_failed_configs=show_failed_configs,
        )

    return _path_to_numpy(path), path_planner


def plan_from_waypoint(
    env,
    object_pose,
    path_planner,
    collision_margin_list=DEFAULT_COLLISION_MARGIN_LIST,
    visulize_result=True,
    show_failed_configs=True,
):
    if path_planner is None or not hasattr(path_planner, "scene_dic"):
        raise TypeError(
            "path_planner must be an initialized Path_planner from "
            "First_path_plan"
        )

    start_config = _planner_config_from_pose(object_pose, "object_pose")
    target_config = _planner_config_from_pose(
        env.goal_region.pose.raw_pose[0],
        "target_pose",
    )
    scene_dic = dict(path_planner.scene_dic)
    path = _plan_with_collision_margins(
        path_planner=path_planner,
        scene_dic=scene_dic,
        start_config=start_config,
        target_config=target_config,
        collision_margin_list=collision_margin_list,
    )

    if visulize_result:
        path_planner.visualize_path(
            object_points=path_planner.object_points,
            obstacle_points=path_planner.obstacle_points,
            path=path,
            start_config=start_config,
            goal_config=target_config,
            map_bounds_x=(-0.6, 0.6),
            map_bounds_y=(-1, 1),
            show_failed_configs=show_failed_configs,
        )

    return _path_to_numpy(path), path_planner


def is_path_point_reached(
    path_points,
    object_pose,
    path_point_index,
    error_radius,
):
    """Return whether the object's XY position is inside a waypoint radius."""
    path_points_array = np.asarray(path_points, dtype=np.float64)
    if path_points_array.ndim != 2 or path_points_array.shape[1] != 3:
        raise ValueError("path_points must have shape (N, 3)")
    if not np.all(np.isfinite(path_points_array)):
        raise ValueError("path_points must contain only finite values")
    if not isinstance(path_point_index, (int, np.integer)):
        raise TypeError("path_point_index must be an integer")

    path_point_index = int(path_point_index)
    if path_point_index < 0 or path_point_index >= len(path_points_array):
        raise ValueError("path_point_index is outside path_points")

    error_radius = float(error_radius)
    if not np.isfinite(error_radius) or error_radius <= 0:
        raise ValueError("error_radius must be finite and positive")

    if isinstance(object_pose, torch.Tensor):
        object_pose_array = object_pose.detach().cpu().numpy()
    else:
        object_pose_array = np.asarray(object_pose, dtype=np.float64)
    if object_pose_array.shape != (7,):
        raise ValueError("object_pose must have shape (7,)")
    if not np.all(np.isfinite(object_pose_array)):
        raise ValueError("object_pose must contain only finite values")

    target_xy = path_points_array[path_point_index, :2]
    distance = np.linalg.norm(target_xy - object_pose_array[:2])
    return bool(distance < error_radius)


# Compute one future push pose toward the requested path waypoint.
def get_future_pose(
    path_points,
    object_pose,
    path_point_index,
    push_step_dict,
):
    """Compute one object-motion target toward an explicit path point."""
    path_points_array = np.asarray(path_points, dtype=np.float64)
    if path_points_array.ndim != 2 or path_points_array.shape[1] != 3:
        raise ValueError("path_points must have shape (N, 3)")
    if not np.all(np.isfinite(path_points_array)):
        raise ValueError("path_points must contain only finite values")
    if not isinstance(path_point_index, (int, np.integer)):
        raise TypeError("path_point_index must be an integer")

    path_point_index = int(path_point_index)
    if path_point_index < 0 or path_point_index >= len(path_points_array):
        raise ValueError("path_point_index is outside path_points")
    if not isinstance(object_pose, torch.Tensor):
        raise TypeError("object_pose must be a torch.Tensor")
    if object_pose.shape != (7,):
        raise ValueError("object_pose must have shape (7,)")
    if not bool(torch.all(torch.isfinite(object_pose)).item()):
        raise ValueError("object_pose must contain only finite values")

    goal_point = path_points_array[path_point_index].copy()
    goal_point[2] %= 2 * np.pi
    goal_point_tensor = torch.as_tensor(
        goal_point,
        dtype=object_pose.dtype,
        device=object_pose.device,
    )

    return interpolatio_compute_next_obj_pose(
        object_pose,
        goal_point_tensor,
        push_step_dict,
    )


def generate_segmentation_mask(image_rgb,points,labels):
    """
    image_rgb: shape (H,W,3);
    points: prompt points (N,2);
    labels: the labels of prompt points, shape (N)
    """
    img_seg_tool = Segmentation_tool()
    img_seg_tool.set_image(image_rgb)
    masks = img_seg_tool.predict(points=points,labels = labels) # shape (1,H,W)

    return masks

def select_points_for_segmentation(camera_segmentation_num):
    # Collect pixels labeled as the target object.
    mask_18 = (camera_segmentation_num == 18).squeeze()
    coords_18 = np.argwhere(mask_18)
    if len(coords_18) < 3:
        print("Error: one camera cannot see the object, segmentation fails!")
    # Randomly choose three positive prompts.
    chosen_18 = coords_18[np.random.choice(len(coords_18), size=3, replace=False)]
    object_points = chosen_18[:, [1, 0]]

    # Collect pixels labeled as background.
    mask_not_18 = (camera_segmentation_num == 16).squeeze()
    coords_not_18 = np.argwhere(mask_not_18)
    if len(coords_not_18) < 2:
        print("Error: one camera view is completely occluded by the object!")
    # Randomly choose two negative prompts.
    chosen_not_18 = coords_not_18[np.random.choice(len(coords_not_18), size=2, replace=False)]
    non_object_points = chosen_not_18[:, [1, 0]]

    # fusion
    points = np.vstack([object_points,non_object_points])
    labels = np.array([1,1,1,0,0])

    return points, labels


def pcd_downsample(points, num_samples=1024):

    if len(points) == 0:
        raise ValueError("Cannot downsample an empty point cloud")

    points_output = points.copy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if len(pcd.points) > num_samples:
        downsampled_pcd = pcd.farthest_point_down_sample(num_samples)
        points_output = np.asarray(downsampled_pcd.points)
    if len(points_output) < num_samples:

        idx = np.random.choice(len(points_output), num_samples - len(points_output), replace=True)
        points_output = np.concatenate([points_output, points_output[idx]], axis=0)

    return points_output

def pointcloud_segmentation_fusion(env,camera_name_list):
    all_pointcloud = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcd_fusion_tool =  Points_fusion_tool()
    for camera_name in camera_name_list:
        # Capture the selected camera observation.
        camera = env._sensors[camera_name]
        camera.capture()
        camera_obs = camera.get_obs()
        # Get the rgb data of this camera
        image_rgb = camera_obs["rgb"].clone().cpu().numpy() #shape (1,512,512,3)

        # Read simulator segmentation labels.
        camera_segmentation_num = camera_obs["segmentation"].clone().cpu().numpy() #shape (1,512,512,1)
        # Get the prompt points
        points, labels = select_points_for_segmentation(camera_segmentation_num)
        # Use sam2 to get the mask of target object
        mask_data = generate_segmentation_mask(image_rgb.squeeze(), points, labels)
        segmentation = torch.tensor(mask_data[..., np.newaxis]).to(device)

        # Read the camera-space position image.
        camera_point_cloud = camera_obs["position"].clone().to(device) #shape (1,512,512,3)
        # Read camera calibration parameters.
        camera_para_dic = camera.get_params()
        # Read the camera-to-world transform.
        camera_cam2world_para = camera_para_dic["cam2world_gl"].clone().to(device)
        # Get the camera intrinsic parameter
        camera_intrinsic_para = camera_para_dic["intrinsic_cv"].clone().to(device)
        # Get the camera intrinsic parameter
        camera_extrinsic = camera_para_dic["extrinsic_cv"].clone().view(3,4).to(device)
        fourth_row = torch.tensor([[0, 0, 0, 1]], dtype=camera_extrinsic.dtype, device=device)
        camera_extrinsic_para = torch.cat([camera_extrinsic, fourth_row], dim=0)

        # Add parameters to Points_fusion_tool
        pcd_fusion_tool.extrinsic.append(camera_extrinsic_para.clone().view(4,4))
        pcd_fusion_tool.intrinsic.append(camera_intrinsic_para.view(3,3))
        pcd_fusion_tool.mask.append(segmentation.clone().squeeze().to(device))

        # Reconstruct the three-dimensional point cloud.
        position = camera_point_cloud.float()
        cam2world = camera_cam2world_para.view(1,4,4).float()

        position[..., :3] = (
                position[..., :3] / 1000.0
            )  # convert the raw depth from millimeters to meters

        # Reconstruct only points retained by the segmentation mask.
        xyzw = torch.cat([position, segmentation], dim=-1).reshape(
            position.shape[0], -1, 4) @ cam2world.transpose(1, 2)

        # Retain homogeneous points with nonzero w.
        mask = xyzw[..., 3] != 0
        filtered_xyzw = xyzw[mask]

        # Keep xyz coordinates from the filtered homogeneous points.
        pointcloud = filtered_xyzw[:,:3]
        all_pointcloud.append(pointcloud)
    output_pointcloud = torch.cat(all_pointcloud,dim=0)
    output_pointcloud = pcd_fusion_tool.filter_point(output_pointcloud,threshold=0.2).cpu().numpy()

    output_pointcloud = pcd_downsample(output_pointcloud,num_samples=1024)

    return output_pointcloud


def record_pushed_pose_and_future_pose(pushed_pose, future_pose, output_dir):
    """Save the final pushed pose and the planned target pose."""
    # Create the output directory before saving.
    save_path = os.path.join(os.fspath(output_dir), "two_pose.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Store both poses in one array with shape (2, 7).
    saved_poses = np.vstack((pushed_pose, future_pose))
    np.save(save_path,saved_poses)


def record_finishing_state(
    fall_state,
    success_state,
    itr_plan,
    path_plan_state,
    segmentation_error,
    output_dir,
):
    # Create a dictionary
    data = {}
    data["fall_state"] = fall_state
    data["success_state"] = f"{success_state}"
    data["push_steps"] = f"{itr_plan}"
    data["path_plan_state"] = path_plan_state
    data["segmentation_error"] = segmentation_error
    if (success_state != True) and (itr_plan == 25):
        data['push_length_error'] = 'Error'
    else:
        data['push_length_error'] = 'Normal'

    file_path = os.path.join(os.fspath(output_dir), "log.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, ensure_ascii=False, indent=4)


def judge_contact_point(env):
    # Judge contact
    have_contact = False
    # The pushed object is the first actor in every contact pair.
    actor0 = env.unwrapped.obj
    # Use the underlying SAPIEN entity for contact queries.
    contact_actor0 = env.unwrapped.obj._bodies[0].entity

    link_panda_2 = env.unwrapped.agent.robot.find_link_by_name("panda_link2")
    link_panda_3 = env.unwrapped.agent.robot.find_link_by_name("panda_link3")
    link_panda_4 = env.unwrapped.agent.robot.find_link_by_name("panda_link4")
    link_panda_5 = env.unwrapped.agent.robot.find_link_by_name("panda_link5")
    link_panda_6 = env.unwrapped.agent.robot.find_link_by_name("panda_link6")
    link_panda_7 = env.unwrapped.agent.robot.find_link_by_name("panda_link7")
    link_panda_8 = env.unwrapped.agent.robot.find_link_by_name("panda_link8")
    link_panda_hand = env.unwrapped.agent.robot.find_link_by_name("panda_hand")
    link_panda_hand_tcp = env.unwrapped.agent.robot.find_link_by_name("panda_hand_tcp")#mass=1*e-6
    link_panda_leftfinger = env.unwrapped.agent.robot.find_link_by_name("panda_leftfinger")
    link_panda_rightfinger = env.unwrapped.agent.robot.find_link_by_name("panda_rightfinger")
    link_panda_leftfinger_pad = env.unwrapped.agent.robot.find_link_by_name("panda_leftfinger_pad")#mass=1*e-6
    link_panda_rightfinger_pad = env.unwrapped.agent.robot.find_link_by_name("panda_rightfinger_pad")#mass=1*e-6
    # Robot links used to aggregate contact forces.
    link_actor1_list = [link_panda_hand_tcp,link_panda_leftfinger,link_panda_rightfinger,link_panda_leftfinger_pad,link_panda_rightfinger_pad,
                        link_panda_2,link_panda_3,link_panda_4,link_panda_5,link_panda_6,link_panda_7,link_panda_8 ,link_panda_hand]
    # Underlying link entities used to query contact pairs.
    contact_actor1_list = [link_panda_hand_tcp._bodies[0].entity, link_panda_leftfinger._bodies[0].entity,
                        link_panda_rightfinger._bodies[0].entity, link_panda_leftfinger_pad._bodies[0].entity, link_panda_rightfinger_pad._bodies[0].entity,
                        link_panda_2._bodies[0].entity, link_panda_3._bodies[0].entity, link_panda_4._bodies[0].entity, link_panda_5._bodies[0].entity,
                       link_panda_6._bodies[0].entity, link_panda_7._bodies[0].entity, link_panda_8._bodies[0].entity,link_panda_hand._bodies[0].entity]
    
    # Record robot-object contacts so invalid trajectories can be filtered.
    # get_contacts is expected to be deprecated by the simulator API.
    all_contacts = env.unwrapped.scene.get_contacts()
    obj_robot_ee_all_contact = get_multiple_pairwise_contacts(all_contacts, actor0 = contact_actor0, actor1_list = contact_actor1_list)
    # Aggregate contact force and reject negligible contacts.
    finger_force = 0
    for actor1 in link_actor1_list:
        actor0_actor1_contact_force = env.unwrapped.scene.get_pairwise_contact_forces(actor0, actor1)
        finger_force = finger_force + torch.sum(actor0_actor1_contact_force)
    if len(obj_robot_ee_all_contact) > 0 and abs(finger_force) >0.001 :
        have_contact = True
    
    return have_contact


# Camera observation notes:
# get_obs() returns RGB and position images shaped (H, W, 3), while depth and
# segmentation images are shaped (H, W, 1). Visualization helpers convert all
# outputs to (H, W, 3); position uses its z channel and segmentation is repeated
# across three channels.
