# Extract each trajectory's current and next point clouds and transform them into the object frame.
import os,glob
import trimesh
import numpy as np
import open3d as o3d
from transforms3d import quaternions
from transforms3d.quaternions import quat2mat, mat2quat
from scipy.spatial.transform import Rotation as R #(x,y,z,w)
from scipy.spatial import ConvexHull
import shutil
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
from scipy.spatial import KDTree
import argparse


# Save a point cloud as a NumPy array.
def save_point_as_numpy(points_np, file_path, file_name):
    all_path = os.path.join(file_path,file_name)
    os.makedirs(os.path.dirname(all_path), exist_ok=True)
    np.save(all_path, points_np)

def transform_to_object_frame(points, obj_position, obj_quaternion):
    """
    Transform points from world frame to object frame

    Args:
        points: Points in world frame (N, 3)
        obj_position: Object position in world frame [x, y, z]
        obj_quaternion: Object orientation in world frame [w, x, y, z]

    Returns:
        Points in object frame (N, 3)
    """
    # Convert the quaternion to (x, y, z, w) order.
    obj_quaternion = np.array([obj_quaternion[1],obj_quaternion[2],obj_quaternion[3],obj_quaternion[0]])
    # Build the world-to-local rotation matrix.
    rot_matrix = R.from_quat(obj_quaternion).as_matrix()  # shape (3, 3)

    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = obj_position[:3]

    # Calculate the relative transform: T_final = inv(T).
    T_final = np.linalg.inv(T)
    points_obj_frame = trimesh.transform_points(points, T_final)  # Apply the transform with trimesh.

    return points_obj_frame

def construct_local_frame(points):
    """Build a valid local-frame pose as XYZ coordinates and a WXYZ quaternion."""
    # Project the points onto the XY plane.
    xy_points = points[:, :2]
    
    # Calculate the convex hull.
    hull = ConvexHull(xy_points)
    hull_vertices = xy_points[hull.vertices]
    
    # Randomly sample a point inside the convex hull using a convex combination.
    if len(hull_vertices) > 0:
        # Generate nonnegative random weights that sum to one.
        weights = np.random.dirichlet(np.ones(len(hull_vertices)))
        # Calculate the weighted-average coordinates.
        x0 = np.dot(weights, hull_vertices[:, 0])
        y0 = np.dot(weights, hull_vertices[:, 1])
    else:  # Handle the degenerate case.
        x0, y0 = np.mean(xy_points, axis=0)
    xyz = np.array([x0, y0, 0.0])


    # Randomly select the x-axis orientation angle.
    theta = np.random.uniform(0, 2*np.pi)
    
    # Construct a quaternion for rotation around the z-axis.
    qw = np.cos(theta/2)
    qz = np.sin(theta/2)
    quaternion = np.array([qw, 0.0, 0.0, qz])  # WXYZ order

    local_frame_pose = np.concatenate([xyz, quaternion])
    
    return local_frame_pose

# Downsample a point cloud.
def fps_downsample(points, num_samples=1024):
    if len(points) >= num_samples:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downsampled_pcd = pcd.farthest_point_down_sample(num_samples)
        points_output = np.asarray(downsampled_pcd.points)
        if len(points_output) < num_samples:
            # Pad with randomly sampled points.
            idx = np.random.choice(len(points_output), num_samples - len(points_output), replace=True)
            points_output = np.concatenate([points_output, points_output[idx]], axis=0)
    else:
        pad_rows = num_samples - len(points)
        # Sample random indices with replacement.
        pad_idx = np.random.choice(len(points), pad_rows, replace=True)
        # Pad the point cloud.
        points_output = np.concatenate([points, points[pad_idx]], axis=0)
    return points_output

def get_target_pose_local_frame(object_pose_gt,future_pose_gt,local_pose):
    """Get the next target pose in the constructed local frame."""
    # Get the object's current pose matrix in the world frame.
    Rotation_current_gt = quat2mat(object_pose_gt[3:7])
    T_current_gt = np.eye(4)
    T_current_gt[:3, :3] = Rotation_current_gt
    T_current_gt[:3, 3] = object_pose_gt[:3]

    # Get the object's next pose matrix in the world frame.
    Rotation_future_gt = quat2mat(future_pose_gt[3:7])
    T_future_gt = np.eye(4)
    T_future_gt[:3, :3] = Rotation_future_gt
    T_future_gt[:3, 3] = future_pose_gt[:3]

    # Get the base-coordinate transform of the object's ground-truth frame.
    T_base_gt = T_future_gt @ np.linalg.inv(T_current_gt) 

    # Get the constructed local frame's pose matrix in the world frame.
    Rotation_current_local = quat2mat(local_pose[3:7])
    T_current_local = np.eye(4)
    T_current_local[:3, :3] = Rotation_current_local
    T_current_local[:3, 3] = local_pose[:3]

    # Apply the base-coordinate transform to the constructed local-frame pose.
    T_future_local = T_base_gt @ T_current_local

    # Calculate the constructed frame's relative transform in local coordinates.
    T_local_transformation = np.linalg.inv(T_current_local) @ T_future_local

    # In the local frame, T_local_transformation is the next pose.
    local_future_pose = np.zeros(7)
    local_future_pose[:3] = T_local_transformation[:3,3]
    local_future_pose[3:7] = mat2quat(T_local_transformation[:3,:3])

    return local_future_pose


# Find the nearest points about the contact points
def find_nearest_points(object_points, contact_points, k=5):
    """
    Find k nearest points on object to each contact point

    Args:
        object_points: Points on the object (N, 3)
        contact_points: Contact points (M, 3)
        k: Number of nearest neighbors to find

    Returns:
        indices: Indices of the nearest points for each contact point (M*k,)
    """
    # Build KD-tree for fast nearest neighbor lookup
    tree = KDTree(object_points)

    # Find k nearest neighbors for each contact point
    _, indices = tree.query(contact_points, k=k)

    # Flatten and get unique indices
    return np.unique(indices.flatten())

def transform_direction_to_local_frame(direction, obj_quaternion):
    """
    Transform a direction vector from world frame to object frame

    Args:
        direction: Direction vector in world frame (3,)
        obj_quaternion: Object orientation in world frame [w, x, y, z]

    Returns:
        Direction vector in object frame (3,)
    """
    # Convert quaternion to rotation matrix
    R_obj_to_wld = quaternions.quat2mat(obj_quaternion)

    # Rotate direction vector
    direction_obj_frame = R_obj_to_wld.T @ direction

    return direction_obj_frame

def convert_save_local_pcd(current_pc_world,object_pose_gt,future_pose_gt,local_pose,world_push_dis,world_push_dir,folder,aug_time):
    """
    Transform the current world-frame point cloud into a constructed local frame
    and calculate the transform to the next state.
    """
    # Get the contact points
    contact_points_dir = os.path.join(folder,"contact_points")
    contact_files = sorted(glob.glob(os.path.join(contact_points_dir, "*contact_points.ply")),
                    key=lambda path: int("".join(filter(str.isdigit, os.path.basename(path).split("contact_points")[0])))
    )
    # Load contact points
    contact_points_mesh = trimesh.load(contact_files[0])
    contact_points = np.asarray(contact_points_mesh.vertices)

    # Resize the point cloud to 1024 - len(contact_points).
    num_samples = 1024 - len(contact_points)
    current_pc_world_lack_contact = fps_downsample(points = current_pc_world.copy(),num_samples = num_samples)
    current_pc_world = np.concatenate([current_pc_world_lack_contact,contact_points.copy()],axis=0)
    # Ensure that current_pc_world contains 1024 points.
    if len(current_pc_world) != 1024:
        print(f'Error: {folder} pointcloud number is not 1024!')
    
    # Unpack parameters for transforming the point cloud into the local frame.
    position_local = local_pose[:3]
    quaternion_local = local_pose[3:7]

    # Convert point cloud from world frame to object frame
    current_pc_local_frame_64 = transform_to_object_frame(points = current_pc_world, obj_position = position_local, obj_quaternion = quaternion_local)
    current_pc_local_frame = current_pc_local_frame_64.astype(np.float16)

    # Transform the world-frame-push-direction to local-frame
    push_direction_local_frame = transform_direction_to_local_frame(direction = world_push_dir, obj_quaternion = quaternion_local)

    # Get the next target pose in the constructed local frame.
    local_future_pose = get_target_pose_local_frame(object_pose_gt,future_pose_gt,local_pose)

    # Convert contact points to object frame
    contact_points_local_frame = transform_to_object_frame(points = contact_points, obj_position = position_local, obj_quaternion = quaternion_local)
    k_neighbors = 5  # Use a smaller k for more precise contact areas
    contact_indices = find_nearest_points(current_pc_local_frame, contact_points_local_frame, k=k_neighbors)
    # Create one-hot mask for contact points (1 for contact, 0 for non-contact)
    quality = np.zeros(len(current_pc_local_frame), dtype=int)
    for idx in contact_indices:
        if idx < len(current_pc_local_frame):  # Safety check
            quality[idx] = 1

    # Normalize the current_pc_local_frame, world_push_dis, local_future_pose(transformation)
    furthest_distance = np.max(np.linalg.norm(current_pc_local_frame, axis=1))
    current_pc_local_frame_nrd = current_pc_local_frame / furthest_distance

    world_push_dis_nrd = world_push_dis / furthest_distance

    local_future_pose_nrd = local_future_pose.copy()
    local_future_pose_nrd[:3] = local_future_pose[:3] / furthest_distance.item()

    # Directory path and name.
    catalog_name = str(aug_time) + "_data"
    file_path = os.path.join(folder,"local_data",catalog_name)

    # Current point-cloud output path and name.
    file_name = "current_pointcloud_local_normalized.npy"
    save_point_as_numpy(points_np = current_pc_local_frame_nrd, file_path = file_path, file_name = file_name)

    # Future point-cloud output path and name.
    file_name = "local_frame_future_pose_normalized.npy"
    save_point_as_numpy(points_np = local_future_pose_nrd, file_path = file_path, file_name = file_name)

    # Local-frame output path and name.
    file_name = "local_frame_pose.npy"
    save_point_as_numpy(points_np = local_pose, file_path = file_path, file_name = file_name)

    # Save the contact-point quality values.
    file_name = "contact_quality.npy"
    save_point_as_numpy(points_np = quality, file_path = file_path, file_name = file_name)

    # Save the world_push_dis_nrd
    file_name = "push_distance_normalized.npy"
    save_point_as_numpy(points_np = world_push_dis_nrd, file_path = file_path, file_name = file_name)

    # Save the world_push_dis_nrd
    file_name = "push_direction_local_frame.npy"
    save_point_as_numpy(points_np = push_direction_local_frame, file_path = file_path, file_name = file_name)



def random_point_deletion(pointcloud):
    """
    Randomly delete points around a certain point in the point cloud's spatial neighborhood.
    Simulate the segmentation points lack.
    """
    N = pointcloud.shape[0]
    # The lack space number
    lack_center_number = np.random.randint(0, 11)

    if lack_center_number == 0:
        return pointcloud  # Don't delete points
    
    center_indices = np.random.choice(N, size=lack_center_number, replace=False)
    centers = pointcloud[center_indices]

    # Generate the point lack number of every lack center
    delete_counts = np.random.randint(0, 40, size=lack_center_number)

    tree = KDTree(pointcloud)

    # Get the indices of deleted points
    delete_indices = []
    for center, count in zip(centers, delete_counts):
        if count == 0:
            continue
        # Query the nearest count+1 points
        distances, indices = tree.query(center, k=count)
        if count ==1:
            indices = np.array([indices])
        delete_indices.extend(indices.tolist())

    # Delete center point
    delete_indices.extend(center_indices.tolist())
    
    # Avoid deleting a point repeatedly
    delete_indices = np.unique(delete_indices)
    
    # Get the pointcloud after deleting
    mask = np.ones(N, dtype=bool)
    mask[delete_indices] = False
    filtered_pointcloud = pointcloud[mask]

    return filtered_pointcloud


def get_current_pointcloud_transformation(folder):

    # Get the current point cloud in world frame
    current_pc_wprld_path = os.path.join(folder,"world_obj_pcd","obj_pcd_before.npy")

    current_pc_world = np.load(current_pc_wprld_path)
    # Delete some points to simulate the segmentation points lack
    current_pc_world = random_point_deletion(current_pc_world)

    # Load the push distance
    world_push_distance_path = os.path.join(folder,"push_distance","distance.npy")
    world_push_distance = np.load(world_push_distance_path)

    # Load the push direction
    pose_ps_path = glob.glob(os.path.join(folder, "push_action_ps", "*action_ps.npy"))[0]
    pose_pe_path = glob.glob(os.path.join(folder, "push_action_pe", "*action_pe.npy"))[0]
    # Calculate direction vector
    action_pose_ps = np.load(pose_ps_path)[:3]
    action_pose_pe = np.load(pose_pe_path)[:3]
    direction = action_pose_pe - action_pose_ps
    # Normalize if non-zero
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        world_push_direction = direction / norm
    else:
        world_push_direction = np.array([0, 0, 1])


    # Define paths for required object pose files
    object_trajectory_path = os.path.join(folder, "object_state")

    #Get the object_pose path
    object_pose_be_path = os.path.join(object_trajectory_path, "object_pose_before.npy")
    object_pose_af_path = os.path.join(object_trajectory_path, "object_pose_after.npy")

    # Load object trajectory (positions and orientations)
    try:
        object_pose_be = np.load(object_pose_be_path)
        object_pose_af = np.load(object_pose_af_path)
    except Exception as e: 
        print(f"The folder is: {folder}")
        print(f"Load trajectory failure: {e}")

    # Data augmentation five times
    for j in range(5):
        # Get the local frame pose
        local_frame_pose = construct_local_frame(points = current_pc_world)

        # Get and save the current pointcloud and future pointcloud
        convert_save_local_pcd(current_pc_world = current_pc_world ,object_pose_gt = object_pose_be,
                future_pose_gt = object_pose_af, local_pose = local_frame_pose,world_push_dis = world_push_distance,
                world_push_dir = world_push_direction,folder=folder, aug_time =j)
        
    return f"Success in processing folder: {folder}."

def delete_local_data(data_dir):
    # Find all trajectory directories.
    # Root data directory.
    scene_dir = Path(data_dir[0])
    # Recursively find all traj_* directories.
    traj_paths = []
    # Recursively find all non_collision directories.
    for scene in scene_dir.iterdir():
        if scene.is_dir() and scene.name.startswith("object_") and scene.name.replace("object_", "").replace(".", "").isdigit():
            for time_name in scene.iterdir():
                if time_name.is_dir() and "_" in time_name.name and time_name.name.replace("_", "").replace(".", "").isdigit():
                    for traj in time_name.iterdir():
                        if traj.is_dir() and traj.name.startswith("traj_"):
                            traj_paths.append(traj)

    # Keep paths named traj_i, where i is numeric.
    all_trajectory_folders = [p for p in traj_paths if p.name[5:].isdigit()]

    i = 0
    for folder in tqdm(all_trajectory_folders, desc="Delete local data"):
        local_data_path = os.path.join(folder,"local_data")
        if os.path.isdir(local_data_path):  # Delete only when the path is a directory.
            shutil.rmtree(local_data_path)
            i +=1

    
    print(f"All local data files have been deleted. The number is {i}.")


def main(args):
    data_dir= [args.demo_dir]

    # Delete local_data directories.
    delete_local_data(data_dir)

    # Find all trajectory directories.
    # Root data directory.
    scene_dir = Path(data_dir[0])
    traj_paths = []
    # Recursively find all traj_* directories.
    for scene in scene_dir.iterdir():
        if scene.is_dir() and scene.name.startswith("object_") and scene.name.replace("object_", "").replace(".", "").isdigit():
            for time_name in scene.iterdir():
                if time_name.is_dir() and "_" in time_name.name and time_name.name.replace("_", "").replace(".", "").isdigit():
                    for traj in time_name.iterdir():
                        if traj.is_dir() and traj.name.startswith("traj_"):
                            traj_paths.append(traj)
    # Keep paths named traj_i, where i is numeric.
    all_trajectory_folders = [p for p in traj_paths if p.name[5:].isdigit()]

    results = []
    with Pool(processes = args.num_procs) as pool:
        with tqdm(total=len(all_trajectory_folders), desc="Processing traj in make point cloud") as pbar:
            for result in pool.imap(get_current_pointcloud_transformation, all_trajectory_folders):
                results.append(result)
                pbar.update(1)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-procs", type=int, default=20, help="Number of processes to use to help parallelize the data process.")
    parser.add_argument("--demo_dir", type=str,
                        default="/data/home/share/object_centric_policy/train_data_1/scene_data",
                        help="Path to original dataset directory")
    return parser.parse_args()

if __name__ == "__main__":

    main(parse_args())
