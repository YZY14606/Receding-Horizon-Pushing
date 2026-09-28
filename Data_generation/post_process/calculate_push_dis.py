import os,glob,re
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
import shutil
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import argparse
import trimesh


def delete_trajectory_folder(traj_dir):
    shutil.rmtree(traj_dir)

def delete_distance_data(data_dir):
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

    i = 0
    for folder in all_trajectory_folders:
        local_data_path = os.path.join(folder,"push_distance")
        if os.path.isdir(local_data_path): 
            shutil.rmtree(local_data_path)
            i +=1
    print(f"All push distance filse have been deleted. The number is {i}.")

def delete_transformation_data(data_dir):
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

    i = 0
    for folder in all_trajectory_folders:
        local_data_path = os.path.join(folder,"object_state/object_transformation.npy")
        if os.path.exists(local_data_path): 
            os.remove(local_data_path)
            i +=1
    print(f"All object_transformation filse have been deleted. The number is {i}.")


def calculate_push_distance(folder):
    # Get the contact_tcp_pose files
    path_dir = os.path.join(folder,"contact_tcp_pose")
    files = glob.glob(os.path.join(path_dir, '*contact_tcp_pose.npy'))

    files.sort(key=lambda x: int(re.search(r'(\d+)contact_tcp_pose\.npy', os.path.basename(x)).group(1)))

    if len(files) <2:
        delete_trajectory_folder(folder)
        return {"Failure":f"Fail in processing folder: {folder}."}
    
    first_file = files[0] if files else None
    last_file = files[-1] if files else None

    contact_start_pose = np.load(first_file)
    contact_end_pose = np.load(last_file)

    push_distance = np.linalg.norm(contact_end_pose[0,:3] - contact_start_pose[0,:3])

    # Save the push distance
    save_path = os.path.join(folder,"push_distance","distance.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path,push_distance)

    return {"Success":f"Success in processing folder: {folder}."}


def the_7d_pose_to_matrix(pose):
    transformation = np.eye(4)
    transformation[:3,:3] = quat2mat(pose[3:7])
    transformation[:3,3] = pose[:3]
    return transformation


def calculate_object_transformation(folder):
    # Calculate the pointcloud motion
    object_pose_before = np.load(f"{folder}/object_state/object_pose_before.npy")
    object_pose_after = np.load(f"{folder}/object_state/object_pose_after.npy")
    object_pose_before_matrix = the_7d_pose_to_matrix(object_pose_before)
    object_pose_after_matrix = the_7d_pose_to_matrix(object_pose_after)
    transformation = object_pose_after_matrix @ np.linalg.inv(object_pose_before_matrix)

    the_7d_transformation = np.zeros(7)
    the_7d_transformation[:3] = transformation[:3,3]
    the_7d_transformation[3:7] = mat2quat(transformation[:3,:3])

    # Save the push distance
    save_path = os.path.join(folder,"object_state",'object_transformation')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path,the_7d_transformation)

    # Load the pointcloud before being pushed
    pcd = np.load(f"{folder}/world_obj_pcd/obj_pcd_before.npy")
    transformed_points = trimesh.transformations.transform_points(pcd, transformation)

    # Save the pointcloud after pushing
    save_path = os.path.join(folder,"world_obj_pcd/obj_pcd_after.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path,transformed_points.astype(np.float32))

    return {"Success":f"Success in processing folder: {folder}."}


def main(args):
    print("Begin to calculate the push_distance/object_transformation!")
    data_dir= [args.demo_dir]

    # Delete push_distance files
    delete_distance_data(data_dir)

    # Delete transformation files
    delete_transformation_data(data_dir)


    # Find all trajectory directories.
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
    all_trajectory_folders = [p for p in traj_paths if p.name[5:].isdigit()]
    results = []
    with Pool(processes = args.num_procs) as pool:
        with tqdm(total=len(all_trajectory_folders), desc="Processing push_distance") as pbar:
            for result in pool.imap(calculate_push_distance, all_trajectory_folders):
                results.append(result)
                pbar.update(1)

    num_failures = sum(1 for r in results if "Failure" in r)
    print(f"The number of failing process push distance is: {num_failures}")

    # Find all trajectory directories.
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
    all_trajectory_folders = [p for p in traj_paths if p.name[5:].isdigit()]
    results = []
    with Pool(processes = args.num_procs) as pool:
        with tqdm(total=len(all_trajectory_folders), desc="Processing object_transformation") as pbar:
            for result in pool.imap(calculate_object_transformation, all_trajectory_folders):
                results.append(result)
                pbar.update(1)

    num_failures = sum(1 for r in results if "Failure" in r)
    print(f"The number of failing process object transformation is: {num_failures}")




def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-procs", type=int, default=20, help="Number of processes to use to help parallelize the data process.")
    parser.add_argument("--demo_dir", type=str,
                        default="scene_data",
                        help="Path to original dataset directory")
    return parser.parse_args()

if __name__ == "__main__":

    main(parse_args())
