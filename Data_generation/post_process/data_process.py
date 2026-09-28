import numpy as np
import os
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
from multiprocessing import Pool



def delete_trajectory_folder(traj_dir):
    shutil.rmtree(traj_dir)



def filter_single_traj(traj_dir):
    # Check whether the object collided with non-gripper robot links.
    other_contact_path = os.path.join(traj_dir,"contact_points", "10001contact_points.ply")
    other_contact_exsit = os.path.exists(other_contact_path)

    if other_contact_exsit:
        delete_trajectory_folder(traj_dir)
        return 'success'

    # Remove the trajectory if the object state did not change after the push.
    target_path = os.path.join(traj_dir, "object_state")
    object_be_state = np.load(os.path.join(target_path,"object_pose_before.npy"))
    object_af_state = np.load(os.path.join(target_path,"object_pose_after.npy"))
    if np.allclose(object_be_state, object_af_state, atol=0.000005):
        delete_trajectory_folder(traj_dir)
        return 'success'

    return 'success'


def main(args):
    # Filter generated data and remove trajectories whose object state did not change.

    # First remove trajectories that contain no valid contact.
    # Root data directory.
    scene_dir = Path(args.demo_dir)
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
    traj_dirs1 = [p for p in traj_paths if p.name[5:].isdigit()]
    # Sort paths by their numeric trajectory index.
    traj_dirs1.sort(key=lambda x: int(os.path.basename(x).split('_')[-1]))
    # Combine both directory lists.
    all_traj_dirs = traj_dirs1
    print("Begin to process data type. The file length is:",len(traj_dirs1))


    with Pool(processes = args.num_procs) as pool:
        with tqdm(total=len(all_traj_dirs), desc="Filter trajectories") as pbar:
            for result in pool.imap(filter_single_traj, all_traj_dirs):
                
                if result != 'success':
                    print('Data filter failed!')

                pbar.update(1)

            



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument("--num-procs", type=int, default=20, help="Number of processes to use to help parallelize the data process.")
    parser.add_argument("--demo_dir", type=str,
                        default="scene_data",
                        help="Path to original dataset directory")
    args = parser.parse_args()
    main(args)






