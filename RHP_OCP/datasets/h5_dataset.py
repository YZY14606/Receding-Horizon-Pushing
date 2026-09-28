import os
import torch
import numpy as np
import trimesh
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
from utils.logging_utils import Logger, LogLevel, log_function
from utils.visualization_utils import VisualizationUtils
from scipy.spatial.transform import Rotation as R #(x,y,z,w)
from tqdm import tqdm
import h5py
import re


class ObjectCentricPushDataset(Dataset):
    """
    Object-centric dataset for push sampling training.
    Loads object meshes and trajectories, and dynamically samples point clouds
    with contact points and push directions.
    """

    @log_function(level=LogLevel.INFO)
    def __init__(self, data_dir,mode = 'train',validation_split = 0.1, logger_level= LogLevel.INFO, debug_dir="debug"):
        """
        Args:
            data_dir: Directory containing trajectory data
            mesh_path: Path to the object mesh file
            num_points: Number of points to sample from the mesh
            debug_dir: Directory to save debug visualizations
        """
        self.logger = Logger.get_instance()
        self.logger.set_level(logger_level)
        # If data_dir is string, convert it to list
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        self.data_dir = data_dir
        self.debug_dir = debug_dir

        # Create debug directory if it doesn't exist and we're in debug mode
        if self.logger.logger.level <= LogLevel.DEBUG.value:
            os.makedirs(self.debug_dir, exist_ok=True)

        subdirs = []
        for data_dir_value in self.data_dir:
            data_dir = Path(data_dir_value)
            if not data_dir.is_dir():
                raise NotADirectoryError(f"Dataset directory not found: {data_dir}")
            subdirs.extend(path for path in data_dir.iterdir() if path.is_dir())

        if mode not in {"train", "val"}:
            raise ValueError(f"Unsupported dataset mode: {mode}")

        # Find all trajectory data
        self.trajectories = []
        for subdir in tqdm(subdirs, desc=f"Load {mode} dataset"):
            sub_trajectories = self._load_trajectories(subdir,mode,validation_split)
            self.trajectories.extend(sub_trajectories)

        self.logger.info(f"Loaded {len(self.trajectories)} trajectories in {mode} dataset.")

        # Create flattened list of (trajectory_idx, step_idx) pairs for indexing
        self.samples = len(self.trajectories)


    @log_function(level=LogLevel.DEBUG)
    def _load_trajectories(self,subdir,mode,validation_split):
        """Load all trajectory data"""

        root = Path(subdir)
        traj_dirs = []
        pattern = re.compile(r"traj_(\d+)$")
        for p in root.iterdir():
            if p.is_dir():
                m = pattern.search(p.name)
                if m:
                    idx = int(m.group(1))
                    traj_dirs.append((idx, p))
        # Sort trajectories by their numeric identifier.
        traj_dirs.sort(key=lambda x: x[0])
        all_trajectory_folders = traj_dirs[:]

        if len(all_trajectory_folders) == 0:
            self.logger.error(f"No trajectory folders found in {self.data_dir}")
            # You might want to check if the path exists
            if not os.path.exists(self.data_dir):
                self.logger.error(f"Data directory {self.data_dir} does not exist!")
            return []

        trajectories = []

        # Split each object directory into training and validation subsets.
        number = int((1-validation_split) * len(all_trajectory_folders))
        if mode == 'train':
            mode_trajectory_folders = all_trajectory_folders[:number]
        elif mode == 'val':
            mode_trajectory_folders = all_trajectory_folders[number:]

        for folder_tuple in mode_trajectory_folders:
            folder = folder_tuple[1]
            h5path = os.path.join(folder,'data.h5')
            with h5py.File(h5path, 'r') as h5f:
                try:
                    # Load the local_frame information
                    for i in range(5):
                        # Load push_distance
                        push_distance_factor = np.array(h5f["local_data"][f"{i}_data/push_distance_normalized"][()])

                        # Load push direction
                        direction = h5f["local_data"][f"{i}_data/push_direction_local_frame"][:]

                        # Load future_local_pose
                        future_local_pose = h5f["local_data"][f"{i}_data/local_frame_future_pose_normalized"][:]

                        # Load object current and future pointcloud
                        current_pointcloud = h5f["local_data"][f"{i}_data/current_pointcloud_local_normalized"][:]

                        # Get the contact quality
                        contact_quality = h5f["local_data"][f"{i}_data/contact_quality"][:]

                        # Add trajectory data
                        trajectory = {
                            "folder" :folder,
                            "current_point_cloud": current_pointcloud,
                            "future_local_pose": future_local_pose,
                            "contact_quality": contact_quality,
                            "direction": direction,
                            "push_distance":push_distance_factor,
                        }

                        # Only add if we have at least one contact frame
                        trajectories.append(trajectory)


                except Exception as e:
                    self.logger.error(f"Error loading trajectory from {folder}: {e}")

        return trajectories

    @log_function(level=LogLevel.DEBUG)
    def __len__(self):
        """Return the total number of samples"""
        return self.samples


    @log_function(level=LogLevel.DEBUG)
    def __getitem__(self, idx):
        """Get a single training sample"""
        # Get trajectory index
        traj_idx = idx
        trajectory = self.trajectories[traj_idx]

        #Get the push_distance from this trajectory
        push_distance = trajectory["push_distance"]

        # 1. Get the contact quality
        quality = trajectory["contact_quality"]
        contact_indices = np.where(quality == 1)[0]

        # 3.Get the current and future point cloud
        current_pc = trajectory["current_point_cloud"]
        future_local_pose = trajectory["future_local_pose"]

        # 7. Transform push direction to object frame
        push_direction_obj_frame = trajectory["direction"]

        # Normalize push direction
        push_dir_norm = np.linalg.norm(push_direction_obj_frame)
        if push_dir_norm > 1e-6:
            push_direction_obj_frame = push_direction_obj_frame / push_dir_norm
        else:
            # If near-zero direction, use a default direction
            push_direction_obj_frame = np.array([0, 0, 1])
            self.logger.warning(
                f"Zero-length push direction in sample traj {traj_idx}"
            )

        # 8. Create orientation tensor for push direction
        orientation = torch.zeros((len(current_pc), 3), dtype=torch.float)
        all_directions = np.zeros((len(current_pc), 3), dtype=np.float32)
        for idx in contact_indices:
            if idx < len(current_pc):  # Safety check
                orientation[idx] = torch.tensor(
                    push_direction_obj_frame, dtype=torch.float
                )
                all_directions[idx] = push_direction_obj_frame

        # Normalize orientations
        orientation = F.normalize(orientation, p=2, dim=1)

        # Set push_distance for every point
        push_distances = torch.zeros((len(current_pc),1), dtype=torch.float)
        if np.ndim(push_distance) == 0:
            push_distances[:] = push_distance.item()
        else:
            self.logger.warning(
                "Push_distance shape isn't (), data load error!"
            )

        # 12. Create visualizations if in debug mode
        if self.logger.logger.level <= LogLevel.DEBUG.value:

            folder = trajectory["folder"]
            h5path = os.path.join(folder,'data.h5')
            with h5py.File(h5path, 'r') as h5f:

                # Get the world frame current pointcloud
                world_current_pc = h5f["world_obj_pcd/obj_pcd_before"][:]

                # Construct the future point cloud.
                fut_pose = future_local_pose.copy()
                fut_pose[3:7] = np.array([fut_pose[4],fut_pose[5],fut_pose[6],fut_pose[3]])
                rot_matrix = R.from_quat(fut_pose[3:7]).as_matrix()  # 3x3
                T2 = np.eye(4)
                T2[:3, :3] = rot_matrix
                T2[:3, 3] = fut_pose[:3]
                # The current frame is the identity, so the relative transform is T2.
                T_final = T2
                future_pc = trimesh.transform_points(current_pc, T_final)

                # Load the contact points in world frame
                contact_points_group = h5f["contact_points"]
                dataset_names = list(contact_points_group.keys())
                def extract_num(name):
                    match = re.match(r"(\d+)", name)
                    return int(match.group(1)) if match else float('inf')
                min_key = min(dataset_names, key=extract_num)
                world_contact_points = contact_points_group[min_key][:]

                # Create one-hot mask for contact points (1 for contact, 0 for non-contact)
                contact_mask = np.zeros(len(current_pc), dtype=bool)
                for idx in contact_indices:
                    if idx < len(current_pc):  # Safety check
                        contact_mask[idx] = True


                # Create comprehensive visualizations
                VisualizationUtils.visualize_object_centric_sample(
                    current_pc=current_pc,
                    contact_mask=contact_mask,
                    directions=all_directions,
                    future_pc=future_pc,
                    world_pc=world_current_pc,
                    world_contacts=world_contact_points,
                    traj_idx=traj_idx,
                    frame_idx=0,
                    debug_dir=self.debug_dir,
                    arrow_length=np.array(push_distances),
                )

        return {
            "current_pc": torch.tensor(current_pc, dtype=torch.float),
            "future_local_pose": torch.tensor(future_local_pose, dtype=torch.float),
            "quality": torch.tensor(quality),
            "orientation": orientation,
            "push_distance":push_distances,
        }
