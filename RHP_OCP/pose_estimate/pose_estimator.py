import numpy as np
import open3d as o3d
from transforms3d.quaternions import quat2mat , mat2quat
from transforms3d.euler import quat2euler,euler2quat


class Pose_Estimator():

    def __init__(self,source_pcd_wld,ori_pose,original_PCA_frame_pose, use_vis = False):

        self.source_pcd = o3d.geometry.PointCloud()
        self.source_pcd.points = o3d.utility.Vector3dVector(source_pcd_wld[:,:3])

        self.starting_pose = ori_pose
        self.starting_pose[2] = 0

        self.diff_transformation = self._calculate_transforamtion_from_ori_PCA_frame_2_ori_pose(original_PCA_frame_pose)

        self.pose_list = []
        self.pose_list.append(self.starting_pose)

        self.use_vis = use_vis

        # Record the push times
        self.push_times = 0


    def _calculate_transforamtion_from_ori_PCA_frame_2_ori_pose(self,original_PCA_frame_pose):
        """
        This function is used to calculate the original transformation from the original_PCA_frame_pose to the original_object_pose in PCA frame
        """

        original_PCA_frame_pose_matrix = self._pose_7d_to_transformation(original_PCA_frame_pose)
        original_object_pose_matrix = self._pose_7d_to_transformation(self.starting_pose.copy())

        diff_transformation = np.linalg.inv(original_PCA_frame_pose_matrix) @ original_object_pose_matrix

        return diff_transformation




    def _voxel_down_sample(self,voxel_size = 0.0001):
        self.source_pcd = self.source_pcd.voxel_down_sample(voxel_size)
        self.target_pcd = self.target_pcd.voxel_down_sample(voxel_size)

    def _pose_7d_to_transformation(self,pose):
        transformation = np.eye(4)
        rot = quat2mat(pose[3:7])
        transformation[:3,:3] = rot
        transformation[:3,3] = pose[:3]
        return transformation
    
    def _transformation_to_pose_7d(self,transformation):
        quat = mat2quat(transformation[:3,:3])
        xyz = transformation[:3,3]

        pose = np.hstack([xyz,quat])

        return pose


    def estimate_transformation_icp(self,source_pcd, max_iteration=1000, threshold=0.01):
        """
        Estimate the source-to-target transformation with ICP.

        Args:
            source_pcd: Source Open3D point cloud.
            max_iteration: Maximum number of ICP iterations.
            threshold: ICP correspondence and convergence threshold.

        Returns:
            transformation: The estimated 4x4 transformation matrix.
            fitness: Registration fitness in the range [0, 1].
        """
        # Configure and run point-to-point ICP.
        initial_pose = np.identity(4)
        icp = o3d.pipelines.registration.registration_icp(
            source_pcd, 
            self.target_pcd,
            threshold,
            initial_pose,  # Use identity when no initial transform is known.
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iteration,
                relative_fitness=threshold,
                relative_rmse=threshold)
        )
        
        return icp.transformation, icp.fitness
    

    def calculate_ori_pcd_in_pca_pose(self,local_frame_pose, ori_pose,last_pose):
        """
        Get the pointcloud by converting original pointcloud to local_frame_pose;
        Input:
        local_frame_pose, the pose of the frame by pca construction.
        ori_pose, the pose of the object in starting state.
        last_pose, the pose of the object before current push, used for modify the pca frame orientation.
        Output:
        transformation, the transformation about the pointcloud from starting state to pca frame.
        new_pose, the pose of the object in pca frame pose.
        self.source_pcd, the starting pointcloud of the object converted to the pca frame.
        """
        source_pcd_np = np.asarray(self.source_pcd.points)
        # Get the pose of ori_pcd in local frame pose
        new_pose = ori_pose.copy()
        new_pose[:2] = local_frame_pose[:2]

        euler_xy_ori = quat2euler(ori_pose[3:7])[:2]

        euler_z_last = quat2euler(last_pose[3:7])[2]
        euler_z_last = (2*np.pi + euler_z_last) if (euler_z_last < 0) else euler_z_last

        euler_z_local_frame = quat2euler(local_frame_pose[3:7])[2]
        euler_z_local_frame = (2*np.pi + euler_z_local_frame) if (euler_z_local_frame < 0) else euler_z_local_frame

        guessed_euler_z_local_frame = (euler_z_local_frame + np.pi) if (euler_z_local_frame < np.pi) else (euler_z_local_frame - np.pi)

        # For plane symmetry object, check the orirntation of the pca frame
        dis_ori_euler_local_frame = min(abs(euler_z_local_frame - euler_z_last),2*np.pi - abs(euler_z_local_frame - euler_z_last))
        dis_guessed_euler_local_frame = min(abs(guessed_euler_z_local_frame - euler_z_last),2*np.pi - abs(guessed_euler_z_local_frame - euler_z_last))
        if dis_ori_euler_local_frame > dis_guessed_euler_local_frame:
            euler_z_local_frame = guessed_euler_z_local_frame
        else:
            euler_z_local_frame = euler_z_local_frame

        euler_new = np.hstack([euler_xy_ori,euler_z_local_frame])
        new_pose[3:7] = euler2quat(euler_new[0],euler_new[1],euler_new[2])

        ori_pose_matrix = self._pose_7d_to_transformation(ori_pose)
        new_pose_matrix = self._pose_7d_to_transformation(new_pose)

        transformation = new_pose_matrix @ np.linalg.inv(ori_pose_matrix)
        source_pcd_homogeneous = np.hstack([source_pcd_np[:,:3], np.ones((source_pcd_np.shape[0], 1))]).T

        transformed_source_pcd = (transformation @ source_pcd_homogeneous).T[:,:3]
        source_pca_frame_pcd = o3d.geometry.PointCloud()
        source_pca_frame_pcd.points = o3d.utility.Vector3dVector(transformed_source_pcd)
        
        if self.use_vis == True:
            source_pca_frame_pcd.paint_uniform_color([0, 1, 0])  # Green

            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

            local_frame_matrix = self._pose_7d_to_transformation(local_frame_pose)
            local_frame_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            local_frame_axis = local_frame_axis.transform(local_frame_matrix)

            o3d.visualization.draw_geometries([source_pca_frame_pcd, self.target_pcd,local_frame_axis,axis])

        return transformation, new_pose , source_pca_frame_pcd
    
    def estimate_object_pose(self,now_pointcloud,local_frame_pose):

        self.target_pcd = o3d.geometry.PointCloud()
        self.target_pcd.points = o3d.utility.Vector3dVector(now_pointcloud[:,:3])

        if self.use_vis == True:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            o3d.visualization.draw_geometries([self.source_pcd, self.target_pcd, axis])

        self._voxel_down_sample()

        # Modify the PCA local frame
        local_frame_pose_matrix = self._pose_7d_to_transformation(local_frame_pose)
        modified_local_frame_pose_matrix = local_frame_pose_matrix @ self.diff_transformation
        modified_local_frame_pose = self._transformation_to_pose_7d(modified_local_frame_pose_matrix)

        # Get the transformation from starting pose to pca frame
        transformation, _, transformed_source_pca_pcd = self.calculate_ori_pcd_in_pca_pose(local_frame_pose = modified_local_frame_pose,
                                             ori_pose = self.starting_pose, last_pose = self.pose_list[self.push_times])
        # Get the ICP transformation
        icp_transformation, _ = self.estimate_transformation_icp(transformed_source_pca_pcd)
        # Get all transformation
        all_transformation = icp_transformation @ transformation

        # Get estimated pose of the object
        starting_pose_matrix = self._pose_7d_to_transformation(self.starting_pose)
        pose_after_icp_matrix = all_transformation @ starting_pose_matrix

        pose_7d = self._transformation_to_pose_7d(pose_after_icp_matrix)

        # Add 1 to push times
        self.push_times = self.push_times + 1

        # Add estimated pose to pose list
        self.pose_list.append(pose_7d)

        # visulize the result of pointcloud transformation
        if self.use_vis == True:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

            transformationed_souce_pcd = o3d.geometry.PointCloud()
            transformationed_souce_pcd.points = o3d.utility.Vector3dVector(self.source_pcd.points)
            transformationed_souce_pcd.paint_uniform_color([0, 1, 0])  # Green

            transformationed_souce_pcd.transform(all_transformation)
            o3d.visualization.draw_geometries([self.source_pcd,transformationed_souce_pcd, self.target_pcd, axis])

        return pose_7d
