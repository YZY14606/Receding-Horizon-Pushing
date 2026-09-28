import numpy as np
import torch
import torch.random
import os
import trimesh
import itertools
import open3d as o3d
from mani_skill.utils.sapien_utils import get_multiple_pairwise_contacts

# Extract contact point information from detected contacts.
def get_contact_points(contacts):
    list_contacts = list(itertools.chain(*contacts.values()))  # Merge contact lists.
    # Extract contact points in the world frame.
    contact_processed = [x[0] for x in list_contacts]
    contact_point_position = []
    for contact in contact_processed:
        single_point_position = []
        for point in contact.points:
            contact_pos = point.position  # Contact position [x, y, z]
            single_point_position.append(contact_pos)
        contact_point_position.append(single_point_position)
        
    return contact_point_position if contact_point_position else None

def save_contact_points(scene_dir,contact_points ,itr_simulator, time_name, traj_id):
    # Create the output path.
    contact_point_filename = str(itr_simulator) + "contact_points.ply"
    point_contact_path = os.path.join(scene_dir,time_name, traj_id, "contact_points", contact_point_filename)
    os.makedirs(os.path.dirname(point_contact_path), exist_ok=True)

    xyz = np.array([item[0] for item in contact_points])
    pcd = trimesh.points.PointCloud(xyz)
    pcd.export(point_contact_path)  # Use Open3D to load the exported point cloud.


# Read and save contact points.
def get_save_contact_point(env, itr_simulator, time_name, traj_id):
    # Judge contact
    have_contact = False
    # Select the actors whose contacts need to be queried.
    # actor0 is used to calculate contact forces.
    actor0 = env.unwrapped.obj
    # contact_actor0 is used to identify contact points.
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
    # link_list is used to calculate contact forces.
    link_actor1_list = [link_panda_hand_tcp,link_panda_leftfinger,link_panda_rightfinger,link_panda_leftfinger_pad,link_panda_rightfinger_pad]
    # contact_actor1_list is used to identify contact points.
    contact_actor1_list = [link_panda_hand_tcp._bodies[0].entity, link_panda_leftfinger._bodies[0].entity,
                        link_panda_rightfinger._bodies[0].entity, link_panda_leftfinger_pad._bodies[0].entity, link_panda_rightfinger_pad._bodies[0].entity]
    
    # Record object contacts with non-gripper robot links for trajectory filtering.
    # link_list is used to calculate contact forces.
    link_actor2_list = [link_panda_2,link_panda_3,link_panda_4,link_panda_5,link_panda_6,link_panda_7,link_panda_8 ,link_panda_hand,]
    # contact_actor1_list is used to identify contact points.
    contact_actor2_list = [link_panda_2._bodies[0].entity, link_panda_3._bodies[0].entity, link_panda_4._bodies[0].entity, link_panda_5._bodies[0].entity,
                       link_panda_6._bodies[0].entity, link_panda_7._bodies[0].entity, link_panda_8._bodies[0].entity,link_panda_hand._bodies[0].entity,]
    
    # Query contact information (get_contacts may be deprecated in the future).
    all_contacts = env.unwrapped.scene.get_contacts()
    # First query contacts between the object and non-gripper robot links.
    obj_robot_other_all_contact = get_multiple_pairwise_contacts(all_contacts, actor0 = contact_actor0, actor1_list = contact_actor2_list)
    # Calculate contact forces.
    other_force = 0
    for actor2 in link_actor2_list:
        actor0_actor2_contact_force = env.unwrapped.scene.get_pairwise_contact_forces(actor0, actor2)
        other_force = other_force + torch.sum(actor0_actor2_contact_force)
    if len(obj_robot_other_all_contact) > 0 and other_force !=0 :
        save_contact_points(scene_dir=env.unwrapped.scene_dir,contact_points = [[0,0,0]], itr_simulator = 10001, time_name =time_name, traj_id = traj_id)

    obj_robot_ee_all_contact = get_multiple_pairwise_contacts(all_contacts, actor0 = contact_actor0, actor1_list = contact_actor1_list)
    # Calculate contact forces and filter contact points.
    finger_force = 0
    for actor1 in link_actor1_list:
        actor0_actor1_contact_force = env.unwrapped.scene.get_pairwise_contact_forces(actor0, actor1)
        finger_force = finger_force + torch.sum(actor0_actor1_contact_force)
    if len(obj_robot_ee_all_contact) > 0 and finger_force !=0 :
        obj_ee_all_contact_points = get_contact_points(obj_robot_ee_all_contact)  # Convert contacts to point coordinates.
        save_contact_points(scene_dir=env.unwrapped.scene_dir,contact_points = obj_ee_all_contact_points, itr_simulator = itr_simulator, time_name =time_name, traj_id = traj_id)
        have_contact = True
    
    return have_contact

# Save the start and end poses for each push.
def get_save_push_ps_pe(scene_dir,ps,pe,time_name,traj_id,itr_simulator):
    # Create the start-pose output path.
    ps_filename = str(itr_simulator)+"action_ps.npy"
    ps_path = os.path.join(scene_dir,time_name, traj_id, "push_action_ps", ps_filename)
    os.makedirs(os.path.dirname(ps_path), exist_ok=True)
    # Create the end-pose output path.
    pe_filename = str(itr_simulator)+"action_pe.npy"
    pe_path = os.path.join(scene_dir,time_name, traj_id, "push_action_pe", pe_filename)
    os.makedirs(os.path.dirname(pe_path), exist_ok=True)

    # Save the start pose.
    np.save(ps_path, np.array(ps).astype(np.float32))
    # Save the end pose.
    np.save(pe_path, np.array(pe).astype(np.float32))


# Record the current TCP pose when contact is detected.
def contact_tcp_pose_record(have_contact,env,itr_simulator,traj_id,time_name):
    if have_contact:
        # Create the contact TCP pose output path.
        contact_tcp_pose_filename = str(itr_simulator)+"contact_tcp_pose.npy"
        contact_tcp_pose_path = os.path.join(env.scene_dir,time_name, traj_id, "contact_tcp_pose", contact_tcp_pose_filename)
        os.makedirs(os.path.dirname(contact_tcp_pose_path), exist_ok=True)

        tcp_pose = env.agent.tcp.pose.raw_pose
        tcp_pose = tcp_pose.cpu().numpy().astype(np.float32)

        # Save the TCP pose.
        np.save(contact_tcp_pose_path, tcp_pose)



# Select the action end point from point-cloud information.
def pointcloud_get_reach_pe(mesh,push_start,obj_idx):
    # Randomly sample a point inside the mesh, including bottle-shaped meshes.
    sampled_points = trimesh.sample.volume_mesh(mesh, 500)

    # Get the mesh centroid.
    mass_center = mesh.center_mass.reshape(1,3)

    if obj_idx ==1 or obj_idx ==7:
        # Sampling strategy for objects 1 and 7.
        if push_start[2] < mass_center[0,2]:
            xyz = np.array(sampled_points)
            filtered_points = filter_points_by_angle(a= mass_center,b = push_start, points = xyz)
            xyz = np.array(filtered_points)[0]
        else: 
            xyz = np.array(sampled_points)[0]
        xyz = generate_point_c(push_start,xyz)
    else:
        # Sampling strategy for objects 2, 3, 4, 5, 6, 8, 9, and 10.
        xyz = generate_point_c(push_start,np.array(sampled_points)[0])

    return xyz

# Select an appropriate end position for object 1.
def filter_points_by_angle(a, b, points, angle_range=(150, 210)):
    """
    Filter points by their planar angle relative to the line from a to b.
    
    Args:
        a (np.array): 3D coordinates of point a, with shape (1, 3).
        b (np.array): 3D coordinates of point b, with shape (3,).
        points (np.array): A set of 3D points with shape (n, 3).
        angle_range (tuple): Accepted angle range in degrees; defaults to (160, 200).
        
    Returns:
        np.array: Filtered points with shape (m, 3).
    """
    
    # Extract the XY coordinates of points a and b.
    a_xy = a[0,:2]
    b_xy = b[:2]

    # Calculate planar vectors from point a to every candidate point.
    vectors = points[:, :2] - a_xy  # Shape: (n, 2)
    # Calculate the planar vector from point a to point b.
    ab_vector = b_xy - a_xy

    # Calculate each point's angle relative to the line from a to b.
    angles = np.array([calculate_angle(v, ab_vector) for v in vectors])

    # Select points whose angles fall within the specified range.
    mask = (angles >= angle_range[0]) & (angles <= angle_range[1])
    filtered_points = points[mask]

    return filtered_points

# Calculate vector angles with np.arctan2 in the range [-pi, pi].
def calculate_angle(v1, v2):
    # Use the cross product to determine clockwise or counterclockwise direction.
    cross_product = np.cross(v1, v2)
    # Calculate the dot product.
    dot_product = np.dot(v1, v2)
    # Calculate the angle and convert radians to degrees.
    angle = np.degrees(np.arctan2(cross_product, dot_product))
        
    # Map negative angles to the range [0, 360].
    if angle < 0:
        angle += 360
    return angle

def generate_point_c(a, b, max_distance=0.05):
    """
    Generate point c beyond b along the a-to-b direction.
    
    Args:
        a: 3D coordinates of point a as a NumPy array with shape (3,).
        b: 3D coordinates of point b as a NumPy array with shape (3,).
        max_distance: Maximum distance between generated point c and b.
    
    Returns:
        The generated 3D point c as a NumPy array with shape (3,). Its z
        coordinate is set to that of point a.
    """
    
    # Calculate the direction vector from a to b.
    direction = b - a
    
    # Normalize the direction vector.
    unit_direction = direction / np.linalg.norm(direction)
    
    # Generate a random distance in the range [0, max_distance].
    distance = np.random.uniform(0, max_distance)
    
    # Generate point c as b + unit_direction * distance.
    c = b + unit_direction * distance
    
    # Keep the z coordinate of c equal to that of a.
    c[2] = a[2]
    
    return c

# Get the observed object point cloud in the world frame.
def get_global_object_pointcloud(env,traj_id ,time_name,camera_name_list,file_name):

    # Resolve the target actor's segmentation ID from the current scene.
    object_segmentation_id = env.obj.per_scene_id

    all_pointcloud = []
    for camera_name in camera_name_list:
        # Get the camera object.
        camera = env._sensors[camera_name]
        # Reconstruct the camera point cloud in the world frame.
        pointcloud = get_camera_world_object_pointcloud(camera, object_segmentation_id)
        all_pointcloud.append(pointcloud)
    all_pointcloud_np = np.concatenate(all_pointcloud,axis=0)
    # More points for delete
    all_pointcloud_np = fps_downsample(points = all_pointcloud_np, num_samples=1500).astype(np.float32)
    # Create the output path.
    save_path = os.path.join(env.scene_dir,time_name,traj_id,"world_obj_pcd",file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path,all_pointcloud_np)



# Save the object pose.
def get_save_object_pose(scene_dir ,object_pose,traj_id,time_name,file_name):
    obj_pose_np = object_pose.clone().cpu().numpy().reshape(-1).astype(np.float32)
    save_path = os.path.join(scene_dir,time_name,traj_id,"object_state",file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, obj_pose_np)




def get_camera_world_object_pointcloud(camera, object_segmentation_id):
    """Get an object's world-frame point cloud from one camera.

    Args:
        camera: A ManiSkill camera object.
    Returns:
        The object point cloud observed by the camera.
    """
    # Get camera observations.
    camera.capture()
    camera_obs = camera.get_obs()
    # Get the camera position data.
    camera_point_cloud = camera_obs["position"] #shape (1,512,512,3)
    # Get the camera segmentation data.
    camera_segmentation_num = camera_obs["segmentation"] #shape (1,512,512,1)


    # Reconstruct the 3D point cloud.
    # Preprocess the data.
    position = camera_point_cloud.float().clone()
    segmentation = camera_segmentation_num

    # Get the camera parameter dictionary.
    camera_para_dic = camera.get_params()
    # Get the camera-to-world transform.
    camera_cam2world_para = camera_para_dic["cam2world_gl"].to(position.device, 
                                                               dtype=position.dtype)
    cam2world = camera_cam2world_para.view(1,4,4).float()

    position[..., :3] = (
            position[..., :3] / 1000.0
        )  # convert the raw depth from millimeters to meters


    object_segmentation_id = object_segmentation_id.to(
        device=segmentation.device,
        dtype=segmentation.dtype,
    )

    # Reconstruct only points belonging to the target segmentation ID.
    xyzw = torch.cat([position, segmentation == object_segmentation_id], dim=-1).reshape(
        position.shape[0], -1, 4) @ cam2world.transpose(1, 2)

    # Filter invalid point-cloud entries.
    mask = xyzw[..., 3] != 0  # Mask entries whose homogeneous w component is nonzero.
    filtered_xyzw = xyzw[mask]  # Shape: [M, 4], where M is the valid point count.

    # Keep the first three coordinates of the object point cloud.
    pointcloud = filtered_xyzw[:,:3].cpu().numpy()

    return pointcloud.astype(np.float32)


# Downsample a point cloud.
def fps_downsample(points, num_samples=1024):
    if len(points) > num_samples:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downsampled_pcd = pcd.farthest_point_down_sample(num_samples)
        points_output = np.asarray(downsampled_pcd.points)
    else:
        pad_rows = num_samples - len(points)
        # Sample random indices with replacement.
        pad_idx = np.random.choice(len(points), pad_rows, replace=True)
        # Pad the point cloud.
        points_output = np.concatenate([points, points[pad_idx]], axis=0)

    return points_output
