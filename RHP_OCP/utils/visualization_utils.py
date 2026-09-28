import os
import numpy as np
import trimesh
from utils.logging_utils import Logger, LogLevel


class VisualizationUtils:
    """
    Utility class for visualizing point clouds, contact points, and push directions.
    """

    @staticmethod
    def save_pointcloud_with_arrows(
        points,
        contact_mask,
        directions,
        output_path,
        target_pc=None,
        arrow_length=0.05,
        arrow_width=0.005,
        arrow_head_length=0.02,
        arrow_segments=8,
        colors=None,
    ):
        """
        Save point cloud as PLY with detailed direction arrows for contact points

        Args:
            points: Point cloud (N, 3)
            contact_mask: Boolean mask for contact points (N,)
            directions: Direction vectors for each point (N, 3)
            output_path: Path to save the PLY file
            target_pc: Optional target/future point cloud (M, 3)
            arrow_length: Length of the arrow
            arrow_width: Width/radius of arrow shaft
            arrow_head_length: Length of the arrow head
            arrow_segments: Number of segments for cylinder
            colors: Optional dictionary with color definitions
        """
        logger = Logger.get_instance()

        # Make sure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Ensure contact_mask is boolean
        if isinstance(contact_mask, np.ndarray) and contact_mask.dtype != bool:
            contact_mask = contact_mask.astype(bool)

        # Handle case where contact_mask is empty
        if contact_mask is None:
            contact_mask = np.zeros(len(points), dtype=bool)

        # Verify that contact_mask and directions have the right dimensions
        if len(contact_mask) != len(points):
            logger.warning(
                f"Contact mask has {len(contact_mask)} entries but point cloud has {len(points)} points"
            )
            # Resize the contact mask to match the point cloud size
            if len(contact_mask) < len(points):
                temp_mask = np.zeros(len(points), dtype=bool)
                temp_mask[: len(contact_mask)] = contact_mask
                contact_mask = temp_mask
            else:
                contact_mask = contact_mask[: len(points)]

        # Check directions array
        if directions is None:
            logger.warning("Direction vectors are None, creating zero vectors")
            directions = np.zeros((len(points), 3))
        elif len(directions) != len(points):
            logger.warning(
                f"Direction vectors has {len(directions)} entries but point cloud has {len(points)} points"
            )
            # Resize the directions array to match the point cloud size
            if len(directions) < len(points):
                temp_dirs = np.zeros((len(points), 3))
                temp_dirs[: len(directions)] = directions
                directions = temp_dirs
            else:
                directions = directions[: len(points)]

        # Set default colors if not provided
        if colors is None:
            colors = {
                "pc": [0.8, 0.8, 0.8, 1.0],  # Light gray for regular points
                "contact": [1.0, 0.0, 0.0, 1.0],  # Red for contact points
                "target": [0.0, 0.6, 1.0, 0.5],  # Light blue, semi-transparent
                "shaft": [0.0, 1.0, 0.0, 1.0],  # Green for arrow shaft
                "head": [0.0, 0.5, 1.0, 1.0],  # Blue for arrow head
            }

        # Create base colors
        all_colors = np.ones((len(points), 4)) * colors["pc"]

        # Find valid contact points
        contact_indices = np.where(contact_mask)[0]

        # Set contact point colors
        all_colors[contact_mask] = colors["contact"]

        # Log the contact points for verification
        logger.debug(
            f"Visualization has {len(contact_indices)} contact points out of {len(points)} total points"
        )

        # If there are no contact points, log a warning
        if len(contact_indices) == 0:
            logger.warning("No contact points to visualize in the point cloud")

        # Start with just the original point cloud
        all_points = points.copy()

        # For each contact point, create arrow visualizations
        arrows_created = 0
        for idx in contact_indices:
            # Get the contact point and its direction
            contact_point = points[idx]
            direction = directions[idx]

            # Skip if direction is too small
            norm = np.linalg.norm(direction)
            if norm <= 1e-6:
                continue

            # Use this point to create an arrow
            arrows_created += 1

            # Normalize direction
            direction = direction / norm

            # Judge arrow_length is single or array
            if isinstance(arrow_length, np.ndarray) :
                real_arrow_length = arrow_length[idx]
            else:
                real_arrow_length = arrow_length

            # Create arrow shaft (line of points)
            num_shaft_points = 10  # Higher resolution for better visualization         
            shaft_points = np.array(
                [
                    contact_point + direction * (j * real_arrow_length / num_shaft_points)
                    for j in range(1, num_shaft_points + 1)
                ]
            )
            # Create arrow head (cone of points)
            # First, create two vectors perpendicular to direction
            if abs(direction[0]) < abs(direction[1]) and abs(direction[0]) < abs(
                direction[2]
            ):
                perp1 = np.array([0, -direction[2], direction[1]])
            elif abs(direction[1]) < abs(direction[2]):
                perp1 = np.array([-direction[2], 0, direction[0]])
            else:
                perp1 = np.array([-direction[1], direction[0], 0])
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(direction, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)

            # Arrow head position
            head_base = contact_point + direction * real_arrow_length

            # Create a circle of points for the base of the arrow head
            head_points = []
            for j in range(arrow_segments):
                angle = 2 * np.pi * j / arrow_segments
                # Point on the circle at the base of the arrow head
                circle_point = (
                    head_base
                    + (perp1 * np.cos(angle) + perp2 * np.sin(angle)) * arrow_width
                )
                head_points.append(circle_point)

            # Add the tip of the arrow
            tip_point = head_base + direction * arrow_head_length
            head_points.append(tip_point)

            # Convert to numpy array
            head_points = np.array(head_points)

            # Add all arrow points to the point cloud
            all_points = np.vstack([all_points, shaft_points, head_points])

            # Create colors for the arrow
            shaft_colors = np.ones((len(shaft_points), 4)) * colors["shaft"]
            head_colors = np.ones((len(head_points), 4)) * colors["head"]

            # Add colors
            all_colors = np.vstack([all_colors, shaft_colors, head_colors])

        # Log the number of arrows created
        logger.debug(f"Created {arrows_created} arrows for visualization")

        # If target point cloud is provided, add it as well
        if target_pc is not None:
            target_colors = np.ones((len(target_pc), 4)) * colors["target"]

            # Add target points and colors
            all_points = np.vstack([all_points, target_pc])
            all_colors = np.vstack([all_colors, target_colors])

        # Create the point cloud with all points and colors
        point_cloud = trimesh.points.PointCloud(all_points, colors=all_colors)

        # Save the point cloud
        point_cloud.export(output_path)
        logger.debug(f"Saved enhanced visualization to {output_path}")

        return output_path

    @staticmethod
    def visualize_object_centric_sample(
        current_pc,
        contact_mask,
        directions,
        future_pc=None,
        world_pc=None,
        world_contacts=None,
        traj_idx=0,
        frame_idx=0,
        debug_dir="debug",
        arrow_length=0.05,
    ):
        """
        Create comprehensive visualizations for an object-centric sample

        Args:
            current_pc: Object-centric point cloud (N, 3)
            contact_mask: Boolean mask for contact points (N,)
            directions: Direction vectors (N, 3)
            future_pc: Target/future point cloud (optional)
            world_pc: World-frame point cloud (optional)
            world_contacts: Original contact points in world frame (optional)
            traj_idx: Trajectory index for naming
            frame_idx: Frame index for naming
            debug_dir: Base directory for debug visualizations
        """
        logger = Logger.get_instance()

        # Only proceed if we're in debug mode
        if logger.logger.level > LogLevel.DEBUG.value:
            return

        # Create a debug directory for this trajectory/frame
        debug_frame_dir = os.path.join(debug_dir, f"traj_{traj_idx}")
        os.makedirs(debug_frame_dir, exist_ok=True)

        # 1. Save the object-centric visualization
        obj_viz_path = os.path.join(debug_frame_dir, "object_frame.ply")
        VisualizationUtils.save_pointcloud_with_arrows(
            current_pc, contact_mask, directions, obj_viz_path, target_pc=future_pc, arrow_length = arrow_length,
        )

        # 2. If world frame data is provided, visualize it
        if world_pc is not None and world_contacts is not None:
            # Create visualization showing world-frame point cloud and original contact points
            world_viz_path = os.path.join(debug_frame_dir, "world_frame.ply")

            # Extract colors for world visualization
            world_colors = {
                "pc": [0.7, 0.7, 0.7, 1.0],  # Gray for world-frame point cloud
                "contact": [0.0, 0.0, 0.0, 0.0],  # Invisible (don't use)
                "target": [1.0, 0.0, 0.0, 1.0],  # Red for original contact points
                "shaft": [0.0, 0.0, 0.0, 0.0],  # Invisible
                "head": [0.0, 0.0, 0.0, 0.0],  # Invisible
            }

            # Create dummy direction vectors (not used)
            dummy_directions = np.zeros_like(world_pc)
            dummy_mask = np.zeros(len(world_pc), dtype=bool)

            # Save world-frame visualization
            VisualizationUtils.save_pointcloud_with_arrows(
                world_pc,
                dummy_mask,
                dummy_directions,
                world_viz_path,
                target_pc=world_contacts,
                arrow_length = arrow_length,
                colors=world_colors,
            )

            # Also save the original contact points separately
            contact_pc = trimesh.points.PointCloud(
                world_contacts,
                colors=np.ones((len(world_contacts), 4)) * [1.0, 0.0, 0.0, 1.0],
            )
            contact_viz_path = os.path.join(debug_frame_dir, "original_contacts.ply")
            contact_pc.export(contact_viz_path)

        # 3. Create a zoomed-in view of the contact area
        if np.any(contact_mask):
            # Find contact point centroid
            contact_pts = current_pc[contact_mask]
            centroid = np.mean(contact_pts, axis=0)

            # Create a subset of points around the centroid
            distances = np.linalg.norm(current_pc - centroid, axis=1)
            radius = np.percentile(
                distances[contact_mask], 50
            )  # Use median distance as radius
            nearby_mask = distances < radius * 3  # 3x radius for good visibility

            # Save zoomed-in view if we have nearby points
            if np.any(nearby_mask):
                zoom_viz_path = os.path.join(debug_frame_dir, "zoom_contacts.ply")

                # Extract nearby points and their properties
                nearby_pc = current_pc[nearby_mask]
                nearby_contacts = contact_mask[nearby_mask]
                nearby_directions = directions[nearby_mask]

                # Save zoomed visualization with brighter colors
                zoom_colors = {
                    "pc": [0.9, 0.9, 0.9, 1.0],  # Lighter gray
                    "contact": [1.0, 0.0, 0.0, 1.0],  # Red
                    "target": [0.0, 0.0, 0.0, 0.0],  # Not used
                    "shaft": [0.0, 1.0, 0.0, 1.0],  # Bright green
                    "head": [0.0, 0.7, 1.0, 1.0],  # Bright blue
                }

                VisualizationUtils.save_pointcloud_with_arrows(
                    nearby_pc,
                    nearby_contacts,
                    nearby_directions,
                    zoom_viz_path,
                    arrow_length = arrow_length,
                    colors=zoom_colors,
                )

        logger.debug(
            f"Created comprehensive visualizations for traj_{traj_idx}_frame_{frame_idx} in {debug_frame_dir}"
        )
