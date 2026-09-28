import os
import numpy as np
from datetime import datetime
import trimesh
from utils.logging_utils import LogLevel, log_function
from utils.visualization_utils import VisualizationUtils




@log_function(LogLevel.DEBUG)
def visulize_pred_results(current_pc_np,future_pc_np,pred_contact_np,
                          pred_orientation_np,pred_push_distance_np,save_path,logger,push_idx = None):
    """
    Create and save detailed object-centric visualizations of model predictions
    with proper contact point and direction visualization using arrows.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Predicted quality and orientation
    # Get the pred_contact_mask
    top_10_indices = np.argsort(pred_contact_np.squeeze())[-10:]
    pred_contact_mask = np.zeros_like(pred_contact_np.squeeze(),dtype=bool)
    pred_contact_mask[top_10_indices] = True

    # Log the number of contact points to verify they exist
    logger.debug(f"Predicted contacts: {np.sum(pred_contact_mask)}")

    # Get current time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



    # Create visualization directory if it doesn't exist
    vis_dir = save_path


    # Setup colors for different visualization components
    colors = {
        "pc": [0.8, 0.8, 0.8, 1.0],  # Light gray for main point cloud
        "contact": [1.0, 0.0, 0.0, 1.0],  # Red for contact points
        "shaft": [0.0, 1.0, 0.0, 1.0],  # Green for arrow shaft
        "head": [0.5, 0.0, 1.0, 1.0],  # Purple for arrow head
        "target": [0.0, 0.6, 1.0, 0.7],  # Light blue for target point cloud
    }


    # 1. Model Prediction Visualization
    logger.debug("Creating prediction visualization")
    pred_filename = os.path.join(vis_dir, "pred_muti_result.ply")

    # Set contact color for predictions to blue
    pred_colors = colors.copy()
    pred_colors["contact"] = [0.0, 0.0, 1.0, 1.0]  # Blue for predicted contacts

    # Use the arrow visualization for predictions
    VisualizationUtils.save_pointcloud_with_arrows(
        points=current_pc_np,
        contact_mask=pred_contact_mask,
        directions=pred_orientation_np,
        output_path=pred_filename,
        target_pc=future_pc_np,
        arrow_length=pred_push_distance_np,
        arrow_width=0.005,
        colors=pred_colors,
    )

    # 2. Visulize the highest contact point
    if push_idx == None:
        # Get the pred_contact_mask
        top_1_indices = np.argsort(pred_contact_np.squeeze())[-1]
    else:
        top_1_indices = push_idx

    pred_contact_mask = np.zeros_like(pred_contact_np.squeeze(),dtype=bool)
    pred_contact_mask[top_1_indices] = True

    single_path = os.path.join(vis_dir, "pred_single_result.ply")

    # Use the arrow visualization for predictions
    VisualizationUtils.save_pointcloud_with_arrows(
        points=current_pc_np,
        contact_mask=pred_contact_mask,
        directions=pred_orientation_np,
        output_path=single_path,
        target_pc=future_pc_np,
        arrow_length=pred_push_distance_np,
        arrow_width=0.005,
        colors=colors,
        )

    # 3. Log probabilities heatmap for further analysis
    # Create a heatmap visualization where point color is determined by probability
    if pred_contact_np is not None:
        logger.debug("Creating probability heatmap visualization")
        heatmap_filename = os.path.join(vis_dir, "heatmap.ply")

        # Create a colormap from blue (0.0) to red (1.0)
        def get_heat_color(prob):
            # Map probability to color from blue (0.0) to red (1.0)
            r = min(1.0, prob * 2)  # Red increases faster
            g = max(0, min(1.0, prob * 2 - 1))  # Green in middle range
            b = max(0, 1.0 - prob * 2)  # Blue decreases faster
            return [r, g, b, 1.0]

        # Create colors based on probability values
        heatmap_colors = np.array(
            [get_heat_color(p) for p in pred_contact_np.squeeze()]
        )

        # Create point cloud with heatmap colors
        heatmap_pc = trimesh.points.PointCloud(current_pc_np, colors=heatmap_colors)
        heatmap_pc.export(heatmap_filename)


    logger.debug(
        f"Saved model prediction visualizations for time {timestamp} to {vis_dir}"
    )
    return vis_dir
