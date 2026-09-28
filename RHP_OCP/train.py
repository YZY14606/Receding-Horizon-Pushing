#!/usr/bin/env python3
import os
import argparse
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
import trimesh
from scipy.spatial.transform import Rotation as R #(x,y,z,w)

# Import your model and dataset
from network.model import PointCloudEncoderDecoder, EncoderDecoderLoss
from datasets.h5_dataset import ObjectCentricPushDataset

# Import the logging utilities
from utils.logging_utils import Logger, LogLevel, log_function
from utils.data_utils import collate_fn
from utils.visualization_utils import VisualizationUtils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PointCloudEncoderDecoder for push prediction"
    )

    # Dataset parameters
    parser.add_argument(
        "--data-dir", nargs='+', type=str, required=True, help='List of data directory paths'
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers", type=int, default=28, help="Number of workers for data loading"
    )

    # Model parameters
    parser.add_argument(
        "--point-dim", type=int, default=3, help="Dimension of point cloud points"
    )
    parser.add_argument(
        "--transformation-dim", type=int, default=7, help="Dimension of target transformation"
    )
    parser.add_argument(
        "--condition-dim", type=int, default=128, help="Dimension of condition features"
    )
    parser.add_argument(
        "--latent-dim", type=int, default=128, help="Dimension of latent space"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Dimension of hidden layers"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--contact-weight", type=float, default=10.0, 
        help="Weight for positive contact examples in loss"
    )
    parser.add_argument(
        "--orientation-weight", type=float, default=0.1, 
        help="Weight for orientation loss term"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )

    # Logging and checkpoints
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Interval for logging training progress",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Epoch interval for saving checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Log level",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="debug",
        help="Directory to save debug visualizations",
    )

    # Weights & Biases options
    parser.add_argument(
        "--wandb-project", type=str, default="push-prediction", help="W&B project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity name"
    )
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    parser.add_argument(
        "--wandb-tags", type=str, nargs="+", default=[], help="W&B tags for this run"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging"
    )

    # Visualization
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--vis-interval",
        type=int,
        default=5,
        help="Epoch interval for creating visualizations",
    )

    return parser.parse_args()


@log_function(LogLevel.INFO)
def setup_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@log_function(LogLevel.INFO)
def get_dataloaders(args, logger):
    """Create training and validation data loaders"""

    logger.info(f"Loading dataset from {args.data_dir}")

    training_dataset = ObjectCentricPushDataset(
        data_dir=args.data_dir,
        debug_dir=args.debug_dir,
        mode = 'train',
        validation_split = args.validation_split,
    )

    validation_dataset = ObjectCentricPushDataset(
        data_dir=args.data_dir,
        debug_dir=args.debug_dir,
        mode = 'val',
        validation_split = args.validation_split,
    )

    logger.info(f"Dataset size: {len(training_dataset) + len(validation_dataset)} samples")
    logger.info(f"Training set: {len(training_dataset)} samples")
    logger.info(f"Validation set: {len(validation_dataset)} samples")

    # Create data loaders with custom collate_fn
    train_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,  # Add custom collate function
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,  # Add custom collate function
    )

    return train_loader, val_loader


@log_function(LogLevel.INFO)
def setup_model(args, device, logger):
    """Initialize model, loss function, optimizer, and scheduler"""
    logger.info("Initializing model, loss function, optimizer, and scheduler")

    # Initialize model
    model = PointCloudEncoderDecoder(
        point_dim=args.point_dim,
        transformation_dim=args.transformation_dim,
        condition_dim=args.condition_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # Initialize loss function
    criterion = EncoderDecoderLoss(
        contact_weight=args.contact_weight,
        orientation_weight=args.orientation_weight,
    ).to(device)

    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        path = args.resume
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        saved_metrics = checkpoint.get("loss_metrics", {})
        best_val_loss = checkpoint.get(
            "best_val_loss",
            saved_metrics.get("val_loss", float("inf")),
        )
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    return model, criterion, optimizer, scheduler, start_epoch, best_val_loss


@log_function(LogLevel.INFO)
def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    args,
    logger,
    global_step=0,
):
    """Train model for one epoch"""
    model.train()
    model.set_epoch(epoch)
    criterion.reset_stats()
    total_loss = 0.0
    total_contact_loss = 0.0
    total_orientation_loss = 0.0
    total_push_distance_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

    for batch_idx, batch_data in enumerate(progress_bar):
        current_pc = batch_data["current_pc"].to(device)
        future_local_pose = batch_data["future_local_pose"].to(device)
        quality = batch_data["quality"].to(device)
        orientation = batch_data["orientation"].to(device)
        push_distance = batch_data["push_distance"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        pred_contact, pred_orientation, pred_push_distance = model(current_pc, future_local_pose)

        # Calculate loss
        loss, contact_loss, orientation_loss, push_distance_loss = criterion(
            pred_contact, pred_orientation, pred_push_distance, quality, orientation, push_distance
        )

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        total_contact_loss += contact_loss.item()
        total_orientation_loss += orientation_loss.item()
        total_push_distance_loss += push_distance_loss.item()

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "c_loss": f"{contact_loss.item():.4f}",
                "o_loss": f"{orientation_loss.item():.4f}",
                "d_loss": f"{push_distance_loss.item():.4f}",
            }
        )

        # Log to Weights & Biases
        if not args.no_wandb and batch_idx % args.log_interval == 0:
            wandb.log(
                {
                    "train/batch/loss": loss.item(),
                    "train/batch/contact_loss": contact_loss.item(),
                    "train/batch/orientation_loss": orientation_loss.item(),
                    "train/batch/push_distance_loss": push_distance_loss.item(),
                    "train/batch/step": global_step,
                },
                step=global_step,
            )

        # Increment global step after logging
        global_step += 1


    # Calculate average losses
    avg_loss = total_loss / len(train_loader)
    avg_contact_loss = total_contact_loss / len(train_loader)
    avg_orientation_loss = total_orientation_loss / len(train_loader)
    avg_push_distance_loss = total_push_distance_loss / len(train_loader)

    # Log average losses for the epoch to Weights & Biases
    if not args.no_wandb:
        wandb.log(
            {
                "train/epoch/loss": avg_loss,
                "train/epoch/contact_loss": avg_contact_loss,
                "train/epoch/orientation_loss": avg_orientation_loss,
                "train/epoch/push_distance_loss": avg_push_distance_loss,
                "epoch": epoch,
            },
            step=global_step,
        )


    logger.info(
        f"Train Epoch: {epoch} "
        f"Loss: {avg_loss:.4f} "
        f"Contact Loss: {avg_contact_loss:.4f} "
        f"Orientation Loss: {avg_orientation_loss:.4f}"
        f"Push_distance Loss: {avg_push_distance_loss:.4f}"
    )


    return avg_loss, avg_contact_loss, avg_orientation_loss, avg_push_distance_loss, global_step


@log_function(LogLevel.INFO)
def validate(model, val_loader, criterion, device, epoch, args, logger, global_step):
    """Validate model on validation data"""
    model.eval()
    total_loss = 0.0
    total_contact_loss = 0.0
    total_orientation_loss = 0.0
    total_push_distance_loss = 0.0

    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")

    with torch.no_grad():
        for batch_data in progress_bar:
            current_pc = batch_data["current_pc"].to(device)
            future_local_pose = batch_data["future_local_pose"].to(device)
            quality = batch_data["quality"].to(device)
            orientation = batch_data["orientation"].to(device)
            push_distance = batch_data["push_distance"].to(device)
            
            # Forward pass
            pred_contact, pred_orientation, pred_push_distance = model(current_pc, future_local_pose)

            # Calculate loss
            loss, contact_loss, orientation_loss, push_distance_loss = criterion(
                pred_contact, pred_orientation, pred_push_distance, quality, orientation, push_distance
            )

            # Update metrics
            total_loss += loss.item()
            total_contact_loss += contact_loss.item()
            total_orientation_loss += orientation_loss.item()
            total_push_distance_loss += push_distance_loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "c_loss": f"{contact_loss.item():.4f}",
                    "o_loss": f"{orientation_loss.item():.4f}",
                    "d_loss": f"{push_distance_loss.item():.4f}",
                }
            )

    # Calculate average losses
    avg_loss = total_loss / len(val_loader)
    avg_contact_loss = total_contact_loss / len(val_loader)
    avg_orientation_loss = total_orientation_loss / len(val_loader)
    avg_push_distance_loss = total_push_distance_loss / len(val_loader)

    # Log average losses for the epoch to Weights & Biases
    if not args.no_wandb:
        # Use the current global_step from training
        wandb.log(
            {
                "val/epoch/loss": avg_loss,
                "val/epoch/contact_loss": avg_contact_loss,
                "val/epoch/orientation_loss": avg_orientation_loss,
                "val/epoch/push_distance_loss": avg_push_distance_loss,
                "epoch": epoch,
            },
            step=global_step,
        )

    logger.info(
        f"Validation Epoch: {epoch} "
        f"Loss: {avg_loss:.4f} "
        f"Contact Loss: {avg_contact_loss:.4f} "
        f"Orientation Loss: {avg_orientation_loss:.4f}"
        f"Push_distance Loss: {avg_push_distance_loss:.4f}"
    )

    return avg_loss, avg_contact_loss, avg_orientation_loss, avg_push_distance_loss


@log_function(LogLevel.DEBUG)
def save_checkpoint(model, optimizer, scheduler, epoch, loss_metrics, best_val_loss, args):
    """Save model checkpoint"""
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss_metrics": loss_metrics,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )

    # Save latest checkpoint (for easy resuming)
    latest_checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss_metrics": loss_metrics,
            "best_val_loss": best_val_loss,
        },
        latest_checkpoint_path,
    )


@log_function(LogLevel.INFO)
def save_model_prediction_visualization(
    model, val_loader, device, epoch, args, logger, global_step=None
):
    """
    Create and save detailed object-centric visualizations of model predictions
    with proper contact point and direction visualization using arrows.
    """
    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    # Always create visualizations at specified intervals
    if epoch % args.vis_interval != 0:
        return

    model.eval()

    # Get a batch of data
    batch_data = next(iter(val_loader))
    current_pc = batch_data["current_pc"].to(device)
    future_local_pose = batch_data["future_local_pose"].to(device)
    gt_quality = batch_data["quality"].to(device)
    gt_orientation = batch_data["orientation"].to(device)
    gt_push_distance = batch_data["push_distance"].to(device)

    with torch.no_grad():
        # Get model predictions
        pred_contact, pred_orientation, pred_push_distance = model(current_pc, future_local_pose)

    # Process for visualization (using first sample in batch)
    sample_idx = 0

    # Convert tensors to numpy arrays
    current_pc_np = current_pc[sample_idx].cpu().numpy()
    future_local_pose_np = future_local_pose[sample_idx].cpu().numpy()

    # Construct the future point cloud.
    fut_pose = future_local_pose_np.copy()
    fut_pose[3:7] = np.array([fut_pose[4],fut_pose[5],fut_pose[6],fut_pose[3]])
    rot_matrix = R.from_quat(fut_pose[3:7]).as_matrix()  # 3x3
    T2 = np.eye(4)
    T2[:3, :3] = rot_matrix
    T2[:3, 3] = fut_pose[:3]
    # The current frame is the identity, so the relative transform is T2.
    T_final = T2
    future_pc_np = trimesh.transform_points(current_pc_np, T_final)

    # Ground truth quality (contact points)
    # Convert to boolean mask for visualization
    gt_quality_np = gt_quality[sample_idx].cpu().numpy()
    gt_contact_mask = gt_quality_np > 0.5  # Convert to boolean mask

    # Ground truth orientations
    gt_orientation_np = gt_orientation[sample_idx].cpu().numpy()

    # Ground truth push_distances
    gt_push_distance_np = gt_push_distance[sample_idx].cpu().numpy()

    # Predicted quality and orientation
    # Convert sigmoid probabilities to mask using threshold
    pred_contact_np = torch.sigmoid(pred_contact[sample_idx]).cpu().numpy()
    pred_contact_mask = pred_contact_np.squeeze() > 0.5  # Threshold at 0.5
    pred_orientation_np = pred_orientation[sample_idx].cpu().numpy()
    pred_push_distance_np = pred_push_distance[sample_idx].cpu().numpy()

    # Log the number of contact points to verify they exist
    logger.info(f"Ground truth contacts: {np.sum(gt_contact_mask)}")
    logger.info(f"Predicted contacts: {np.sum(pred_contact_mask)}")

    # Create visualization filename base
    base_filename = f"epoch_{epoch}_prediction"

    # Create visualization directory if it doesn't exist
    vis_epoch_dir = os.path.join(args.vis_dir, f"epoch_{epoch}")
    os.makedirs(vis_epoch_dir, exist_ok=True)

    # Setup colors for different visualization components
    colors = {
        "pc": [0.8, 0.8, 0.8, 1.0],  # Light gray for main point cloud
        "contact": [1.0, 0.0, 0.0, 1.0],  # Red for contact points
        "shaft": [0.0, 1.0, 0.0, 1.0],  # Green for arrow shaft
        "head": [0.5, 0.0, 1.0, 1.0],  # Purple for arrow head
        "target": [0.0, 0.6, 1.0, 0.7],  # Light blue for target point cloud
    }

    # 1. Ground Truth Visualization
    logger.info("Creating ground truth visualization")
    gt_filename = os.path.join(vis_epoch_dir, f"{base_filename}_gt.ply")

    # Use the arrow visualization for ground truth
    VisualizationUtils.save_pointcloud_with_arrows(
        points=current_pc_np,
        contact_mask=gt_contact_mask,
        directions=gt_orientation_np,
        output_path=gt_filename,
        target_pc=future_pc_np,
        arrow_length=gt_push_distance_np,
        arrow_width=0.005,
        colors=colors,
    )

    # 2. Model Prediction Visualization
    logger.info("Creating prediction visualization")
    pred_filename = os.path.join(vis_epoch_dir, f"{base_filename}_pred.ply")

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

    # 4. Create a side-by-side visualization with both ground truth and predictions
    logger.info("Creating side-by-side visualization")

    # Offset the point clouds for side by side comparison
    # Find the bounding box dimensions to determine offset
    x_min = np.min(current_pc_np[:, 0])
    x_max = np.max(current_pc_np[:, 0])

    # Calculate offset based on size of point cloud (add 20% padding)
    x_offset = (x_max - x_min) * 1.2

    # Create offset point clouds
    gt_pc = current_pc_np.copy()
    pred_pc = current_pc_np.copy() + np.array([x_offset, 0, 0])
    # Create a combined point cloud
    combined_pc = np.vstack([gt_pc, pred_pc])

    # Create colors for both point clouds
    gt_pc_colors = np.ones((len(gt_pc), 4)) * colors["pc"]
    gt_pc_colors[gt_contact_mask] = colors["contact"]

    pred_pc_colors = np.ones((len(pred_pc), 4)) * colors["pc"]
    pred_pc_colors[pred_contact_mask] = pred_colors["contact"]

    # Combine colors
    combined_colors = np.vstack([gt_pc_colors, pred_pc_colors])

    # Create a combined point cloud
    combined_point_cloud = trimesh.points.PointCloud(
        combined_pc, colors=combined_colors
    )

    # Save the combined visualization
    combined_filename = os.path.join(vis_epoch_dir, f"{base_filename}_side_by_side.ply")
    combined_point_cloud.export(combined_filename)

    # 5. Log probabilities heatmap for further analysis
    # Create a heatmap visualization where point color is determined by probability
    if pred_contact_np is not None:
        logger.info("Creating probability heatmap visualization")
        heatmap_filename = os.path.join(vis_epoch_dir, f"{base_filename}_heatmap.ply")

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

    # Log only stats to wandb if enabled
    if not args.no_wandb:
        try:
            wandb_log_args = {
                "prediction_stats/num_gt_contacts": np.sum(gt_contact_mask),
                "prediction_stats/num_pred_contacts": np.sum(pred_contact_mask),
                "prediction_stats/max_prob": np.max(pred_contact_np),
                "prediction_stats/min_prob": np.min(pred_contact_np),
                "prediction_stats/mean_prob": np.mean(pred_contact_np),
            }

            # Only specify step if we have a valid global_step
            if global_step is not None and global_step >= 1:
                wandb.log(wandb_log_args, step=global_step)
            else:
                wandb.log(wandb_log_args)  # Let wandb handle the step

        except Exception as e:
            logger.warning(f"Failed to log prediction stats to wandb: {e}")

    logger.info(
        f"Saved model prediction visualizations for epoch {epoch} to {vis_epoch_dir}"
    )
    return vis_epoch_dir


@log_function(LogLevel.INFO)
def main():
    args = parse_args()

    # Setup logging
    logger = Logger.get_instance()
    logger.set_level(args.log_level)

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    # Create debug directory if in debug mode
    if logger.logger.level <= LogLevel.DEBUG.value:
        os.makedirs(args.debug_dir, exist_ok=True)
        logger.debug(f"Created debug directory at {args.debug_dir}")

    # Setup random seed
    setup_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize Weights & Biases
    if not args.no_wandb:
        run_name = (
            args.wandb_name
            or f"encoder_decoder_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=args.wandb_tags,
            config=vars(args),
        )
        logger.info(f"Initialized W&B run: {run_name}")

        # Save configuration to W&B
        wandb.config.update(vars(args))

        # Log system info
        wandb.run.summary["system/device"] = str(device)
        wandb.run.summary["system/cuda_version"] = (
            torch.version.cuda if torch.cuda.is_available() else "N/A"
        )
        wandb.run.summary["system/torch_version"] = torch.__version__
    else:
        logger.info("Weights & Biases logging disabled")

    # Get data loaders
    train_loader, val_loader = get_dataloaders(args, logger)

    # Setup model, criterion, optimizer, and scheduler
    model, criterion, optimizer, scheduler, start_epoch, best_val_loss = setup_model(
        args, device, logger
    )

    # Log model architecture
    logger.info(f"Model architecture:\n{model}")
    if not args.no_wandb:
        # Log model architecture to W&B
        wandb.run.summary["model/architecture"] = str(model)

        # Watch the model to track gradients, parameters, etc.
        wandb.watch(model, log="all", log_freq=args.log_interval)

    # Initialize global step counter for wandb
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_loss, train_c_loss, train_o_loss, train_d_loss, global_step = (
            train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                args,
                logger,
                global_step,
            )
        )

        # Validate
        val_loss, val_c_loss, val_o_loss, val_d_loss = validate(
            model, val_loader, criterion, device, epoch, args, logger, global_step
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Log learning rate to W&B
        if not args.no_wandb:
            for param_group in optimizer.param_groups:
                wandb.log({"train/learning_rate": param_group["lr"]}, step=global_step)


        # Save checkpoint
        loss_metrics = {
            "train_loss": train_loss,
            "train_contact_loss": train_c_loss,
            "train_orientation_loss": train_o_loss,
            "train_push_distance_loss": train_d_loss,
            "val_loss": val_loss,
            "val_contact_loss": val_c_loss,
            "val_orientation_loss": val_o_loss,
            "val_push_distance_loss": val_d_loss,
        }

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(
                args.checkpoint_dir, "checkpoint_best.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss_metrics": loss_metrics,
                    "best_val_loss": best_val_loss,
                },
                best_checkpoint_path,
            )
            logger.info(f"Saved best model with val_loss={best_val_loss:.4f}")

            # Log best model to W&B
            if not args.no_wandb:
                wandb.run.summary["best_model/epoch"] = epoch
                wandb.run.summary["best_model/val_loss"] = best_val_loss
                wandb.run.summary["best_model/path"] = best_checkpoint_path

                # Save the best model as an artifact
                artifact = wandb.Artifact(f"model-best-{wandb.run.id}", type="model")
                artifact.add_file(best_checkpoint_path)
                wandb.log_artifact(artifact)

        # Save periodic checkpoint
        if (epoch+1) % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, loss_metrics, best_val_loss, args)
            logger.info(f"Saved checkpoint at epoch {epoch}")

            # Log checkpoint to W&B as artifact
            if not args.no_wandb:
                checkpoint_path = os.path.join(
                    args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
                )
                artifact = wandb.Artifact(
                    f"model-checkpoint-e{epoch}", type="checkpoint"
                )
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)

        # Create visualizations
        if (
            epoch % args.vis_interval == 0
            or logger.logger.level <= LogLevel.DEBUG.value
        ):
            save_model_prediction_visualization(
                model, val_loader, device, epoch, args, logger
            )
            logger.info(f"Saved visualizations for epoch {epoch}")

    # Save final model
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, loss_metrics, best_val_loss, args)
    logger.info("Training complete!")

    if not args.no_wandb:
        wandb.run.summary["final_model/epoch"] = args.epochs - 1
        wandb.run.summary["final_model/val_loss"] = val_loss

        # Close wandb run
        wandb.finish()


if __name__ == "__main__":
    main()
