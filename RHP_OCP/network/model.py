import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np

# Import the Logger from your existing code
from utils.logging_utils import Logger, LogLevel, log_function


class FullConnectedLayerFeatureExtractor(nn.Module):
    """
    Full-connected-layer feature extractor for transformations.
    Processes a transformation [x,y,z,qw,qx,qy,qz].

    Args:
        input_channels: Number of input channels (typically 7 for XYZ and quat)
        output_channels: Number of output channels for transformation features
    """

    def __init__(
        self,
        input_channels: int = 7,
        output_channels: int = 128,
        name: str = "FullConnectedLayerFeatureExtractor",
    ):
        super(FullConnectedLayerFeatureExtractor, self).__init__()
        self.name = name  # Add name for debugging
        self.logger = Logger.get_instance()

        self.mlp = nn.Sequential(
                # Input layer (7 → 256)
                nn.Linear(input_channels, 256),
                nn.LayerNorm(256),
                nn.ReLU(),  # ReLU
    
                # Hidden layer 1 (256 → 256)
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
    
                # Hidden layer 2 (256 → 128) output layer
                nn.Linear(256, output_channels)
            )

    @log_function(level=LogLevel.DEBUG)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            x: Transformation tensor of shape [B, C] where:
               B = batch size, C = input channels

        Returns:
            Point features of shape [B, output_channels]
        """
        self.logger.debug(
            f"{self.name} input shape: {x.shape}, data range: [{x.min().item():.3f}, {x.max().item():.3f}]"
        )

        # Track if we have NaN values in input
        if torch.isnan(x).any():
            self.logger.warning(f"{self.name}: NaN detected in input tensor")

        transformation_features = self.mlp(x)  # [B, output_channels]

        # Log feature statistics
        self.logger.debug(
            f"{self.name} output shape: {transformation_features.shape}, "
            f"mean: {transformation_features.mean().item():.3f}, "
            f"std: {transformation_features.std().item():.3f}, "
            f"min: {transformation_features.min().item():.3f}, "
            f"max: {transformation_features.max().item():.3f}"
        )

        # Check for activation saturation (all close to 0 or 1 after ReLU)
        near_zero = (transformation_features.abs() < 1e-6).float().mean().item()
        self.logger.debug(f"{self.name} near-zero activations: {near_zero:.2%}")

        return transformation_features



class PointNetFeatureExtractor(nn.Module):
    """
    PointNet-based feature extractor for point clouds.
    Processes a point cloud into per-point features using 1D convolutions.

    Args:
        input_channels: Number of input channels (typically 3 for XYZ coordinates)
        output_channels: Number of output channels for point features
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 128,
        name: str = "PointNetFeatureExtractor",
    ):
        super(PointNetFeatureExtractor, self).__init__()
        self.name = name  # Add name for debugging
        self.logger = Logger.get_instance()

        self.mlp = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv1d(128, output_channels, 1),
            nn.GroupNorm(16, output_channels),
            nn.ReLU(),
        )

    @log_function(level=LogLevel.DEBUG)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            x: Point cloud tensor of shape [B, N, C] where:
               B = batch size, N = number of points, C = input channels

        Returns:
            Point features of shape [B, N, output_channels]
        """
        self.logger.debug(
            f"{self.name} input shape: {x.shape}, data range: [{x.min().item():.3f}, {x.max().item():.3f}]"
        )

        # Track if we have NaN values in input
        if torch.isnan(x).any():
            self.logger.warning(f"{self.name}: NaN detected in input tensor")

        x = x.transpose(2, 1)  # [B, C, N]
        point_features = self.mlp(x)  # [B, output_channels, N]
        point_features = point_features.transpose(2, 1)  # [B, N, output_channels]

        # Log feature statistics
        self.logger.debug(
            f"{self.name} output shape: {point_features.shape}, "
            f"mean: {point_features.mean().item():.3f}, "
            f"std: {point_features.std().item():.3f}, "
            f"min: {point_features.min().item():.3f}, "
            f"max: {point_features.max().item():.3f}"
        )

        # Check for activation saturation (all close to 0 or 1 after ReLU)
        near_zero = (point_features.abs() < 1e-6).float().mean().item()
        self.logger.debug(f"{self.name} near-zero activations: {near_zero:.2%}")

        return point_features


class GlobalFeatureAggregator(nn.Module):
    """
    Aggregates per-point features into a global feature vector using max pooling.

    Args:
        input_channels: Number of input feature channels
        output_channels: Number of output feature channels
    """

    def __init__(
        self,
        input_channels: int = 128,
        output_channels: int = 128,
        name: str = "GlobalFeatureAggregator",
    ):
        super(GlobalFeatureAggregator, self).__init__()
        self.name = name
        self.logger = Logger.get_instance()

        self.mlp = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 1),
            nn.GroupNorm(16, output_channels),
            nn.ReLU(),
        )

    @log_function(level=LogLevel.DEBUG)
    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate point features into a global feature vector.

        Args:
            point_features: Per-point features of shape [B, N, C] where:
                           B = batch size, N = number of points, C = channels

        Returns:
            Global feature vector of shape [B, output_channels]
        """
        x = point_features.transpose(2, 1)  # [B, C, N]

        # Log pre-pooling statistics
        self.logger.debug(
            f"{self.name} pre-pooling shape: {x.shape}, "
            f"mean: {x.mean().item():.3f}, std: {x.std().item():.3f}"
        )

        # Check max pooling diversity (how many points contribute to the max)
        max_indices = torch.argmax(x, dim=2)  # [B, C]
        unique_indices = torch.zeros(x.size(0), device=x.device)

        for b in range(x.size(0)):
            unique_indices[b] = torch.unique(max_indices[b]).size(0)

        avg_unique = unique_indices.float().mean().item()
        self.logger.debug(
            f"{self.name} max pooling avg unique contributors: {avg_unique:.1f} / {x.size(-1)}"
        )

        global_feature = torch.max(x, 2, keepdim=True)[0]  # [B, C, 1]
        global_feature = self.mlp(global_feature)  # [B, output_channels, 1]
        global_feature = global_feature.squeeze(2)  # [B, output_channels]

        # Log global feature statistics
        self.logger.debug(
            f"{self.name} output shape: {global_feature.shape}, "
            f"mean: {global_feature.mean().item():.3f}, "
            f"std: {global_feature.std().item():.3f}"
        )

        return global_feature


class Encoder(nn.Module):
    """
    Encoder processes the current point cloud conditioned on the target's global feature.
    It extracts per-point features, computes a global feature, and outputs a latent feature vector.

    Args:
        point_dim: Dimension of input points (typically 3 for XYZ)
        condition_dim: Dimension of the condition/target feature
        latent_dim: Dimension of the latent space
        hidden_dim: Dimension of hidden layers
    """

    def __init__(
        self,
        point_dim: int = 3,
        condition_dim: int = 128,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super(Encoder, self).__init__()
        self.logger = Logger.get_instance()
        self.latent_dim = latent_dim

        # Extract features from current point cloud
        self.point_feature_extractor = PointNetFeatureExtractor(
            input_channels=point_dim,
            output_channels=hidden_dim,
            name="Encoder_PointFeatureExtractor",
        )
        # Separate network for global feature aggregation
        self.global_feature_aggregator = GlobalFeatureAggregator(
            input_channels=hidden_dim,
            output_channels=hidden_dim,
            name="Encoder_GlobalFeatureAggregator",
        )

        # Combine current global feature with target condition
        self.combine_layer = nn.Sequential(
            nn.Linear(hidden_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Direct output to latent feature space (no distribution parameters)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

    @log_function(level=LogLevel.DEBUG)
    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.

        Args:
            x: Current point cloud of shape [B, N, point_dim]
            condition: Target global feature of shape [B, condition_dim]

        Returns:
            Tuple containing:
              - latent: Latent feature vector [B, latent_dim]
              - point_features: Per-point features [B, N, hidden_dim]
        """
        # Extract per-point features
        point_features = self.point_feature_extractor(x)

        # Aggregate to global feature
        global_feature = self.global_feature_aggregator(point_features)

        # Combine with condition
        self.logger.debug(
            f"Encoder global feature shape: {global_feature.shape}, condition shape: {condition.shape}"
        )

        combined = torch.cat([global_feature, condition], dim=1)
        combined = self.combine_layer(combined)

        self.logger.debug(
            f"Encoder combined feature stats - mean: {combined.mean().item():.3f}, "
            f"std: {combined.std().item():.3f}"
        )

        # Direct mapping to latent space (no distribution parameters)
        latent = self.fc_latent(combined)

        # Log latent statistics
        self.logger.debug(
            f"Encoder latent stats - mean: {latent.mean().item():.3f}, std: {latent.std().item():.3f}, "
            f"min: {latent.min().item():.3f}, max: {latent.max().item():.3f}"
        )

        return latent, point_features


class Decoder(nn.Module):
    """
    Decoder predicts contact probability and orientation for each point.
    Uses the latent code, target condition, and current point cloud's per-point features.

    Args:
        point_dim: Dimension of input points (typically 3 for XYZ)
        condition_dim: Dimension of the condition/target feature
        latent_dim: Dimension of the latent space
        hidden_dim: Dimension of hidden layers
    """

    def __init__(
        self,
        point_dim: int = 3,
        condition_dim: int = 128,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super(Decoder, self).__init__()
        self.logger = Logger.get_instance()
        self.hidden_dim = hidden_dim

        # Process latent code and target condition
        self.combine_layer = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Process per-point features with the expanded latent-condition vector
        self.point_decoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Contact probability head (binary classification)
        self.contact_head = nn.Linear(hidden_dim, 1)

        # Orientation head (3D vector)
        self.orientation_head = nn.Linear(hidden_dim, point_dim)

        # Push_distance
        self.push_distance_head = nn.Linear(hidden_dim, 1)

    @log_function(level=LogLevel.DEBUG)
    def forward(
        self, latent: torch.Tensor, condition: torch.Tensor, point_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder.

        Args:
            latent: Latent feature vector of shape [B, latent_dim]
            condition: Target global feature of shape [B, condition_dim]
            point_features: Per-point features of shape [B, N, hidden_dim]

        Returns:
            Tuple containing:
              - contact_prob: Contact probability logits [B, N, 1]
              - orientation: Orientation predictions [B, N, point_dim] (normalized)
        """
        batch_size, num_points, _ = point_features.shape

        # Combine latent and condition
        self.logger.debug(
            f"Decoder latent shape: {latent.shape}, condition shape: {condition.shape}"
        )

        combined = torch.cat([latent, condition], dim=1)
        combined = self.combine_layer(combined)

        self.logger.debug(
            f"Decoder combined feature stats - mean: {combined.mean().item():.3f}, "
            f"std: {combined.std().item():.3f}"
        )

        # Expand combined feature to every point
        combined_expanded = combined.unsqueeze(1).expand(-1, num_points, -1)
        point_input = torch.cat(
            [combined_expanded, point_features], dim=2
        )  # [B, N, 2*hidden_dim]

        # Reshape for efficient processing
        point_input = point_input.reshape(-1, point_input.size(2))
        point_output = self.point_decoder(point_input)  # [B*N, hidden_dim]

        # Log point decoder output stats
        flat_output_stats = {
            "mean": point_output.mean().item(),
            "std": point_output.std().item(),
            "near_zero": (point_output.abs() < 1e-6).float().mean().item(),
        }
        self.logger.debug(
            f"Decoder point output stats - mean: {flat_output_stats['mean']:.3f}, "
            f"std: {flat_output_stats['std']:.3f}, "
            f"near-zero: {flat_output_stats['near_zero']:.2%}"
        )

        # Predict contact probability (binary) and orientation (3D vector)
        contact_prob = self.contact_head(point_output)  # [B*N, 1]
        orientation = self.orientation_head(point_output)  # [B*N, point_dim]
        push_distance = self.push_distance_head(point_output)

        # Reshape back to batch format
        contact_prob = contact_prob.view(batch_size, num_points, 1)
        orientation = orientation.view(batch_size, num_points, -1)
        push_distance = push_distance.view(batch_size, num_points, 1)

        # Log contact logits distribution before sigmoid
        self.logger.debug(
            f"Contact logits - mean: {contact_prob.mean().item():.3f}, "
            f"std: {contact_prob.std().item():.3f}, "
            f"min: {contact_prob.min().item():.3f}, "
            f"max: {contact_prob.max().item():.3f}"
        )

        # Very important: check if all contact predictions are heavily biased toward 0
        neg_bias = (contact_prob < -5).float().mean().item()
        self.logger.debug(f"Contact strong negative bias proportion: {neg_bias:.2%}")

        # Get sigmoid probabilities for additional insights
        contact_probs_sigmoid = torch.sigmoid(contact_prob)
        high_prob = (contact_probs_sigmoid > 0.5).float().mean().item()
        self.logger.debug(f"Contact points proportion (p > 0.5): {high_prob:.2%}")

        # Log push_distance distribution
        self.logger.debug(
            f"push distance -- mean: {push_distance.mean().item():.3f}, "
            f"std: {push_distance.std().item():.3f}, "
            f"min: {push_distance.min().item():.3f}, "
            f"max: {push_distance.max().item():.3f}"
        )

        # Normalize orientation vectors
        orientation_norm = torch.norm(orientation, p=2, dim=2, keepdim=True)
        self.logger.debug(
            f"Orientation norm - mean: {orientation_norm.mean().item():.3f}, "
            f"std: {orientation_norm.std().item():.3f}"
        )

        orientation = F.normalize(orientation, p=2, dim=2)

        return contact_prob, orientation, push_distance


class PointCloudEncoderDecoder(nn.Module):
    """
    Encoder-Decoder architecture for point cloud processing.
    Predicts contact points and their orientations between current and target point clouds.

    Args:
        point_dim: Dimension of input points (typically 3 for XYZ)
        transformation_dim: Dimension of input transformation (x,y,z,qw,qx,qy,qz)
        condition_dim: Dimension of the condition feature
        latent_dim: Dimension of the latent space
        hidden_dim: Dimension of hidden layers
    """

    def __init__(
        self,
        point_dim: int = 3,
        transformation_dim: int =7,
        condition_dim: int = 128,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super(PointCloudEncoderDecoder, self).__init__()
        self.logger = Logger.get_instance()
        self.latent_dim = latent_dim

        # Target transformation processing
        self.condition_feature_extractor = FullConnectedLayerFeatureExtractor(
            input_channels=transformation_dim,
            output_channels=condition_dim,
            name="EncoderDecoder_ConditionFeatureExtractor",
        )

        # Main encoder-decoder architecture
        self.encoder = Encoder(
            point_dim=point_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        self.decoder = Decoder(
            point_dim=point_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

        # Initialize tracking of epoch stats
        self.epoch = 0
        self.train_stats = {"contact_pred": [], "pos_ratio": []}

    @log_function(level=LogLevel.DEBUG)
    def forward(
        self, x: torch.Tensor, goal_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Encoder-Decoder.

        Args:
            x: Current point cloud [B, N, point_dim]
            goal_x: Target transformation [B, M]

        Returns:
            Tuple containing:
              - contact_prob: Predicted contact probability logits [B, N, 1]
              - orientation: Predicted orientation per point [B, N, point_dim]
        """
        # Log input statistics
        self.logger.debug(
            f"EncoderDecoder forward - Input x shape: {x.shape}, goal_x shape: {goal_x.shape}"
        )
        self.logger.debug(
            f"Input x stats - mean: {x.mean().item():.3f}, std: {x.std().item():.3f}"
        )
        self.logger.debug(
            f"Goal x stats - mean: {goal_x.mean().item():.3f}, std: {goal_x.std().item():.3f}"
        )

        # Extract condition from target point cloud
        condition_features = self.condition_feature_extractor(goal_x)

        # Encode current point cloud to get latent representation
        latent, point_features = self.encoder(x, condition_features)

        # Decode to predictions
        # Contact output remains as logits; the loss applies the sigmoid.
        contact_prob, orientation, push_distance = self.decoder(
            latent,
            condition_features,
            point_features,
        )

        # Track training statistics for epoch-level monitoring
        if self.training:
            contact_sigmoid = torch.sigmoid(contact_prob)
            pos_ratio = (contact_sigmoid > 0.5).float().mean().item()

            self.train_stats["contact_pred"].append(contact_sigmoid.mean().item())
            self.train_stats["pos_ratio"].append(pos_ratio)

            # Log rolling average once every 10 batches
            if len(self.train_stats["contact_pred"]) % 10 == 0:
                avg_contact = np.mean(self.train_stats["contact_pred"][-10:])
                avg_pos = np.mean(self.train_stats["pos_ratio"][-10:])

                self.logger.debug(
                    f"Epoch {self.epoch} - Last 10 batches: "
                    f"avg contact: {avg_contact:.4f}, "
                    f"avg pos ratio: {avg_pos:.4f}"
                )

        return contact_prob, orientation, push_distance

    def set_epoch(self, epoch: int):
        """Set the current epoch number for logging"""
        self.epoch = epoch
        self.train_stats = {"contact_pred": [], "pos_ratio": []}
        self.logger.info(f"Starting epoch {epoch}")


class EncoderDecoderLoss(nn.Module):
    """
    Custom loss function for the Encoder-Decoder with contact point prediction and orientation.
    Compatible with the existing collate_fn that pads data.

    Args:
        contact_weight: Weight for positive contact points (for handling class imbalance)
        orientation_weight: Weight for orientation loss
    """

    def __init__(
        self,
        contact_weight: float = 10.0,
        orientation_weight: float = 0.1,
        distance_weight:float = 1,
    ):
        super(EncoderDecoderLoss, self).__init__()
        self.logger = Logger.get_instance()
        self.contact_weight = contact_weight
        self.orientation_weight = orientation_weight
        self.distance_weight = distance_weight

        # Register pos_weight as a buffer to ensure it gets moved to the right device
        self.register_buffer("pos_weight", torch.tensor([contact_weight]))

        # Initialize loss tracking
        self.batch_count = 0
        self.reset_stats()

    def reset_stats(self):
        """Reset tracking statistics"""
        self.stats = {
            "total_loss": [],
            "contact_loss": [],
            "orientation_loss": [],
            "push_distance_loss":[],
            "gt_pos_ratio": [],
            "pred_pos_ratio": [],
        }

    @log_function(level=LogLevel.DEBUG)
    def forward(
        self,
        pred_contact: torch.Tensor,
        pred_orientation: torch.Tensor,
        pred_push_distance: torch.Tensor,
        gt_contact: torch.Tensor,
        gt_orientation: torch.Tensor,
        gt_push_distance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the loss components.

        Args:
            pred_contact: Predicted contact logits [B, N, 1]
            pred_orientation: Predicted orientations [B, N, point_dim]
            gt_contact: Ground truth contact labels [B, N, 1] (binary)
            gt_orientation: Ground truth orientations [B, N, point_dim]

        Returns:
            total_loss: Combined loss value
            contact_loss: Loss for contact prediction
            orientation_loss: Loss for orientation prediction
        """
        # Get the batch size.
        batch_size = pred_contact.shape[0]

        # Log ground truth statistics
        gt_positive = (gt_contact > 0.5).float().mean().item()
        self.logger.debug(f"Ground truth positive ratio: {gt_positive:.4f}")

        # Critical check: if there are no positive examples, warn loudly
        if gt_positive == 0:
            self.logger.warning(
                "NO POSITIVE EXAMPLES IN BATCH - all ground truth contact points are 0!"
            )

        # Convert integer contact points to float and ensure right shape for BCE loss
        gt_contact_float = (
            gt_contact.float()
            if gt_contact.dim() == 3
            else gt_contact.float().unsqueeze(-1)
        )  # [B, N, 1]

        # Contact point loss with positive class weighting
        contact_loss = F.binary_cross_entropy_with_logits(
            pred_contact,
            gt_contact_float,
            pos_weight=self.pos_weight.to(pred_contact.device),
        )

        # Predict contact probabilities for logging
        pred_probs = torch.sigmoid(pred_contact)
        pred_positive = (pred_probs > 0.5).float().mean().item()
        self.logger.debug(f"Predicted positive ratio: {pred_positive:.4f}")

        # Log prediction statistics to help identify issues.
        if pred_positive == 0:
            self.logger.warning(
                "NO POSITIVE EXAMPLES IN BATCH - all predicted contact points are 0!"
            )

        # Track alignment between predictions and ground truth
        correct_neg = (
            ((pred_probs < 0.5) & (gt_contact_float < 0.5)).float().sum().item()
        )
        correct_pos = (
            ((pred_probs > 0.5) & (gt_contact_float > 0.5)).float().sum().item()
        )
        total_neg = (gt_contact_float < 0.5).float().sum().item()
        total_pos = (gt_contact_float > 0.5).float().sum().item()

        neg_accuracy = correct_neg / max(total_neg, 1)
        pos_accuracy = correct_pos / max(total_pos, 1)

        self.logger.debug(
            f"Negative accuracy: {neg_accuracy:.4f}, Positive accuracy: {pos_accuracy:.4f}"
        )

        # Create contact mask (for considering only contact points in orientation loss)
        contact_mask = (gt_contact_float > 0.5).float()  # [B, N, 1]

        # Count contacts per batch item to handle batch normalization correctly
        contacts_per_batch = torch.sum(contact_mask.view(batch_size, -1), dim=1)  # [B]

        # Make sure both orientation vectors are normalized
        pred_orientation = F.normalize(pred_orientation, p=2, dim=2)
        gt_orientation = F.normalize(gt_orientation, p=2, dim=2)

        # Initialize per-batch orientation losses
        batch_orientation_losses = torch.zeros(batch_size, device=pred_contact.device)

        # Calculate orientation loss per batch item
        for b in range(batch_size):
            batch_contacts = contacts_per_batch[b]
            if batch_contacts > 0:
                # MSE loss
                diff = pred_orientation[b] - gt_orientation[b]  # [N, D]
                squared_diff = torch.sum(diff**2, dim=1)  # [N]
                masked_squared_diff = squared_diff * contact_mask[b].squeeze(
                    -1
                )  # [N]
                batch_orientation_losses[b] = (
                    torch.sum(masked_squared_diff) / batch_contacts
                )

        # Average orientation loss across the batch
        orientation_loss = torch.mean(batch_orientation_losses)

        # Apply weighting to orientation loss
        orientation_loss = self.orientation_weight * orientation_loss

        # Initialize per-batch push_distance losses
        batch_push_distance_losses = torch.zeros(batch_size, device=pred_contact.device)

        # Calculate push_distance loss per batch item
        for b in range(batch_size):
            batch_contacts = contacts_per_batch[b]
            if batch_contacts > 0:
                # MSE loss
                diff = pred_push_distance[b] - gt_push_distance[b]  # [N, 1]
                squared_diff = torch.sum(diff**2, dim=1)  # [N]
                masked_squared_diff = squared_diff * contact_mask[b].squeeze(
                    -1
                )  # [N]
                batch_push_distance_losses[b] = (
                    torch.sum(masked_squared_diff) / batch_contacts
                )

        # Average orientation loss across the batch
        push_distance_loss = torch.mean(batch_push_distance_losses)

        # Apply weighting to orientation loss
        push_distance_loss = self.distance_weight * push_distance_loss

        # Log loss components
        self.logger.debug(
            f"Loss components - contact: {contact_loss.item():.4f}, "
            f"orientation: {orientation_loss.item():.4f}"
            f"push_distance: {push_distance_loss.item():.4f}"
        )

        # Check for NaN in loss
        if torch.isnan(contact_loss) or torch.isnan(orientation_loss):
            self.logger.critical("NaN detected in loss calculation!")

        # Total loss
        total_loss = contact_loss + orientation_loss + push_distance_loss

        # Track loss statistics
        self.stats["total_loss"].append(total_loss.item())
        self.stats["contact_loss"].append(contact_loss.item())
        self.stats["orientation_loss"].append(orientation_loss.item())
        self.stats["push_distance_loss"].append(push_distance_loss.item())
        self.stats["gt_pos_ratio"].append(gt_positive)
        self.stats["pred_pos_ratio"].append(pred_positive)

        self.batch_count += 1

        # Log rolling average every 10 batches
        if self.batch_count % 10 == 0:
            self.logger.debug(
                f"Loss stats (last 10 batches): "
                f"total={np.mean(self.stats['total_loss'][-10:]):.4f}, "
                f"contact={np.mean(self.stats['contact_loss'][-10:]):.4f}, "
                f"gt_pos={np.mean(self.stats['gt_pos_ratio'][-10:]):.4f}, "
                f"pred_pos={np.mean(self.stats['pred_pos_ratio'][-10:]):.4f}"
            )

        return total_loss, contact_loss, orientation_loss, push_distance_loss


# Additional helper function to debug training process


# Example usage with debug logging
