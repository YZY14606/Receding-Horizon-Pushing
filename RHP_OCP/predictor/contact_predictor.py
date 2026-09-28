import os
from predictor import config as args
from predictor.predictor_utils import visulize_pred_results
import torch
import numpy as np
from utils.logging_utils import Logger, LogLevel, log_function
from network.model import PointCloudEncoderDecoder
from scipy.spatial.transform import Rotation as R #(x,y,z,w)
import trimesh
import torch.nn.functional as F
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from scipy.special import expit

class contact_predictor:

    def __init__(self):
        # Setup logging
        self.logger = Logger.get_instance()
        self.logger.set_level(LogLevel.INFO)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        model_path = 'checkpoints/model_100'
        model_name = "checkpoint_epoch_149.pth"
        self.model_path = os.path.join(model_path, model_name)
        # Set model
        self.model = self.set_model()

        # Visualize
        self.vis = args.visualization

    @log_function(level=LogLevel.DEBUG)
    def set_model(self):
        """Initialize model, loss function, optimizer, and scheduler"""
        self.logger.info("Initializing model")

        # Initialize model
        model = PointCloudEncoderDecoder(
            point_dim=args.point_dim,
            condition_dim=args.condition_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            ).to(self.device)

        # Load model state
        path = self.model_path
        if os.path.isfile(path):
            self.logger.info(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        return model.eval()
    

    @log_function(level=LogLevel.DEBUG)
    def predict(
        self,
        current_pc,
        local_frame_future_pose,
        visualization_dir=None,
    ):
        # Normalize the current pointcloud
        furthest_distance = np.max(np.linalg.norm(current_pc, axis=1))
        current_pc_nrd = current_pc.copy() / furthest_distance.item()

        local_future_pose_nrd = local_frame_future_pose.copy()
        local_future_pose_nrd[:3] = local_frame_future_pose[:3] / furthest_distance.item()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        with torch.no_grad():
            # Convert pointcloud to input shape (B,N,3)
            input_current_pc = torch.tensor(current_pc_nrd,dtype=torch.float).view(1,-1,args.point_dim).to(self.device)
            input_local_frame_future_pose = torch.tensor(local_future_pose_nrd,dtype=torch.float).view(1,args.transformation_dim).to(self.device)

            # Forward pass
            pred_contact, pred_orientation, pred_push_distance = self.model(input_current_pc, input_local_frame_future_pose)




        # Convert tensors to numpy 
        normalized_orientation = F.normalize(pred_orientation, p=2, dim=-1)
        pred_orientation_np = normalized_orientation[0].cpu().detach().numpy()
        pred_push_distance_np = pred_push_distance[0].cpu().detach().numpy()

        # Predicted quality and orientation
        # Convert sigmoid probabilities to mask using threshold
        pred_contact_np = torch.sigmoid(pred_contact[0]).cpu().detach().numpy()

        # Get the max value index
        max_index = self.select_contact_point(current_pc_nrd.copy(),pred_contact_np.squeeze(),pred_orientation_np.copy())

        # Get the output shape
        output_contact = current_pc[max_index]
        output_orientation = pred_orientation_np[max_index]
        output_push_distance = pred_push_distance_np[max_index] * furthest_distance.item()

        # Visualize
        if self.vis:
            # Construct the future point cloud.
            fut_pose = input_local_frame_future_pose.view(args.transformation_dim).clone().cpu().numpy()
            fut_pose[3:7] = np.array([fut_pose[4],fut_pose[5],fut_pose[6],fut_pose[3]])
            rot_matrix = R.from_quat(fut_pose[3:7]).as_matrix()  # 3x3
            T2 = np.eye(4)
            T2[:3, :3] = rot_matrix
            T2[:3, 3] = fut_pose[:3]
            # The current frame is the identity, so the relative transform is T2.
            T_final = T2
            future_pc = trimesh.transform_points(current_pc_nrd, T_final)
            # Batch evaluation passes a fully resolved directory. The global
            # default remains only for standalone predictor debugging.
            save_path = os.fspath(
                args.vis_dir if visualization_dir is None else visualization_dir
            )
            self.vis_dir = visulize_pred_results(current_pc_np = current_pc_nrd,future_pc_np =future_pc,
                                  pred_contact_np = pred_contact_np, pred_orientation_np =pred_orientation_np, push_idx = max_index,
                                  pred_push_distance_np = pred_push_distance_np,save_path = save_path, logger = self.logger)
            

        return output_contact, output_orientation, output_push_distance
    

    def _self_modified_sigmoid(self,original_score):
        input_score = 5 * original_score
        final_score = (expit(input_score) - 0.5 ) * 2
        return final_score
 

    def select_contact_point(self,current_pc_nrd,pred_contact_np,pred_orientation_np):

        valid_indices = np.arange(current_pc_nrd.shape[0])
        valid_probs = pred_contact_np.copy()

        # Keep points whose predicted probability exceeds 0.5.
        pos_mask = valid_probs > 0.5
        pos_idx = valid_indices[pos_mask]
        if len(pos_idx) >= 10:
            result_idx = pos_idx
        else:
            # Fall back to the ten highest-probability points.
            top10_rel = np.argsort(valid_probs)[-10:][::-1]
            result_idx = valid_indices[top10_rel]

        z_all = current_pc_nrd[:, 2]
        mask = z_all < z_all.min() + 0.25 * (z_all.max() - z_all.min())
        xy_points = current_pc_nrd[mask, :2]
        # Build the planar convex hull.
        hull = ConvexHull(xy_points)
        hull_polygon = Polygon(xy_points[hull.vertices])
        # Score each candidate using its projected distance and height.
        final_scores = []
        for idx in result_idx:
            z = current_pc_nrd[idx, 2]
            orientation_nrd = pred_orientation_np[idx][:2] / np.linalg.norm(pred_orientation_np[idx][:2])

            d_score = self.calculate_projection(hull_polygon,orientation_nrd)

            # Compute the geometric score d / z.
            if z > z_all.min() + 0.02 * (z_all.max() - z_all.min()):
                score = d_score / z
                score = self._self_modified_sigmoid(original_score = score)
            else:
                score = 0  # Avoid unstable division near the lowest surface.
            actor_score = pred_contact_np[idx]
            final_score = actor_score * score

            final_scores.append(final_score)

        # Select the candidate with the highest combined score.
        max_idx = result_idx[np.argmax(final_scores)]

        return max_idx
    
    def calculate_projection(self,hull_polygon,orientation_nrd):

        scores = []

        # Polygon exterior coordinates include the closing vertex.
        coords = np.array(hull_polygon.exterior.coords)
        for p1, p2 in zip(coords[:-1], coords[1:]):
            edge = p2 - p1
            n = np.array([edge[1], -edge[0]])  # Rotate 90 degrees for a normal.
            n /= np.linalg.norm(n)

            # Orient every normal outward.
            if np.dot(n, p1) < 0:
                n = -n

            proj = np.dot(orientation_nrd, n)
            if proj > 0:  # Keep only edges facing the push direction.
                dist = np.dot(n, p1)
                scores.append(dist/proj)

        if scores:
            score = min(scores)
            return score
