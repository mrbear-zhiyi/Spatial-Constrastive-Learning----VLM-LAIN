# utils/spatial_contrastive.py
# Module: Spatial Contrastive Learning for Human-Object Interaction
# This module implements spatial relation encoding and spatial-aware contrastive loss for HOI tasks.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class SpatialRelationEncoder(nn.Module):
    """
    Module: Spatial Relation Encoder
    Purpose: Encode pairwise spatial features between human bounding box and object bounding box into fixed-dimension embedding.
    Input: human bounding boxes, object bounding boxes (x1,y1,x2,y2 format)
    Output: spatial embedding vector
    """
    def __init__(self, output_dim: int = 256):
        super().__init__()
        # Build MLP encoder + layer normalization to map 25-d spatial raw features to output embedding
        self.encoder = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, human_boxes: torch.Tensor, object_boxes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass entry
        Args:
            human_boxes: batch of human bboxes, shape [B,4]
            object_boxes: batch of object bboxes, shape [B,4]
        Returns:
            encoded spatial embedding tensor
        """
        spatial_feats = self._compute_spatial_features(human_boxes, object_boxes)
        return self.encoder(spatial_feats)

    def _compute_spatial_features(self, human_boxes: torch.Tensor, object_boxes: torch.Tensor) -> torch.Tensor:
        """
        Submodule: Raw spatial feature calculator
        Compute all 25-dimensional handcrafted spatial features between human and object bboxes, then concatenate.
        Feature components: rel_pos(4) + scale_ratio(2) + iou(1) + direction(4) + contact(1) + pos_relation(6) + orientation(4) + scale_relation(3)
        """
        B = human_boxes.size(0)
        device = human_boxes.device

        # 1. Calculate center coordinate and width/height size of human and object bounding boxes
        human_center = (human_boxes[:, :2] + human_boxes[:, 2:]) / 2
        object_center = (object_boxes[:, :2] + object_boxes[:, 2:]) / 2
        human_size = human_boxes[:, 2:] - human_boxes[:, :2]
        object_size = object_boxes[:, 2:] - object_boxes[:, :2]

        # 2. Relative position feature (4-dim): normalized center offset + normalized size ratio
        rel_pos = torch.cat([
            (human_center - object_center) / (object_size + 1e-6),
            human_size / (object_size + 1e-6)
        ], dim=-1)

        # 3. Area scale ratio feature (2-dim): human/object area and object/human area
        human_area = human_size[:, 0] * human_size[:, 1]
        object_area = object_size[:, 0] * object_size[:, 1]
        scale_ratio = torch.stack([
            human_area / (object_area + 1e-6),
            object_area / (human_area + 1e-6)
        ], dim=-1)

        # 4. IoU feature (1-dim): Intersection over Union between human and object bboxes
        inter_x1 = torch.max(human_boxes[:, 0], object_boxes[:, 0])
        inter_y1 = torch.max(human_boxes[:, 1], object_boxes[:, 1])
        inter_x2 = torch.min(human_boxes[:, 2], object_boxes[:, 2])
        inter_y2 = torch.min(human_boxes[:, 3], object_boxes[:, 3])
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union_area = human_area + object_area - inter_area
        iou = (inter_area / (union_area + 1e-6)).unsqueeze(-1)

        # 5. Direction binary encoding (4-dim): judge human center above/below, left/right relative to object
        direction = torch.zeros(B, 4, device=device)
        direction[:, 0] = (human_center[:, 1] < object_center[:, 1]).float()
        direction[:, 1] = (human_center[:, 1] > object_center[:, 1]).float()
        direction[:, 2] = (human_center[:, 0] < object_center[:, 0]).float()
        direction[:, 3] = (human_center[:, 0] > object_center[:, 0]).float()

        # 6. Distance and contact degree feature (1-dim): gap distance between boxes, contact score by exponential decay
        dx = torch.max(
            torch.zeros_like(human_boxes[:, 0]),
            torch.max(human_boxes[:, 0] - object_boxes[:, 2], object_boxes[:, 0] - human_boxes[:, 2])
        )
        dy = torch.max(
            torch.zeros_like(human_boxes[:, 1]),
            torch.max(human_boxes[:, 1] - object_boxes[:, 3], object_boxes[:, 1] - human_boxes[:, 3])
        )
        distance = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        contact = torch.exp(-distance * 10).unsqueeze(-1)

        # 7. Extra spatial relation sub-features
        # pos_relation(6): horizontal, vertical and overlap flag; orientation(4): angle bin one-hot; scale_relation(3): area ratio bin
        pos_relation = self._encode_positional_relation(human_boxes, object_boxes)
        orientation = self._encode_orientation(human_boxes, object_boxes)
        scale_relation = self._encode_scale_relation(human_boxes, object_boxes)

        # Concatenate all spatial components to get final 25-dimensional raw spatial feature
        spatial_features = torch.cat([
            rel_pos, scale_ratio, iou, direction, contact,
            pos_relation, orientation, scale_relation
        ], dim=-1)

        return spatial_features

    def _encode_positional_relation(self, human_boxes, object_boxes):
        """
        Submodule: Positional relation encoding (6 dims)
        Encode horizontal relative position, vertical relative position and axis-aligned overlap status
        """
        B = human_boxes.size(0)
        device = human_boxes.device
        horizontal = torch.stack([
            (human_boxes[:, 0] < object_boxes[:, 0]).float(),
            (human_boxes[:, 0] > object_boxes[:, 0]).float()
        ], dim=-1)
        vertical = torch.stack([
            (human_boxes[:, 1] < object_boxes[:, 1]).float(),
            (human_boxes[:, 1] > object_boxes[:, 1]).float()
        ], dim=-1)
        overlap = torch.stack([
            ((human_boxes[:, 2] > object_boxes[:, 0]) & (human_boxes[:, 0] < object_boxes[:, 2])).float(),
            ((human_boxes[:, 3] > object_boxes[:, 1]) & (human_boxes[:, 1] < object_boxes[:, 3])).float()
        ], dim=-1)
        return torch.cat([horizontal, vertical, overlap], dim=-1)

    def _encode_orientation(self, human_boxes, object_boxes):
        """
        Submodule: Orientation angle encoding (4 dims)
        Compute direction angle from object center to human center, quantize angle into 4 bins with one-hot encoding
        """
        B = human_boxes.size(0)
        device = human_boxes.device
        center_diff = (human_boxes[:, :2] + human_boxes[:, 2:]) / 2 - (object_boxes[:, :2] + object_boxes[:, 2:]) / 2
        angle = torch.atan2(center_diff[:, 1], center_diff[:, 0])
        angle_deg = (angle * 180 / 3.1415926) % 360
        angle_bin = torch.floor(angle_deg / 90).long().clamp(0, 3)
        orientation = torch.zeros(B, 4, device=device)
        orientation[torch.arange(B), angle_bin] = 1.0
        return orientation

    def _encode_scale_relation(self, human_boxes, object_boxes):
        """
        Submodule: Scale relation encoding (3 dims)
        Quantize human/object area ratio into three bins: smaller, 1~2x, >=2x
        """
        B = human_boxes.size(0)
        device = human_boxes.device
        human_area = (human_boxes[:, 2] - human_boxes[:, 0]) * (human_boxes[:, 3] - human_boxes[:, 1])
        object_area = (object_boxes[:, 2] - object_boxes[:, 0]) * (object_boxes[:, 3] - object_boxes[:, 1])
        scale_ratio = human_area / (object_area + 1e-6)
        scale_bin = torch.stack([
            (scale_ratio < 1.0).float(),
            ((scale_ratio >= 1.0) & (scale_ratio < 2.0)).float(),
            (scale_ratio >= 2.0).float()
        ], dim=-1)
        return scale_bin


class SpatialContrastiveLoss(nn.Module):
    """
    Module: Spatial Aware Contrastive Loss
    Purpose: Contrastive loss combining visual feature similarity and spatial embedding similarity,
             with curriculum dynamic threshold for rare class samples and cosine annealing weight.
    """
    def __init__(
            self,
            temperature: float = 0.07,
            base_threshold: float = 0.5,
            delta_theta: float = 0.2,
            e_trans: int = 15,
            e_warm: int = 3,
            total_epochs: int = 20
    ):
        super().__init__()
        self.temperature = temperature
        self.base_threshold = base_threshold
        self.delta_theta = delta_theta
        self.e_trans = e_trans
        self.e_warm = e_warm
        self.total_epochs = total_epochs
        # Initialize spatial relation encoder submodule
        self.spatial_encoder = SpatialRelationEncoder(output_dim=256)

        # Print initialization log: No emoji/icon inside this print statement
        print(f"[SpatialContrastiveLoss] Init: temp={temperature}, base_th={base_threshold}, delta={delta_theta}")
        # Print curriculum scheduler log: No emoji/icon inside this print statement
        print(
            f"[SpatialContrastiveLoss] Curriculum schedule: E_trans={e_trans}, E_warm={e_warm} (implement smooth interpolation to guarantee Lipschitz continuity)")

    def forward(
            self,
            iam_features: torch.Tensor,
            human_boxes: torch.Tensor,
            object_boxes: torch.Tensor,
            hois_labels: torch.Tensor,
            is_rare_mask: torch.Tensor,  # [B], bool tensor, mark rare-class samples in current batch
            current_epoch: int,
            return_metrics: bool = False
    ):
        B = iam_features.size(0)
        device = iam_features.device

        # ==========================================
        # Submodule 1: Smooth interpolation strategy (Paper Eq.4 & Eq.5)
        # Objective: Address reviewer concerns on Lipschitz continuity and experimental validation.
        # Use continuous linear interpolation to avoid step jump between epoch 15 and 16.
        # ==========================================
        if current_epoch <= self.e_trans:
            current_threshold = self.base_threshold
            interpolation_weight = 0.0
        elif current_epoch < self.e_trans + self.e_warm:
            # Phase 2 Transition: linear smooth interpolation to maintain Lipschitz continuity
            interpolation_weight = (current_epoch - self.e_trans) / self.e_warm
            current_threshold = self.base_threshold + self.delta_theta * interpolation_weight
        else:
            # Phase3 Stable: reach maximum relaxed threshold
            current_threshold = self.base_threshold + self.delta_theta
            interpolation_weight = 1.0

        # Cosine annealing for dynamic contrast loss weight β(E)
        progress = current_epoch / self.total_epochs
        beta_weight = 0.02 + (0.1 - 0.02) * 0.5 * (1 - math.cos(math.pi * progress))

        # ==========================================
        # Submodule 2: Pairwise similarity matrix calculation
        # ==========================================
        iam_norm = F.normalize(iam_features, dim=-1, p=2)
        visual_sim = (iam_norm @ iam_norm.T) / self.temperature

        spatial_embeddings = self.spatial_encoder(human_boxes, object_boxes)
        spatial_norm = F.normalize(spatial_embeddings, dim=-1, p=2)
        spatial_sim = (spatial_norm @ spatial_norm.T) / self.temperature

        # Joint similarity (Paper Eq.2): weighted sum of visual similarity and spatial similarity
        alpha_weight = 1.0 - beta_weight
        joint_sim = visual_sim * alpha_weight + spatial_sim * beta_weight

        # ==========================================
        # Submodule3: Frequency-Adaptive Positive Sample Masking
        # ==========================================
        label_sim = (hois_labels @ hois_labels.T > 0).float()
        label_sim.fill_diagonal_(0)
        positive_mask = label_sim.clone()

        spatial_sim_no_diag = spatial_sim.clone()
        torch.diagonal(spatial_sim_no_diag).fill_(-float('inf'))

        # For rare samples: treat pair as positive if spatial similarity >= dynamic threshold
        rare_mask_2d = is_rare_mask.unsqueeze(1) & is_rare_mask.unsqueeze(0)
        spatial_positive = (spatial_sim_no_diag >= current_threshold).float() * rare_mask_2d.float()

        positive_mask = torch.max(positive_mask, spatial_positive)
        positive_mask.fill_diagonal_(0)

        # ==========================================
        # Submodule4: Contrastive loss calculation with numerical stability protection
        # ==========================================
        pos_count = positive_mask.sum().item()
        if pos_count == 0:
            loss = torch.tensor(0.0, device=device)
        else:
            joint_sim_max = joint_sim.max(dim=1, keepdim=True)[0]
            exp_sim = torch.exp(joint_sim - joint_sim_max)

            pos_sim = (exp_sim * positive_mask).sum(dim=1, keepdim=True) + 1e-8
            neg_mask = 1.0 - positive_mask
            neg_mask.fill_diagonal_(0)
            neg_sim = (exp_sim * neg_mask).sum(dim=1, keepdim=True) + 1e-8

            ratio = torch.clamp(pos_sim / (pos_sim + neg_sim), min=1e-8, max=1 - 1e-8)
            loss = -torch.log(ratio).mean()
            loss = torch.clamp(loss, min=0.0, max=10.0) * beta_weight

        # ==========================================
        # Submodule5: Diagnostic metrics for rebuttal experimental validation
        # ==========================================
        if return_metrics:
            metrics = {
                'loss_value': loss.item(),
                'current_epoch': current_epoch,
                'current_threshold': current_threshold,
                'interpolation_weight': interpolation_weight,
                'beta_weight': beta_weight,
                'positive_pairs_ratio': pos_count / (B * (B - 1) + 1e-8),
            }
            return loss, metrics
        else:
            return loss, None


# ==========================================
# Module: Auxiliary helper function for plotter callback
# ==========================================
def log_stability_metrics(
        metrics: Dict[str, float],
        plotter_instance: Optional[object] = None,
        save_plot_at_epoch: int = 20
):
    """
    Helper Function: log and feed stability metrics to visualization plotter
    Invoked at end of each training epoch. Pass metrics to plotter and save rebuttal validation figure at target epoch.

    Args:
        metrics: metric dict returned by forward function of SpatialContrastiveLoss
        plotter_instance: instance of CurriculumStabilityPlotter
        save_plot_at_epoch: target epoch to save final validation figure (usually total_epochs)
    """
    if plotter_instance is not None:
        plotter_instance.update(metrics)
        if metrics.get('current_epoch', 0) >= save_plot_at_epoch:
            plotter_instance.plot_and_save(filename=f"stability_validation_ep{save_plot_at_epoch}.png")


__all__ = ['SpatialRelationEncoder', 'SpatialContrastiveLoss', 'log_stability_metrics']
