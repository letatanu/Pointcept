import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.utils.structure import Point
from pointcept.models.losses import build_criteria
import pointops

class PointSorter(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_orders=1):
        super().__init__()
        self.num_orders = num_orders
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_orders)
        )
        
        # Initialize the last layer to near-zero.
        # This ensures the model starts by outputting the base Z-order (delta ~ 0)
        # and slowly learns to deviate as training progresses.
        nn.init.normal_(self.mlp[-1].weight, std=0.001)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, point):
        # 1. Flatten Input (Standardize to N, C)
        coord = point.coord.view(-1, 3)
        N = coord.shape[0]
        
        if point.feat is not None:
            feat = point.feat.view(N, -1) 
            inp = torch.cat([coord, feat.detach()], dim=1)
        else:
            inp = coord

        # 2. Base Score: Z-order / Hilbert Curve
        # We use the pre-computed serialization as a robust starting point.
        if hasattr(point, "serialized_inverse"):
            total_elements = point.serialized_inverse.numel()
            if total_elements == N * self.num_orders:
                # Reshape to (N, Num_Orders)
                base_score = point.serialized_inverse.view(self.num_orders, N).transpose(0, 1).float()
                base_score = base_score / N 
            elif total_elements == N:
                base_score = point.serialized_inverse.float().view(N, 1) / N
                base_score = base_score.expand(-1, self.num_orders)
            else:
                z = coord[:, 2:3]
                base_score = (z - z.min()) / (z.max() - z.min() + 1e-6)
                base_score = base_score.view(N, 1).expand(-1, self.num_orders)
        else:
            z = coord[:, 2:3]
            base_score = (z - z.min()) / (z.max() - z.min() + 1e-6)
            base_score = base_score.view(N, 1).expand(-1, self.num_orders)

        # 3. Predict Residual
        # REMOVED: residual_scale and tanh. We allow the network to learn full range.
        delta = self.mlp(inp)
        
        # 4. Final Score
        scores = base_score + delta

        # 5. Generate Orders
        if hasattr(point, "batch"):
             batch = point.batch.view(N)
        else:
             batch = torch.zeros(N, device=coord.device, dtype=torch.long)

        # Offset batches to ensure sorting happens within each scene independently
        batch_offset = batch.unsqueeze(1) * (scores.max().detach() + 10.0)
        scores_with_batch = scores + batch_offset

        orders_list = []
        inverses_list = []
        for k in range(self.num_orders):
            order = torch.argsort(scores_with_batch[:, k])
            inv = torch.zeros_like(order)
            inv[order] = torch.arange(order.shape[0], device=order.device)
            orders_list.append(order)
            inverses_list.append(inv)
        
        learned_order = torch.stack(orders_list, dim=0)
        learned_inverse = torch.stack(inverses_list, dim=0)
        
        return scores, learned_order, learned_inverse

@MODELS.register_module("OPTNet")
class OPTNet(PointTransformerV3):
    def __init__(self, 
                 in_channels=6, 
                 ordering_loss_weight=0.1,  # Reduced default weight
                 ordering_k=16, 
                 warmup_epoch=5,            # Added warmup default
                 **kwargs):
        
        super().__init__(in_channels=in_channels, **kwargs)
        
        self.ordering_loss_weight = ordering_loss_weight
        self.ordering_k = ordering_k
        self.warmup_epoch = warmup_epoch
        
        # Input channels: Coord (3) + Color (3) = 6. 
        # Sorter takes coord + feat.
        self.sorter = PointSorter(
            in_channels=in_channels + 3, 
            hidden_channels=64, 
            num_orders=len(self.order)
        )

    def compute_ordering_loss(self, point, scores):
        """
        Loss Objective:
        1. Semantic Smoothness: Points of the SAME class that are SPATIALLY close 
           should have similar scores. This keeps the curve "pure" locally.
        2. Distribution Regularization: Scores should span [0, 1] uniformly.
        """
        # --- 1. Semantic Local Smoothness ---
        # Query k-Nearest Neighbors in 3D space
        idx = pointops.knn_query(self.ordering_k, point.coord, point.offset)[0].long() # (N, k)
        
        # Fetch scores: (N, num_orders) -> (N, k, num_orders)
        neighbor_scores = scores[idx]
        center_scores = scores.unsqueeze(1)
        
        # Fetch segments: (N,) -> (N, k)
        neighbor_seg = point.segment[idx]
        center_seg = point.segment.unsqueeze(1)
        
        # Mask: Only penalize score differences if neighbors belong to the SAME class
        # This allows the curve to "jump" across object boundaries (e.g. floor -> table)
        valid_mask = (neighbor_seg != -1) & (center_seg != -1)
        same_class_mask = (neighbor_seg == center_seg) & valid_mask
        same_class_mask = same_class_mask.float().unsqueeze(-1) # (N, k, 1)
        
        # MSE Loss masked
        score_diff = (center_scores - neighbor_scores) ** 2
        loss_semantic = (score_diff * same_class_mask).sum() / (same_class_mask.sum() + 1e-6)

        # --- 2. Distribution Regularization ---
        # Prevent all scores from collapsing to a single value
        loss_dist_list = []
        N = scores.shape[0]
        
        # Downsample for efficiency if N is large
        if N > 4096:
            perm = torch.randperm(N, device=scores.device)[:4096]
            scores_sample = scores[perm]
        else:
            scores_sample = scores

        for k in range(self.sorter.num_orders):
            # Sort scores and compare to a perfect uniform linear ramp
            s_sorted, _ = torch.sort(scores_sample[:, k])
            target = torch.linspace(0, 1, steps=s_sorted.shape[0], device=scores.device)
            loss_dist_list.append(((s_sorted - target) ** 2).mean())
        
        loss_dist = sum(loss_dist_list) / len(loss_dist_list)

        # Weighting: 
        # Semantic smoothness is the primary goal. 
        # Distribution is a regularization term (typically smaller weight).
        return loss_semantic + loss_dist * 0.5

    def forward(self, data_dict):
        point = Point(data_dict)
        current_epoch = data_dict.get("epoch", 0) if self.training else float('inf')

        # 1. Standard Serialization (Populates default Z-order / Hilbert)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        # 2. Run OPTNet Sorter
        # We always run the sorter to get 'scores' for the loss
        scores, learned_order, learned_inverse = self.sorter(point)

        # 3. Apply Order
        # ONLY apply the learned order after the warmup period.
        # This allows the backbone to learn stable features using Z-order first.
        if current_epoch >= self.warmup_epoch:
            if self.training:
                # Add noise during training to make backbone robust to slight ordering shifts
                noise = torch.randn_like(scores) * 0.01
                noisy_scores = scores + noise
                
                # Re-generate order from noisy scores
                batch_offset = point.batch.unsqueeze(1) * (noisy_scores.max().detach() + 10.0)
                scores_with_batch = noisy_scores + batch_offset
                
                orders_list = []
                inverses_list = []
                for k in range(self.sorter.num_orders):
                    order = torch.argsort(scores_with_batch[:, k])
                    inv = torch.zeros_like(order)
                    inv[order] = torch.arange(order.shape[0], device=order.device)
                    orders_list.append(order)
                    inverses_list.append(inv)
                
                point.serialized_order = torch.stack(orders_list, dim=0)
                point.serialized_inverse = torch.stack(inverses_list, dim=0)
            else:
                # Validation/Test: Use the clean learned order
                point.serialized_order = learned_order
                point.serialized_inverse = learned_inverse

        # 4. Calculate Loss
        if self.training and self.ordering_loss_weight > 0.0:
            loss_ord = self.compute_ordering_loss(point, scores)
            point["ordering_loss"] = loss_ord * self.ordering_loss_weight

        # 5. Backbone Forward
        point = self.embedding(point)
        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)

        return point


@MODELS.register_module("OPTNetSegmentor")
class OPTNetSegmentor(nn.Module):
    def __init__(self, backbone, criteria, num_classes, backbone_out_channels=None):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        self.criteria = build_criteria(criteria)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, data_dict):
        point = self.backbone(data_dict)
        seg_logits = self.seg_head(point.feat)

        if self.training:
            loss = self.criteria(seg_logits, data_dict["segment"])
            return_dict = dict(seg_loss=loss)
            
            if "ordering_loss" in point:
                loss = loss + point["ordering_loss"]
                return_dict["ordering_loss"] = point["ordering_loss"]
            
            return_dict["loss"] = loss
            return return_dict
        
        elif "segment" in data_dict:
            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
            
        return dict(seg_logits=seg_logits)