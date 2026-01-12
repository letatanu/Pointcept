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

    def forward(self, point):
        # 1. Prepare Input
        if point.feat is not None:
            inp = torch.cat([point.coord, point.feat.detach()], dim=1)
        else:
            inp = point.coord

        # 2. Predict Scores
        scores = torch.sigmoid(self.mlp(inp))

        # 3. Generate Orders
        # Add large offset so batches don't mix during global argsort
        batch_offset = point.batch.unsqueeze(1) * (scores.max().detach() + 10.0)
        scores_with_batch = scores + batch_offset

        orders_list = []
        inverses_list = []
        
        scores_t = scores_with_batch.transpose(0, 1) # (num_orders, N)
        for i in range(self.num_orders):
            order = torch.argsort(scores_t[i])
            inverse = torch.zeros_like(order)
            inverse[order] = torch.arange(len(order), device=order.device)
            
            orders_list.append(order)
            inverses_list.append(inverse)

        return scores, torch.stack(orders_list), torch.stack(inverses_list)


class ContextAwareSorter(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_orders=1, k=32):
        super().__init__()
        self.k = k
        self.num_orders = num_orders
        
        # PointNet-style Local Aggregation
        # Input: [x_i, x_j - x_i] -> captures local shape
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )
        
        # Global MLP to predict score from aggregated features
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_orders)
        )

    def forward(self, point):
        # 1. Local Neighborhood Aggregation
        # Query neighbors (k-NN)
        idx = pointops.knn_query(self.k, point.coord, point.offset)[0] # (N, k)
        
        # Group coordinates: (N, k, 3)
        neighbor_xyz = point.coord[idx.long()] 
        center_xyz = point.coord.unsqueeze(1).repeat(1, self.k, 1)
        
        # Feature: Relative coordinates [center, neighbor - center]
        # Shape: (N, k, 6) -> Flatten to (N*k, 6) for Linear
        local_feat = torch.cat([center_xyz, neighbor_xyz - center_xyz], dim=-1)
        
        # Apply MLP to every edge
        N, k, C = local_feat.shape
        local_feat = self.local_mlp(local_feat.view(N*k, C)) # (N*k, Hidden)
        local_feat = local_feat.view(N, k, -1) # (N, k, Hidden)
        
        # Max Pooling to get shape descriptor
        shape_feat = local_feat.max(dim=1)[0] # (N, Hidden)

        # 2. Predict Scores from Shape Context
        scores = torch.sigmoid(self.global_mlp(shape_feat))

        # 3. Generate Orders (Standard)
        batch_offset = point.batch.unsqueeze(1) * (scores.max().detach() + 10.0)
        scores_with_batch = scores + batch_offset

        orders_list = []
        inverses_list = []
        scores_t = scores_with_batch.transpose(0, 1) 
        
        for i in range(self.num_orders):
            order = torch.argsort(scores_t[i])
            inverse = torch.zeros_like(order)
            inverse[order] = torch.arange(len(order), device=order.device)
            orders_list.append(order)
            inverses_list.append(inverse)

        return scores, torch.stack(orders_list), torch.stack(inverses_list)

@MODELS.register_module("OPTNet")
class OPTNet(PointTransformerV3):
    def __init__(self, 
                 in_channels=6, 
                 ordering_loss_weight=1.0, 
                 ordering_k=16, 
                 warmup_epoch=0,
                 **kwargs):
        
        super().__init__(in_channels=in_channels, **kwargs)
        
        self.ordering_loss_weight = ordering_loss_weight
        self.ordering_k = ordering_k
        self.warmup_epoch = warmup_epoch
        
        self.sorter = ContextAwareSorter(
            in_channels=3, 
            hidden_channels=64, 
            num_orders=len(self.order), 
            k=self.ordering_k
        )

    # def compute_ordering_loss(self, point, scores):
    #     """
    #     1. Locality Loss: Neighbors should have similar scores.
    #     2. Distribution Loss: Scores should be uniformly distributed in [0, 1] (prevents collapse).
    #     """
    #     # --- 1. Locality Loss ---
    #     idx = pointops.knn_query(self.ordering_k, point.coord, point.offset)[0]
    #     neighbor_scores = scores[idx.long()]
        
    #     # (Score - Neighbor_Score)^2
    #     diff = scores.unsqueeze(1) - neighbor_scores
    #     loss_locality = (diff ** 2).sum(dim=1).mean()
        
    #     # --- 2. Distribution Loss (New) ---
    #     # Sort the scores and compare against a perfect linear ramp [0, ..., 1]
    #     # This forces the model to use the full range of values.
    #     sorted_scores, _ = torch.sort(scores.view(-1))
    #     target = torch.linspace(0, 1, steps=len(sorted_scores), device=scores.device)
    #     loss_dist = ((sorted_scores - target) ** 2).mean()
        
    #     # Combine losses (Weighted equally or adjust as needed)
    #     return loss_locality + loss_dist

    def compute_ordering_loss(self, point, scores):
        # 1. Locality & Distribution (Standard)
        idx = pointops.knn_query(self.ordering_k, point.coord, point.offset)[0]
        neighbor_scores = scores[idx.long()]
        diff = scores.unsqueeze(1) - neighbor_scores
        loss_local = (diff ** 2).sum(dim=1).mean()
        
        sorted_scores, _ = torch.sort(scores.view(-1))
        target = torch.linspace(0, 1, steps=len(sorted_scores), device=scores.device)
        loss_dist = ((sorted_scores - target) ** 2).mean()
        
        # 2. Hilbert/Z-order Teacher Regularization
        loss_teacher = 0.0
        if "serialized_code" in point.keys():
             # point.serialized_code is (4, N) (e.g., Z, Z-trans, Hilbert, Hilbert-trans)
             # We transpose it to (N, 4) to match 'scores'
             t_codes = point.serialized_code.float().transpose(0, 1).to(scores.device)
             
             # Normalize each column independently to [0, 1] range
             # This handles the different coordinate ranges of Z vs Hilbert
             t_min = t_codes.min(dim=0, keepdim=True)[0]
             t_max = t_codes.max(dim=0, keepdim=True)[0]
             
             # Safety: add epsilon to avoid div by zero
             t_norm = (t_codes - t_min) / (t_max - t_min + 1e-6)
             
             # MSE: Force Head 0->Z, Head 1->Z-trans, etc.
             loss_teacher = 0.1 * ((scores - t_norm) ** 2).mean()

        return loss_local + loss_dist + loss_teacher

    def forward(self, data_dict):
        point = Point(data_dict)
        current_epoch = data_dict["epoch"] if self.training else float('inf')

        # 1. Standard Serialization (Populates default Z-order)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        # 2. Run OPTNet Sorter
        scores, learned_order, learned_inverse = self.sorter(point)

        # 3. Apply Order Strategy
        if current_epoch >= self.warmup_epoch:
            # We update the serialized_order. The Window Attention in PTv3 
            # uses this to shuffle the points before computing attention.
            point.serialized_order = learned_order
            point.serialized_inverse = learned_inverse
            # Note: We do NOT update point.serialized_code. 
            # Pooling layers must rely on geometric Morton codes to merge neighbors correctly.

        # 4. Calculate Loss
        if self.training and self.ordering_loss_weight > 0:
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

        # 1. Training Mode
        if self.training:
            loss = self.criteria(seg_logits, data_dict["segment"])
            return_dict = dict(seg_loss=loss)
            
            if "ordering_loss" in point:
                loss = loss + point["ordering_loss"]
                return_dict["ordering_loss"] = point["ordering_loss"]
            
            return_dict["loss"] = loss
            return return_dict
        
        # 2. Validation Mode (Not training, but has labels)
        elif "segment" in data_dict:
            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
            
        # 3. Test Mode (No labels)
        return dict(seg_logits=seg_logits)