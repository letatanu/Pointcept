import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.utils.structure import Point
from pointcept.models.losses import build_criteria
import pointops

class SoftSort(nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, scores, features):
        """
        scores: (B, N, 1)
        features: (B, N, C)
        """
        # 1. Compute Soft Permutation Matrix P (B, N, N)
        # P_ij = Probability that element j is sorted to position i
        # We use deterministic SoftSort based on pairwise score differences
        score_diff = scores.transpose(1, 2) - scores # (B, 1, N) - (B, N, 1) -> (B, N, N)
        P = torch.softmax(score_diff / self.tau, dim=-1)

        # 2. Apply Permutation to Features
        # (B, N, N) @ (B, N, C) -> (B, N, C)
        sorted_feat = torch.bmm(P, features)
        return sorted_feat

# class PointSorter(nn.Module):
#     def __init__(self, in_channels, hidden_channels=64, num_orders=1):
#         super().__init__()
#         self.num_orders = num_orders
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, hidden_channels),
#             nn.BatchNorm1d(hidden_channels),
#             nn.GELU(),
#             nn.Linear(hidden_channels, hidden_channels),
#             nn.BatchNorm1d(hidden_channels),
#             nn.GELU(),
#             nn.Linear(hidden_channels, num_orders)
#         )

#     def forward(self, point):
#         # 1. Prepare Input
#         if point.feat is not None:
#             inp = torch.cat([point.coord, point.feat.detach()], dim=1)
#         else:
#             inp = point.coord

#         # 2. Predict Scores
#         scores = torch.sigmoid(self.mlp(inp))

#         # 3. Generate Orders
#         # Add large offset so batches don't mix during global argsort
#         batch_offset = point.batch.unsqueeze(1) * (scores.max().detach() + 10.0)
#         scores_with_batch = scores + batch_offset

#         orders_list = []
#         inverses_list = []
        
#         scores_t = scores_with_batch.transpose(0, 1) # (num_orders, N)
#         for i in range(self.num_orders):
#             order = torch.argsort(scores_t[i])
#             inverse = torch.zeros_like(order)
#             inverse[order] = torch.arange(len(order), device=order.device)
            
#             orders_list.append(order)
#             inverses_list.append(inverse)

#         return scores, torch.stack(orders_list), torch.stack(inverses_list)


class PointSorter(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_orders=1, residual_scale=0.1):
        super().__init__()
        self.num_orders = num_orders
        self.residual_scale = residual_scale # Limit movement to 10% of the scene
        
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
        # 1. Flatten Input (Standardize to N, C)
        coord = point.coord.view(-1, 3)
        N = coord.shape[0]
        
        # Robust Feature Flattening
        if point.feat is not None:
            # Flatten everything after dim 0 into dim 1
            feat = point.feat.view(N, -1) 
            inp = torch.cat([coord, feat.detach()], dim=1)
        else:
            inp = coord

        # 2. Base Score: Handle Multi-Head Serialization
        if hasattr(point, "serialized_inverse"):
            total_elements = point.serialized_inverse.numel()
            
            if total_elements == N * self.num_orders:
                # Case A: (Num_Orders, N) flattened -> Reshape to (N, Num_Orders)
                # PTv3 usually stores as (Order, N), so we view then transpose.
                base_score = point.serialized_inverse.view(self.num_orders, N).transpose(0, 1).float()
                base_score = base_score / N # Normalize to 0-1
            elif total_elements == N:
                # Case B: Single Order (N) -> Reshape (N, 1) -> Expand (N, Num_Orders)
                base_score = point.serialized_inverse.float().view(N, 1) / N
                base_score = base_score.expand(-1, self.num_orders)
            else:
                # Unexpected size? Fallback to Z-coord to avoid crash
                z = coord[:, 2:3]
                base_score = (z - z.min()) / (z.max() - z.min() + 1e-6)
                base_score = base_score.view(N, 1).expand(-1, self.num_orders)
        else:
            # Fallback (No serialization info)
            z = coord[:, 2:3]
            base_score = (z - z.min()) / (z.max() - z.min() + 1e-6)
            base_score = base_score.view(N, 1).expand(-1, self.num_orders)

        # 3. Predict Residual
        # Output (N, num_orders)
        delta = torch.tanh(self.mlp(inp)) * self.residual_scale
        
        # 4. Final Score
        # base_score is now guaranteed to be (N, num_orders)
        scores = base_score + delta

        # 5. Generate Orders
        # Standardize batch
        if hasattr(point, "batch"):
             batch = point.batch.view(N)
        else:
             batch = torch.zeros(N, device=coord.device, dtype=torch.long)

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
        
        # self.sorter = ContextAwareSorter(
        #     in_channels=3, 
        #     hidden_channels=64, 
        #     num_orders=len(self.order), 
        #     k=self.ordering_k
        # )
        self.sorter = PointSorter(
            in_channels=in_channels + 3, 
            hidden_channels=64, 
            num_orders=len(self.order), 
            residual_scale=0.1
        )

    def compute_ordering_loss(self, point, scores):
        loss_list = []
        
        # 1. Config
        chunk_size = 1024 
        num_chunks = 4
        N = scores.shape[0]
        
        # Margin: We want different classes to be separated by score 0.01
        margin = 0.01 
        
        if N < chunk_size:
            return torch.tensor(0.0, device=scores.device)

        for _ in range(num_chunks):
            # --- CHANGE: Local Sampling instead of Global Random ---
            # 1. Pick a random center point
            center_idx = torch.randint(0, N, (1,), device=scores.device)
            center_coord = point.coord[center_idx] # (1, 3)
            
            # 2. Find 1024 Nearest Neighbors (Euclidean Distance)
            # This ensures we sort a LOCAL patch where Residual Sorter works
            dists = ((point.coord - center_coord) ** 2).sum(dim=1)
            _, rand_idx = torch.topk(dists, chunk_size, largest=False)
            # ------------------------------------------------------
            
            # Ground Truth Labels for this chunk
            labels = point.segment[rand_idx].long()
            labels[labels == -1] = 0
            
            # Create Pairwise Masks (B, N, N)
            labels_expanded = labels.unsqueeze(1)
            mask_pos = (labels_expanded == labels_expanded.t()).float()
            mask_neg = 1.0 - mask_pos
            
            # Remove diagonal
            diag = torch.eye(chunk_size, device=scores.device)
            mask_pos = mask_pos - diag

            for k in range(self.sorter.num_orders):
                # Get scores for this chunk: (chunk_size, 1)
                s = scores[rand_idx, k].view(chunk_size, 1)
                
                # Compute Pairwise Distance Matrix
                dist_mat = torch.abs(s - s.t())
                
                # Discriminative Loss (Contrastive)
                # Pull same classes together
                pos_term = (dist_mat * mask_pos).sum() / (mask_pos.sum() + 1e-6)
                
                # Push different classes apart (up to margin)
                neg_term = (torch.relu(margin - dist_mat) * mask_neg).sum() / (mask_neg.sum() + 1e-6)
                
                loss_list.append(pos_term + neg_term)

        # Distribution Regularization (Prevent Collapse)
        loss_dist_list = []
        for k in range(self.sorter.num_orders):
            sorted_scores_k, _ = torch.sort(scores[:, k])
            target = torch.linspace(0, 1, steps=N, device=scores.device)
            loss_dist_list.append(((sorted_scores_k - target) ** 2).mean())
        
        loss_dist = sum(loss_dist_list) / len(loss_dist_list)

        # Final Loss
        return (sum(loss_list) / len(loss_list)) * 100.0 + loss_dist * 10.0
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

    # def compute_ordering_loss(self, point, scores):
    #     # 1. Locality & Distribution (Standard)
    #     idx = pointops.knn_query(self.ordering_k, point.coord, point.offset)[0]
    #     neighbor_scores = scores[idx.long()]
    #     diff = scores.unsqueeze(1) - neighbor_scores
    #     loss_local = (diff ** 2).sum(dim=1).mean()
        
    #     sorted_scores, _ = torch.sort(scores.view(-1))
    #     target = torch.linspace(0, 1, steps=len(sorted_scores), device=scores.device)
    #     loss_dist = ((sorted_scores - target) ** 2).mean()
        
    #     # 2. Hilbert/Z-order Teacher Regularization
    #     # point.serialized_code is (4, N) (e.g., Z, Z-trans, Hilbert, Hilbert-trans)
    #     # We transpose it to (N, 4) to match 'scores'
    #     t_codes = point.serialized_code.float().transpose(0, 1).to(scores.device)
        
    #     # Normalize each column independently to [0, 1] range
    #     # This handles the different coordinate ranges of Z vs Hilbert
    #     t_min = t_codes.min(dim=0, keepdim=True)[0]
    #     t_max = t_codes.max(dim=0, keepdim=True)[0]
        
    #     # Safety: add epsilon to avoid div by zero
    #     t_norm = (t_codes - t_min) / (t_max - t_min + 1e-6)
        
    #     # MSE: Force Head 0->Z, Head 1->Z-trans, etc.
    #     loss_teacher = 0.1 * ((scores - t_norm) ** 2).mean()

    #     return loss_local + loss_dist + loss_teacher

    # def compute_ordering_loss(self, point, scores, teacher_weight=0.0):
    #     # Retrieve neighbors (N, k)
    #     idx = pointops.knn_query(self.ordering_k, point.coord, point.offset)[0].long()
        
    #     # 1. Semantic Locality Loss
    #     # We only want to enforce score smoothness between neighbors OF THE SAME CLASS.
    #     # This allows the curve to "break" at object boundaries.
    #     if hasattr(point, "segment"):
    #         # (N, k) - Labels of neighbors
    #         neighbor_seg = point.segment[idx] 
    #         # (N, 1) - Label of center
    #         center_seg = point.segment.unsqueeze(1)
            
    #         # Mask: 1 if neighbor is same class, 0 otherwise
    #         mask = (neighbor_seg == center_seg).float()
    #         mask = mask.unsqueeze(-1) # (N, k) -> (N, k, 1)
            
    #         # Get scores
    #         neighbor_scores = scores[idx]
    #         diff = scores.unsqueeze(1) - neighbor_scores
            
    #         # Weighted MSE: Only penalize distance if they are the same class
    #         # Add epsilon to mask_sum to prevent division by zero
    #         loss_local = ((diff ** 2) * mask).sum() / (mask.sum() + 1e-6)
    #     else:
    #         # Fallback for validation/test if needed (though loss usually train-only)
    #         neighbor_scores = scores[idx]
    #         diff = scores.unsqueeze(1) - neighbor_scores
    #         loss_local = (diff ** 2).mean()

    #     # 2. Distribution Loss (Keep this to prevent collapse to a single value)
    #     sorted_scores, _ = torch.sort(scores.view(-1))
    #     target = torch.linspace(0, 1, steps=len(sorted_scores), device=scores.device)
    #     loss_dist = ((sorted_scores - target) ** 2).mean()
        
    #     # 3. Teacher Loss (REDUCE WEIGHT or REMOVE)
    #     # If you want to learn a BETTER order, you must relax the teacher.
    #     # Only use this for stability in the first few epochs.
    #     loss_teacher = 0.0
    #     if teacher_weight > 0: # Or decay based on epoch
    #          # ... (your existing teacher code) ...
    #          # Reduce weight significantly, e.g., to 0.01 or 0.0
    #            #     t_codes = point.serialized_code.float().transpose(0, 1).to(scores.device)
        
    #         #     # Normalize each column independently to [0, 1] range
    #         #     # This handles the different coordinate ranges of Z vs Hilbert
    #         t_codes = point.serialized_code.float().transpose(0, 1).to(scores.device)
    #         t_min = t_codes.min(dim=0, keepdim=True)[0]
    #         t_max = t_codes.max(dim=0, keepdim=True)[0]
            
    #         # Safety: add epsilon to avoid div by zero
    #         t_norm = (t_codes - t_min) / (t_max - t_min + 1e-6)
            
    #         # MSE: Force Head 0->Z, Head 1->Z-trans, etc.
    #         loss_teacher = teacher_weight * ((scores - t_norm) ** 2).mean()

    #     # High weight on semantic locality is key
    #     return loss_local * 10.0 + loss_dist + loss_teacher

    def forward(self, data_dict):
        point = Point(data_dict)
        current_epoch = data_dict["epoch"] if self.training else float('inf')

        # 1. Standard Serialization (Populates default Z-order)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        # 2. Run OPTNet Sorter
        scores, learned_order, learned_inverse = self.sorter(point)
        if self.training:
            # Add small random noise to scores before generating order.
            # This prevents the backbone from overfitting to a fixed "perfect" order.
            noise_magnitude = 0.05  # Start with 5% noise
            noise = torch.randn_like(scores) * noise_magnitude
            noisy_scores = scores + noise
            
            # Re-calculate order based on noisy scores
            # We use the noisy order for the backbone, but the CLEAN scores for the Loss
            batch_offset = point.batch.unsqueeze(1) * (noisy_scores.max().detach() + 10.0)
            scores_with_batch = noisy_scores + batch_offset
            
            noisy_orders_list = []
            noisy_inverses_list = []
            for k in range(self.sorter.num_orders):
                order = torch.argsort(scores_with_batch[:, k])
                inv = torch.zeros_like(order)
                inv[order] = torch.arange(order.shape[0], device=order.device)
                noisy_orders_list.append(order)
                noisy_inverses_list.append(inv)
            
            learned_order = torch.stack(noisy_orders_list, dim=0)
            learned_inverse = torch.stack(noisy_inverses_list, dim=0)
        # ----------------------------------------------

        # 2. Apply Order (Noisy for Train, Clean for Val)
        if current_epoch >= self.warmup_epoch:
            point.serialized_order = learned_order
            point.serialized_inverse = learned_inverse

        # 3. Calculate Loss (Use CLEAN scores)
        # We want the Sorter to output clean scores, even if we feed noise to the backbone
        if self.training and self.ordering_loss_weight > 0.0:
            loss_ord = self.compute_ordering_loss(point, scores) # Pass CLEAN scores here
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