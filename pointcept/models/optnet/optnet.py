import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.utils.structure import Point
from pointcept.models.losses import build_criteria
from pointcept.models.utils.serialization import encode
import pointops
import torch.nn.init as init
import torch.nn.functional as F



from pointcept.models.utils.structure import Point
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import SerializedPooling

# class SortedWindowPooling(nn.Module):
#     """
#     O(N) pooling directly on the 1D sorted order from PointSorter.
    
#     Valid because _locality_loss enforces:  score_neighbors ≈ score_i
#     → consecutive points in sorted order ARE 3D spatial neighbors.
    
#     Every `pool_size` consecutive sorted points → 1 super-point.
#     No grid hashing, no scatter, no kNN — just reshape + mean.
#     """
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         pool_size=8,        # 8 ≈ stride=2 in PTv3 octree (2^3 = 8 cells merged)
#         code_depth=10,
#         traceable=True,
#         norm_layer=nn.LayerNorm,
#         act_layer=nn.GELU,
#     ):
#         super().__init__()
#         self.pool_size = pool_size
#         self.code_depth = code_depth
#         self.traceable = traceable
#         self.proj = nn.Linear(in_channels, out_channels, bias=False)
#         self.norm = norm_layer(out_channels)
#         self.act = act_layer()

#     def forward(self, point: Point):
#         order = point.serialized_order[0]   # [N]: sorted_pos → orig_idx
#         P = self.pool_size
#         device = order.device

#         # ------------------------------------------------------------------
#         # 1. Batch-safe clipping
#         # Batch index lives in high bits of serialized_code, so sorted order
#         # naturally groups per-batch. We clip each batch's tail to a multiple
#         # of P to avoid cross-batch window pollution.
#         # ------------------------------------------------------------------
#         sorted_batch = point.batch[order]
#         _, counts = torch.unique_consecutive(sorted_batch, return_counts=True)
#         per_batch_valid = (counts // P) * P        # floor to multiple of P
#         valid_ends = torch.cumsum(per_batch_valid, dim=0)
#         valid_starts = torch.cat([
#             torch.zeros(1, dtype=torch.long, device=device), valid_ends[:-1]
#         ])

#         valid_orig_idx = torch.cat([
#             order[valid_starts[b]:valid_ends[b]]
#             for b in range(len(counts))
#             if valid_ends[b] > valid_starts[b]
#         ], dim=0)                                   # [N_valid]

#         N_valid = valid_orig_idx.shape[0]
#         n_groups = N_valid // P

#         # ------------------------------------------------------------------
#         # 2. Pool features: sort → reshape → mean  (the key O(N) operation)
#         # ------------------------------------------------------------------
#         sorted_feat = point.feat[valid_orig_idx]                    # [N_valid, C_in]
#         pooled_feat = sorted_feat.view(n_groups, P, -1).mean(dim=1) # [n_groups, C_in]

#         # ------------------------------------------------------------------
#         # 3. Representative point per group (center of window)
#         # ------------------------------------------------------------------
#         center_pos = torch.arange(n_groups, device=device) * P + (P // 2)
#         center_orig_idx = valid_orig_idx[center_pos]               # [n_groups]

#         # ------------------------------------------------------------------
#         # 4. Build new Point
#         # ------------------------------------------------------------------
#         new_point = Point()
#         new_point.coord      = point.coord[center_orig_idx]
#         new_point.grid_coord = torch.div(
#             point.grid_coord[center_orig_idx], P, rounding_mode='floor'
#         )
#         new_point.batch  = point.batch[center_orig_idx]
#         new_point.offset = pointops.compute_offset(new_point.batch)
#         new_point.feat   = self.act(self.norm(self.proj(pooled_feat)))

#         # ------------------------------------------------------------------
#         # 5. Pool scores → re-serialize for next Transformer block
#         # ------------------------------------------------------------------
#         sorted_scores  = point.scores[valid_orig_idx]                    # [N_valid, 1]
#         pooled_scores  = sorted_scores.view(n_groups, P, -1).mean(dim=1) # [n_groups, 1]
#         new_point.scores = pooled_scores

#         code       = scores_to_hierarchical_code(pooled_scores, new_point.batch, depth=self.code_depth)
#         code_2d    = code.unsqueeze(0)
#         new_order  = torch.argsort(code_2d, dim=1)
#         new_inverse = torch.zeros_like(new_order).scatter_(
#             1, new_order,
#             torch.arange(new_order.shape[1], device=device).unsqueeze(0).expand_as(new_order)
#         )
#         new_point.serialized_code    = code_2d
#         new_point.serialized_order   = new_order
#         new_point.serialized_inverse = new_inverse
#         new_point.serialized_depth   = self.code_depth

#         # ------------------------------------------------------------------
#         # 6. Traceability for decoder unpooling
#         # ------------------------------------------------------------------
#         if self.traceable:
#             group_ids        = torch.arange(n_groups, device=device).repeat_interleave(P)
#             pooling_inverse  = torch.full((order.shape[0],), n_groups - 1,
#                                           dtype=torch.long, device=device)
#             pooling_inverse[valid_orig_idx] = group_ids
#             new_point.pooling_inverse = pooling_inverse
#             new_point.pooling_parent  = point

#         return new_point

class OffsetSorter(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64):
        super().__init__()
        # Tiny, fast MLP. No Fourier features required.
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh() # Bounds the output to [-1, 1]
        )
        # Scale factor limits how far a point can jump in the sorted list
        self.offset_scale = 0.05 

    def forward(self, point):
        # 1. Get the perfect deterministic spatial code already calculated by PTv3
        # z_code = point.serialized_code[0].float()
        z = point.serialized_code
        # Handle both [1, N] tensor and list of tensors
        if hasattr(z, "dim") and z.dim() == 2:
            z = z[0]
        elif isinstance(z, (list, tuple)):
            z = z[0]
            
        z_code = z.float()

        
        # Normalize the base score to [0, 1]
        base_score = (z_code - z_code.min()) / (z_code.max() - z_code.min() + 1e-6)
        
        # 2. Predict the semantic offset using Coord(3) + Color(3)
        inp = point.feat if point.feat is not None else point.coord
        offset = self.mlp(inp).squeeze(-1)
        
        # 3. Final Score = Spatial Base + Semantic Nudge
        final_scores = base_score + (offset * self.offset_scale)
        
        # Clamp to avoid extreme outliers
        return torch.clamp(final_scores, min=1e-5, max=1 - 1e-5).unsqueeze(1)

# class PointSorter(nn.Module):
#     """
#     Lightweight Sorter that uses PTv3's perfect spatial Z-code as a base,
#     and applies a learned semantic offset via a simple MLP.
#     """
#     def __init__(self, in_channels=6, hidden_channels=64, ordering_k=16, loss_weights=[1, 0, 0, 1], tau=0.1, num_classes=13):
#         super().__init__()
#         self.num_classes = num_classes
        
#         # Tiny, fast MLP. No Fourier features or ResNet blocks required.
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, hidden_channels),
#             nn.BatchNorm1d(hidden_channels),
#             nn.ReLU(),
#             nn.Linear(hidden_channels, 1),
#             nn.Tanh() # Bounds the output to [-1, 1]
#         )
        
#         # Scale factor limits how far a point can jump in the sorted list
#         self.offset_scale = 0.05 
        
#         # NOTE: Make sure your loss_weights has a value > 0 for the first element (locality)
#         self.ordering_loss = OrderingLoss(ordering_k=ordering_k, loss_weights=loss_weights, tau=tau, ignore_index=-1)

#     def forward(self, point):
#         # 1. Get the perfect deterministic spatial code calculated by PTv3
#         z_code = point.serialized_code[0].float()
        
#         # Normalize the base score to [0, 1]
#         base_score = (z_code - z_code.min()) / (z_code.max() - z_code.min() + 1e-6)
        
#         # 2. Predict the semantic offset using Coord(3) + Color(3)
#         inp = point.feat if point.feat is not None else point.coord
#         offset = self.mlp(inp).squeeze(-1)
        
#         # 3. Final Score = Spatial Base + Semantic Nudge
#         final_scores = base_score + (offset * self.offset_scale)
        
#         # Clamp to avoid extreme outliers
#         learned_scores = torch.clamp(final_scores, min=1e-5, max=1 - 1e-5).unsqueeze(1)
        
#         # Store for loss computation
#         if self.training:
#             self._last_feat = self.mlp[0:3](inp) # Save hidden features for global loss
            
#         return learned_scores

#     def compute_loss(self, scores, coords, batch_ids, offset, z_target=None, labels=None):
#         features = getattr(self, '_last_feat', None)
#         return self.ordering_loss(
#             scores, coords, batch_ids, offset, 
#             z_target=z_target, 
#             features=features,
#             labels=labels # IMPORTANT: Pass labels down to OrderingLoss
#         )

class SortedWindowPooling(nn.Module):
    def __init__(self, in_channels, out_channels, 
            pool_size=8, stride=2, 
            code_depth=10, traceable=True, 
            norm_layer=nn.LayerNorm, 
            act_layer=nn.GELU, **kwargs):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride  # Fix: explicitly define the spatial stride
        self.code_depth = code_depth
        self.traceable = traceable

        # Added for Centroid-Aware Voxelization: MLP for offset encoding
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, in_channels, bias=False),
            norm_layer(in_channels),
            act_layer(),
            nn.Linear(in_channels, in_channels, bias=False)
        )
        
        self.proj = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, point: Point):
        order = point.serialized_order[0]
        P = self.pool_size
        device = order.device

        # 1. Batch-safe clipping
        sorted_batch = point.batch[order]
        _, counts = torch.unique_consecutive(sorted_batch, return_counts=True)
        per_batch_valid = (counts // P) * P
        valid_ends = torch.cumsum(per_batch_valid, dim=0)
        valid_starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), valid_ends[:-1]])

        valid_orig_idx = torch.cat([
            order[valid_starts[b]:valid_ends[b]]
            for b in range(len(counts)) if valid_ends[b] > valid_starts[b]
        ], dim=0)

        N_valid = valid_orig_idx.shape[0]
        n_groups = N_valid // P

        # ------------------------------------------------------------------
        # 2. Centroid-Aware Voxelization 
        # ------------------------------------------------------------------
        sorted_coord = point.coord[valid_orig_idx].view(n_groups, P, 3)     
        sorted_feat = point.feat[valid_orig_idx].view(n_groups, P, -1)      

        # Calculate true physical centroid 
        centroid = sorted_coord.mean(dim=1)                                 
        
        # Calculate local spatial offset and encode to positional features
        offset = sorted_coord - centroid.unsqueeze(1)                       
        pos_enc = self.pos_mlp(offset)                                      
        
        # Inject positional encoding to point features before pooling
        encoded_feat = sorted_feat + pos_enc                                
        pooled_feat = encoded_feat.mean(dim=1)                              

        # ------------------------------------------------------------------
        # 3. Build new Point
        # ------------------------------------------------------------------
        center_pos = torch.arange(n_groups, device=device) * P + (P // 2)
        center_orig_idx = valid_orig_idx[center_pos]

        new_point = Point()
        new_point.coord = centroid # Use computed centroid as coordinate
        
        # Divide grid_coord by `stride` (e.g., 2), NOT `pool_size`
        new_point.grid_coord = torch.div(
            point.grid_coord[center_orig_idx], self.stride, rounding_mode='floor'
        )
        new_point.batch  = point.batch[center_orig_idx]
        new_point.offset = pointops.compute_offset(new_point.batch)
        new_point.feat   = self.act(self.norm(self.proj(pooled_feat)))

        # ------------------------------------------------------------------
        # 4. Pool scores & serialization logic
        # ------------------------------------------------------------------
        # sorted_scores  = point.scores[valid_orig_idx]
        # pooled_scores  = sorted_scores.view(n_groups, P, -1).mean(dim=1)
        # new_point.scores = pooled_scores

        # code       = scores_to_hierarchical_code(pooled_scores, new_point.batch, depth=self.code_depth)
        # code_2d    = code.unsqueeze(0)
        # new_order  = torch.argsort(code_2d, dim=1)
        # new_inverse = torch.zeros_like(new_order).scatter_(
        #     1, new_order,
        #     torch.arange(new_order.shape[1], device=device).unsqueeze(0).expand_as(new_order)
        # )
        # Create a dummy code that just counts up [0, 1, 2, ..., n_groups-1]
        # We add the batch ID to the high bits just like standard PTv3 serialization
        batch_shifted = new_point.batch.long() << 30 # Shift batch ID to high bits
        sequential_code = torch.arange(n_groups, dtype=torch.long, device=device)
        new_code = batch_shifted + sequential_code

        new_code_2d = new_code.unsqueeze(0) # Shape: [1, n_groups]

        # 2. Because they are sequential, the order is just 0 to n_groups-1
        new_order = torch.arange(n_groups, dtype=torch.long, device=device).unsqueeze(0)

        # 3. And the inverse mapping is identical to the order
        new_inverse = new_order.clone()

        new_point.serialized_code    = new_code_2d
        new_point.serialized_order   = new_order
        new_point.serialized_inverse = new_inverse
        new_point.serialized_depth   = self.code_depth

        if self.traceable:
            group_ids        = torch.arange(n_groups, device=device).repeat_interleave(P)
            pooling_inverse  = torch.full((order.shape[0],), n_groups - 1, dtype=torch.long, device=device)
            pooling_inverse[valid_orig_idx] = group_ids
            new_point.pooling_inverse = pooling_inverse
            new_point.pooling_parent  = point

        return new_point


class GridPoolingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, code_depth=10,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, traceable=True):
        super().__init__()
        self.stride = stride
        self.code_depth = code_depth
        self.traceable = traceable
        self.proj = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, point: Point):
        # 1. Geometric downsampling
        new_grid_coord = torch.div(point.grid_coord, self.stride, rounding_mode='floor')
        cluster_idx, unique_idx = pointops.unique_and_cluster(new_grid_coord, point.batch)

        # 2. Build new Point (coord/batch from representative points)
        new_point = Point()
        new_point.grid_coord = new_grid_coord[unique_idx]
        new_point.coord      = point.coord[unique_idx]
        new_point.batch      = point.batch[unique_idx]
        new_point.offset     = pointops.compute_offset(new_point.batch)

        # 3. Pool features
        pooled_feat    = pointops.scatter_mean(point.feat, cluster_idx, dim=0)
        new_point.feat = self.act(self.norm(self.proj(pooled_feat)))

        # 4. ← KEY: Reuse existing codes directly from _build_serialization_from_scores
        #    unique_idx are the representative points — their codes are already computed.
        #    No need to re-run scores_to_hierarchical_code.
        existing_codes = point.serialized_code[0][unique_idx]   # [n_new_points]

        code_2d     = existing_codes.unsqueeze(0)
        new_order   = torch.argsort(code_2d, dim=1)
        new_inverse = torch.zeros_like(new_order).scatter_(
            1, new_order,
            torch.arange(new_order.shape[1], device=new_order.device)
                .unsqueeze(0).expand_as(new_order)
        )

        new_point.serialized_code    = code_2d
        new_point.serialized_order   = new_order
        new_point.serialized_inverse = new_inverse
        new_point.serialized_depth   = self.code_depth

        # 5. Traceability for decoder unpooling
        if self.traceable:
            new_point.pooling_inverse = cluster_idx
            new_point.pooling_parent  = point

        return new_point

# ---------------------------------------------------------------------------
# Helper: score → hierarchical integer code
# ---------------------------------------------------------------------------

def scores_to_hierarchical_code(scores, batch, depth=10):
    """
    Quantize scores [0,1] to depth*3 bits and pack with batch index.
    Compatible with PTv3's SerializedPooling which does code >> (k*3).
    With depth=10: 2^30 levels, rank(code)==rank(score), no ties for N<1B.
    """
    scores_flat = scores.view(-1)
    max_code    = (1 << (depth * 3)) - 1
    quantized   = (scores_flat * max_code).long().clamp(0, max_code)
    code        = batch.long() << (depth * 3) | quantized
    return code


# ---------------------------------------------------------------------------
# Ordering Loss  (locality + distribution + Z-regression, no contrastive)
# ---------------------------------------------------------------------------

class OrderingLoss(nn.Module):
    def __init__(self, ordering_k=16, loss_weights=[0,0,0,1], tau=0.1, ignore_index=-1):
        super().__init__()
        self.ordering_k = ordering_k
        # Unpack the list: [locality, distribution, z_regression, global_feature]
        self.w_loc, self.w_dist, self.w_z, self.w_glob = loss_weights
        self.tau = tau
        self.ignore_index = ignore_index 


    def forward(self, scores, coords, batch_ids, offset, z_target=None, features=None, labels=None):
        scores_1d = scores.view(-1, 1)
        total = torch.tensor(0.0, device=scores.device)
        loss_dict = {}

        # 1. Locality Loss
        if self.w_loc > 0:
            if offset is None:
                _, counts = torch.unique_consecutive(batch_ids.long(), return_counts=True)
                offset = torch.cumsum(counts, dim=0).int()
            else:
                offset = offset.int()
            if labels is not None:
                loss_locality = self.semantic_locality_loss(scores_1d, coords, labels, offset) * self.w_loc
            else:
                loss_locality = self._locality_loss(scores_1d, coords, offset) * self.w_loc
            total = total + loss_locality
            loss_dict["locality"] = loss_locality.item()

        # 2. Distribution Loss
        if self.w_dist > 0:
            if offset is None:
                _, counts = torch.unique_consecutive(batch_ids.long(), return_counts=True)
                offset = torch.cumsum(counts, dim=0).int()
            else:
                offset = offset.int()
            loss_distribution = self._fps_distribution_loss(scores_1d, coords, batch_ids, offset) * self.w_dist
            total = total + loss_distribution
            loss_dict["distribution"] = loss_distribution.item()


        # 3. Z-Regression Loss
        if z_target is not None and self.w_z > 0:
            s = scores_1d.squeeze(1)
            if z_target.dim() == 1:
                best_target = z_target
            else:
                with torch.no_grad():
                    mse_per_order = ((z_target - s.unsqueeze(0)) ** 2).mean(dim=1)
                    best_k = mse_per_order.argmin()
                best_target = z_target[best_k]

            loss_z = F.mse_loss(s, best_target) * self.w_z
            if not torch.isnan(loss_z):
                total = total + loss_z
                loss_dict["z_regression"] = loss_z.item()

        # 4. Global Feature Loss (ICCV 2023)
        if features is not None and self.w_glob > 0:
            loss_global = self._global_feature_loss(scores_1d, features, batch_ids) * self.w_glob
            total = total + loss_global
            loss_dict["global_feature"] = loss_global.item()

        return total, loss_dict

    def _locality_loss(self, scores, coords, offset):
        N = scores.shape[0]
        if N < 2 or torch.isnan(coords).any():
            return torch.tensor(0.0, device=scores.device)
        k = min(self.ordering_k, N - 1)
        idx = pointops.knn_query(k, coords.contiguous(), offset)[0].long()
        idx = torch.clamp(idx, 0, N - 1)
        neighbor_scores = scores[idx]
        return F.mse_loss(scores.unsqueeze(1).expand_as(neighbor_scores), neighbor_scores)

    def _fps_distribution_loss(self, scores, coords, batch_ids, offset, num_centroids=128):
        """
        Uses Farthest Point Sampling to find geometrically distant anchor points,
        then forces their scores to be uniformly distributed across [0, 1].
        """
        batch_ids_long = batch_ids.long()
        num_batches = int(batch_ids_long.max().item()) + 1
        total_dist_loss = torch.tensor(0.0, device=scores.device)
        count = 0
        
        # In Pointcept, offset is the cumulative sum of points per batch.
        # If it's None, we calculate it.
        if offset is None:
            _, counts = torch.unique_consecutive(batch_ids_long, return_counts=True)
            offset = torch.cumsum(counts, dim=0).int()
        else:
            offset = offset.int()

        # The Pointcept FPS function requires a target offset array to know
        # how many points to sample per batch.
        # We want `num_centroids` points per batch, so we create a new offset array:
        # [128, 256, 384, ...]
        new_counts = torch.full((num_batches,), num_centroids, dtype=torch.int32, device=scores.device)
        new_offset = torch.cumsum(new_counts, dim=0).int()

        # We can only perform FPS if EVERY batch has at least `num_centroids` points.
        # Otherwise, the CUDA kernel will crash.
        min_points_in_batch = offset[0] if num_batches == 1 else min(offset[0], (offset[1:] - offset[:-1]).min())
        if min_points_in_batch < num_centroids:
            # If a room has too few points, we skip the distribution loss for this step
            # to avoid crashing the FPS CUDA kernel.
            return torch.tensor(0.0, device=scores.device)

        # 1. Run Farthest Point Sampling using Pointcept's 1D flattened format
        # Returns the indices of the sampled points in the flattened array
        fps_idx = pointops.farthest_point_sampling(coords.contiguous(), offset, new_offset).long()
        
        # 2. Compute loss per batch
        for b in range(num_batches):
            # Extract the FPS indices for this specific batch
            start_idx = b * num_centroids
            end_idx = (b + 1) * num_centroids
            b_fps_idx = fps_idx[start_idx:end_idx]
            
            # Extract the predicted scores for these distant anchors
            anchor_scores = scores[b_fps_idx].flatten()
            
            # 3. Force these distant points to stretch across the [0, 1] spectrum
            sorted_anchors, _ = torch.sort(anchor_scores)
            target = torch.linspace(0, 1, steps=num_centroids, device=scores.device)
            
            total_dist_loss = total_dist_loss + F.l1_loss(sorted_anchors, target)
            count += 1
            
        return total_dist_loss / max(count, 1)


    def _global_feature_loss(self, scores, features, batch_ids):
        """
        Computes the global feature contribution loss.
        Target score for point i: 
        s_target = mean( 2 * sigmoid( (f_i - F_global) / tau ) )
        """
        batch_ids_long = batch_ids.long()
        num_batches = int(batch_ids_long.max().item()) + 1
        
        # 1. Compute global feature per batch (Max Pooling)
        global_max = torch.empty((num_batches, features.shape[1]), device=features.device)
        for b in range(num_batches):
            mask = batch_ids_long == b
            if mask.any():
                global_max[b] = features[mask].max(dim=0)[0]
            else:
                global_max[b] = 0.0
                
        # 2. Expand global feature to per-point shape [N, D]
        g = global_max[batch_ids_long]  
        
        # 3. Compute the normalized difference between point feature and global feature
        # The paper uses tau (temperature) to scale the difference before sigmoid
        diff = (features - g) / self.tau
        
        # 4. Map the difference to [0, 1] using Sigmoid, scaled by 2 as per ICCV'23
        # Then average across all feature dimensions (D) to get a single scalar target per point
        target_scores = (2.0 * torch.sigmoid(diff)).mean(dim=1, keepdim=True)
        
        # 5. The loss is the MSE between the Sorter's predicted scores and the target feature difference
        return F.mse_loss(scores, target_scores.detach())

    def semantic_locality_loss(self, scores, coords, labels, offset):
        N = scores.shape[0]
        if N < 2 or torch.isnan(coords).any():
            return torch.tensor(0.0, device=scores.device)
            
        k = min(self.ordering_k, N - 1)
        
        # 1. Find K-nearest spatial neighbors
        idx = pointops.knn_query(k, coords.contiguous(), offset)[0].long()
        idx = torch.clamp(idx, 0, N - 1)
        
        neighbor_scores = scores[idx] # [N, K, 1]
        neighbor_labels = labels[idx] # [N, K]
        
        curr_scores = scores.unsqueeze(1).expand_as(neighbor_scores)
        curr_labels = labels.unsqueeze(1).expand_as(neighbor_labels)
        
        # 2. Create masks for same class vs different class
        valid_mask = (curr_labels != self.ignore_index) & (neighbor_labels != self.ignore_index)
        same_class = (curr_labels == neighbor_labels) & valid_mask
        diff_class = (curr_labels != neighbor_labels) & valid_mask
        
        # 3. PULL: Spatially close points of the SAME class should have identical 1D scores
        pull_loss = F.mse_loss(curr_scores[same_class], neighbor_scores[same_class]) if same_class.any() else 0.0
        
        # 4. PUSH: Spatially close points of DIFFERENT classes should be separated in the 1D sequence
        margin = 0.02 # A small margin creates a local sequence boundary without breaking global spatial Z-order
        score_diff = torch.abs(curr_scores[diff_class] - neighbor_scores[diff_class])
        push_loss = F.relu(margin - score_diff).mean() if diff_class.any() else 0.0
        
        return pull_loss + push_loss


# ---------------------------------------------------------------------------
# PointSorter
# ---------------------------------------------------------------------------

# class PointSorter(nn.Module):
#     """
#     Predicts a scalar score per point.

#     Input: point.feat  — already = coord(3) + color(3) = 6 channels
#            (set by feat_keys=('coord', 'color') in the dataset config)

#     During training, GT segment labels are fused residually.
#     """

#     def __init__(
#         self,
#         in_channels=6,     # == len(feat_keys) channels: coord(3)+color(3)
#         hidden_channels=64,
#         ordering_k=16,
#         z_loss_weight=10.0,
#         num_classes=13,
#     ):
#         super().__init__()
#         self.num_classes = num_classes

#         self.mlp_shared = nn.Sequential(
#             nn.Linear(in_channels, hidden_channels),
#             nn.LayerNorm(hidden_channels),
#             nn.GELU(),
#             nn.Linear(hidden_channels, hidden_channels),
#             nn.LayerNorm(hidden_channels),
#             nn.GELU(),
#         )

#         self.label_proj = nn.Sequential(
#             nn.Linear(num_classes, hidden_channels),
#             nn.GELU(),
#         )

#         self.score_head = nn.Linear(hidden_channels, 1)

#         self.ordering_loss = OrderingLoss(
#             ordering_k=ordering_k,
#             z_loss_weight=z_loss_weight,
#         )

#         self._zorder_init(in_channels, hidden_channels)

#     def _zorder_init(self, in_channels, hidden_channels):
#         device = next(self.parameters()).device \
#             if list(self.parameters()) else 'cpu'

#         for layer in self.mlp_shared:
#             if isinstance(layer, nn.Linear):
#                 init.kaiming_normal_(layer.weight, mode='fan_in',
#                                      nonlinearity='relu')
#                 if layer.bias is not None:
#                     init.zeros_(layer.bias)

#         N, depth, grid_scale = 4096, 10, 1023
#         dummy_grid   = torch.randint(0, grid_scale + 1, (N, 3), device=device).int()
#         dummy_batch  = torch.zeros(N, device=device).long()
#         dummy_coords = (dummy_grid.float() / grid_scale) * 2 - 1

#         # dummy input matching point.feat layout: coord(3) + zeros for rest
#         dummy_inp = torch.cat(
#             [dummy_coords,
#              torch.zeros(N, in_channels - 3, device=device)],
#             dim=1
#         ) if in_channels > 3 else dummy_coords

#         z_key    = encode(dummy_grid, dummy_batch, depth, order="z")
#         z_scores = (z_key.float() - z_key.float().min()) / \
#                    (z_key.float().max() - z_key.float().min() + 1e-6)
#         z_scores     = z_scores * 0.98 + 0.01
#         target_logit = torch.log(z_scores / (1 - z_scores)).unsqueeze(1)

#         with torch.no_grad():
#             feat = dummy_inp
#             for layer in self.mlp_shared:
#                 feat = layer(feat)

#             feat_mean   = feat.mean(dim=0)
#             target_mean = target_logit.mean(dim=0)
#             feat_c      = feat - feat_mean
#             target_c    = target_logit - target_mean

#             I      = torch.eye(feat_c.shape[1], device=device) * 1e-3
#             weight = torch.linalg.solve(
#                 feat_c.T @ feat_c + I, feat_c.T @ target_c).T
#             bias   = target_mean - weight @ feat_mean

#             self.score_head.weight.copy_(weight)
#             self.score_head.bias.copy_(bias.squeeze())

#     def forward(self, point, segment=None):
#         # point.feat = coord(3) + color(3) = 6 channels — use directly
#         inp = point.feat if point.feat is not None else point.coord
#         inp = torch.nan_to_num(inp, nan=0.0)
#         inp = torch.clamp(inp, -10.0, 10.0)

#         feat = self.mlp_shared(inp)                          # [N, hidden]

#         # Residual semantic fusion during training
#         if segment is not None and self.training:
#             seg_clamped = segment.clamp(0, self.num_classes - 1)
#             valid       = (segment >= 0)
#             one_hot     = F.one_hot(seg_clamped, self.num_classes).float()
#             one_hot[~valid] = 0.0
#             feat = feat + self.label_proj(one_hot)

#         logits         = self.score_head(feat)
#         learned_scores = torch.sigmoid(logits)
#         learned_scores = torch.clamp(learned_scores, min=1e-5, max=1 - 1e-5)
#         return learned_scores                                 # [N, 1]

#     def compute_loss(self, scores, coords, batch_ids, offset, z_target=None):
#         return self.ordering_loss(
#             scores, coords, batch_ids, offset, z_target=z_target)


class FourierEmbedding(nn.Module):
    def __init__(self, in_channels=3, num_freqs=10, max_freq_log2=None):
        """
        Projects coords into high-freq features: [sin(2^0 pi x), cos(2^0 pi x), ...]
        Essential for learning space-filling curves (Z-order/Hilbert).
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_freqs = num_freqs
        
        if max_freq_log2 is None:
            max_freq_log2 = num_freqs - 1
            
        # Frequencies: 2^0, 2^1, ..., 2^(L-1)
        self.freq_bands = 2.0 ** torch.linspace(
            0.0, max_freq_log2, steps=num_freqs
        )

    def forward(self, x):
        # x: [N, 3]
        # output: [N, 3 + 3 * 2 * num_freqs]
        
        embed = [x]
        for freq in self.freq_bands.to(x.device):
            embed.append(torch.sin(x * freq * torch.pi))
            embed.append(torch.cos(x * freq * torch.pi))
            
        return torch.cat(embed, dim=-1)


class ResidualBlock(nn.Module):
    """Simple ResNet block for better gradient flow."""
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.fc(x))


class PointSorter(nn.Module):
    """
    Lightweight Sorter that uses PTv3's perfect spatial Z-code as a base,
    and applies a learned semantic offset via a simple MLP.
    """
    def __init__(self, in_channels=6, hidden_channels=64, ordering_k=16, loss_weights=[1, 0, 0, 1], tau=0.1, num_classes=13, ignore_index=-1):
        super().__init__()
        self.num_classes = num_classes
        # self.ignore_index = ignore_index
        
        # Tiny, fast MLP. No Fourier features or ResNet blocks required.
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh() # Bounds the output to [-1, 1]
        )
        
        # Scale factor limits how far a point can jump in the sorted list
        self.offset_scale = 0.05 
        
        # NOTE: Make sure your loss_weights has a value > 0 for the first element (locality)
        self.ordering_loss = OrderingLoss(ordering_k=ordering_k, loss_weights=loss_weights, tau=tau, ignore_index=-1)

    def forward(self, point):
        # 1. Get the perfect deterministic spatial code calculated by PTv3
        # z_code = point.serialized_code[0].float()
        z = point.serialized_code
        # Handle both [1, N] tensor and list of tensors
        if hasattr(z, "dim") and z.dim() == 2:
            z = z[0]
        elif isinstance(z, (list, tuple)):
            z = z[0]
            
        z_code = z.float()
        
        # Normalize the base score to [0, 1]
        base_score = (z_code - z_code.min()) / (z_code.max() - z_code.min() + 1e-6)
        
        # 2. Predict the semantic offset using Coord(3) + Color(3)
        inp = point.feat if point.feat is not None else point.coord
        offset = self.mlp(inp).squeeze(-1)
        
        # 3. Final Score = Spatial Base + Semantic Nudge
        final_scores = base_score + (offset * self.offset_scale)
        
        # Clamp to avoid extreme outliers
        learned_scores = torch.clamp(final_scores, min=1e-5, max=1 - 1e-5).unsqueeze(1)
        
        # Store for loss computation
        if self.training:
            self._last_feat = self.mlp[0:3](inp) # Save hidden features for global loss
            
        return learned_scores

    def compute_loss(self, scores, coords, batch_ids, offset, z_target=None, labels=None):
        features = getattr(self, '_last_feat', None)
        return self.ordering_loss(
            scores, coords, batch_ids, offset, 
            z_target=z_target, 
            features=features,
            labels=labels # IMPORTANT: Pass labels down to OrderingLoss
        )


class PointSorter_old(nn.Module):
    """
    Upgraded Sorter with Fourier Embeddings + ResNet.
    Input: point.feat (coord + color)
    """

    def __init__(
        self,
        in_channels=6,      # coord(3) + color(3)
        hidden_channels=128, # Increased width for capacity
        ordering_k=16,
        loss_weights=[0,0,0,1], # Only global feature loss for now
        tau=0.1,
        num_classes=13,
        num_freqs=10,       # 10 freqs cover grid resolution ~1024 (2^10)
        ignore_index=-1,    # For semantic locality loss, if using labels
    ):
        super().__init__()
        self.num_classes = num_classes

        # 1. Coordinate Encoding
        # Maps 3 coords -> 3 + 60 = 63 dims
        self.pos_enc = FourierEmbedding(in_channels=3, num_freqs=num_freqs)
        embed_dim = 3 + (3 * 2 * num_freqs)
        
        # 2. Input Projection
        # We process (encoded_coord) and (color) together
        # in_channels - 3 gives us the color channel count
        self.input_proj = nn.Linear(embed_dim + (in_channels - 3), hidden_channels)
        
        # 3. Residual Backbone (3 blocks)
        self.backbone = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
        )

        # # 4. Semantic Fusion (Training only)
        # self.label_proj = nn.Sequential(
        #     nn.Linear(num_classes, hidden_channels),
        #     nn.GELU(),
        # )

        # 5. Head
        self.score_head = nn.Linear(hidden_channels, 1)

        # Loss
        self.ordering_loss = OrderingLoss(
            ordering_k=ordering_k,
            loss_weights=loss_weights,
            tau=tau, ignore_index=ignore_index
        )

        # We skip the heavy solver init now, standard initialization works 
        # better with residual connections + PE.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, point):
        # 1. Split coord and color
        # point.feat is [coord(3), color(3)]
        coords = point.coord
        if point.feat is not None:
            colors = point.feat[:, 3:6]  # Assuming color channels are after the first 3 coord channels
        else:
            colors = torch.zeros(coords.shape[0], 0, device=coords.device)
        

        # 2. Fourier Encode Coords
        # [N, 3] -> [N, 63]
        pos_feat = self.pos_enc(coords)

        # 3. Fuse with Color
        # [N, 63 + 3] -> [N, hidden]
        inp = torch.cat([pos_feat, colors], dim=1)
        feat = self.input_proj(inp)

        # 4. Residual Backbone
        feat = self.backbone(feat)

        # Store for compute_loss — only needed during training
        if self.training:
            self._last_feat = feat  # [N, hidden_channels]

        # 6. Prediction
        logits = self.score_head(feat)
        learned_scores = torch.sigmoid(logits)
        
        # Avoid 0/1 collapse for loss stability
        learned_scores = torch.clamp(learned_scores, min=1e-5, max=1 - 1e-5)
        
        return learned_scores

    def compute_loss(self, scores, coords, batch_ids, offset, z_target=None, labels=None):
        # Retrieve stored features — None if not training (global loss skipped automatically)
        features = getattr(self, '_last_feat', None)
        return self.ordering_loss(
            scores, coords, batch_ids, offset, z_target=z_target, features=features, labels=labels)


# ---------------------------------------------------------------------------
# OPTNet
# ---------------------------------------------------------------------------

@MODELS.register_module("OPTNet")
class OPTNet(PointTransformerV3):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        ordering_loss_weight=1.0,
        ordering_k=16,
        warmup_epoch=5,
        enable_score_concat=True,
        tau=0.1,
        loss_weights=[0,0,0,1], # Only global feature loss for now
        num_classes=13,
        code_depth=10,
        pool_sizes = [4, 8, 8, 16], # For the 3 pooling layers in PTv3
        use_labels_in_loss=False, # Whether to fuse GT segment labels into the Sorter loss
        **kwargs,
    ):
        single_order = [order[0]] if isinstance(order, (list, tuple)) else [order]
        self.enable_score_concat = enable_score_concat
        self.code_depth          = code_depth
        self.use_labels_in_loss   = use_labels_in_loss

        backbone_in_channels = in_channels + 1 if enable_score_concat else in_channels
        super().__init__(
            in_channels=backbone_in_channels,
            order=single_order,
            **kwargs
        )

        self.ordering_loss_weight = ordering_loss_weight
        self.warmup_epoch         = warmup_epoch
        self.all_orders           = list(order) if isinstance(order, (list, tuple)) else [order]
        # e.g. ["z", "z-trans", "hilbert", "hilbert-trans"]


        # in_channels == point.feat channels == len(feat_keys) * 3 == 6
        self.sorter = PointSorter(
            in_channels=in_channels,
            hidden_channels=64,
            ordering_k=ordering_k,
            num_classes=num_classes,
            loss_weights=loss_weights,
            tau=tau, 
            ignore_index=kwargs.get("ignore_index", -1)
        )
        self.code_depth = code_depth
        self._replace_pooling_layers(stride=kwargs['stride'], pool_sizes=pool_sizes)

    # def _replace_pooling_layers(self, pool_sizes=None):
    #     # Recursively find and replace SerializedPooling with GridPoolingLayer
    #     for name, module in self.named_modules():
    #         # Check if it's a PointSequential or similar container that holds layers
    #         if hasattr(module, "__iter__"): 
    #             continue # Skip, we need to modify the parent
            
    #     # Safer approach: Iterate the encoder specific structure
    #     # PTv3 stores stages in self.enc (a PointSequential)
    #     for i, stage in enumerate(self.enc):
    #         if isinstance(stage, SerializedPooling):
    #             # Retrieve config from the existing layer
    #             in_c = stage.in_channels
    #             out_c = stage.out_channels
    #             stride = stage.stride
                
    #             # Create replacement
    #             grid_pool = GridPoolingLayer(
    #                 in_channels=in_c,
    #                 out_channels=out_c,
    #                 stride=stride,
    #                 code_depth=self.code_depth,
    #                 traceable=True
    #             )
                
    #             # Replace in the sequential container
    #             self.enc[i] = grid_pool

    def _replace_pooling_layers(self, stride=(2, 2, 2, 2), pool_sizes=(8, 8, 8, 16)):
        pool_idx = 0
        for i, layer in enumerate(self.enc):
            if isinstance(layer, SerializedPooling):
                ps = pool_sizes[pool_idx] if pool_idx < len(pool_sizes) else pool_sizes[-1]
                st = stride[pool_idx] if pool_idx < len(stride) else stride[-1]
                self.enc[i] = SortedWindowPooling(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    pool_size=ps,
                    stride=st,
                    code_depth=self.code_depth,
                    traceable=True,
                )
                pool_idx += 1


    def _build_serialization_from_scores(self, scores, point):
        # 1. We must sort per-batch. To do this with floats, we can add a large batch offset.
        # Scores are in [0, 1]. If we add (batch_id * 10.0), then batch 0 is [0, 1], batch 1 is [10, 11], etc.
        # This guarantees points are sorted first by batch, then by score.
        batch_offsets = point.batch.float() * 10.0
        batch_aware_scores = scores.view(-1) + batch_offsets
        
        # 2. Sort the float scores directly
        order = torch.argsort(batch_aware_scores).unsqueeze(0) # [1, N]
        
        # 3. Compute inverse mapping
        inverse = torch.zeros_like(order).scatter_(
            dim=1, 
            index=order, 
            src=torch.arange(order.shape[1], device=order.device).unsqueeze(0).expand_as(order)
        )
        
        # 4. PTv3 needs `serialized_code` to know batch boundaries during attention.
        # We can just create a simple sequential integer array, but put the batch ID in the high bits.
        # This perfectly mimics what PTv3 expects without needing to quantize the actual floats.
        N = order.shape[1]
        sorted_batch = point.batch[order[0]]
        sequential_code = torch.arange(N, dtype=torch.long, device=order.device)
        
        # Shift batch ID to top bits (depth*3) just like original PTv3, add sequential ID
        shift_bits = self.code_depth * 3
        new_code = (sorted_batch.long() << shift_bits) | sequential_code
        
        # We need to map this sorted code BACK to the original point order 
        # so that `point.serialized_code` matches `point.coord`.
        original_order_code = torch.zeros_like(new_code)
        original_order_code[order[0]] = new_code
        code_2d = original_order_code.unsqueeze(0)

        # 5. Assign to point
        point.serialized_code = code_2d
        point.serialized_order = order
        point.serialized_inverse = inverse
        point.serialized_depth = self.code_depth


    def forward(self, data_dict):
        point         = Point(data_dict)
        # Ensure PTv3 serialization exists for OffsetSorter (needed in eval too)
        point.serialization(order=self.order, shuffle_orders=False, compute_codes=True)

        current_epoch = data_dict.get("epoch", 0) if self.training else float("inf")
        use_learned   = current_epoch >= self.warmup_epoch

        # ------------------------------------------------------------------
        # 1. Z-order serialization — training only
        #    Used for warmup backbone ordering + Z-regression loss target
        # ------------------------------------------------------------------
        if self.training:
            # 1. Create a lightweight temporary point just for multi-order targets
            # (Zero memory overhead because we omit the heavy .feat tensor)
            tmp_point = Point({
                "grid_coord": point.grid_coord,
                "batch": point.batch
            })
            
            tmp_point.serialization(
                order=self.all_orders,   # Pass all K orders here
                shuffle_orders=False,    # Targets don't need shuffling
                compute_codes=True,
            )
            # Save the [4, N] target codes
            target_codes = tmp_point.serialized_code.float().clone()

            # 2. Run the normal single-order serialization for the actual backbone
            if not use_learned:
                point.serialization(
                    order=self.order,        # Uses your single self.order
                    shuffle_orders=True,   # Shuffle only before warmup
                    compute_codes=True,
                )
        else:
            target_codes = None
            point.serialization(
                order=self.order,        
                shuffle_orders=False,  
                compute_codes=True,
            )


        # ------------------------------------------------------------------
        # 2. Run PointSorter  (uses point.feat directly — 6 channels)
        # ------------------------------------------------------------------
        learned_scores = self.sorter(point)
        point.scores = learned_scores  # Store scores for potential use in pooling layers

        # ------------------------------------------------------------------
        # 3. Build serialization from learned scores
        #    Post-warmup training: override Z-order codes
        #    Inference: always use learned codes (no Z computed above)
        # ------------------------------------------------------------------
        if use_learned:
            self._build_serialization_from_scores(learned_scores, point)

        # ------------------------------------------------------------------
        # 4. Compute ordering loss (training only)
        # ------------------------------------------------------------------
        if self.training and self.ordering_loss_weight > 0:
            # target_codes is already [4, N]
            code_mins = target_codes.min(dim=1, keepdim=True)[0]
            code_maxs = target_codes.max(dim=1, keepdim=True)[0]
            z_targets = (target_codes - code_mins) / (code_maxs - code_mins + 1e-6)
            labels = data_dict.get("segment", None) if self.use_labels_in_loss else None
            loss_ord, loss_dict = self.sorter.compute_loss(
                learned_scores,
                point.coord,
                point.batch,
                point.offset,
                z_target=z_targets, # Passes shape [4, N] natively
                labels=labels
            )

            
            loss_ord            = torch.clamp(loss_ord, max=10.0)
            point.ordering_loss = loss_ord * self.ordering_loss_weight

        # ------------------------------------------------------------------
        # 5. Concatenate scores into backbone features
        # ------------------------------------------------------------------
        if self.enable_score_concat:
            if point.feat is None:
                point.feat = learned_scores
            else:
                point.feat = torch.cat([point.feat, learned_scores], dim=1)

        # ------------------------------------------------------------------
        # 6. Sparsify → PTv3 backbone
        # ------------------------------------------------------------------
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point


# ---------------------------------------------------------------------------
# OPTNetSegmentor
# ---------------------------------------------------------------------------

@MODELS.register_module("OPTNetSegmentor")
class OPTNetSegmentor(nn.Module):
    def __init__(self, backbone, criteria, num_classes, backbone_out_channels=None):
        super().__init__()
        self.backbone    = MODELS.build(backbone)
        self.criteria    = build_criteria(criteria)
        self.seg_head    = nn.Linear(backbone_out_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, data_dict):
        point      = self.backbone(data_dict)
        seg_logits = self.seg_head(point.feat)

        if self.training:
            target = data_dict["segment"]
            if target.dtype != torch.int64:
                target = target.long()

            valid_mask = (target >= 0) & (target < self.num_classes)
            if not valid_mask.all():
                target = target.clone()
                target[~valid_mask] = -1

            seg_loss    = self.criteria(seg_logits, target)
            return_dict = dict(seg_loss=seg_loss)

            if "ordering_loss" in point:
                ordering_loss                = point["ordering_loss"]
                return_dict["ordering_loss"] = ordering_loss
                return_dict["loss"]          = seg_loss + ordering_loss
            else:
                return_dict["loss"] = seg_loss

            return return_dict

        elif "segment" in data_dict:
            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)

        return dict(seg_logits=seg_logits)
