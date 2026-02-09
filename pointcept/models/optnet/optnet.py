import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.utils.structure import Point
from pointcept.models.losses import build_criteria
import pointops
import torch.nn.init as init  # Add if missing
import torch.nn.functional as F




import torch
import torch.nn as nn
import torch.nn.init as init

class PointSorter(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_orders=1):
        super().__init__()
        self.num_orders = num_orders
        
        # Standard MLP structure
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_orders)
        )

        # Initialize weights specifically to mimic Z-order curve
        self._zorder_init(in_channels)
        
        # ADD THIS:
        self.ss_loss = SelfSupervisedOrderingLoss(
            k_near=16, k_far=32, temp_locality=0.1, temp_contrastive=0.5
        )

    def _zorder_init(self, in_channels):
        """
        Initializes the MLP such that it approximates the Z-order curve.
        Early layers are Kaiming init; Last layer is solved via Least Squares.
        """
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
        
        # 1. Initialize early layers with Kaiming (good for GELU)
        for i, layer in enumerate(self.mlp[:-1]):
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        # 2. Create Dummy Data to fit the Z-curve
        # We simulate the coordinate part (first 3 channels)
        N = 4096
        # Random coords in range [-1, 1]
        dummy_coords = torch.rand(N, 3, device=device) * 2 - 1
        
        # If input has features (in_channels > 3), pad with zeros or noise
        if in_channels > 3:
            dummy_feats = torch.zeros(N, in_channels - 3, device=device)
            dummy_inp = torch.cat([dummy_coords, dummy_feats], dim=1)
        else:
            dummy_inp = dummy_coords

        # 3. Compute Target Z-scores (The Ground Truth)
        z_key = self.morton_encode(dummy_coords)
        # Normalize to [0.01, 0.99] to avoid infinities in logit
        z_scores = (z_key - z_key.min()) / (z_key.max() - z_key.min() + 1e-6)
        z_scores = z_scores * 0.98 + 0.01
        
        # Target Logits: inverse sigmoid(z) = log(z / (1-z))
        # Shape: (N, 1)
        target_logit = torch.log(z_scores / (1 - z_scores)).unsqueeze(1)

        # 4. Get Intermediate Features from the initialized early layers
        with torch.no_grad():
            feat = dummy_inp
            # Pass through all layers except the last Linear
            for layer in self.mlp[:-1]:
                feat = layer(feat)
            
            # feat shape: (N, hidden_channels)
            
            # 5. Least Squares Solution: feat @ W.T + b = target
            
            feat_mean = feat.mean(dim=0)          # (hidden,)
            target_mean = target_logit.mean(dim=0)# (1,)

            # Center data
            feat_c = feat - feat_mean
            target_c = target_logit - target_mean

            # Solve A @ X = B  -> feat_c @ W.T = target_c
            # solution shape: (hidden, 1)
            # using lstsq for robustness
            solution = torch.linalg.lstsq(feat_c, target_c).solution
            weight = solution.T  # (1, hidden)

            # Calculate bias: b = mean_y - W @ mean_x
            bias = target_mean - weight @ feat_mean # (1,)

            # 6. Apply to Last Layer
            last_layer = self.mlp[-1]
            
            # Replicate weights for all orders (start all orders as Z-order)
            last_layer.weight.copy_(weight.repeat(self.num_orders, 1))
            last_layer.bias.copy_(bias.repeat(self.num_orders))

    def morton_encode(self, coords):
        """Encodes 3D coordinates into a Z-order curve value."""
        # Normalize coordinates to [0, 1023] integers
        # Assumes coords are roughly [-1, 1] or similar scale.
        coords = (coords - coords.min(0)[0]) / (coords.max(0)[0] - coords.min(0)[0] + 1e-6)
        vals = (coords * 1023).long()
        
        x = vals[:, 0]
        y = vals[:, 1]
        z = vals[:, 2]
        
        key = torch.zeros_like(x)
        for i in range(10):
            # Interleave bits: ... z_i y_i x_i ...
            key |= (x >> i & 1) << (3 * i + 0)
            key |= (y >> i & 1) << (3 * i + 1)
            key |= (z >> i & 1) << (3 * i + 2)
            
        return key.float()

    def forward(self, point):
        # 1. Prepare Input
        if point.feat is not None:
            inp = torch.cat([point.coord, point.feat.detach()], dim=1)
        else:
            inp = point.coord

        # DEBUG: Check input
        if torch.isnan(inp).any() or torch.isinf(inp).any():
            print(f"[DEBUG] NaN/Inf in PointSorter input BEFORE clamping: coord NaN={torch.isnan(point.coord).any()}, feat NaN={torch.isnan(point.feat).any() if point.feat is not None else False}")
            inp = torch.nan_to_num(inp, nan=0.0, posinf=1.0, neginf=-1.0)
            inp = torch.clamp(inp, -10.0, 10.0)

        # 2. Predict Scores
        logits = self.mlp(inp)
        
        # DEBUG: Check logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[DEBUG] NaN/Inf in MLP logits! logits stats: min={logits.min()}, max={logits.max()}, has_nan={torch.isnan(logits).any()}")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=5.0, neginf=-5.0)
        
        scores = torch.clamp(torch.sigmoid(logits), min=1e-6, max=1-1e-6)
        
        # DEBUG: Check scores
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print(f"[DEBUG] NaN/Inf in scores after sigmoid! This should not happen.")
            scores = torch.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)

        # 3. Generate Orders
        batch_offset = point.batch.unsqueeze(1) * 100000.0
        scores_for_sort = scores.detach()
        scores_with_batch = scores_for_sort + batch_offset

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

    
    def compute_loss(self, scores, coords, batch_ids, offset):
        return self.ss_loss(scores, coords, batch_ids, offset)

# ==============================================================================
# Self-Supervised Ordering Loss
# ==============================================================================

class SelfSupervisedOrderingLoss(nn.Module):
    """
    Self-supervised loss for learning ordering scores.
    Uses k-NN locality and contrastive learning.
        
    Key principles:
    1. Neighboring points should have similar scores (locality)
    2. Far-away points should have different scores (contrastive)
    3. Scores should span the full [0, 1] range (distribution)
    """
    def __init__(self, k_near=8, k_far=16, temp_locality=0.1, temp_contrastive=0.5):
        super().__init__()
        self.k_near = k_near
        self.k_far = k_far
        self.temp_locality = temp_locality
        self.temp_contrastive = temp_contrastive
    
    def forward(self, scores, coords, batch_ids, offset=None):
        # Handle multi-order
        if scores.dim() > 1:
            scores_mean = scores.mean(dim=1, keepdim=True)
        else:
            scores_mean = scores.unsqueeze(1) if scores.dim() == 1 else scores
        
        # DEBUG
        if torch.isnan(scores_mean).any() or torch.isinf(scores_mean).any():
            print(f"[DEBUG LOSS] NaN/Inf in scores_mean at loss entry!")
            scores_mean = torch.nan_to_num(scores_mean, nan=0.5)
        
        if torch.isnan(coords).any() or torch.isinf(coords).any():
            print(f"[DEBUG LOSS] NaN/Inf in coords at loss entry!")
            coords = torch.nan_to_num(coords, nan=0.0)
        
        N = scores_mean.shape[0]
        device = scores_mean.device
        dtype = scores_mean.dtype
        
        if N < self.k_near:
            return torch.tensor(0.0, device=device, dtype=dtype), {}
        
        # Prepare offset
        if offset is None:
            batch_ids_long = batch_ids.long()
            unique_ids, counts = torch.unique_consecutive(batch_ids_long, return_counts=True)
            offset = torch.cumsum(counts, dim=0).int()
        else:
            offset = offset.int()
        
        # Compute losses with try-except
        try:
            loss_locality = self._compute_locality_loss(scores_mean, coords, offset)
        except Exception as e:
            print(f"[DEBUG LOSS] locality failed: {e}")
            loss_locality = torch.tensor(0.0, device=device, dtype=dtype)
        
        try:
            loss_contrastive = self._compute_contrastive_loss(scores_mean, coords, offset)
        except Exception as e:
            print(f"[DEBUG LOSS] contrastive failed: {e}")
            loss_contrastive = torch.tensor(0.0, device=device, dtype=dtype)
        
        try:
            # loss_distribution = self._compute_distribution_loss(scores_mean, batch_ids)
            loss_distribution = torch.tensor(0.0, device=device, dtype=dtype)  # TEMP DISABLE

        except Exception as e:
            print(f"[DEBUG LOSS] distribution failed: {e}")
            loss_distribution = torch.tensor(0.0, device=device, dtype=dtype)
        
        try:
            loss_smoothness = self._compute_smoothness_loss(scores_mean, coords, offset)
        except Exception as e:
            print(f"[DEBUG LOSS] smoothness failed: {e}")
            loss_smoothness = torch.tensor(0.0, device=device, dtype=dtype)
        
        # Check each loss component
        if torch.isnan(loss_locality):
            print(f"[DEBUG LOSS] loss_locality is NaN!")
            loss_locality = torch.tensor(0.0, device=device, dtype=dtype)
        if torch.isnan(loss_contrastive):
            print(f"[DEBUG LOSS] loss_contrastive is NaN!")
            loss_contrastive = torch.tensor(0.0, device=device, dtype=dtype)
        if torch.isnan(loss_distribution):
            print(f"[DEBUG LOSS] loss_distribution is NaN!")
            loss_distribution = torch.tensor(0.0, device=device, dtype=dtype)
        if torch.isnan(loss_smoothness):
            print(f"[DEBUG LOSS] loss_smoothness is NaN!")
            loss_smoothness = torch.tensor(0.0, device=device, dtype=dtype)
        
        total_loss = 1.0 * loss_locality + 0.5 * loss_contrastive + 0.3 * loss_distribution + 0.2 * loss_smoothness
        
        if torch.isnan(total_loss):
            print(f"[DEBUG LOSS] TOTAL LOSS IS NaN after combination!")
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True), {}
        
        return total_loss, {
            "locality": loss_locality.item(),
            "contrastive": loss_contrastive.item(),
            "distribution": loss_distribution.item(),
            "smoothness": loss_smoothness.item()
        }


    
    def _compute_locality_loss(self, scores, coords, offset):
        """Neighboring points should have similar scores."""
        device = scores.device
        dtype = scores.dtype
        
        try:
            k_actual = min(self.k_near, scores.shape[0] - 1)
            if k_actual < 1:
                return torch.tensor(0.0, device=device, dtype=dtype)
            
            # if pointops is not None:
            idx = pointops.knn_query(k_actual, coords.contiguous(), offset)[0]
            # else:
            #     idx = self._manual_knn(coords, offset, k_actual)
            
            neighbor_scores = scores[idx.long()]
            score_diff = (scores.unsqueeze(1) - neighbor_scores).abs()
            
            # Weighted by distance
            neighbor_coords = coords[idx.long()]
            dists = torch.norm(coords.unsqueeze(1) - neighbor_coords, dim=2)
            weights = torch.exp(-dists / self.temp_locality)
            
            loss = (weights * score_diff.pow(2)).sum() / weights.sum().clamp(min=1e-8)
            
        except Exception as e:
            loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        return loss
    
    def _compute_contrastive_loss(self, scores, coords, offset):
        """Far-away points should have different scores."""
        device = scores.device
        dtype = scores.dtype
        N = scores.shape[0]
        
        k_near = min(self.k_near, N - 1)
        k_far = min(self.k_far, N - 1)
        
        if k_far <= k_near:
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        idx_far_all = pointops.knn_query(k_far, coords.contiguous(), offset)[0]
        idx_near = idx_far_all[:, :k_near]
        idx_far = idx_far_all[:, k_near:]
 
        # Positive pairs (near): high similarity
        scores_near = scores[idx_near.long()]
        sim_pos = 1.0 - (scores.unsqueeze(1) - scores_near).abs()
        
        # Negative pairs (far): low similarity
        scores_far = scores[idx_far.long()]
        sim_neg = 1.0 - (scores.unsqueeze(1) - scores_far).abs()
        
        # Contrastive loss
        logits_pos = sim_pos / self.temp_contrastive
        logits_neg = sim_neg / self.temp_contrastive
        
        loss_pos = -torch.log(torch.sigmoid(logits_pos) + 1e-8).mean()
        loss_neg = -torch.log(1 - torch.sigmoid(logits_neg) + 1e-8).mean()
        
        loss = loss_pos + loss_neg
        

        
        return loss
    
    def _compute_distribution_loss(self, scores, batch_ids):
        """
        Scores should be uniformly distributed in [0, 1].
        Handles multi-order case (scores: N, num_orders).
        """
        device = scores.device
        dtype = scores.dtype
        
        batch_ids_long = batch_ids.long()
        num_batches = batch_ids_long.max().item() + 1
        
        total_loss = 0.0
        count = 0
        
        for b in range(num_batches):
            mask = (batch_ids_long == b)
            if mask.sum() < 2: 
                continue
                
            scores_b = scores[mask]  # (M, num_orders) where M = mask.sum()
            M = scores_b.shape[0]
            
            # OPTION 1: Average across orders (recommended)
            scores_flat = scores_b.mean(dim=1)  # (M,)
            sorted_scores, _ = torch.sort(scores_flat)
            target = torch.linspace(0, 1, M, device=device, dtype=dtype)
            loss_b = F.mse_loss(sorted_scores, target)
            
            # OPTION 2: Per-order (uncomment if you want separate uniformity)
            # for o in range(scores_b.shape[1]):
            #     sorted_scores, _ = torch.sort(scores_b[:, o])
            #     target = torch.linspace(0, 1, M, device=device, dtype=dtype)
            #     loss_b += F.mse_loss(sorted_scores, target)
            # loss_b /= scores_b.shape[1]
            
            total_loss += loss_b
            count += 1
        
        return total_loss / max(count, 1) if count > 0 else torch.tensor(0.0, device=device, dtype=dtype)

    def _compute_smoothness_loss(self, scores, coords, offset):
        """Encourage smooth score transitions."""
        device = scores.device
        dtype = scores.dtype
        k_smooth = min(8, scores.shape[0] - 1)
        if k_smooth < 1:
            return torch.tensor(0.0, device=device, dtype=dtype)

        idx = pointops.knn_query(k_smooth, coords.contiguous(), offset)[0]
        neighbor_scores = scores[idx.long()]
        neighbor_mean = neighbor_scores.mean(dim=1)
        
        loss = F.mse_loss(scores, neighbor_mean)
            

        
        return loss


@MODELS.register_module("OPTNet")
class OPTNet(PointTransformerV3):
    def __init__(self,
                 in_channels=6,
                 ordering_loss_weight=1.0,
                 ordering_k=16,
                 warmup_epoch=0,
                 contrastive_k_far=32,
                 contrastive_temp=0.5,
                 contrastive_weight=0.5,
                 **kwargs):
        super().__init__(in_channels=in_channels, **kwargs)
        self.ordering_loss_weight = ordering_loss_weight
        self.ordering_k = ordering_k
        self.warmup_epoch = warmup_epoch
        
        # NEW: Contrastive loss parameters
        self.contrastive_k_far = contrastive_k_far
        self.contrastive_temp = contrastive_temp
        self.contrastive_weight = contrastive_weight
        
        self.sorter = PointSorter(
            in_channels=in_channels + 3,
            hidden_channels=64,
            num_orders=len(self.order)
        )

    

    def forward(self, data_dict):
        point = Point(data_dict)
        current_epoch = data_dict.get("epoch", 0) if self.training else float('inf')
        
        # 1. Standard Serialization
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        
        # 2. Run OPTNet Sorter
        scores, learned_order, learned_inverse = self.sorter(point)
        
        # 3. Apply Order Strategy
        if current_epoch >= self.warmup_epoch:
            point.serialized_order = learned_order
            point.serialized_inverse = learned_inverse
        
        # 4. Calculate Loss
        if self.training and self.ordering_loss_weight > 0:
            loss_ord, loss_dict = self.sorter.compute_loss(
            scores, 
            point.coord, 
            point.batch, 
            point.offset
            )
            point.ordering_loss = loss_ord * self.ordering_loss_weight
 
        
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
        
        # 2. Validation Mode
        elif "segment" in data_dict:
            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        
        # 3. Test Mode
        return dict(seg_logits=seg_logits)
