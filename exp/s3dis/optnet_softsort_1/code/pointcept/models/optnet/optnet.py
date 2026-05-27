import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.utils.structure import Point
from pointcept.models.losses import build_criteria
import pointops
import torch_scatter
from pointcept.models.modules import PointModule, PointSequential
from functools import partial
class SoftSort(nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, scores, features):
        """
        scores: (B, N, 1)
        features: (B, N, C)
        """
        # Compute Soft Permutation Matrix P (B, N, N)
        score_diff = scores.transpose(1, 2) - scores
        P = torch.softmax(score_diff / self.tau, dim=-1)
        
        # Apply Permutation to Features
        sorted_feat = torch.bmm(P, features)
        return sorted_feat


class PointSorter(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_orders=1, residual_scale=0.1, tau=1.0):
        super().__init__()
        self.num_orders = num_orders
        self.residual_scale = residual_scale
        self.tau = tau
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_orders)
        )
        
        self.soft_sort = SoftSort(tau=tau)

    def forward(self, point):
        coord = point.coord.view(-1, 3)
        N = coord.shape[0]
        
        if point.feat is not None:
            feat = point.feat.view(N, -1) 
            inp = torch.cat([coord, feat.detach()], dim=1)
        else:
            inp = coord

        # Use Z-coordinate as base score (no serialization needed)
        z = coord[:, 2:3]
        base_score = (z - z.min()) / (z.max() - z.min() + 1e-6)
        base_score = base_score.view(N, 1).expand(-1, self.num_orders)

        # Predict residual
        delta = torch.tanh(self.mlp(inp)) * self.residual_scale
        scores = base_score + delta

        # Handle batch offsets
        if hasattr(point, "batch"):
            batch = point.batch.view(N)
        else:
            batch = torch.zeros(N, device=coord.device, dtype=torch.long)

        batch_offset = batch.unsqueeze(1) * (scores.max().detach() + 10.0)
        scores_with_batch = scores + batch_offset

        # Generate orders
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


class WindowPooling(PointModule):
    """
    Window-based pooling that selects n-th percentile 3D point from ordered 1D windows.
    Replaces SerializedPooling without requiring serialization codes.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        percentile=50,
        traceable=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.percentile = percentile
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        else:
            self.norm = None
        if act_layer is not None:
            self.act = PointSequential(act_layer())
        else:
            self.act = None

    def forward(self, point: Point):
        """
        Pool points by selecting n-th percentile from windows along learned order
        """
        # Get ordering from point (should be set by PointSorter)
        if hasattr(point, 'serialized_order') and point.serialized_order is not None:
            order = point.serialized_order[0]  # Use first order
        else:
            # Fallback: order by z-coordinate
            order = torch.argsort(point.coord[:, 2])
        
        N = len(order)
        window_size = self.stride
        
        # Process each batch separately
        unique_batches = torch.unique(point.batch)
        all_selected_indices = []
        all_clusters = []
        cluster_offset = 0
        
        for batch_id in unique_batches:
            batch_mask = point.batch == batch_id
            batch_indices = torch.where(batch_mask)[0]
            
            # Get order for this batch
            batch_order_mask = torch.isin(order, batch_indices)
            batch_order = order[batch_order_mask]
            
            N_batch = len(batch_order)
            num_windows = (N_batch + window_size - 1) // window_size
            
            selected_indices = []
            
            for w in range(num_windows):
                window_start = w * window_size
                window_end = min((w + 1) * window_size, N_batch)
                window_indices = batch_order[window_start:window_end]
                
                if len(window_indices) == 0:
                    continue
                
                # Get 3D coordinates for points in this window
                window_coords = point.coord[window_indices]
                
                # Compute centroid
                centroid = window_coords.mean(dim=0, keepdim=True)
                
                # Compute distances to centroid
                distances = torch.norm(window_coords - centroid, dim=1, p=2)
                
                # Select n-th percentile
                k = int(len(distances) * self.percentile / 100.0)
                k = max(0, min(k, len(distances) - 1))
                
                # Get index of k-th smallest distance
                sorted_dist_indices = torch.argsort(distances)
                selected_local_idx = sorted_dist_indices[k]
                selected_global_idx = window_indices[selected_local_idx]
                
                selected_indices.append(selected_global_idx)
            
            all_selected_indices.extend(selected_indices)
            
            # Create cluster mapping: each point in batch maps to its window
            for local_idx, global_idx in enumerate(batch_order):
                window_id = local_idx // window_size
                all_clusters.append(cluster_offset + window_id)
            
            cluster_offset += num_windows
        
        selected_indices = torch.tensor(all_selected_indices, dtype=torch.long, device=point.coord.device)
        cluster = torch.tensor(all_clusters, dtype=torch.long, device=point.coord.device)
        
        # Sort by cluster for segment_csr
        _, indices = torch.sort(cluster)
        counts = torch.bincount(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        
        # Aggregate features using max pooling over each window
        feat_pooled = torch_scatter.segment_csr(
            self.proj(point.feat)[indices], idx_ptr, reduce="max"
        )
        
        # Use selected points for coords and other attributes
        coord_pooled = point.coord[selected_indices]
        grid_coord_pooled = point.grid_coord[selected_indices] >> 1
        batch_pooled = point.batch[selected_indices]
        
        # CRITICAL FIX: Generate new ordering for downsampled points
        # The serialized_order and serialized_inverse must be tensors, not Point objects
        num_selected = len(selected_indices)
        
        # Get number of order dimensions from input
        if hasattr(point, 'serialized_order'):
            num_orders = point.serialized_order.shape[0]
        else:
            num_orders = 1
        
        # Create new ordering based on pooled points
        new_order_list = []
        new_inverse_list = []
        
        for k in range(num_orders):
            # Simple ordering: preserve relative order from input
            new_order = torch.arange(num_selected, device=point.coord.device)
            new_inverse = torch.arange(num_selected, device=point.coord.device)
            new_order_list.append(new_order)
            new_inverse_list.append(new_inverse)
        
        new_serialized_order = torch.stack(new_order_list, dim=0)  # Shape: (num_orders, num_selected)
        new_serialized_inverse = torch.stack(new_inverse_list, dim=0)  # Shape: (num_orders, num_selected)
        
        # Build output point dict
        point_dict = {
            'feat': feat_pooled,
            'coord': coord_pooled,
            'grid_coord': grid_coord_pooled,
            'batch': batch_pooled,
            'serialized_order': new_serialized_order,  # Must be tensor
            'serialized_inverse': new_serialized_inverse,  # Must be tensor
        }
        
        if 'condition' in point.keys():
            point_dict['condition'] = point.condition
        if 'context' in point.keys():
            point_dict['context'] = point.context
        
        if self.traceable:
            point_dict['pooling_inverse'] = cluster
            point_dict['pooling_parent'] = point
        
        point_out = Point(point_dict)
        
        if self.norm is not None:
            point_out = self.norm(point_out)
        if self.act is not None:
            point_out = self.act(point_out)
        
        point_out.sparsify()
        return point_out




@MODELS.register_module("OPTNet")
class OPTNet(PointTransformerV3):
    def __init__(self, 
                 in_channels=6, 
                 ordering_loss_weight=1.0, 
                 ordering_k=16, 
                 warmup_epoch=0,
                 pool_percentile=50,
                 soft_tau=1.0,
                 **kwargs):
        
        # Initialize parent WITHOUT calling serialization
        super().__init__(in_channels=in_channels, **kwargs)
        
        self.ordering_loss_weight = ordering_loss_weight
        self.ordering_k = ordering_k
        self.warmup_epoch = warmup_epoch
        self.pool_percentile = pool_percentile
        
        # Differentiable sorter
        self.sorter = PointSorter(
            in_channels=in_channels + 3, 
            hidden_channels=64, 
            num_orders=len(self.order), 
            residual_scale=0.1,
            tau=soft_tau
        )
        
        # Replace SerializedPooling in encoder with WindowPooling
        self._replace_pooling_layers()
    
    def _replace_pooling_layers(self):
        """Replace all SerializedPooling layers with WindowPooling"""
        from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import SerializedPooling
        
        # Replace in encoder
        for stage_name, stage_module in self.enc.named_children():
            for name, module in stage_module.named_children():
                if isinstance(module, SerializedPooling):
                    # Create WindowPooling with same parameters
                    window_pool = WindowPooling(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        stride=module.stride,
                        norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01) if hasattr(module, 'norm') else None,
                        act_layer=nn.GELU if hasattr(module, 'act') else None,
                        percentile=self.pool_percentile,
                        traceable=module.traceable,
                    )
                    setattr(stage_module, name, window_pool)

    def compute_ordering_loss_contrastive(self, point, scores, temperature=0.1):
        """
        Contrastive loss: geometrically close 3D points should have similar scores
        """
        N = scores.shape[0]
        chunk_size = min(1024, N)
        num_chunks = 4
        
        if N < 64:
            return torch.tensor(0.0, device=scores.device)
        
        loss_list = []
        
        for _ in range(num_chunks):
            # Sample local region
            center_idx = torch.randint(0, N, (1,), device=scores.device)
            center_coord = point.coord[center_idx]
            
            # Find nearest neighbors in 3D
            dists_3d = ((point.coord - center_coord) ** 2).sum(dim=1)
            _, local_idx = torch.topk(dists_3d, chunk_size, largest=False)
            
            local_coords = point.coord[local_idx]
            local_scores = scores[local_idx]
            
            # Pairwise 3D distances
            coord_diff = local_coords.unsqueeze(1) - local_coords.unsqueeze(0)
            geo_dist = torch.norm(coord_diff, dim=2)
            geo_sim = torch.exp(-geo_dist / (geo_dist.std() + 1e-6))
            
            for k in range(self.sorter.num_orders):
                s_k = local_scores[:, k:k+1]
                score_diff = torch.abs(s_k - s_k.t())
                
                # Contrastive: minimize score distance for geometrically close points
                contrastive_term = (geo_sim * score_diff).sum() / (geo_sim.sum() + 1e-6)
                
                # Repulsion: push apart distant points
                margin = 0.1
                geo_dissim = 1.0 - geo_sim
                repulsion_term = (geo_dissim * torch.relu(margin - score_diff)).sum() / (geo_dissim.sum() + 1e-6)
                
                loss_list.append(contrastive_term + 0.5 * repulsion_term)
        
        # Distribution regularization
        loss_dist_list = []
        for k in range(self.sorter.num_orders):
            sorted_scores_k, _ = torch.sort(scores[:, k])
            target = torch.linspace(0, 1, steps=N, device=scores.device)
            loss_dist_list.append(((sorted_scores_k - target) ** 2).mean())
        
        loss_dist = sum(loss_dist_list) / len(loss_dist_list)
        contrastive_loss = sum(loss_list) / len(loss_list)
        
        return contrastive_loss * 10.0 + loss_dist * 5.0

    def forward(self, data_dict):
        point = Point(data_dict)
        current_epoch = data_dict["epoch"] if self.training else float('inf')

        # REMOVED: point.serialization() - no serialization needed!
        point.sparsify()

        # Run differentiable sorter to generate ordering
        scores, learned_order, learned_inverse = self.sorter(point)
        
        # Set the learned order on the point
        point.serialized_order = learned_order
        point.serialized_inverse = learned_inverse

        # Compute contrastive ordering loss
        if self.training and self.ordering_loss_weight > 0.0:
            loss_ord = self.compute_ordering_loss_contrastive(point, scores)
            point["ordering_loss"] = loss_ord * self.ordering_loss_weight

        # Backbone forward (uses WindowPooling instead of SerializedPooling)
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
