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
    Ultra-fast strided sampling: simply take every stride-th point
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        traceable=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = PointSequential(norm_layer(out_channels)) if norm_layer is not None else None
        self.act = PointSequential(act_layer()) if act_layer is not None else None

    def forward(self, point: Point):
        """
        Simply take every stride-th point along the ordering
        """
        # Get ordering
        if hasattr(point, 'serialized_order') and point.serialized_order is not None:
            order = point.serialized_order[0]
        else:
            order = torch.argsort(point.coord[:, 2])
        
        N = len(order)
        stride = self.stride
        
        # Process each batch separately
        batch = point.batch
        unique_batches = torch.unique(batch, sorted=True)
        
        all_selected = []
        all_clusters = []
        cluster_offset = 0
        
        for b in unique_batches:
            batch_mask = batch == b
            batch_indices = torch.where(batch_mask)[0]
            
            # Get ordering for this batch
            batch_order_mask = torch.isin(order, batch_indices)
            batch_order = order[batch_order_mask]
            
            # Simply take every stride-th point
            selected = batch_order[::stride]
            all_selected.append(selected)
            
            # Assign clusters
            n_batch = len(batch_order)
            window_ids = torch.arange(n_batch, device=order.device) // stride + cluster_offset
            
            for idx in batch_order:
                local_pos = (batch_order == idx).nonzero(as_tuple=True)[0].item()
                all_clusters.append(window_ids[local_pos].item())
            
            cluster_offset += (n_batch + stride - 1) // stride
        
        selected_indices = torch.cat(all_selected)
        cluster = torch.tensor(all_clusters, dtype=torch.long, device=order.device)
        
        # Aggregate features
        _, indices = torch.sort(cluster)
        num_clusters = cluster.max().item() + 1
        counts = torch.bincount(cluster, minlength=num_clusters)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        
        feat_pooled = torch_scatter.segment_csr(
            self.proj(point.feat)[indices], idx_ptr, reduce="max"
        )
        
        # Downsampled attributes
        coord_pooled = point.coord[selected_indices]
        grid_coord_pooled = point.grid_coord[selected_indices] >> 1
        batch_pooled = point.batch[selected_indices]
        
        # New ordering
        num_selected = len(selected_indices)
        num_orders = point.serialized_order.shape[0] if hasattr(point, 'serialized_order') else 1
        new_serialized_order = torch.arange(num_selected, device=order.device).unsqueeze(0).expand(num_orders, -1)
        new_serialized_inverse = new_serialized_order.clone()
        
        point_dict = {
            'feat': feat_pooled,
            'coord': coord_pooled,
            'grid_coord': grid_coord_pooled,
            'batch': batch_pooled,
            'serialized_order': new_serialized_order,
            'serialized_inverse': new_serialized_inverse,
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
        # self.pool_percentile = pool_percentile
        
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
                        # percentile=self.pool_percentile,
                        traceable=module.traceable,
                    )
                    setattr(stage_module, name, window_pool)

    def compute_ordering_loss_simple(self, point, scores):
        """
        Ultra-simple loss: neighbors should have similar scores + distribution regularization
        """
        N = scores.shape[0]
        
        if N < 64:
            return torch.tensor(0.0, device=scores.device)
        
        # Find k nearest neighbors
        k = min(16, N - 1)
        knn_idx = pointops.knn_query(k, point.coord, point.offset)[0]  # (N, k)
        
        total_loss = 0.0
        
        for order_idx in range(self.sorter.num_orders):
            # Get scores for this order
            order_scores = scores[:, order_idx]  # (N,)
            
            # Get neighbor scores
            neighbor_scores = order_scores[knn_idx.long()]  # (N, k)
            
            # Compute differences: broadcast (N,) vs (N, k)
            # neighbor_scores - order_scores[:, None] creates proper (N, k) differences
            diff = neighbor_scores - order_scores.unsqueeze(1)  # (N, k)
            
            # Simple L2 loss: neighbors should have similar scores
            loss_local = (diff ** 2).mean()
            
            # Distribution loss: use full [0, 1] range
            sorted_s, _ = torch.sort(order_scores)
            target = torch.linspace(0, 1, steps=N, device=scores.device)
            loss_dist = ((sorted_s - target) ** 2).mean()
            
            total_loss += loss_local + loss_dist
        
        return total_loss / self.sorter.num_orders



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
            loss_ord = self.compute_ordering_loss_simple(point, scores)
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
