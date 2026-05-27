import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.utils.structure import Point
from pointcept.models.losses import build_criteria
from pointcept.models.utils.serialization import encode
import pointops
import torch.nn.functional as F
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import SerializedPooling

# REMOVED: neuralop imports — GNOBlock uses cdist internally, OOM at point cloud scale
# from neuralop.layers.gno_block import GNOBlock
# from neuralop.layers.neighbor_search import NeighborSearch


# ---------------------------------------------------------------------------
# KNN-based Graph Neural Operator layer
# Implements the integral: (K u)(x_i) = mean_{j in KNN(i)} MLP([x_i-x_j, f_j])
# Uses pointops.knn_query — O(N*k), never builds N×N matrix
# ---------------------------------------------------------------------------
class GNOKernelLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.kernel_mlp = nn.Sequential(
            nn.Linear(3 + in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.self_linear = nn.Linear(in_channels, out_channels)

    def forward(self, coords, features, offset):
        N = coords.shape[0]
        k = min(self.k, N - 1)
        idx = pointops.knn_query(k, coords.contiguous(), offset)[0].long()  # (N, k)
        idx = torch.clamp(idx, 0, N - 1)

        neighbor_coords = coords[idx]                                        # (N, k, 3)
        neighbor_features = features[idx]                                    # (N, k, C)

        rel_pos = neighbor_coords - coords.unsqueeze(1)                      # (N, k, 3)
        kernel_input = torch.cat([rel_pos, neighbor_features], dim=-1)      # (N, k, 3+C)

        kernel_out = self.kernel_mlp(kernel_input)                           # (N, k, out)
        aggregated = kernel_out.mean(dim=1)                                  # (N, out)

        return aggregated + self.self_linear(features)


# ---------------------------------------------------------------------------
# Neural Operator-based PointSorter
# ---------------------------------------------------------------------------
class PointSorterNO(nn.Module):
    """
    Graph Neural Operator PointSorter.
    Each point's score is computed by integrating over its KNN neighborhood.
    Locality is architectural — no loss_locality term needed.
    Memory: O(N*k), safe for N > 1M points.
    """

    def __init__(self, in_channels, hidden_channels=64, num_orders=1, k=16):
        super().__init__()
        self.num_orders = num_orders

        self.lift = nn.Linear(in_channels, hidden_channels)
        self.gno1 = GNOKernelLayer(hidden_channels, hidden_channels, k=k)
        self.gno2 = GNOKernelLayer(hidden_channels, hidden_channels, k=k)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.score_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, num_orders),
        )

    def forward(self, point):
        if point.feat is not None:
            inp = torch.cat([point.coord, point.feat.detach()], dim=1)
        else:
            inp = point.coord

        coords = point.coord
        h = self.lift(inp)

        h = self.norm1(torch.relu(self.gno1(coords, h, point.offset)))
        h = self.norm2(torch.relu(self.gno2(coords, h, point.offset)))

        scores = torch.sigmoid(self.score_head(h))  # (N, num_orders)

        batch_offset = point.batch.unsqueeze(1) * (scores.max().detach() + 10.0)
        scores_with_batch = scores + batch_offset

        orders_list, inverses_list = [], []
        scores_t = scores_with_batch.transpose(0, 1)
        for i in range(self.num_orders):
            order = torch.argsort(scores_t[i])
            inverse = torch.zeros_like(order)
            inverse[order] = torch.arange(len(order), device=order.device)
            orders_list.append(order)
            inverses_list.append(inverse)

        return scores, torch.stack(orders_list), torch.stack(inverses_list)

    def compute_loss(self, scores, coords, batch_ids, offset,
                     z_target=None, labels=None):
        sorted_scores, _ = torch.sort(scores.view(-1))
        target = torch.linspace(0, 1, steps=len(sorted_scores), device=scores.device)
        loss_dist = F.mse_loss(sorted_scores, target)
        return loss_dist, {"distribution": loss_dist.item()}


# ---------------------------------------------------------------------------
# SemanticGridPoolingLayer
# ---------------------------------------------------------------------------
class SemanticGridPoolingLayer(nn.Module):
    """
    Combines 3D physical voxels with learned 1D scores to prevent
    feature smearing across semantic boundaries.
    """

    def __init__(self, in_channels, out_channels, stride=2, code_depth=10,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, traceable=True,
                 num_score_buckets=8, **kwargs):
        super().__init__()
        self.stride = stride
        self.code_depth = code_depth
        self.traceable = traceable
        self.num_score_buckets = num_score_buckets

        self.proj = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, point: Point):
        device = point.coord.device

        new_grid_coord = torch.div(point.grid_coord, self.stride, rounding_mode='floor')

        scores = (point.scores.squeeze(-1) if point.scores is not None
                  else torch.zeros(point.coord.shape[0], device=device))
        score_buckets = (scores * self.num_score_buckets).long()
        score_buckets = torch.clamp(score_buckets, 0, self.num_score_buckets - 1)

        semantic_grid_coord = torch.cat(
            [new_grid_coord, score_buckets.unsqueeze(-1)], dim=1)

        cluster_idx, unique_idx = pointops.unique_and_cluster(
            semantic_grid_coord, point.batch)

        new_point = Point()
        new_point.grid_coord = new_grid_coord[unique_idx]
        new_point.coord = point.coord[unique_idx]
        new_point.batch = point.batch[unique_idx]
        new_point.offset = pointops.compute_offset(new_point.batch)

        pooled_feat = pointops.scatter_mean(point.feat, cluster_idx, dim=0)
        new_point.feat = self.act(self.norm(self.proj(pooled_feat)))

        if point.scores is not None:
            new_point.scores = pointops.scatter_mean(point.scores, cluster_idx, dim=0)

        n_groups = unique_idx.shape[0]
        batch_shifted = new_point.batch.long() << (self.code_depth * 3)
        sequential_code = torch.arange(n_groups, dtype=torch.long, device=device)
        new_code = batch_shifted + sequential_code
        new_code_2d = new_code.unsqueeze(0)

        new_order = torch.arange(n_groups, dtype=torch.long, device=device).unsqueeze(0)
        new_inverse = new_order.clone()

        new_point.serialized_code = new_code_2d
        new_point.serialized_order = new_order
        new_point.serialized_inverse = new_inverse
        new_point.serialized_depth = self.code_depth

        if self.traceable:
            new_point.pooling_inverse = cluster_idx
            new_point.pooling_parent = point

        return new_point


# ---------------------------------------------------------------------------
# PointSorter (MLP-based, kept for ablation)
# ---------------------------------------------------------------------------
class PointSorter(nn.Module):
    """
    Lightweight MLP Sorter: PTv3 Z-code base + learned semantic offset.
    """

    def __init__(self, in_channels=6, hidden_channels=64, ordering_k=16,
                 loss_weights=(1, 0, 0, 1), tau=0.1, num_classes=13, ignore_index=-1):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh()
        )
        self.offset_scale = 0.05
        self.ordering_loss = OrderingLoss(
            ordering_k=ordering_k,
            loss_weights=loss_weights,
            tau=tau,
            ignore_index=-1
        )

    def forward(self, point):
        z = point.serialized_code
        if hasattr(z, "dim") and z.dim() == 2:
            z = z[0]
        elif isinstance(z, (list, tuple)):
            z = z[0]

        z_code = z.float()
        base_score = (z_code - z_code.min()) / (z_code.max() - z_code.min() + 1e-6)

        inp = point.feat if point.feat is not None else point.coord
        offset = self.mlp(inp).squeeze(-1)

        final_scores = base_score + (offset * self.offset_scale)
        learned_scores = torch.clamp(final_scores, min=1e-5, max=1 - 1e-5).unsqueeze(1)

        if self.training:
            self._last_feat = self.mlp[0:3](inp)

        return learned_scores

    def compute_loss(self, scores, coords, batch_ids, offset,
                     z_target=None, labels=None):
        features = getattr(self, '_last_feat', None)
        return self.ordering_loss(
            scores, coords, batch_ids, offset,
            z_target=z_target,
            features=features,
            labels=labels
        )


# ---------------------------------------------------------------------------
# OrderingLoss
# ---------------------------------------------------------------------------
class OrderingLoss(nn.Module):
    def __init__(self, ordering_k=16, loss_weights=(0, 0, 0, 1),
                 tau=0.1, ignore_index=-1):
        super().__init__()
        self.ordering_k = ordering_k
        self.w_loc, self.w_dist, self.w_z, self.w_glob = loss_weights
        self.tau = tau
        self.ignore_index = ignore_index

    def forward(self, scores, coords, batch_ids, offset,
                z_target=None, features=None, labels=None):
        scores_1d = scores.view(-1, 1)
        total = torch.tensor(0.0, device=scores.device)
        loss_dict = {}

        if self.w_loc > 0:
            if offset is None:
                _, counts = torch.unique_consecutive(
                    batch_ids.long(), return_counts=True)
                offset = torch.cumsum(counts, dim=0).int()
            else:
                offset = offset.int()
            if labels is not None:
                loss_locality = self.semantic_locality_loss(
                    scores_1d, coords, labels, offset) * self.w_loc
            else:
                loss_locality = self._locality_loss(
                    scores_1d, coords, offset) * self.w_loc
            total = total + loss_locality
            loss_dict["locality"] = loss_locality.item()

        if self.w_dist > 0:
            if offset is None:
                _, counts = torch.unique_consecutive(
                    batch_ids.long(), return_counts=True)
                offset = torch.cumsum(counts, dim=0).int()
            else:
                offset = offset.int()
            loss_distribution = self._fps_distribution_loss(
                scores_1d, coords, batch_ids, offset) * self.w_dist
            total = total + loss_distribution
            loss_dict["distribution"] = loss_distribution.item()

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

        if features is not None and self.w_glob > 0:
            loss_global = self._global_feature_loss(
                scores_1d, features, batch_ids) * self.w_glob
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
        return F.mse_loss(scores.unsqueeze(1).expand_as(neighbor_scores),
                          neighbor_scores)

    def _fps_distribution_loss(self, scores, coords, batch_ids, offset,
                                num_centroids=128):
        batch_ids_long = batch_ids.long()
        num_batches = int(batch_ids_long.max().item()) + 1
        total_dist_loss = torch.tensor(0.0, device=scores.device)
        count = 0

        if offset is None:
            _, counts = torch.unique_consecutive(batch_ids_long, return_counts=True)
            offset = torch.cumsum(counts, dim=0).int()
        else:
            offset = offset.int()

        new_counts = torch.full((num_batches,), num_centroids,
                                dtype=torch.int32, device=scores.device)
        new_offset = torch.cumsum(new_counts, dim=0).int()

        min_points_in_batch = (offset[0] if num_batches == 1
                               else min(offset[0], (offset[1:] - offset[:-1]).min()))
        if min_points_in_batch < num_centroids:
            return torch.tensor(0.0, device=scores.device)

        fps_idx = pointops.farthest_point_sampling(
            coords.contiguous(), offset, new_offset).long()

        for b in range(num_batches):
            start_idx = b * num_centroids
            end_idx = (b + 1) * num_centroids
            b_fps_idx = fps_idx[start_idx:end_idx]
            anchor_scores = scores[b_fps_idx].flatten()
            sorted_anchors, _ = torch.sort(anchor_scores)
            target = torch.linspace(0, 1, steps=num_centroids, device=scores.device)
            total_dist_loss = total_dist_loss + F.l1_loss(sorted_anchors, target)
            count += 1

        return total_dist_loss / max(count, 1)

    def _global_feature_loss(self, scores, features, batch_ids):
        batch_ids_long = batch_ids.long()
        num_batches = int(batch_ids_long.max().item()) + 1

        global_max = torch.empty((num_batches, features.shape[1]),
                                 device=features.device)
        for b in range(num_batches):
            mask = batch_ids_long == b
            if mask.any():
                global_max[b] = features[mask].max(dim=0)[0]
            else:
                global_max[b] = 0.0

        g = global_max[batch_ids_long]
        diff = (features - g) / self.tau
        target_scores = (2.0 * torch.sigmoid(diff)).mean(dim=1, keepdim=True)
        return F.mse_loss(scores, target_scores.detach())

    def semantic_locality_loss(self, scores, coords, labels, offset):
        N = scores.shape[0]
        if N < 2 or torch.isnan(coords).any():
            return torch.tensor(0.0, device=scores.device)

        k = min(self.ordering_k, N - 1)
        idx = pointops.knn_query(k, coords.contiguous(), offset)[0].long()
        idx = torch.clamp(idx, 0, N - 1)

        neighbor_scores = scores[idx]
        neighbor_labels = labels[idx]

        curr_scores = scores.unsqueeze(1).expand_as(neighbor_scores)
        curr_labels = labels.unsqueeze(1).expand_as(neighbor_labels)

        valid_mask = ((curr_labels != self.ignore_index) &
                      (neighbor_labels != self.ignore_index))
        same_class = (curr_labels == neighbor_labels) & valid_mask
        diff_class = (curr_labels != neighbor_labels) & valid_mask

        pull_loss = (F.mse_loss(curr_scores[same_class], neighbor_scores[same_class])
                     if same_class.any() else 0.0)

        margin = 0.02
        score_diff = torch.abs(curr_scores[diff_class] - neighbor_scores[diff_class])
        push_loss = (F.relu(margin - score_diff).mean()
                     if diff_class.any() else 0.0)

        return pull_loss + push_loss


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
        loss_weights=(0, 0, 0, 1),
        num_classes=13,
        code_depth=10,
        pool_sizes=(4, 8, 8, 16),
        use_labels_in_loss=False,
        num_score_buckets=8,
        sorter_k=16,                # replaces sorter_radius — uses KNN not radius
        sorter_hidden_channels=64,
        **kwargs,
    ):
        single_order = [order[0]] if isinstance(order, (list, tuple)) else [order]
        self.enable_score_concat = enable_score_concat
        self.code_depth = code_depth
        self.use_labels_in_loss = use_labels_in_loss

        backbone_in_channels = in_channels + 1 if enable_score_concat else in_channels
        super().__init__(
            in_channels=backbone_in_channels,
            order=single_order,
            **kwargs
        )

        self.ordering_loss_weight = ordering_loss_weight
        self.warmup_epoch = warmup_epoch
        self.all_orders = list(order) if isinstance(order, (list, tuple)) else [order]

        self.sorter = PointSorterNO(
            in_channels=in_channels + 3,
            hidden_channels=sorter_hidden_channels,
            num_orders=len(self.order),
            k=sorter_k,              # O(N*k) — no cdist, no OOM
        )

        self.num_score_buckets = num_score_buckets
        self._replace_pooling_layers(
            stride=kwargs['stride'], pool_sizes=pool_sizes)

    def _replace_pooling_layers(self, stride=(2, 2, 2, 2), pool_sizes=(8, 8, 8, 16)):
        pool_idx = 0
        for i, layer in enumerate(self.enc):
            if isinstance(layer, SerializedPooling):
                st = stride[pool_idx] if pool_idx < len(stride) else stride[-1]
                self.enc[i] = SemanticGridPoolingLayer(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    stride=st,
                    code_depth=self.code_depth,
                    traceable=True,
                    num_score_buckets=self.num_score_buckets
                )
                pool_idx += 1

    def _build_serialization_from_scores(self, scores, point):
        batch_offsets = point.batch.float() * 10.0
        batch_aware_scores = scores.view(-1) + batch_offsets
        order = torch.argsort(batch_aware_scores).unsqueeze(0)
        inverse = torch.zeros_like(order).scatter_(
            dim=1, index=order,
            src=torch.arange(order.shape[1], device=order.device
                             ).unsqueeze(0).expand_as(order)
        )

        N = order.shape[1]
        sorted_batch = point.batch[order[0]]
        sequential_code = torch.arange(N, dtype=torch.long, device=order.device)

        shift_bits = self.code_depth * 3
        new_code = (sorted_batch.long() << shift_bits) | sequential_code

        original_order_code = torch.zeros_like(new_code)
        original_order_code[order[0]] = new_code
        code_2d = original_order_code.unsqueeze(0)

        point.serialized_code = code_2d
        point.serialized_order = order
        point.serialized_inverse = inverse
        point.serialized_depth = self.code_depth

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=False,
                            compute_codes=True)

        current_epoch = data_dict.get("epoch", 0) if self.training else float("inf")
        use_learned = current_epoch >= self.warmup_epoch

        if self.training:
            tmp_point = Point({
                "grid_coord": point.grid_coord,
                "batch": point.batch
            })
            tmp_point.serialization(
                order=self.all_orders,
                shuffle_orders=False,
                compute_codes=True,
            )
            target_codes = tmp_point.serialized_code.float().clone()

            if not use_learned:
                point.serialization(
                    order=self.order,
                    shuffle_orders=True,
                    compute_codes=True,
                )
        else:
            target_codes = None
            point.serialization(
                order=self.order,
                shuffle_orders=False,
                compute_codes=True,
            )

        # AFTER (fixed) — unpack the tuple immediately
        scores, learned_order, learned_inverse = self.sorter(point)
        point.scores = scores                    # only the (N, num_orders) score tensor

        if use_learned:
            # _build_serialization_from_scores also needs just scores
            self._build_serialization_from_scores(scores, point)

        if self.training and self.ordering_loss_weight > 0:
            code_mins = target_codes.min(dim=1, keepdim=True)[0]
            code_maxs = target_codes.max(dim=1, keepdim=True)[0]
            z_targets = (target_codes - code_mins) / (code_maxs - code_mins + 1e-6)
            labels = (data_dict.get("segment", None)
                    if self.use_labels_in_loss else None)
            loss_ord, loss_dict = self.sorter.compute_loss(
                scores,           # <-- just the score tensor now
                point.coord,
                point.batch,
                point.offset,
                z_target=z_targets,
                labels=labels
            )
            loss_ord = torch.clamp(loss_ord, max=10.0)
            point.ordering_loss = loss_ord * self.ordering_loss_weight

        if self.enable_score_concat:
            if point.feat is None:
                point.feat = scores              # <-- scores, not the tuple
            else:
                point.feat = torch.cat([point.feat, scores], dim=1)


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
        self.backbone = MODELS.build(backbone)
        self.criteria = build_criteria(criteria)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, data_dict):
        point = self.backbone(data_dict)
        seg_logits = self.seg_head(point.feat)

        if self.training:
            target = data_dict["segment"]
            if target.dtype != torch.int64:
                target = target.long()

            valid_mask = (target >= 0) & (target < self.num_classes)
            if not valid_mask.all():
                target = target.clone()
                target[~valid_mask] = -1

            seg_loss = self.criteria(seg_logits, target)
            return_dict = dict(seg_loss=seg_loss)

            if "ordering_loss" in point:
                ordering_loss = point["ordering_loss"]
                return_dict["ordering_loss"] = ordering_loss
                return_dict["loss"] = seg_loss + ordering_loss
            else:
                return_dict["loss"] = seg_loss

            return return_dict

        elif "segment" in data_dict:
            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)

        return dict(seg_logits=seg_logits)
