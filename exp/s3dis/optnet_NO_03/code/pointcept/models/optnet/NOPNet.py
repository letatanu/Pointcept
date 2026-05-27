import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from addict import Dict

import pointops

from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
    SerializedPooling,
)


class SuperpointNeuralOperator(nn.Module):
    """
    Models superpoint assignment as a fixed-point of a kernel integral equation.

    Iterative update (T steps):
        s_{t+1}(x) = sigma( integral_{B(x,r)} G_theta(x,y,s_t(x),s_t(y)) * s_t(y) dy
                           + W * s_t(x) )

    The Green's kernel G_theta learns:
    - G(x,y) ~ 1 when x,y are in the same superpoint (smooth interior)
    - G(x,y) ~ 0 when x,y cross a semantic boundary (discontinuity)
    """

    def __init__(self, in_channels, hidden_channels=64, k=16, T=3):
        super().__init__()
        self.T = T
        self.k = k

        self.lift = nn.Linear(in_channels, hidden_channels)

        self.green_kernel = nn.Sequential(
            nn.Linear(3 + hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )

        self.W = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(T)]
        )

        self.project = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, point):
        coords = point.coord
        N = coords.shape[0]
        k = min(self.k, N - 1)

        idx = pointops.knn_query(
            k, coords.contiguous(), point.offset
        )[0].long()
        idx = torch.clamp(idx, 0, N - 1)
        rel_pos = coords[idx] - coords.unsqueeze(1)

        inp = (
            torch.cat([point.coord, point.feat.detach()], dim=1)
            if point.feat is not None
            else point.coord
        )
        v = self.lift(inp)

        for t in range(self.T):
            v_j = v[idx]
            v_i = v.unsqueeze(1).expand(-1, k, -1)

            kernel_input = torch.cat([rel_pos, v_i, v_j], dim=-1)
            G = self.green_kernel(kernel_input)

            integral = (G * v_j).mean(dim=1)
            v = self.norms[t](torch.relu(integral + self.W(v)))

        scores = self.project(v)

        v_j_final = v[idx]
        v_i_final = v.unsqueeze(1).expand(-1, k, -1)
        kernel_input_final = torch.cat([rel_pos, v_i_final, v_j_final], dim=-1)
        w_ij = self.green_kernel(kernel_input_final).squeeze(-1)

        return scores, idx, w_ij, v

    def compute_loss(self, scores, idx, w_ij, v, coords):
        v_j = v[idx]
        feat_diff_sq = ((v.unsqueeze(1) - v_j) ** 2).sum(-1)

        compactness = (w_ij * feat_diff_sq).mean()
        sharpness = ((1 - w_ij) * feat_diff_sq.detach()).mean()

        coord_diff = ((coords.unsqueeze(1) - coords[idx]) ** 2).sum(-1)
        geo_prior = torch.exp(-coord_diff / 0.02)
        geo_loss = F.mse_loss(w_ij, geo_prior.detach())

        score_j = scores[idx].squeeze(-1)
        pc_loss = (
            w_ij.detach() * (scores.expand(-1, idx.shape[1]) - score_j) ** 2
        ).mean()

        loss = compactness + geo_loss + 0.1 * pc_loss - 0.3 * sharpness
        return loss, {
            "compactness": compactness.item(),
            "geo_prior": geo_loss.item(),
            "piecewise_const": pc_loss.item(),
        }


class NOSuperpointPooling(PointModule):
    """
    PTv3-compatible pooling that keeps original serialization untouched.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        shuffle_orders=True,
        traceable=True,
        hidden_channels=64,
        k=16,
        T=3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        self.stride = stride
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer is not None else None
        self.act = act_layer() if act_layer is not None else None

        self.operator = SuperpointNeuralOperator(
            in_channels=in_channels + 3,
            hidden_channels=hidden_channels,
            k=k,
            T=T,
        )

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0

        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(point.keys()), "Run point.serialization() before NOSuperpointPooling"

        scores, idx, w_ij, v = self.operator(point)

        code = point.serialized_code >> (pooling_depth * 3)
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )

        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        weights = scores.clamp(min=1e-4)
        weights_sorted = weights[indices]

        feat_proj = self.proj(point.feat)
        feat_sorted = feat_proj[indices]

        feat_num = torch_scatter.segment_csr(
            feat_sorted * weights_sorted,
            idx_ptr,
            reduce="sum",
        )
        weight_den = torch_scatter.segment_csr(
            weights_sorted,
            idx_ptr,
            reduce="sum",
        ).clamp_min(1e-6)
        pooled_feat = feat_num / weight_den

        coord_num = torch_scatter.segment_csr(
            point.coord[indices] * weights_sorted,
            idx_ptr,
            reduce="sum",
        )
        pooled_coord = coord_num / weight_den

        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0], device=code.device)
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # Compute counts of points in each batch, then cumulative sum to get the offset
        _, counts = torch.unique_consecutive(point.batch[head_indices], return_counts=True)
        new_offset = torch.cumsum(counts, dim=0).int()

        point_dict = Dict(
            feat=pooled_feat,
            coord=pooled_coord,
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
            offset=new_offset,  # <-- Use the locally computed offset here
        )


        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point

        # Accumulate the energy loss if training
        if self.training:
            loss, _ = self.operator.compute_loss(
                scores, idx, w_ij, v, point.coord
            )
            # Safely fetch existing loss using get()
            existing_loss = point.get("no_pooling_loss", torch.tensor(0.0, device=point.coord.device))
            point_dict["no_pooling_loss"] = existing_loss + loss
        else:
            # Keep propagating zero or existing value during inference
            point_dict["no_pooling_loss"] = point.get("no_pooling_loss", torch.tensor(0.0, device=point.coord.device))


        new_point = Point(point_dict)

        if self.norm is not None:
            new_point.feat = self.norm(new_point.feat)
        if self.act is not None:
            new_point.feat = self.act(new_point.feat)

        new_point.sparsify()
        return new_point



@MODELS.register_module("OPTNet")
class OPTNet(PointTransformerV3):
    """
    PTv3 backbone with NO-based pooling only.

    Important:
    - Serialization is unchanged from PTv3.
    - No learned re-ordering is applied.
    - NOSuperpointPooling replaces SerializedPooling in encoder stages.
    """

    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        ordering_loss_weight=1.0,
        ordering_k=16,
        warmup_epoch=5,
        enable_score_concat=False,
        tau=0.1,
        loss_weights=(0, 0, 0, 1),
        num_classes=13,
        code_depth=10,
        pool_sizes=(4, 8, 8, 16),
        use_labels_in_loss=False,
        num_score_buckets=8,
        sorter_k=16,
        sorter_hidden_channels=64,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            order=order,
            **kwargs,
        )

        self.no_pooling_loss_weight = ordering_loss_weight
        self.sorter_hidden_channels = sorter_hidden_channels
        self.sorter_k = sorter_k

        stride = kwargs["stride"] if "stride" in kwargs else (2, 2, 2, 2)
        self._replace_pooling_layers(stride=stride)

    def _replace_pooling_layers(self, stride=(2, 2, 2, 2)):
        pool_idx = 0
        for _, stage in self.enc._modules.items():
            for name, layer in stage._modules.items():
                if isinstance(layer, SerializedPooling):
                    st = stride[pool_idx] if pool_idx < len(stride) else stride[-1]
                    stage._modules[name] = NOSuperpointPooling(
                        in_channels=layer.in_channels,
                        out_channels=layer.out_channels,
                        stride=st,
                        norm_layer=nn.LayerNorm,
                        act_layer=nn.GELU,
                        shuffle_orders=layer.shuffle_orders,
                        traceable=layer.traceable,
                        hidden_channels=self.sorter_hidden_channels,
                        k=self.sorter_k,
                        T=3,
                    )
                    pool_idx += 1

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=False, compute_codes=True)
        point.sparsify()

        if self.training:
            point["no_pooling_loss"] = torch.tensor(0.0, device=point.coord.device,
                                                    requires_grad=False)

        point = self.embedding(point)
        point = self.enc(point)

        # ── save BEFORE decoder overwrites point ──
        no_pooling_loss = None
        if self.training and "no_pooling_loss" in point.keys():
            no_pooling_loss = point["no_pooling_loss"] * self.no_pooling_loss_weight

        if not self.enc_mode:
            point = self.dec(point)

        # ── re-attach after decoder ──
        if no_pooling_loss is not None:
            point["no_pooling_loss"] = no_pooling_loss

        return point




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
            
            # The dictionary Pointcept uses for logging
            return_dict = dict(seg_loss=seg_loss)
            total_loss = seg_loss

            if "no_pooling_loss" in point:
                no_pooling_loss = point["no_pooling_loss"]
                # Add to return_dict so Pointcept's AverageMeter tracks and logs it
                return_dict["no_pooling_loss"] = no_pooling_loss
                total_loss = total_loss + no_pooling_loss

            # The 'loss' key is the only one used for loss.backward()
            return_dict["loss"] = total_loss
            return return_dict

        elif "segment" in data_dict:
            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)

        return dict(seg_logits=seg_logits)

