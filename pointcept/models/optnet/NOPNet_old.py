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


# ─────────────────────────────────────────────────────────────────────────────
# Energy function from SPT (Robert et al., ICCV 2023, arXiv:2306.08045)
#
# E(s, G, λ) = Σ_i ||f_i - s_i||²          (data fidelity)
#            + λ Σ_{(i,j)∈G} w_ij * ||s_i - s_j||²  (smoothness / boundary)
#
# We relax the ℓ0 boundary term to a differentiable form:
#   boundary ≈ Σ_{(i,j)} w_ij * (1 - exp(-||s_i - s_j||² / τ))
# where w_ij ∈ [0,1] is predicted by the Green's kernel (high = same region).
#
# The neural operator learns to make w_ij → 1 inside superpoints (smooth)
# and w_ij → 0 across boundaries (discontinuous), mimicking cut-pursuit.
# ─────────────────────────────────────────────────────────────────────────────


class SuperpointNeuralOperator(nn.Module):
    """
    Models superpoint assignment as a fixed-point of a kernel integral equation.

    Iterative update (T steps):
        s_{t+1}(x) = sigma( ∫_{B(x,r)} G_theta(x,y,s_t(x),s_t(y)) * s_t(y) dy
                           + W * s_t(x) )

    The Green's kernel G_theta learns:
    - G(x,y) ~ 1 when x,y are in the same superpoint (smooth interior)
    - G(x,y) ~ 0 when x,y cross a semantic boundary (discontinuity)

    This directly mimics the ℓ0-cut pursuit energy from SPT:
      E = data_fidelity + λ * Σ_{edges} 𝟙[s_i ≠ s_j]
    relaxed to a differentiable form via w_ij = G_theta(i,j).
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

        # Projects hidden state → soft assignment logit (scalar score per point)
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

        # Build KNN graph on point coords — used ONLY for energy supervision
        idx = pointops.knn_query(
            k, coords.contiguous(), point.offset
        )[0].long()                                          # [N, k]
        idx = torch.clamp(idx, 0, N - 1)
        rel_pos = coords[idx] - coords.unsqueeze(1)          # [N, k, 3]

        inp = (
            torch.cat([point.coord, point.feat.detach()], dim=1)
            if point.feat is not None
            else point.coord
        )
        v = self.lift(inp)                                   # [N, H]

        for t in range(self.T):
            v_j = v[idx]                                     # [N, k, H]
            v_i = v.unsqueeze(1).expand(-1, k, -1)           # [N, k, H]
            kernel_input = torch.cat([rel_pos, v_i, v_j], dim=-1)
            G = self.green_kernel(kernel_input)              # [N, k, 1]
            integral = (G * v_j).mean(dim=1)                 # [N, H]
            v = self.norms[t](torch.relu(integral + self.W(v)))

        scores = self.project(v)                             # [N, 1]

        # Final kernel weights for energy loss
        v_j_final = v[idx]
        v_i_final = v.unsqueeze(1).expand(-1, k, -1)
        kernel_input_final = torch.cat([rel_pos, v_i_final, v_j_final], dim=-1)
        w_ij = self.green_kernel(kernel_input_final).squeeze(-1)  # [N, k]

        return scores, idx, w_ij, v

    def compute_spt_energy_loss(self, scores, idx, w_ij, v, coords, lam=1.0, tau=0.05):
        """
        Differentiable relaxation of the SPT ℓ0-cut pursuit energy:

          E = Σ_i ||v_i - s_i||²
            + λ Σ_{(i,j)∈G} w_ij * (1 - exp(-||s_i - s_j||² / τ))

        Terms:
          data_fidelity : encourages s_i (scores) to remain close to lifted
                          features v_i  (piecewise-constant interior)
          boundary      : w_ij acts as learned edge weight; penalizes score
                          discontinuities where the kernel says "same region"
                          (w_ij~1) and encourages them where kernel says
                          "boundary" (w_ij~0), matching the SPT ℓ0 term
          geo_prior     : spatial proximity → high w_ij (geometric coherence)
          piecewise_const: score consistency within predicted soft clusters
        """
        # Data fidelity: scores (projected v) should track v's structure
        # Use score as a 1-D proxy for the constant component per superpoint
        s_i = scores                                         # [N, 1]
        s_j = scores[idx].squeeze(-1)                        # [N, k]

        # ── (1) Boundary term: relaxed ℓ0 edge penalty ───────────────────────
        diff_sq = (s_i - s_j) ** 2                          # [N, k]
        # w_ij ~ 1 → same region → penalize diff (smooth inside)
        # w_ij ~ 0 → boundary   → don't penalize diff (allow discontinuity)
        boundary = (w_ij * (1.0 - torch.exp(-diff_sq / tau))).mean()

        # ── (2) Data fidelity: feature space smoothness inside superpoints ────
        v_j = v[idx]                                         # [N, k, H]
        feat_diff_sq = ((v.unsqueeze(1) - v_j) ** 2).sum(-1)  # [N, k]
        data_fidelity = (w_ij * feat_diff_sq).mean()

        # ── (3) Geometric coherence prior (spatial proximity → high w_ij) ────
        coord_diff = ((coords.unsqueeze(1) - coords[idx]) ** 2).sum(-1)
        geo_prior_target = torch.exp(-coord_diff / 0.02)
        geo_loss = F.mse_loss(w_ij, geo_prior_target.detach())

        # ── (4) Piecewise-constant score consistency ──────────────────────────
        pc_loss = (
            w_ij.detach() * (s_i.expand(-1, idx.shape[1]) - s_j) ** 2
        ).mean()

        loss = data_fidelity + lam * boundary + geo_loss + 0.1 * pc_loss
        return loss, {
            "data_fidelity": data_fidelity.item(),
            "boundary": boundary.item(),
            "geo_prior": geo_loss.item(),
            "piecewise_const": pc_loss.item(),
        }


class NOSuperpointPooling(PointModule):
    """
    PTv3-compatible pooling replacing SerializedPooling.

    Key design:
    - Pooling is done via SOFT SCORE-WEIGHTED ASSIGNMENT over KNN clusters,
      NOT by serialized space-filling codes.
    - Space-filling codes are preserved untouched for the PTv3 group attention.
    - The neural operator is supervised by the SPT energy function
      (Robert et al., ICCV 2023, arXiv:2306.08045), making superpoints
      geometrically coherent and semantically pure.

    Pooling mechanism:
      1. SuperpointNeuralOperator predicts per-point scores s_i ∈ [0,1]
         and edge weights w_ij encoding region membership.
      2. Points are clustered by their top-K score via differentiable
         weighted pooling: pooled_feat = Σ s_i * f_i / Σ s_i per cluster.
      3. Clusters are formed by strided subsampling of the sorted score
         order, so the number of output points = ceil(N / stride).
      4. The SPT energy loss supervises the operator during training.
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
        spt_lambda=1.0,
        spt_tau=0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        self.stride = stride
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable
        self.spt_lambda = spt_lambda
        self.spt_tau = spt_tau

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer is not None else None
        self.act = act_layer() if act_layer is not None else None

        self.operator = SuperpointNeuralOperator(
            in_channels=in_channels + 3,
            hidden_channels=hidden_channels,
            k=k,
            T=T,
        )

    def _pool_by_scores(self, point, scores, stride):
        """
        Score-based pooling (replaces serialized-code clustering):

        For each batch element:
          1. Sort points by score (descending) → high-score points become
             superpoint "seeds".
          2. Assign remaining points to the nearest seed via KNN (greedy).
          3. Weighted-average features and coords within each cluster.

        This produces ceil(N_batch / stride) superpoints per batch element,
        respecting batch boundaries via point.offset.
        """
        coords = point.coord          # [N, 3]
        feats  = self.proj(point.feat)  # [N, C_out]
        batch  = point.batch          # [N]
        offset = point.offset         # [B]  cumulative counts

        all_pooled_feat  = []
        all_pooled_coord = []
        all_pooled_batch = []
        cluster_map      = torch.zeros(
            coords.shape[0], dtype=torch.long, device=coords.device
        )

        start = 0
        global_seed_idx = 0
        for b, end in enumerate(offset.tolist()):
            end = int(end)
            n   = end - start
            if n == 0:
                start = end
                continue

            s_b   = scores[start:end, 0]   # [n]
            c_b   = coords[start:end]       # [n, 3]
            f_b   = feats[start:end]        # [n, C_out]

            # Number of superpoints for this batch element
            n_sp  = max(1, math.ceil(n / self.stride))

            # Select seeds = top-n_sp scoring points
            _, sorted_idx = torch.sort(s_b, descending=True)
            seed_local    = sorted_idx[:n_sp]              # [n_sp]
            seed_coords   = c_b[seed_local]                # [n_sp, 3]

            # Assign every point to its nearest seed
            dist = torch.cdist(c_b, seed_coords)           # [n, n_sp]
            assign = dist.argmin(dim=1)                    # [n]

            # Record global cluster indices for traceable inverse
            cluster_map[start:end] = global_seed_idx + assign

            # Weighted pooling within each cluster
            w_b = s_b.clamp(min=1e-4)                      # [n]
            pooled_f = torch_scatter.scatter(
                f_b * w_b.unsqueeze(-1),
                assign, dim=0, dim_size=n_sp, reduce="sum"
            ) / torch_scatter.scatter(
                w_b, assign, dim=0, dim_size=n_sp, reduce="sum"
            ).clamp_min(1e-6).unsqueeze(-1)                # [n_sp, C_out]

            pooled_c = torch_scatter.scatter(
                c_b * w_b.unsqueeze(-1),
                assign, dim=0, dim_size=n_sp, reduce="sum"
            ) / torch_scatter.scatter(
                w_b, assign, dim=0, dim_size=n_sp, reduce="sum"
            ).clamp_min(1e-6).unsqueeze(-1)                # [n_sp, 3]

            all_pooled_feat.append(pooled_f)
            all_pooled_coord.append(pooled_c)
            all_pooled_batch.append(
                torch.full((n_sp,), b, dtype=torch.long, device=coords.device)
            )
            global_seed_idx += n_sp
            start = end

        pooled_feat  = torch.cat(all_pooled_feat,  dim=0)
        pooled_coord = torch.cat(all_pooled_coord, dim=0)
        pooled_batch = torch.cat(all_pooled_batch, dim=0)

        _, counts = torch.unique_consecutive(pooled_batch, return_counts=True)
        new_offset = torch.cumsum(counts, dim=0).int()

        return pooled_feat, pooled_coord, pooled_batch, new_offset, cluster_map

    def forward(self, point: Point):
        # ── Serialization must already exist for PTv3 group attention ─────────
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(point.keys()), \
            "Run point.serialization() before NOSuperpointPooling"

        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0

        # ── Neural operator: scores + edge weights ────────────────────────────
        scores, idx, w_ij, v = self.operator(point)

        # ── Score-based pooling (NOT serialized-code clustering) ──────────────
        pooled_feat, pooled_coord, pooled_batch, new_offset, cluster_map = \
            self._pool_by_scores(point, scores, self.stride)

        M = pooled_coord.shape[0]

        # ── Re-derive serialized codes for the NEW (coarser) point set ────────
        # Quantize pooled coords to the same grid as parent, shifted by depth
        # so PTv3 group-attention still works on the coarser level.
        grid_coord = (
            pooled_coord / point.grid_coord.float().max() *
            (2 ** (point.serialized_depth - pooling_depth) - 1)
        ).long().clamp(min=0)

        # Encode to Morton codes (z-order) over the coarser grid
        # We reuse the existing serialized_code logic: map via grid_coord
        # For simplicity, derive codes from pooled grid_coord directly
        code = self._encode_morton(grid_coord, pooled_batch, point)

        order   = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=0,
            index=order,
            src=torch.arange(M, device=order.device),
        )
        code    = code.unsqueeze(0)   # [1, M] — single order curve
        order   = order.unsqueeze(0)
        inverse = inverse.unsqueeze(0)

        if self.shuffle_orders:
            # PTv3 expects multiple shuffled orders; pad with permutations
            n_orders = point.serialized_code.shape[0]
            codes_list   = [code]
            orders_list  = [order]
            inverses_list= [inverse]
            for _ in range(n_orders - 1):
                perm  = torch.randperm(M, device=code.device)
                c_p   = code[0][perm].unsqueeze(0)
                o_p   = torch.argsort(c_p[0]).unsqueeze(0)
                iv_p  = torch.zeros_like(o_p).scatter_(
                    dim=1, index=o_p,
                    src=torch.arange(M, device=o_p.device).unsqueeze(0)
                )
                codes_list.append(c_p)
                orders_list.append(o_p)
                inverses_list.append(iv_p)
            code    = torch.cat(codes_list,    dim=0)
            order   = torch.cat(orders_list,   dim=0)
            inverse = torch.cat(inverses_list, dim=0)

        point_dict = Dict(
            feat=pooled_feat,
            coord=pooled_coord,
            grid_coord=grid_coord,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=pooled_batch,
            offset=new_offset,
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster_map
            point_dict["pooling_parent"]  = point

        # ── SPT energy loss supervision ───────────────────────────────────────
        if self.training:
            loss, loss_components = self.operator.compute_spt_energy_loss(
                scores, idx, w_ij, v, point.coord,
                lam=self.spt_lambda, tau=self.spt_tau,
            )
            existing = point.get(
                "no_pooling_loss",
                torch.tensor(0.0, device=point.coord.device)
            )
            point_dict["no_pooling_loss"] = existing + loss
        else:
            point_dict["no_pooling_loss"] = point.get(
                "no_pooling_loss",
                torch.tensor(0.0, device=point.coord.device)
            )

        new_point = Point(point_dict)

        if self.norm is not None:
            new_point.feat = self.norm(new_point.feat)
        if self.act is not None:
            new_point.feat = self.act(new_point.feat)

        new_point.sparsify()
        return new_point

    @staticmethod
    def _encode_morton(grid_coord, batch, parent_point):
        """
        Simple per-batch-element Morton (z-order) encoding for the coarser
        pooled points. Offsets batch index into upper bits so codes are
        globally unique across the batch.
        """
        x = grid_coord[:, 0].long()
        y = grid_coord[:, 1].long()
        z = grid_coord[:, 2].long()

        def spread_bits(v):
            v = v & 0x1FFFFF  # 21 bits
            v = (v | (v << 32)) & 0x1F00000000FFFF
            v = (v | (v << 16)) & 0x1F0000FF0000FF
            v = (v | (v <<  8)) & 0x100F00F00F00F00F
            v = (v | (v <<  4)) & 0x10C30C30C30C30C3
            v = (v | (v <<  2)) & 0x1249249249249249
            return v

        code = spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)
        # Shift batch index into the top bits
        code = code + batch.long() * (2 ** 62 // max(batch.max().item() + 1, 1))
        return code



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

