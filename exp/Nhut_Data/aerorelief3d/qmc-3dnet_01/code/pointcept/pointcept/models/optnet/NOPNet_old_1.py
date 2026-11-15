import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import torch.utils.checkpoint as cp
from addict import Dict

import pointops

from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
    SerializedPooling,
    SerializedUnpooling,
)


class SuperpointNeuralOperator(nn.Module):

    def __init__(self, in_channels, hidden_channels=64, k=16, T=3, k_anchor=8):
        super().__init__()
        self.k = k  
        self.T = T
        self.k_anchor = k_anchor  # only assign to k_anchor nearest anchors

        self.lift = nn.Linear(in_channels, hidden_channels)
        
        # --> Split the Green Kernel into pre-computable parts <--
        self.green_pos = nn.Linear(3, hidden_channels)
        self.green_vi  = nn.Linear(hidden_channels, hidden_channels)
        self.green_vj  = nn.Linear(hidden_channels, hidden_channels, bias=False)
        
        self.green_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.W     = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(T)])
        self.assign_key   = nn.Linear(hidden_channels, hidden_channels)
        self.assign_query = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, point, n_sp):
        coords = point.coord
        N = coords.shape[0]
        k = min(self.k, N - 1)

        # KNN graph for the Green's kernel integral
        idx = pointops.knn_query(k, coords.contiguous(), point.offset)[0].long()
        idx = torch.clamp(idx, 0, N - 1)
        rel_pos = coords[idx] - coords.unsqueeze(1)

        inp = torch.cat([coords, point.feat.detach()], dim=1) \
              if point.feat is not None else coords
        
        v = self.lift(inp)
        pos_proj = self.green_pos(rel_pos) # Compute once!

        # ──> NEW: Checkpointed GNN Step <──
        def gnn_step(v_curr, p_proj, current_idx):
            # Broadcasting unsqueeze(1) saves memory over expand()
            vi_proj = self.green_vi(v_curr).unsqueeze(1) 
            vj_proj = self.green_vj(v_curr)[current_idx]
            
            G_feat = p_proj + vi_proj + vj_proj
            G = self.green_mlp(G_feat)
            
            integral = (G * v_curr[current_idx]).mean(dim=1)
            return integral, G

        w_ij = None
        for t in range(self.T):
            # Use checkpointing during training to save massive amounts of memory
            if self.training and v.requires_grad:
                integral, G = cp.checkpoint(gnn_step, v, pos_proj, idx, use_reentrant=False)
            else:
                integral, G = gnn_step(v, pos_proj, idx)
                
            v = self.norms[t](torch.relu(integral + self.W(v)))
            
            # Save the final G to use as w_ij
            if t == self.T - 1 and self.training:
                w_ij = G.squeeze(-1).clamp(1e-6, 1.0 - 1e-6)

        # ── Sparse local assignment ───────────────────────────────────────────
        # OLD: Global Top-K (Removes spatial uniformity)
        # scores     = v.norm(dim=-1)                        
        # _, top_idx = torch.topk(scores, n_sp, dim=0)       

        # NEW: Local Argmax along the Space-Filling Curve
        scores = v.norm(dim=-1)
        stride = math.ceil(N / n_sp)
        
        # --> VECTORIZED LOCAL ARGMAX (Lightning Fast) <--
        pad_len = n_sp * stride - N
        if pad_len > 0:
            padded_scores = F.pad(scores, (0, pad_len), value=-1e9)
        else:
            padded_scores = scores
            
        reshaped_scores = padded_scores.view(n_sp, stride)
        local_argmax = reshaped_scores.argmax(dim=1)
        
        base_indices = torch.arange(n_sp, device=coords.device) * stride
        top_idx = torch.clamp(base_indices + local_argmax, 0, N - 1)

        # Find anchor neighbors...
        k_anchor = min(self.k_anchor, n_sp)
        anchor_nn = pointops.knn_query(
            k_anchor, coords[top_idx].contiguous(), 
            torch.tensor([n_sp], device=coords.device, dtype=torch.int),
            coords.contiguous(), point.offset,
        )[0].long()
        anchor_nn = torch.clamp(anchor_nn, 0, n_sp - 1)

        keys    = self.assign_key(v)                       # [N, H]
        queries = self.assign_query(v[top_idx])            # [n_sp, H]

        # AMP FIX: Scale BEFORE multiplication to prevent fp16 overflow
        H_dim = v.shape[-1]
        scale = math.pow(H_dim, 0.25)
        keys_scaled    = keys / scale
        queries_scaled = queries / scale

        local_queries = queries_scaled[anchor_nn]                 
        local_logits  = (keys_scaled.unsqueeze(1) * local_queries).sum(-1) 
        local_A = torch.softmax(local_logits, dim=-1)      
        # local_A[i, j] = soft prob that point i belongs to anchor anchor_nn[i,j]

        return local_A, anchor_nn, top_idx, v, idx, w_ij

    def energy_loss(self, local_A, anchor_nn, v, idx, w_ij, lam=1.0, beta=0.1):
        v_target = v.detach()

        # (1) Local data fidelity - cast to float32
        anchor_v      = v_target[anchor_nn]                      
        v_reconstruct = (local_A.unsqueeze(-1) * anchor_v).sum(1)
        data_fidelity = F.mse_loss(v_reconstruct.float(), v_target.float())

        # (2) Boundary - cast to float32 BEFORE squaring
        vr_j     = v_reconstruct[idx]                            
        vr_i     = v_reconstruct.unsqueeze(1).expand_as(vr_j)
        diff_sq  = ((vr_i.float() - vr_j.float()) ** 2).sum(-1)                  
        boundary = (w_ij.float().clamp(0, 1) * diff_sq.clamp(max=1e4)).mean()

        # (3) Entropy
        local_A_safe = local_A.float().clamp(min=1e-6, max=1.0)
        entropy = -(local_A_safe * local_A_safe.log()).sum(dim=-1).mean()

        loss = data_fidelity + lam * boundary + beta * entropy

        if not torch.isfinite(loss):
            loss = torch.tensor(0.0, device=v.device, requires_grad=True)

        return loss, {
            "data_fidelity": data_fidelity.item(),
            "boundary":      boundary.item(),
            "entropy":       entropy.item(),
        }
    
class NOSuperpointPooling(PointModule):
    """
    PTv3-compatible pooling replacing SerializedPooling.

    Pooling mechanism:
      - SuperpointNeuralOperator predicts per-point hidden states v and
        edge weights w_ij via an iterative Green's kernel integral.
      - Sparse local soft assignment A[N, k_anchor] is computed via
        dot-product attention between each point and its k_anchor nearest
        superpoint anchors (selected by hidden-state norm).
      - Features and coords are pooled via differentiable scatter over A.
      - Self-supervised by the SPT energy (Robert et al., ICCV 2023):
          E = data_fidelity + λ * boundary + β * entropy

    Space-filling codes are preserved untouched for PTv3 group attention.
    Memory cost is O(N * k_anchor) instead of O(N * K).
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
        k_anchor=8,
        spt_lambda=1.0,
        spt_beta=0.1,
    ):
        super().__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        self.stride         = stride
        self.shuffle_orders = shuffle_orders
        self.traceable      = traceable
        self.k_anchor       = k_anchor
        self.spt_lambda     = spt_lambda
        self.spt_beta       = spt_beta

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer is not None else None
        self.act  = act_layer()              if act_layer  is not None else None

        self.operator = SuperpointNeuralOperator(
            in_channels=in_channels + 3,
            hidden_channels=hidden_channels,
            k=k,
            T=T,
            k_anchor=k_anchor,
        )

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0

        assert {
            "serialized_code", "serialized_order",
            "serialized_inverse", "serialized_depth",
        }.issubset(point.keys()), \
            "Run point.serialization() before NOSuperpointPooling"

        coords = point.coord
        feats  = self.proj(point.feat)
        batch  = point.batch
        offset = point.offset

        if not torch.isfinite(feats).all():
            feats = torch.nan_to_num(feats, nan=0.0, posinf=1e4, neginf=-1e4)

        all_pooled_feat   = []
        all_pooled_coord  = []
        all_pooled_batch  = []
        all_global_anchor = []   # global anchor indices for code inheritance
        all_local_A       = []
        all_anchor_nn     = []
        all_v             = []
        all_idx           = []
        all_w_ij          = []
        all_coords        = []
        all_global_anchor_nn = []

        cluster_map   = torch.zeros(
            coords.shape[0], dtype=torch.long, device=coords.device
        )
        start         = 0
        global_sp_idx = 0

        for b, end in enumerate(offset.tolist()):
            end  = int(end)
            n    = end - start
            if n == 0:
                start = end
                continue

            n_sp = max(1, math.ceil(n / self.stride))

            sub_point = Point(Dict(
                coord=coords[start:end],
                feat=point.feat[start:end],
                offset=torch.tensor([n], device=coords.device, dtype=torch.int),
                batch=torch.zeros(n, dtype=torch.long, device=coords.device),
            ))

            local_A, anchor_nn, top_idx, v, idx, w_ij = self.operator(sub_point, n_sp)

            # Guard operator outputs
            local_A = torch.nan_to_num(local_A, nan=1.0 / local_A.shape[1])
            local_A = local_A / local_A.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            if w_ij is not None:
                w_ij = torch.nan_to_num(w_ij, nan=0.5).clamp(1e-6, 1.0 - 1e-6)

            v = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)

            # Global anchor indices — used to inherit serialized codes
            global_anchor = start + top_idx                  # [n_sp]
            all_global_anchor.append(global_anchor)

            # Sparse differentiable pooling
            k_a     = local_A.shape[1]
            f_b     = feats[start:end]
            c_b     = coords[start:end]

            f_exp   = f_b.unsqueeze(1).expand(-1, k_a, -1)
            c_exp   = c_b.unsqueeze(1).expand(-1, k_a, -1)

            # ── Sparse differentiable pooling ──────────────────────────────────────
            # Expand and flatten (same as before)...
            flat_anchor = anchor_nn.reshape(-1)
            flat_w      = local_A.reshape(-1)
            flat_f      = f_exp.reshape(-1, f_b.shape[-1])
            flat_c      = c_exp.reshape(-1, 3)

            # AMP FIX: Perform scatter accumulation in float32
            feat_num  = torch_scatter.scatter(
                (flat_f * flat_w.unsqueeze(-1)).float(),
                flat_anchor, dim=0, dim_size=n_sp, reduce="sum",
            )
            coord_num = torch_scatter.scatter(
                (flat_c * flat_w.unsqueeze(-1)).float(),
                flat_anchor, dim=0, dim_size=n_sp, reduce="sum",
            )
            # Find empty anchors before clamping
            w_den = torch_scatter.scatter(
                flat_w.float(), flat_anchor, dim=0, dim_size=n_sp, reduce="sum",
            )
            empty_mask = w_den < 1e-5
            w_den_safe = w_den.clamp_min(1e-5)

            # Convert back to original precision
            pooled_f = (feat_num  / w_den_safe.unsqueeze(-1)).to(f_b.dtype)
            pooled_c = (coord_num / w_den_safe.unsqueeze(-1)).to(c_b.dtype)

            #  Fallback to the anchor's original features/coords if empty
            if empty_mask.any():
                pooled_f[empty_mask] = f_b[top_idx[empty_mask]]
                pooled_c[empty_mask] = c_b[top_idx[empty_mask]]

            # Fill any bad superpoints with batch mean
            bad_feat  = ~torch.isfinite(pooled_f).all(dim=-1)
            bad_coord = ~torch.isfinite(pooled_c).all(dim=-1)
            if bad_feat.any():
                pooled_f[bad_feat]  = f_b.mean(0, keepdim=True).expand(bad_feat.sum(), -1)
            if bad_coord.any():
                pooled_c[bad_coord] = c_b.mean(0, keepdim=True).expand(bad_coord.sum(), -1)

            hard_assign = anchor_nn[
                torch.arange(n, device=coords.device), local_A.argmax(dim=-1)
            ]
            cluster_map[start:end] = global_sp_idx + hard_assign

            all_pooled_feat.append(pooled_f)
            all_pooled_coord.append(pooled_c)
            all_pooled_batch.append(
                torch.full((n_sp,), b, dtype=torch.long, device=coords.device)
            )
            all_local_A.append(local_A)
            all_anchor_nn.append(anchor_nn)
            all_v.append(v)
            all_idx.append(idx)
            all_w_ij.append(w_ij)
            all_coords.append(c_b)
            all_global_anchor_nn.append(global_sp_idx + anchor_nn)

            global_sp_idx += n_sp
            start = end

        pooled_feat  = torch.cat(all_pooled_feat,  dim=0)   # [M, C_out]
        pooled_coord = torch.cat(all_pooled_coord, dim=0)   # [M, 3]
        pooled_batch = torch.cat(all_pooled_batch, dim=0)   # [M]
        global_anchors = torch.cat(all_global_anchor, dim=0) # [M]
        M = pooled_feat.shape[0]

        _, counts  = torch.unique_consecutive(pooled_batch, return_counts=True)
        new_offset = torch.cumsum(counts, dim=0).int()

        # ── Inherit serialized codes from anchor points (no Morton needed) ─────────
        code    = point.serialized_code[:, global_anchors] >> (pooling_depth * 3)
        order   = torch.argsort(code, dim=1)
        n_orders = code.shape[0]
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(M, device=order.device)
                .unsqueeze(0).expand(n_orders, -1),
        )

        if self.shuffle_orders:
            perm    = torch.randperm(n_orders, device=code.device)
            code    = code[perm]
            order   = order[perm]
            inverse = inverse[perm]

        # ── Assemble output Point ──────────────────────────────────────────────────
        point_dict = Dict(
            feat=pooled_feat,
            coord=pooled_coord,
            grid_coord=point.grid_coord[global_anchors] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=pooled_batch,
            offset=new_offset,
        )

        if "condition" in point.keys(): point_dict["condition"] = point.condition
        if "context"   in point.keys(): point_dict["context"]   = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster_map
            point_dict["pooling_parent"]  = point
            point_dict["pooling_local_A"] = torch.cat(all_local_A, dim=0)
            point_dict["pooling_anchor_nn"] = torch.cat(all_global_anchor_nn, dim=0)

        # ── SPT energy loss ────────────────────────────────────────────────────────
        if self.training:
            total_loss = torch.tensor(0.0, device=coords.device)
            num_batches = len(all_local_A) # Number of items in this batch
            
            for local_A, anchor_nn, v, idx, w_ij, c in zip(
                all_local_A, all_anchor_nn, all_v, all_idx, all_w_ij, all_coords
            ):
                loss, _ = self.operator.energy_loss(
                    local_A, anchor_nn, v, idx, w_ij,
                    lam=self.spt_lambda, beta=self.spt_beta,
                )
                total_loss = total_loss + (loss / num_batches) # -> CHANGE: Average across batch items

            if not torch.isfinite(total_loss):
                total_loss = torch.tensor(0.0, device=coords.device, requires_grad=False)

            existing = point.get(
                "no_pooling_loss",
                torch.tensor(0.0, device=coords.device)
            )
            point_dict["no_pooling_loss"] = existing + total_loss
        else:
            point_dict["no_pooling_loss"] = point.get(
                "no_pooling_loss",
                torch.tensor(0.0, device=coords.device)
            )

        new_point = Point(point_dict)
        if self.norm is not None:
            new_point.feat = self.norm(new_point.feat)
        if self.act is not None:
            new_point.feat = self.act(new_point.feat)

        new_point.sparsify()
        return new_point

class NOSuperpointUnpooling(PointModule):
    """
    Symmetric unpooling for NOSuperpointPooling.
    Reconstructs fine-grained features by applying the exact soft 
    assignment weights (local_A) that were calculated during pooling.
    """
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            act_layer() if act_layer is not None else nn.Identity(),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            act_layer() if act_layer is not None else nn.Identity(),
        )

    def forward(self, point: Point):
        parent = point.pop("pooling_parent")
        
        # Retrieve the soft assignments saved during the encoder phase
        local_A = point.pop("pooling_local_A", None)
        anchor_nn = point.pop("pooling_anchor_nn", None)
        
        if local_A is not None and anchor_nn is not None:
            # ───> OOM FIX: Incremental Weighted Sum <───
            Fine_N, k_anchor = local_A.shape
            Channels = point.feat.shape[-1]
            
            # Pre-allocate output tensor [Fine_N, Channels]
            up_feat = torch.zeros((Fine_N, Channels), dtype=point.feat.dtype, device=point.feat.device)
            
            # Detach local_A so PyTorch does NOT save gigabytes of data for the backward pass
            local_A_detached = local_A.detach()
            
            # Loop incrementally over the k anchors (very fast, uses virtually 0 VRAM)
            for k in range(k_anchor):
                w = local_A_detached[:, k:k+1]      # [Fine_N, 1]
                idx = anchor_nn[:, k]               # [Fine_N]
                
                # Multiply and accumulate in-place
                up_feat += point.feat[idx] * w
        else:
            # Fallback to hard unpooling if soft weights are missing
            inverse = point.pop("pooling_inverse")
            up_feat = point.feat[inverse]

        # Combine upsampled features with the skip connection from the encoder
        parent.feat = self.proj(up_feat) + self.proj_skip(parent.feat)
        
        # SpConv modules rely on `sparse_conv_feat` matching `feat` perfectly.
        if "sparse_conv_feat" in parent.keys():
            parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)
            
        return parent


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
        # 1. Replace Pooling in Encoder
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
                    
        # 2. Replace Unpooling in Decoder
        for _, stage in self.dec._modules.items():
            for name, layer in stage._modules.items():
                if isinstance(layer, SerializedUnpooling):
                    # Dynamically extract the channel dimensions from the original PTv3 layer
                    in_channels = layer.proj[0].in_features
                    skip_channels = layer.proj_skip[0].in_features
                    out_channels = layer.proj[0].out_features
                    
                    stage._modules[name] = NOSuperpointUnpooling(
                        in_channels=in_channels,
                        skip_channels=skip_channels,
                        out_channels=out_channels,
                        norm_layer=nn.LayerNorm,
                        act_layer=nn.GELU,
                    )

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

        # Guard backbone output
        if not torch.isfinite(point.feat).all():
            point.feat = torch.nan_to_num(point.feat, nan=0.0, posinf=1e4, neginf=-1e4)

        seg_logits = self.seg_head(point.feat)

        # Guard logits before criterion
        if not torch.isfinite(seg_logits).all():
            seg_logits = torch.nan_to_num(seg_logits, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.training:
            target = data_dict["segment"]
            if target.dtype != torch.int64:
                target = target.long()

            valid_mask = (target >= 0) & (target < self.num_classes)
            if not valid_mask.all():
                target = target.clone()
                target[~valid_mask] = -1

            seg_loss = self.criteria(seg_logits, target)

            # Hard stop on bad seg loss
            if not torch.isfinite(seg_loss):
                zero = seg_logits.sum() * 0.0
                return dict(
                    seg_loss=zero.detach(),
                    no_pooling_loss=point.get(
                        "no_pooling_loss",
                        torch.tensor(0.0, device=seg_logits.device)
                    ).detach(),
                    loss=zero
                )

            return_dict = dict(seg_loss=seg_loss)
            total_loss = seg_loss

            if "no_pooling_loss" in point:
                no_pooling_loss = point["no_pooling_loss"]
                if not torch.isfinite(no_pooling_loss):
                    no_pooling_loss = torch.tensor(0.0, device=seg_logits.device)
                return_dict["no_pooling_loss"] = no_pooling_loss
                total_loss = total_loss + no_pooling_loss

            if not torch.isfinite(total_loss):
                total_loss = seg_logits.sum() * 0.0

            return_dict["loss"] = total_loss
            return return_dict

        elif "segment" in data_dict:
            target = data_dict["segment"]
            if target.dtype != torch.int64:
                target = target.long()
            valid_mask = (target >= 0) & (target < self.num_classes)
            if not valid_mask.all():
                target = target.clone()
                target[~valid_mask] = -1

            loss = self.criteria(seg_logits, target)
            if not torch.isfinite(loss):
                loss = seg_logits.sum() * 0.0
            return dict(loss=loss, seg_logits=seg_logits)

        return dict(seg_logits=seg_logits)

