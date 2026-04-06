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

# ─────────────────────────────────────────────────────────────────────────────
# 1. Fast Point Transformer (FPT) Local Attention Block
# ─────────────────────────────────────────────────────────────────────────────
class FPTLightweightAttentionBlock(PointModule):
    def __init__(self, channels, num_heads=4, k=16, use_normal=True):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.use_normal = use_normal

        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels)

        # Upgrade Positional Encoding to accept 6D (Rel Pos + Rel Normal)
        pos_dim = 6 if use_normal else 3
        self.pos_enc = nn.Sequential(
            nn.Linear(pos_dim, channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels, bias=False)
        )

        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        
    def forward(self, point: Point):
        coords = point.coord
        feats = point.feat
        offset = point.offset
        
        N = coords.shape[0]
        k = min(self.k, N - 1) 
        if k <= 0: return point

        idx = pointops.knn_query(k, coords.contiguous(), offset)[0].long()
        idx = torch.clamp(idx, 0, N - 1)

        h = self.norm1(feats)
        qkv = self.qkv(h).view(N, 3, self.num_heads, self.head_dim)
        q, k_v, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        k_v_gathered = k_v[idx] 
        v_gathered = v[idx]     

        # ─── Geometric Encoding with Normals ───
        rel_pos = coords.unsqueeze(1) - coords[idx] # [N, K, 3]
        
        if self.use_normal:
            if "normal" not in point.keys() or point.normal is None:
                raise ValueError(
                    "FPTLightweightAttentionBlock expects 'normal' in the Point dictionary "
                    "because use_normal=True, but 'normal' was not found. "
                    "Please add 'normal' to your dataset config's 'Collect' feat_keys."
                )
            normals = point.normal
            rel_normal = normals.unsqueeze(1) - normals[idx] # [N, K, 3]
            pos_feat = torch.cat([rel_pos, rel_normal], dim=-1) # [N, K, 6]
        else:
            pos_feat = rel_pos

        pos_emb = self.pos_enc(pos_feat).view(N, k, self.num_heads, self.head_dim)

        # 4. Lightweight Self-Attention
        k_v_gathered = k_v_gathered + pos_emb
        
        # ─── MEMORY OPTIMIZED ATTENTION ───
        # OLD: attn = (q.unsqueeze(1) * k_v_gathered).sum(dim=-1) / math.sqrt(self.head_dim)
        # NEW: einsum matches Query [N, Heads, HeadDim] with Keys [N, K, Heads, HeadDim]
        attn = torch.einsum('nhd,nkhd->nkh', q, k_v_gathered) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=1) 

        # OLD: out = (attn.unsqueeze(-1) * v_gathered).sum(dim=1)
        # NEW: einsum matches Attn [N, K, Heads] with Values [N, K, Heads, HeadDim]
        out = torch.einsum('nkh,nkhd->nhd', attn, v_gathered)
        out = out.view(N, self.channels)


        feats = feats + self.proj(out)
        feats = feats + self.ffn(self.norm2(feats))

        point.feat = feats
        return point

# ─────────────────────────────────────────────────────────────────────────────
# 2. Superpoint Neural Operator (Unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class SuperpointNeuralOperator(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, k=16, T=3, k_anchor=8):
        super().__init__()
        self.k = k  
        self.T = T
        self.k_anchor = k_anchor  

        self.lift = nn.Linear(in_channels, hidden_channels) # Now dynamically larger
        self.green_kernel = nn.Sequential(
            nn.Linear(3 + hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.W = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(T)])
        self.assign_key = nn.Linear(hidden_channels, hidden_channels)
        self.assign_query = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, point, n_sp):
        coords = point.coord
        N = coords.shape[0]
        k = min(self.k, N - 1)

        idx = pointops.knn_query(k, coords.contiguous(), point.offset)[0].long()
        idx = torch.clamp(idx, 0, N - 1)
        rel_pos = coords[idx] - coords.unsqueeze(1)

        # ─── Concatenate Coords + Normals + Features ───
        feat_list = [coords]
        if "normal" in point.keys():
            feat_list.append(point.normal)
        if point.feat is not None:
            feat_list.append(point.feat.detach())
            
        inp = torch.cat(feat_list, dim=1)
        v = self.lift(inp)

        for t in range(self.T):
            v_j = v[idx]
            v_i = v.unsqueeze(1).expand(-1, k, -1)
            G   = self.green_kernel(torch.cat([rel_pos, v_i, v_j], dim=-1))
            integral = (G * v_j).mean(dim=1)
            v = self.norms[t](torch.relu(integral + self.W(v)))

        if self.training:
            v_j  = v[idx]
            v_i  = v.unsqueeze(1).expand(-1, k, -1)
            w_ij = self.green_kernel(
                       torch.cat([rel_pos, v_i, v_j], dim=-1)
                   ).squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
        else:
            w_ij = None

        scores = v.norm(dim=-1)                        
        _, top_idx = torch.topk(scores, n_sp, dim=0)       

        k_anchor = min(self.k_anchor, n_sp)
        anchor_nn = pointops.knn_query(
            k_anchor, coords[top_idx].contiguous(), 
            torch.tensor([n_sp], device=coords.device, dtype=torch.int),
            coords.contiguous(), point.offset,
        )[0].long()
        anchor_nn = torch.clamp(anchor_nn, 0, n_sp - 1)

        keys = self.assign_key(v)                       
        queries = self.assign_query(v[top_idx])            

        scale = math.pow(v.shape[-1], 0.25)
        local_logits = ((keys / scale).unsqueeze(1) * (queries / scale)[anchor_nn]).sum(-1) 
        local_A = torch.softmax(local_logits, dim=-1)      

        return local_A, anchor_nn, top_idx, v, idx, w_ij

    def energy_loss(self, local_A, anchor_nn, v, idx, w_ij, lam=1.0, beta=0.1):
        v_target = v.detach()
        anchor_v = v_target[anchor_nn]                      
        v_reconstruct = (local_A.unsqueeze(-1) * anchor_v).sum(1)
        data_fidelity = F.mse_loss(v_reconstruct.float(), v_target.float())

        vr_j = v_reconstruct[idx]                            
        vr_i = v_reconstruct.unsqueeze(1).expand_as(vr_j)
        diff_sq = ((vr_i.float() - vr_j.float()) ** 2).sum(-1)                  
        boundary = (w_ij.float().clamp(0, 1) * diff_sq.clamp(max=1e4)).mean()

        local_A_safe = local_A.float().clamp(min=1e-6, max=1.0)
        entropy = -(local_A_safe * local_A_safe.log()).sum(dim=-1).mean()

        loss = data_fidelity + lam * boundary + beta * entropy
        if not torch.isfinite(loss): loss = torch.tensor(0.0, device=v.device, requires_grad=True)
        return loss

# ─────────────────────────────────────────────────────────────────────────────
# 3. Clean NOSuperpointPooling (No PTv3 Serialization!)
# ─────────────────────────────────────────────────────────────────────────────
class NOSuperpointPooling(PointModule):
    def __init__(self, in_channels, out_channels, stride=2, k_anchor=8, use_normal=True):
        super().__init__()
        self.stride = stride
        self.use_normal = use_normal
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )
        
        # Calculate input size for the Operator (Coords=3 + Normal=3 + Feats=in_channels)
        op_in_channels = in_channels + 6 if use_normal else in_channels + 3
        self.operator = SuperpointNeuralOperator(
            in_channels=op_in_channels, hidden_channels=64, k=16, k_anchor=k_anchor
        )

    def forward(self, point: Point):
        coords, feats, offset = point.coord, self.proj(point.feat), point.offset
        normals = point.get("normal", None)

        all_pooled_feat, all_pooled_coord, all_pooled_batch = [], [], []
        all_pooled_normal = [] # Track pooled normals
        all_local_A, all_anchor_nn = [], []
        
        cluster_map = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        start, global_sp_idx = 0, 0
        total_loss = torch.tensor(0.0, device=coords.device)

        for b, end in enumerate(offset.tolist()):
            end = int(end)
            n = end - start
            n_sp = max(1, math.ceil(n / self.stride))

            # ─── Build Sub-Point Cloud with Normals ───
            sub_point_dict = dict(
                coord=coords[start:end], 
                feat=point.feat[start:end],
                offset=torch.tensor([n], device=coords.device, dtype=torch.int)
            )
            if normals is not None:
                sub_point_dict["normal"] = normals[start:end]
                
            sub_point = Point(Dict(sub_point_dict))

            local_A, anchor_nn, _, v, idx, w_ij = self.operator(sub_point, n_sp)

            if self.training:
                total_loss += self.operator.energy_loss(local_A, anchor_nn, v, idx, w_ij)

            # Pooling math
            flat_anchor = anchor_nn.reshape(-1)
            flat_w = local_A.reshape(-1)
            f_exp = feats[start:end].unsqueeze(1).expand(-1, local_A.shape[1], -1).reshape(-1, feats.shape[-1])
            c_exp = coords[start:end].unsqueeze(1).expand(-1, local_A.shape[1], -1).reshape(-1, 3)

            feat_num = torch_scatter.scatter((f_exp * flat_w.unsqueeze(-1)).float(), flat_anchor, dim=0, dim_size=n_sp, reduce="sum")
            coord_num = torch_scatter.scatter((c_exp * flat_w.unsqueeze(-1)).float(), flat_anchor, dim=0, dim_size=n_sp, reduce="sum")
            w_den = torch_scatter.scatter(flat_w.float(), flat_anchor, dim=0, dim_size=n_sp, reduce="sum").clamp_min(1e-3)

            pooled_f = (feat_num / w_den.unsqueeze(-1)).to(feats.dtype)
            pooled_c = (coord_num / w_den.unsqueeze(-1)).to(coords.dtype)

            # ─── Pool and Normalize the Normal Vectors ───
            if normals is not None:
                n_exp = normals[start:end].unsqueeze(1).expand(-1, local_A.shape[1], -1).reshape(-1, 3)
                normal_num = torch_scatter.scatter((n_exp * flat_w.unsqueeze(-1)).float(), flat_anchor, dim=0, dim_size=n_sp, reduce="sum")
                pooled_n = (normal_num / w_den.unsqueeze(-1)).to(normals.dtype)
                pooled_n = F.normalize(pooled_n, p=2, dim=-1) # Ensure they remain valid unit vectors
                all_pooled_normal.append(pooled_n)

            cluster_map[start:end] = global_sp_idx + anchor_nn[torch.arange(n), local_A.argmax(dim=-1)]
            
            all_pooled_feat.append(pooled_f)
            all_pooled_coord.append(pooled_c)
            all_pooled_batch.append(torch.full((n_sp,), b, dtype=torch.long, device=coords.device))
            all_local_A.append(local_A)
            all_anchor_nn.append(global_sp_idx + anchor_nn) 

            global_sp_idx += n_sp
            start = end

        pooled_batch = torch.cat(all_pooled_batch, dim=0)
        _, counts = torch.unique_consecutive(pooled_batch, return_counts=True)
        
        # ─── Assemble the New Point ───
        new_point_dict = dict(
            feat=torch.cat(all_pooled_feat, dim=0),
            coord=torch.cat(all_pooled_coord, dim=0),
            batch=pooled_batch,
            offset=torch.cumsum(counts, dim=0).int(),
        )
        if all_pooled_normal:
            new_point_dict["normal"] = torch.cat(all_pooled_normal, dim=0)

        new_point = Point(Dict(new_point_dict))
        
        point.pooling_local_A = torch.cat(all_local_A, dim=0)
        point.pooling_anchor_nn = torch.cat(all_anchor_nn, dim=0)
        point.pooling_inverse = cluster_map
        
        if self.training:
            new_point.no_pooling_loss = point.get("no_pooling_loss", 0.0) + total_loss

        return new_point

# ─────────────────────────────────────────────────────────────────────────────
# 4. Clean NOSuperpointUnpooling
# ─────────────────────────────────────────────────────────────────────────────
class NOSuperpointUnpooling(PointModule):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels), nn.LayerNorm(out_channels), nn.GELU())
        self.proj_skip = nn.Sequential(nn.Linear(skip_channels, out_channels), nn.LayerNorm(out_channels), nn.GELU())

    def forward(self, enc_point: Point, dec_point: Point):
        """
        enc_point: The fine-grained point cloud saved during the Encoder forward pass.
        dec_point: The coarse point cloud coming up from the Decoder.
        """
        if hasattr(enc_point, "pooling_local_A") and hasattr(enc_point, "pooling_anchor_nn"):
            # Scatter coarse features back to fine points using the soft weights
            feat_anchors = dec_point.feat[enc_point.pooling_anchor_nn] # [N_fine, k_anchor, C]
            up_feat = (feat_anchors * enc_point.pooling_local_A.unsqueeze(-1)).sum(dim=1) 
        else:
            # Fallback to hard nearest neighbor
            up_feat = dec_point.feat[enc_point.pooling_inverse]

        # Combine upsampled features with the skip connection
        enc_point.feat = self.proj(up_feat) + self.proj_skip(enc_point.feat)
        return enc_point

# ─────────────────────────────────────────────────────────────────────────────
# 5. Standalone OPTNet Backbone (Pure U-Net Structure)
# ─────────────────────────────────────────────────────────────────────────────
@MODELS.register_module("OPTNet")
class OPTNet(nn.Module):
    def __init__(
        self,
        in_channels=6,
        base_channels=32,
        enc_depths=(2, 2, 2, 6, 2),
        dec_depths=(2, 2, 2, 2),
        fpt_k=12,
        ordering_loss_weight=1.0,
        **kwargs
    ):
        super().__init__()
        self.no_pooling_loss_weight = ordering_loss_weight
        
        # Channels at each U-Net stage
        channels = [base_channels * (2 ** i) for i in range(len(enc_depths))]

        # Initial Embedding
        self.embedding = nn.Sequential(
            nn.Linear(in_channels, channels[0]),
            nn.LayerNorm(channels[0]),
            nn.GELU()
        )

        # ─── ENCODER ───
        self.enc_blocks = nn.ModuleList()
        self.enc_pools = nn.ModuleList()
        
        for i in range(len(enc_depths)):
            # FPT blocks for this stage
            stage_blocks = nn.Sequential(*[
                FPTLightweightAttentionBlock(channels[i], num_heads=max(2, channels[i]//32), k=fpt_k)
                for _ in range(enc_depths[i])
            ])
            self.enc_blocks.append(stage_blocks)
            
            # Pooling (except for the very last bottleneck stage)
            if i < len(enc_depths) - 1:
                self.enc_pools.append(NOSuperpointPooling(channels[i], channels[i+1], stride=2))

        # ─── DECODER ───
        self.dec_blocks = nn.ModuleList()
        self.dec_unpools = nn.ModuleList()
        
        for i in range(len(dec_depths) - 1, -1, -1):
            # Unpooling: takes input from decoder stage i+1 and skip connection from encoder stage i
            self.dec_unpools.append(NOSuperpointUnpooling(
                in_channels=channels[i+1], skip_channels=channels[i], out_channels=channels[i]
            ))
            
            # FPT blocks for this stage
            stage_blocks = nn.Sequential(*[
                FPTLightweightAttentionBlock(channels[i], num_heads=max(2, channels[i]//32), k=fpt_k)
                for _ in range(dec_depths[i])
            ])
            self.dec_blocks.append(stage_blocks)

    def forward(self, data_dict):
        point = Point(data_dict)
        point.feat = self.embedding(point.feat)

        if self.training:
            point.no_pooling_loss = torch.tensor(0.0, device=point.coord.device)

        # Encoder pass
        skip_points = []
        for i in range(len(self.enc_blocks)):
            point = self.enc_blocks[i](point)
            skip_points.append(point) # Save for skip connections
            
            if i < len(self.enc_pools):
                point = self.enc_pools[i](point)

        # Decoder pass
        for i in range(len(self.dec_unpools)):
            # Pop the corresponding skip connection (reverse order)
            skip_point = skip_points[-(i + 2)] 
            
            # Unpool combines coarse 'point' with fine 'skip_point'
            point = self.dec_unpools[i](skip_point, point) 
            point = self.dec_blocks[i](point)

        if self.training and hasattr(point, "no_pooling_loss"):
            point.no_pooling_loss = point.no_pooling_loss * self.no_pooling_loss_weight

        return point

# ─────────────────────────────────────────────────────────────────────────────
# 6. Segmentor Wrapper (Unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@MODELS.register_module("OPTNetSegmentor")
class OPTNetSegmentor(nn.Module):
    def __init__(self, backbone, criteria, num_classes, backbone_out_channels=32):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        self.criteria = build_criteria(criteria)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, data_dict):
        point = self.backbone(data_dict)
        seg_logits = self.seg_head(point.feat)

        if self.training:
            target = data_dict["segment"].long()
            valid_mask = (target >= 0) & (target < self.num_classes)
            target[~valid_mask] = -1

            seg_loss = self.criteria(seg_logits, target)
            total_loss = seg_loss

            return_dict = dict(seg_loss=seg_loss)
            if hasattr(point, "no_pooling_loss"):
                return_dict["no_pooling_loss"] = point.no_pooling_loss
                total_loss += point.no_pooling_loss

            return_dict["loss"] = total_loss
            return return_dict

        loss = self.criteria(seg_logits, data_dict["segment"].long()) if "segment" in data_dict else 0.0
        return dict(loss=loss, seg_logits=seg_logits)