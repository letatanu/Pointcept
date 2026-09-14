"""PTv3 encoder with a genuine Haar wavelet-domain context branch.

Companion files: haar_wno.py and neighbor_quatrpe.py in the same directory.
Registers new model names, so the original implementation can coexist.
This model is NOT structurally SO(3)-equivariant. Retrain after these changes.
The head has multiscale skip features but no transformer decoder blocks.
"""
import torch
import torch.nn as nn
import torch_scatter
from addict import Dict
from functools import partial
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point
from pointcept.models.losses import build_criteria
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import Block, Embedding
from .haar_wno import WNO3dBlock, NOGlobalBranch, QuatRPE, apply_quat_rotation_to_features

class GridPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride        = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce         = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable      = traceable

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
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.empty_like(point.coord, dtype=torch.long)
            for scene in torch.unique(point.batch):
                idx = torch.nonzero(point.batch == scene, as_tuple=False).flatten()
                coords = point.coord[idx]
                grid_coord[idx] = torch.div(
                    coords - coords.amin(dim=0), point.grid_size,
                    rounding_mode="floor",
                ).long()
        else:
            raise AssertionError(
                "[grid_coord] or [coord, grid_size] should be in the Point"
            )
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="floor").long()
        # Explicit batch column avoids bit-packing assumptions and collisions.
        keys = torch.cat((point.batch[:, None].long(), grid_coord), dim=1)
        keys, cluster, counts = torch.unique(
            keys, sorted=True, return_inverse=True, return_counts=True, dim=0,
        )
        grid_coord = keys[:, 1:]
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
        )
        for key in ("origin_coord", "condition", "context", "name", "split"):
            if key in point.keys():
                point_dict[key] = (
                    torch_scatter.segment_csr(point[key][indices], idx_ptr, reduce="mean")
                    if key == "origin_coord" else point[key]
                )
        if "color" in point.keys():
            point_dict["color"] = torch_scatter.segment_csr(
                point.color[indices], idx_ptr, reduce="mean"
            )
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"]  = point
            point_dict["idx_ptr"]         = idx_ptr

        order = point.order
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.serialization(order=order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        return point

class QuatRPEBlockWrapper(PointModule):
    def __init__(self, block: nn.Module, channels: int, **quatrpe_kwargs):
        super().__init__()
        self.block = block
        self.rpe = QuatRPE(channels, num_freqs=8, **quatrpe_kwargs)
        
    def forward(self, point: Point) -> Point:
        # Inject geometry directly into features before attention
        pos_enc = self.rpe(point).to(point.feat.dtype)
        point.feat = point.feat + pos_enc
        
        # Update sparse feature cache if it exists
        if hasattr(point, "sparse_conv_feat") and point.sparse_conv_feat is not None:
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
            
        return self.block(point)

class NOFusedGridPooling(PointModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm_layer=None,
        act_layer=None,
        enable_no: bool = False,
        fusion: str = "add",
        reduce: str = "max",
        shuffle_orders: bool = True,
        universal_no_branch: nn.Module = None,
        universal_dim: int = 64,
    ):
        super().__init__()
        self.pool = GridPooling(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            norm_layer=norm_layer,
            act_layer=act_layer,
            reduce=reduce,
            shuffle_orders=shuffle_orders,
            traceable=True,
        )
        self.enable_no  = enable_no
        if fusion not in ("add", "concat", "gated_concat"):
            raise ValueError("fusion must be add, concat, or gated_concat")
        self.fusion = fusion

        if enable_no:
            assert universal_no_branch is not None, \
                "universal_no_branch must be provided when enable_no=True"
            self.no_branch = universal_no_branch

            # Per-layer lightweight adapters (cheap)
            self.down_proj = nn.Linear(in_channels, universal_dim)
            self.up_proj   = nn.Linear(universal_dim, out_channels)

            self.gate = (nn.Parameter(torch.full((1,), -4.0))
                         if fusion in ("add", "gated_concat") else None)
            self.proj_concat = (nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.LayerNorm(out_channels),
            ) if fusion in ("concat", "gated_concat") else None)

    def forward(self, point: Point) -> Point:
        if self.enable_no:
            # 1. Project input to universal dim (N_in, universal_dim)
            feat_down = self.down_proj(point.feat)
            
            # 2. Extract Global Context via WNO (N_in, universal_dim)
            global_feat = self.no_branch(feat_down, point)

        # 3. Downsample the backbone point cloud (N_in -> N_out)
        point = self.pool(point)

        if self.enable_no:
            inv = point.pooling_inverse
            
            # 4. OPTIMIZATION: Pool the narrow universal_dim FIRST (N_out, universal_dim)
            global_feat_coarse = torch_scatter.scatter_max(global_feat, inv, dim=0)[0]
            
            # 5. Project to out_channels on the much smaller N_out point set!
            feat_no_coarse = self.up_proj(global_feat_coarse)

            if self.fusion == "add":
                point.feat = point.feat + self.gate.sigmoid() * feat_no_coarse
            else:
                context = (self.gate.sigmoid() * feat_no_coarse
                           if self.gate is not None else feat_no_coarse)
                point.feat = self.proj_concat(torch.cat((point.feat, context), dim=-1))

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

class NOLightweightUpsampleHead(PointModule):
    def __init__(
        self,
        stage_channels,          # list: enc_channels, e.g. [32, 64, 128, 256, 512]
        out_channels: int = 64,
        fusion: str = "sum",     # "sum" | "concat"
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()
        assert fusion in ("sum", "concat")
        self.num_stages = len(stage_channels)
        self.fusion = fusion

        # Lightweight per-stage projections
        self.stage_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, out_channels),
                norm_layer(out_channels),
            )
            for c in stage_channels
        ])

        # Learnable blend weights (softmax-normalized at runtime)
        self.stage_weights = nn.Parameter(
            torch.full((self.num_stages,), 1.0 / self.num_stages)
        )

        in_ch = self.num_stages * out_channels if fusion == "concat" else out_channels
        self.output_proj = nn.Sequential(
            nn.Linear(in_ch, out_channels),
            norm_layer(out_channels),
            act_layer(),
        )

    def _upsample_to_stage0(self, feat, stage_points, s):
        """Walk pooling_inverse chain from stage s back to stage 0."""
        for t in range(s, 0, -1):
            inv = stage_points[t].pooling_inverse
            feat = feat[inv]
        return feat

    def forward(self, stage_points):
        weights = torch.softmax(self.stage_weights, dim=0)  # (num_stages,)
        upsampled = []
        for s in range(self.num_stages):
            feat_s = self.stage_proj[s](stage_points[s].feat)   # (N_s, out_ch)
            if s > 0:
                feat_s = self._upsample_to_stage0(feat_s, stage_points, s)
            upsampled.append(weights[s] * feat_s)

        fused = (
            torch.stack(upsampled, dim=0).sum(dim=0)
            if self.fusion == "sum"
            else torch.cat(upsampled, dim=-1)
        )
        fused = self.output_proj(fused)

        point = stage_points[0]
        point.feat = fused
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(fused)
        return point

@MODELS.register_module("PT-HaarWNO-EncoderOnly")
class PointTransformerV3_HaarWNO_EncoderOnly(PointModule):
    def __init__(
        self,
        in_channels: int = 6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio: int = 4,
        drop_path: float = 0.3,
        pre_norm: bool = True,
        shuffle_orders: bool = True,
        enable_flash: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        # ---- NO parameters ----
        no_stages=(True, True, True, True),   # one per transition (num_stages - 1)
        fno_modes: int = 8,  # Legacy config compatibility; unused by Haar WNO
        base_grid_size: tuple = (64, 64, 64),
        fusion: str = "concat",               # pooling NO fusion: "add" | "concat"
        share_no_branch: bool = True,
        universal_dim: int = 64,
        NO_type: str = "WNO",  # Only WNO is supported in this revision
        wavelet_levels: int = 3,
        wavelet_global_rank: int = 16,
        wavelet_global_mixing: bool = True,
        use_quatrpe: bool = True,  # Disables BOTH positional injection sites
        quatrpe_k: int = 16,
        quatrpe_backend: str = "auto",
        quatrpe_chunk_size: int = 512,
        quatrpe_edge_chunk_size: int = 2048,
        quatrpe_distance_scale: float = 1.0,
        pool_reduce: str = "max",
        # ---- Upsample head parameters ----
        head_out_channels: int = 64,
        head_fusion: str = "sum",             # head fusion: "sum" | "concat"
    ):
        super().__init__()
        self.num_stages     = len(enc_depths)
        self.order          = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders

        bn_layer  = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer  = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(in_channels, enc_channels[0], bn_layer, act_layer)

        if NO_type != "WNO":
            raise ValueError("This implementation supports only explicit Haar WNO")
        if len(stride) != self.num_stages - 1 or len(no_stages) != self.num_stages - 1:
            raise ValueError("stride and no_stages must match the number of transitions")
        if any(len(v) != self.num_stages for v in (enc_channels, enc_num_head, enc_patch_size)):
            raise ValueError("Encoder parameter lengths must match enc_depths")
        quatrpe_kwargs = dict(
            k=quatrpe_k, backend=quatrpe_backend, chunk_size=quatrpe_chunk_size,
            edge_chunk_size=quatrpe_edge_chunk_size, distance_scale=quatrpe_distance_scale,
        )
        def make_branch():
            return NOGlobalBranch(
                channels=universal_dim, grid_size=base_grid_size, norm_layer=ln_layer,
                levels=wavelet_levels, global_rank=wavelet_global_rank,
                global_mixing=wavelet_global_mixing, use_quatrpe=use_quatrpe,
                quatrpe_k=quatrpe_k, quatrpe_backend=quatrpe_backend,
                quatrpe_chunk_size=quatrpe_chunk_size,
                quatrpe_edge_chunk_size=quatrpe_edge_chunk_size,
                quatrpe_distance_scale=quatrpe_distance_scale,
            )
        self.universal_no_branch = make_branch() if share_no_branch and any(no_stages) else None

        # ------------------------------------------------------------------ #
        # Encoder stages
        # ------------------------------------------------------------------ #
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]

        self.enc_stages = nn.ModuleList()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]): sum(enc_depths[:s + 1])]
            enc = PointSequential()

            if s > 0:
                enc.add(
                    NOFusedGridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        enable_no=no_stages[s - 1],
                        fusion=fusion,
                        reduce=pool_reduce,
                        shuffle_orders=shuffle_orders,
                        universal_no_branch=(self.universal_no_branch if share_no_branch else
                                             make_branch() if no_stages[s - 1] else None), # type: ignore
                        universal_dim=universal_dim,
                    ),
                    name="down",
                )

            for i in range(enc_depths[s]):
                # Create the standard block
                base_block = Block(
                    channels=enc_channels[s],
                    num_heads=enc_num_head[s],
                    patch_size=enc_patch_size[s],
                    mlp_ratio=mlp_ratio,
                    drop_path=enc_drop_path_[i],
                    norm_layer=ln_layer,
                    act_layer=act_layer,
                    pre_norm=pre_norm,
                    order_index=i % len(self.order),
                    cpe_indice_key=f"stage{s}",
                    enable_flash=enable_flash,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                )
                
                # Optional quaternion positional encoding (not an SO(3) guarantee)
                enc.add(
                    QuatRPEBlockWrapper(base_block, enc_channels[s], **quatrpe_kwargs) if use_quatrpe else base_block,
                    name=f"block{i}",
                )


            self.enc_stages.append(enc)  

        # ------------------------------------------------------------------ #
        # Lightweight upsample head (replaces decoder)
        # ------------------------------------------------------------------ #
        self.head = NOLightweightUpsampleHead(
            stage_channels=enc_channels,
            out_channels=head_out_channels,
            fusion=head_fusion,
            norm_layer=ln_layer,
            act_layer=act_layer,
        )

    def forward(self, data_dict: dict) -> Point:
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)

        stage_points = []
        for stage in self.enc_stages:   
            point = stage(point)
            stage_points.append(point)

        return self.head(stage_points)
    
    
@MODELS.register_module()
class DefaultSegmentorHaarWNO(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.freeze_backbone:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)

        # Collect OPTNet-style RPE aux loss if available
        aux_rpe_loss = getattr(point, "aux_rpe_loss", None)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            if aux_rpe_loss is not None:
                loss = loss + 0.5 * aux_rpe_loss
            return_dict["loss"] = loss
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict