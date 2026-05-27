import torch
import torch.nn as nn
import torch_scatter
from functools import partial
from addict import Dict

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.point_prompt_training import PDNorm

# Import local PTv3 basic blocks to build our standalone architecture
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    Block, Embedding, SerializedPooling, SerializedUnpooling
)

class FNO3dBlock(nn.Module):
    """3D Fourier Neural Operator block for dense grids (AMP safe)."""
    def __init__(self, channels, modes=8):
        super().__init__()
        self.modes = modes
        scale = 1 / (channels * channels)
        
        # FIX: Store real and imaginary parts as standard float32/float16
        # instead of a single cfloat tensor to prevent AMP GradScaler crash.
        self.weight_real = nn.Parameter(
            scale * torch.rand(channels, channels, modes, modes, modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.rand(channels, channels, modes, modes, modes)
        )
        self.bypass = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        # FFT to spectral domain
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros_like(x_ft)
        
        # Bound modes to prevent indexing errors on very small spatial grids
        mx = min(self.modes, x_ft.shape[2])
        my = min(self.modes, x_ft.shape[3])
        mz = min(self.modes, x_ft.shape[4])
        
        # FIX: Reconstruct the complex weight tensor on the fly
        weight = torch.complex(self.weight_real, self.weight_imag)
        
        # Apply spectral weights
        out_ft[:, :, :mx, :my, :mz] = torch.einsum(
            "bcxyz,cdxyz->bdxyz",
            x_ft[:, :, :mx, :my, :mz],
            weight[:, :, :mx, :my, :mz]
        )
        
        # Inverse FFT back to spatial domain
        x_no = torch.fft.irfftn(out_ft, s=(X, Y, Z), dim=[-3, -2, -1])
        return (x_no + self.bypass(x)).contiguous()
    

class NOGlobalBranch(PointModule):
    """Maps serialized points to grid, applies FNO, maps back to points."""
    def __init__(self, channels, modes=8, norm_layer=nn.LayerNorm):
        super().__init__()
        self.fno = FNO3dBlock(channels, modes)
        self.norm = norm_layer(channels)

    def forward(self, point: Point):
        feat = point.feat
        coord = point.grid_coord
        
        # Shift coords to 0-based for dense grid bounds
        origin = coord.min(dim=0).values
        shifted = coord - origin
        grid_shape = shifted.max(dim=0).values + 1
        X, Y, Z = grid_shape.tolist()
        C = feat.shape[1]

        # Flatten indices for scatter
        flat_idx = shifted[:, 0] * (Y * Z) + shifted[:, 1] * Z + shifted[:, 2]
        grid_flat = torch.zeros(X * Y * Z, C, device=feat.device, dtype=feat.dtype)
        count = torch.zeros(X * Y * Z, 1, device=feat.device, dtype=feat.dtype)

        # Scatter to grid
        grid_flat.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, C), feat)
        # count.scatter_add_(0, flat_idx.unsqueeze(1), torch.ones_like(count))

        count.scatter_add_(0, flat_idx.unsqueeze(1), 
                           torch.ones(feat.shape[0], 1, device=feat.device, dtype=feat.dtype))
        
        grid_flat = grid_flat / count.clamp(min=1)
        grid = grid_flat.view(1, X, Y, Z, C).permute(0, 4, 1, 2, 3)

        # Apply global operator
        grid_out = self.fno(grid)

        # Exact index lookup back to points (no trilinear needed)
        feat_out = grid_out.squeeze(0).permute(1, 2, 3, 0).reshape(-1, C)[flat_idx]
        return self.norm(feat_out)

class NOFusedPooling(PointModule):
    def __init__(self, in_channels, out_channels, stride=2, modes=8, norm_layer=None, act_layer=None, enable_no=False, fusion="add"):
        super().__init__()
        self.pool = SerializedPooling(in_channels, out_channels, stride, norm_layer, act_layer)
        self.enable_no = enable_no
        self.fusion = fusion
        if enable_no:
            self.no_branch = NOGlobalBranch(in_channels, modes, norm_layer=norm_layer if norm_layer else nn.LayerNorm)
            self.proj_no = nn.Linear(in_channels, out_channels)

    def forward(self, point: Point):
        if self.enable_no:
            feat_global = self.proj_no(self.no_branch(point))

        point = self.pool(point)

        if self.enable_no:
            inv = point.pooling_inverse
            # Max pooling from fine NO features to coarse points
            feat_no_coarse = torch_scatter.scatter_max(feat_global, inv, dim=0)[0]
            if self.fusion == "add":
                point.feat = point.feat + feat_no_coarse
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

class NOFusedUnpooling(PointModule):
    def __init__(self, in_channels, skip_channels, out_channels, modes=8, norm_layer=None, act_layer=None, enable_no=False, use_skip=True, fusion="add"):
        super().__init__()
        self.unpool = SerializedUnpooling(in_channels, skip_channels, out_channels, norm_layer, act_layer)
        self.enable_no = enable_no
        self.use_skip = use_skip
        self.fusion = fusion
        if enable_no:
            self.no_branch = NOGlobalBranch(in_channels, modes, norm_layer=norm_layer if norm_layer else nn.LayerNorm)
            self.proj_no = nn.Linear(in_channels, out_channels)

    def forward(self, point: Point):
        if self.enable_no:
            feat_global = self.proj_no(self.no_branch(point))

        # Ablation: Zero out skip features before unpooling if testing pure NO
        if not self.use_skip:
            parent = point.pooling_parent
            parent.feat = torch.zeros_like(parent.feat)

        inverse = point.pooling_inverse  # Capture inverse before unpool consumes it
        point = self.unpool(point)       # Returns the FINE (parent) points

        if self.enable_no:
            feat_no_fine = feat_global[inverse]  # Broadcast coarse NO to fine
            if self.fusion == "add":
                point.feat = point.feat + feat_no_fine
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

@MODELS.register_module("PT-v3m1-NO")
class PointTransformerV3_NO(PointModule):
    """
    Standalone PTv3 architecture integrated with GINO-style Global Neural Operator blocks.
    """
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        # NO parameters
        no_stages=(False, False, True, True),  # Apply NO only at deeper stages to prevent OOM
        fno_modes=8,
        use_skip=True,
        fusion="add",
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders

        # Standard layers setup
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(in_channels, enc_channels[0], bn_layer, act_layer)
        
        # --- ENCODER ---
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    NOFusedPooling(
                        in_channels=enc_channels[s - 1], out_channels=enc_channels[s],
                        stride=stride[s - 1], modes=fno_modes, norm_layer=bn_layer, act_layer=act_layer,
                        enable_no=no_stages[s - 1], fusion=fusion
                    ), name="down"
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s], num_heads=enc_num_head[s], patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio, drop_path=enc_drop_path_[i], norm_layer=ln_layer, act_layer=act_layer,
                        pre_norm=pre_norm, order_index=i % len(self.order), cpe_indice_key=f"stage{s}",
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax
                    ), name=f"block{i}"
                )
            self.enc.add(module=enc, name=f"enc{s}")

        # --- DECODER ---
        dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
        self.dec = PointSequential()
        dec_channels_ = list(dec_channels) + [enc_channels[-1]]
        for s in reversed(range(self.num_stages - 1)):
            dec_drop_path_ = dec_drop_path[sum(dec_depths[:s]) : sum(dec_depths[: s + 1])][::-1]
            dec = PointSequential()
            dec.add(
                NOFusedUnpooling(
                    in_channels=dec_channels_[s + 1], skip_channels=enc_channels[s], out_channels=dec_channels_[s],
                    modes=fno_modes, norm_layer=bn_layer, act_layer=act_layer,
                    enable_no=no_stages[s], use_skip=use_skip, fusion=fusion
                ), name="up"
            )
            for i in range(dec_depths[s]):
                dec.add(
                    Block(
                        channels=dec_channels_[s], num_heads=dec_num_head[s], patch_size=dec_patch_size[s],
                        mlp_ratio=mlp_ratio, drop_path=dec_drop_path_[i], norm_layer=ln_layer, act_layer=act_layer,
                        pre_norm=pre_norm, order_index=i % len(self.order), cpe_indice_key=f"stage{s}",
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax
                    ), name=f"block{i}"
                )
            self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        return point