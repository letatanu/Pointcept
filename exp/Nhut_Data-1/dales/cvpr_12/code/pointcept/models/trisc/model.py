"""Convolution-free blocks with configurable local and cross attention order."""

from functools import partial
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .geometry import (
    batch_from_data,
    centered_coordinates,
    make_patches,
    make_voxels,
    make_grid_stride_voxels,
    group_mean,
)
from .layout import (
    compact_samples, cross_layout, grid_cross_tables, make_grid_cross_layout,
    sample_cross, sample_central_cross, sample_grid_rows,
)
from .attention import PackedAttention, ROTARY_MODES
from .coarse import CoarseGlobalContext, crop_bounds


ATTENTION_ORDERS = {
    "local_then_cross": ("local", "cross"),
    "cross_then_local": ("cross", "local"),
}


def mlp(cin, cout, ratio=2):
    return nn.Sequential(
        nn.LayerNorm(cin),
        nn.Linear(cin, cin * ratio),
        nn.GELU(),
        nn.Linear(cin * ratio, cout),
    )


class LearnedPooling(nn.Module):
    """Small scalar score per point; channel projection only AFTER pooling."""

    def __init__(self, cin, cout, score_hidden=16):
        super().__init__()
        self.norm = nn.LayerNorm(cin)
        self.score = nn.Sequential(
            nn.Linear(cin + 3, score_hidden), nn.GELU(), nn.Linear(score_hidden, 1)
        )
        self.project = mlp(cin, cout)

    def forward(self, feature, coord, mass, cells, size, return_weights=False):
        group, count = cells.inverse, len(cells.counts)
        with torch.no_grad():
            center, total_mass = group_mean(coord, group, count, mass)
        relative = ((coord - center[group]) / size).to(feature.dtype)
        score = (
            self.score(torch.cat((self.norm(feature), relative), -1))
            .squeeze(-1)
            .float()
        )
        maximum = score.new_full((count,), -torch.inf)
        maximum.scatter_reduce_(
            0, group, score.detach(), reduce="amax", include_self=True
        )
        exp = (score - maximum[group]).exp()
        denominator = exp.new_zeros(count).index_add(0, group, exp)
        weight = exp / denominator[group].clamp_min(1e-12)
        # Do not create a full FP32 N x Cin weighted tensor: accumulate per
        # feature chunk. Autograd still saves required backward information.
        pooled = feature.new_zeros((count, feature.shape[1]), dtype=torch.float32)
        for begin in range(0, len(feature), 65536):
            end = begin + 65536
            pooled.index_add_(
                0,
                group[begin:end],
                weight[begin:end, None] * feature[begin:end].float(),
            )
        output = self.project(pooled.to(feature.dtype))
        result = (output, center, total_mass, cells.indices[:, 0])
        return (*result, weight) if return_weights else result


class MeanPooling(nn.Module):
    """Equal current-token mean -> same post-pooling MLP as learned pooling.

    Coordinates/mass handling match LearnedPooling for a controlled ablation.
    This is not PTv3's projection-before-reduction feature rule.
    """
    def __init__(self, cin, cout):
        super().__init__()
        self.project = mlp(cin, cout)

    def forward(self, feature, coord, mass, cells, size, return_weights=False):
        group, count = cells.inverse, len(cells.counts)
        pooled, _ = group_mean(feature, group, count)
        with torch.no_grad():
            center, total_mass = group_mean(coord, group, count, mass)
        result = (self.project(pooled), center, total_mass, cells.indices[:, 0])
        if return_weights:
            weight = cells.counts[group].float().reciprocal()
            return (*result, weight)
        return result


class PTv3Pooling(nn.Module):
    """PTv3 feature rule: Linear -> segment reduction -> norm -> GELU.

    Reference: Pointcept point_transformer_v3m1_base.py SerializedPooling.
    Default reduce=max, BatchNorm1d(eps=1e-3, momentum=.01). Clusters come
    from the backbone's configured partition; this is not a Pointcept Point
    object or sparse-convolution adapter. No torch_scatter dependency.
    """
    def __init__(self, cin, cout, reduce="max", norm="batch"):
        super().__init__()
        if reduce not in ("mean", "max", "sum", "min"):
            raise ValueError("pooling_reduce must be mean, max, sum or min")
        if norm not in ("batch", "layer", "none"):
            raise ValueError("pooling_norm must be batch, layer or none")
        self.reduce = reduce
        self.proj = nn.Linear(cin, cout)
        self.norm = (nn.BatchNorm1d(cout, eps=1e-3, momentum=.01) if norm == "batch" else
                     nn.LayerNorm(cout) if norm == "layer" else nn.Identity())
        self.act = nn.GELU()

    def forward(self, feature, coord, mass, cells, size):
        group, count = cells.inverse, len(cells.counts)
        value = self.proj(feature)
        if self.reduce == "mean":
            pooled, _ = group_mean(value, group, count)
        elif self.reduce == "sum":
            pooled = value.new_zeros((count, value.shape[1]), dtype=torch.float32)
            pooled.index_add_(0, group, value.float())
            pooled = pooled.to(value.dtype)
        else:
            fill = -torch.inf if self.reduce == "max" else torch.inf
            pooled = value.new_full((count, value.shape[1]), fill, dtype=torch.float32)
            pooled.scatter_reduce_(0, group[:, None].expand_as(value), value.float(),
                                   reduce="amax" if self.reduce == "max" else "amin",
                                   include_self=True)
            pooled = pooled.to(value.dtype)
        with torch.no_grad():
            # Upstream averages CURRENT token coordinates, not original-point mass.
            center, _ = group_mean(coord, group, count)
            total_mass = mass.new_zeros(count).index_add_(0, group, mass)
        return self.act(self.norm(pooled)), center, total_mass, cells.indices[:, 0]


class Fusion(nn.Module):
    def __init__(self, fine, coarse):
        super().__init__()
        self.project = mlp(coarse, fine)
        self.gate = nn.Sequential(
            nn.LayerNorm(fine * 2), nn.Linear(fine * 2, fine), nn.Sigmoid()
        )
        self.refine = mlp(fine, fine)

    def forward(self, coarse, skip, inverse):
        lifted = self.project(coarse)[inverse]
        mixed = skip + self.gate(torch.cat((skip, lifted), -1)) * lifted
        return mixed + self.refine(mixed)


class Block(nn.Module):
    def __init__(
        self,
        channels,
        heads,
        ffn_ratio,
        local_mode,
        cross_rotary,
        enable_cross,
        group_size,
        queries,
        keys,
        dilation,
        backend,
        chunk_size,
        min_freq,
        max_freq,
        enable_local=True,
        cross_mode="grouped",
        cross_window_size=5,
        cross_include_center=True,
        cross_grid_chunk_size=1024,
        attention_order="local_then_cross",
    ):
        super().__init__()
        self.ffn = mlp(channels, channels, ffn_ratio)
        self.scale = nn.Parameter(torch.full((channels,), 0.1))
        self.local = PackedAttention(
            channels, heads, local_mode, backend, chunk_size, min_freq, max_freq
        ) if enable_local else None
        self.cross = (
            PackedAttention(
                channels, heads, cross_rotary, backend, chunk_size, min_freq, max_freq
            )
            if enable_cross
            else None
        )
        self.group_size, self.queries, self.keys, self.dilation = (
            group_size,
            queries,
            keys,
            dilation,
        )
        self.cross_mode, self.cross_window_size = cross_mode, cross_window_size
        self.cross_include_center = cross_include_center
        self.cross_grid_chunk_size = cross_grid_chunk_size
        self.attention_order = attention_order

    def forward(
        self, feature, coord, batch, patches, spatial_scale, return_debug=False, grid_layout=None
    ):
        feature = feature + self.scale.to(feature.dtype) * self.ffn(feature)
        members = patches.members
        record = (dict(local_members=members.detach(), local_enabled=self.local is not None,
                       attention_order=self.attention_order) if return_debug else None)
        for branch in ATTENTION_ORDERS[self.attention_order]:
            if branch == "local" and self.local is not None:
                width = members.shape[1]
                feature = self.local(feature, coord, members, members, width, width, spatial_scale)
            elif branch == "cross" and self.cross is not None:
                feature = self._cross_update(
                    feature, coord, batch, patches, grid_layout, spatial_scale, record)
        return (feature, record) if return_debug else feature

    def _cross_update(self, feature, coord, batch, patches, grid_layout, spatial_scale, record):
        if self.cross_mode == "grid":
            return self._grid_cross_update(feature, coord, grid_layout, spatial_scale, record)
        if self.cross_mode == "central":
            q, k = sample_central_cross(
                patches, batch, self.cross_window_size, self.dilation,
                self.queries, self.keys, self.training,
                include_center=self.cross_include_center)
            max_keys = k.shape[1]
        else:
            layout = cross_layout(
                patches, batch, self.group_size, self.dilation, self.queries, self.keys)
            q, k = sample_cross(patches, layout, self.group_size, self.training)
            max_keys = self.group_size * self.keys
        if record is not None:
            record.update(cross_queries=q.detach(), cross_keys=k.detach(),
                          cross_mode=self.cross_mode, dilation=self.dilation)
            if self.cross_mode == "grouped":
                record.update(patch_group=layout.group.detach(), patch_slot=layout.slot.detach())
        return self.cross(feature, coord, q, k, self.queries, max_keys, spatial_scale)

    def _grid_cross_update(self, feature, coord, layout, spatial_scale, record):
        # Pooled stages have one token per retained grid cell. Avoid padding
        # those stages to the stage-1 sampling budgets.
        max_occupancy = layout.max_occupancy
        budget_q = min(self.queries, max_occupancy)
        budget_k = min(self.keys, max_occupancy)
        query_ids, query_slots = compact_samples(sample_grid_rows(layout, budget_q, self.training))
        key_ids, key_slots = compact_samples(sample_grid_rows(layout, budget_k, self.training))
        windows = grid_cross_tables(layout, self.cross_window_size, self.dilation,
                                    query_slots, key_slots, self.cross_grid_chunk_size)
        if record is not None:
            windows = list(windows)
            if windows:
                q, k = map(torch.cat, zip(*windows))
            else:
                q = query_slots[:0]
                k = key_slots.new_empty((0, (self.cross_window_size**3 - 1) * budget_k))
            record.update(cross_queries=query_ids[q.clamp_min(0)].masked_fill(q < 0, -1),
                          cross_keys=key_ids[k.clamp_min(0)].masked_fill(k < 0, -1),
                          cross_mode="grid", dilation=self.dilation,
                          grid_cells=layout.cells, grid_inverse=layout.inverse)
        qids, update = self.cross.sampled_window_updates(
            feature, coord, query_ids, key_ids, windows, spatial_scale)
        cell_update, _ = group_mean(update, layout.inverse[qids], len(layout.cells))
        # All center-cell members get the mean of their sampled query updates.
        # Empty neighborhoods have a zero update.
        return feature + cell_update[layout.inverse].to(feature.dtype)


class SerializedLocalCross(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=64,
        channels=(48, 96, 192, 256),
        heads=(3, 6, 12, 16),
        depths=(2, 2, 3, 2),
        patch_sizes=(128, 128, 128, 128),
        pool_sizes=(0.4, 0.8, 1.6),
        position_scales=(0.4, 0.8, 1.6, 3.2),
        cross_group_size=2,
        cross_queries=(16, 16, 16, 16),
        cross_keys=(32, 32, 32, 32),
        cross_every=2,
        cross_dilations=(1, 2),
        local_rotary="none",
        cross_rotary="qk",
        ffn_ratio=2,
        score_hidden=16,
        chunk_size=128,
        attention_backend="flash_varlen",
        serialization_resolution=0.05,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_orders=True,
        serialization_backend="pointcept",
        axis_orders=None,
        rotary_min_frequency=0.25,
        rotary_max_frequency=2.0,
        checkpoint_blocks=False,
        enable_local_attention=True,
        enable_cross_attention=True,
        enable_coarse_attention=False,
        cross_mode="grouped",
        cross_window_size=5,
        coarse_grid_size=(8, 8, 8),
        coarse_channels=64,
        coarse_heads=8,
        coarse_depth=1,
        coarse_rotary="qk",
        coarse_position_scale=32.0,
        coarse_stages=None,
        coarse_placement="stage_end",
        pooling_mode="learned",
        pooling_reduce="max",
        pooling_norm="batch",
        pooling_partition="metric",
        pooling_grid_size=0.2,
        pooling_strides=None,
        coarse_mode="independent",
        coarse_cell_pool="mean_ln_project",
        cross_include_center=True,
        cross_grid_chunk_size=1024,
        attention_order="local_then_cross",
    ):
        super().__init__()
        stages = len(channels)
        if pooling_mode not in ("learned", "mean", "ptv3"):
            raise ValueError("pooling_mode must be learned, mean or ptv3")
        if pooling_partition not in ("metric", "grid_stride"):
            raise ValueError("pooling_partition must be metric or grid_stride")
        pooling_strides = (2,) * (stages - 1) if pooling_strides is None else tuple(pooling_strides)
        if pooling_partition == "grid_stride":
            if (len(pooling_strides) != stages - 1 or any(
                    type(v) is not int or v < 2 or v & (v - 1) for v in pooling_strides)):
                raise ValueError("pooling_strides must have one power-of-two integer >=2 per transition")
            if not math.isfinite(pooling_grid_size) or pooling_grid_size <= 0:
                raise ValueError("pooling_grid_size must be finite and positive")
        self.pooling_partition = pooling_partition
        self.pooling_grid_size = pooling_grid_size
        self.pooling_strides = pooling_strides
        self.pooling_mode = pooling_mode
        if coarse_mode not in ("independent", "shared_final", "shared_fused"):
            raise ValueError("Unknown coarse_mode")
        if coarse_placement not in ("stage_end", "every_block", "decoder_entry"):
            raise ValueError("Unknown coarse_placement")
        shared = enable_coarse_attention and coarse_mode != "independent"
        late_stages = (stages - 2, stages - 1)
        if shared and coarse_stages is None:
            coarse_stages = late_stages
        coarse_stages = (stages - 1,) if coarse_stages is None else tuple(coarse_stages)
        if (any(type(s) is not int or s < 0 or s >= stages for s in coarse_stages)
                or len(set(coarse_stages)) != len(coarse_stages)):
            raise ValueError("coarse_stages must contain unique zero-based encoder stage indices")
        if shared and (set(coarse_stages) != set(late_stages) or coarse_placement != "decoder_entry"):
            raise ValueError("Shared coarse attention needs both late stages and decoder_entry placement")
        if enable_coarse_attention and coarse_placement == "decoder_entry" and (
                stages < 2 or any(s not in late_stages for s in coarse_stages)):
            raise ValueError("decoder_entry supports only the final two encoder stages")
        self.coarse_stages = coarse_stages
        self.coarse_placement = coarse_placement
        if any(type(v) is not bool for v in (enable_local_attention, enable_cross_attention, enable_coarse_attention)):
            raise ValueError("attention enable flags must be booleans")
        if attention_order not in ATTENTION_ORDERS:
            raise ValueError(f"attention_order must be one of {tuple(ATTENTION_ORDERS)}")
        if cross_mode not in ("grouped", "central", "grid"):
            raise ValueError("cross_mode must be grouped, central or grid")
        if type(cross_include_center) is not bool:
            raise ValueError("cross_include_center must be a boolean")
        if enable_cross_attention and cross_mode == "grouped" and not cross_include_center:
            raise ValueError("Neighbor-only cross attention requires cross_mode='central'")
        if cross_mode in ("central", "grid") and (
                type(cross_window_size) is not int or cross_window_size < 3 or cross_window_size % 2 != 1):
            raise ValueError("cross_window_size must be odd and >=3")
        self.use_grid_cross = enable_cross_attention and cross_mode == "grid"
        if self.use_grid_cross and (pooling_partition != "grid_stride" or cross_include_center):
            raise ValueError("Grid cross attention requires grid_stride pooling and cross_include_center=False")
        if (
            stages < 1
            or any(
                len(v) != stages
                for v in (
                    heads,
                    depths,
                    patch_sizes,
                    position_scales,
                    cross_queries,
                    cross_keys,
                )
            )
            or len(pool_sizes) != stages - 1
        ):
            raise ValueError("Stage setting lengths do not match")
        integers = (
            *channels,
            *heads,
            *depths,
            *patch_sizes,
            *cross_queries,
            *cross_keys,
            *cross_dilations,
            cross_group_size,
            cross_every,
            cross_grid_chunk_size,
            ffn_ratio,
            score_hidden,
            chunk_size,
            in_channels,
            out_channels,
        )
        if any(not isinstance(v, int) or v < 1 for v in integers):
            raise ValueError(
                "Dimensions, depths, sampling counts, intervals must be positive integers"
            )
        if (
            not cross_dilations
            or cross_group_size < 2
            or (cross_mode == "grouped" and any(m < cross_group_size for m in cross_queries))
        ):
            raise ValueError(
                "Need >=2 patches/group and >=1 query budget per member patch"
            )
        if any(c % h for c, h in zip(channels, heads)):
            raise ValueError("Channels must be divisible by heads")
        if attention_backend not in ("flash_varlen", "reference"):
            raise ValueError("Unknown attention backend")
        if attention_backend == "flash_varlen" and any(
            (c // h) % 8 for c, h in zip(channels, heads)
        ):
            raise ValueError("FlashAttention head dimensions must be multiples of 8")
        if local_rotary not in ROTARY_MODES or cross_rotary not in ROTARY_MODES:
            raise ValueError(f"Rotary mode must be one of {ROTARY_MODES}")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (*pool_sizes, *position_scales, serialization_resolution)
        ):
            raise ValueError("Spatial scales must be finite and positive")
        if any(b <= a for a, b in zip(pool_sizes, pool_sizes[1:])):
            raise ValueError("Pooling sizes must increase")
        if axis_orders is not None:
            if not axis_orders or any(sorted(a) != [0, 1, 2] for a in axis_orders):
                raise ValueError("Invalid legacy serialization axes")
        elif not order or any(
            o not in ("z", "z-trans", "hilbert", "hilbert-trans") for o in order
        ):
            raise ValueError("Invalid serialization order names")
        if serialization_backend not in ("pointcept", "reference"):
            raise ValueError("Unknown serialization backend")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.patch_sizes, self.pool_sizes, self.position_scales = (
            patch_sizes,
            pool_sizes,
            position_scales,
        )
        self.resolution, self.axis_orders, self.checkpoint_blocks = (
            serialization_resolution,
            axis_orders,
            checkpoint_blocks,
        )
        self.order, self.shuffle_orders, self.serialization_backend = (
            tuple(order),
            shuffle_orders,
            serialization_backend,
        )
        self.stem = mlp(in_channels, channels[0])
        stages_list, cross_index = [], 0
        for s, c in enumerate(channels):
            blocks = []
            for b in range(depths[s]):
                enabled = enable_cross_attention and (b + 1) % cross_every == 0
                dilation = cross_dilations[cross_index % len(cross_dilations)]
                blocks.append(
                    Block(
                        c,
                        heads[s],
                        ffn_ratio,
                        local_rotary,
                        cross_rotary,
                        enabled,
                        cross_group_size,
                        cross_queries[s],
                        cross_keys[s],
                        dilation,
                        attention_backend,
                        chunk_size,
                        rotary_min_frequency,
                        rotary_max_frequency,
                        enable_local_attention,
                        cross_mode,
                        cross_window_size,
                        cross_include_center,
                        cross_grid_chunk_size,
                        attention_order,
                    )
                )
                cross_index += int(enabled)
            stages_list.append(nn.ModuleList(blocks))
        self.stages = nn.ModuleList(stages_list)
        self.coarse = CoarseGlobalContext(
            channels[-1], coarse_channels, coarse_heads, coarse_depth,
            coarse_grid_size, coarse_rotary, coarse_position_scale,
            attention_backend, chunk_size, rotary_min_frequency, rotary_max_frequency,
            cell_pool=coarse_cell_pool
        ) if enable_coarse_attention and not shared and stages - 1 in coarse_stages else None
        # Preserve the original final-stage module/key names for old v2 checkpoints.
        # Earlier stages have independent modules. In every_block mode, a stage's
        # module is reused after each block of that stage (weights are shared).
        self.coarse_contexts = nn.ModuleDict({
            str(s): CoarseGlobalContext(
                channels[s], coarse_channels, coarse_heads, coarse_depth,
                coarse_grid_size, coarse_rotary, coarse_position_scale,
                attention_backend, chunk_size, rotary_min_frequency, rotary_max_frequency,
                cell_pool=coarse_cell_pool)
            for s in coarse_stages if s != stages - 1
        }) if enable_coarse_attention and not shared else nn.ModuleDict()
        self.shared_coarse = CoarseGlobalContext(
            channels[-1], coarse_channels, coarse_heads, coarse_depth,
            coarse_grid_size, coarse_rotary, coarse_position_scale,
            attention_backend, chunk_size, rotary_min_frequency, rotary_max_frequency,
            fine_channels=channels[-2], use_fine_features=coarse_mode == "shared_fused",
            cell_pool=coarse_cell_pool
        ) if shared else None
        self.pools = nn.ModuleList([
            LearnedPooling(a, b, score_hidden) if pooling_mode == "learned" else
            MeanPooling(a, b) if pooling_mode == "mean" else
            PTv3Pooling(a, b, pooling_reduce, pooling_norm)
            for a, b in zip(channels, channels[1:])])
        self.decoders = nn.ModuleList(
            [Fusion(a, b) for a, b in zip(channels, channels[1:])]
        )
        self.head = mlp(channels[0], out_channels)

    def _coarse_update(self, feature, coord, batch, mass, stage, debug, block=None):
        module = (self.coarse if stage == len(self.stages) - 1 else
                  self.coarse_contexts[str(stage)] if str(stage) in self.coarse_contexts else None)
        if module is None:
            return feature
        if debug is not None:
            feature, record = module(feature, coord, batch, mass, True)
            debug.append(dict(branch="coarse", stage=stage, block=block, **record))
            return feature
        return module(feature, coord, batch, mass)

    def forward(self, data, return_debug=False):
        coord, raw = data["coord"], data["feat"]
        if (
            coord.ndim != 2
            or coord.shape[1] != 3
            or not len(coord)
            or raw.shape != (len(coord), self.in_channels)
        ):
            raise ValueError("Expected coord (N,3), feat (N,in_channels), N>0")
        if (
            coord.device != raw.device
            or not coord.is_floating_point()
            or not raw.is_floating_point()
        ):
            raise ValueError("Floating features and coordinates must share a device")
        if not bool(torch.isfinite(coord).all()) or not bool(torch.isfinite(raw).all()):
            raise ValueError("Inputs must be finite")
        batch = batch_from_data(data, len(coord), coord.device)
        with torch.no_grad():
            coord = centered_coordinates(coord.float(), batch)
        bounds = crop_bounds(coord, batch) if self.shared_coarse is not None else None
        delayed = self.coarse_placement == "decoder_entry" and (
            self.shared_coarse is not None or self.coarse is not None or bool(self.coarse_contexts)
        )
        feature = self.stem(raw)
        mass = coord.new_ones(len(coord))
        pool_grid = (torch.floor(coord / self.pooling_grid_size).long()
                     if self.pooling_partition == "grid_stride" else None)
        effective_grid_size = self.pooling_grid_size
        records, debug = [], []
        for s, blocks in enumerate(self.stages):
            layouts = {}
            grid_layout = None
            if self.use_grid_cross and any(block.cross is not None for block in blocks):
                grid_layout = make_grid_cross_layout(pool_grid, batch)
            available = self.axis_orders if self.axis_orders is not None else self.order
            if self.shuffle_orders and self.training:
                permutation = torch.randperm(len(available)).tolist()
                available = tuple(available[i] for i in permutation)
            for b, block in enumerate(blocks):
                axes = available[(s + b) % len(available)]
                if axes not in layouts:
                    layouts[axes] = make_patches(
                        coord,
                        batch,
                        self.resolution,
                        axes,
                        self.patch_sizes[s],
                        self.serialization_backend,
                    )
                operation = partial(
                    block,
                    coord=coord,
                    batch=batch,
                    patches=layouts[axes],
                    spatial_scale=self.position_scales[s],
                    grid_layout=grid_layout,
                )
                if return_debug:
                    feature, record = operation(feature, return_debug=True)
                    record.update(stage=s, block=b, batch=batch.detach())
                    record["serialization_order"] = axes
                    debug.append(record)
                elif self.checkpoint_blocks and self.training:
                    feature = checkpoint(
                        operation, feature, use_reentrant=False, preserve_rng_state=True
                    )
                else:
                    feature = operation(feature)
                if self.coarse_placement == "every_block":
                    feature = self._coarse_update(
                        feature, coord, batch, mass, s, debug if return_debug else None, b)
            if self.coarse_placement == "stage_end":
                feature = self._coarse_update(
                    feature, coord, batch, mass, s, debug if return_debug else None)
            if delayed and s == len(self.stages) - 2:
                fine_state = feature, coord, batch, mass
            if s < len(self.pools):
                if self.pooling_partition == "grid_stride":
                    cells = make_grid_stride_voxels(pool_grid, batch, self.pooling_strides[s])
                    pool_grid = cells.indices[:, 1:]
                    effective_grid_size *= self.pooling_strides[s]
                    size = effective_grid_size
                else:
                    size = self.pool_sizes[s]
                    cells = make_voxels(coord, batch, size)
                records.append((feature, cells.inverse))
                feature, coord, mass, batch = self.pools[s](
                    feature, coord, mass, cells, size
                )
        if delayed:
            stage = len(self.stages) - 1
            if self.shared_coarse is not None:
                skip, feature, record = self.shared_coarse.forward_shared(
                    fine_state, (feature, coord, batch, mass), bounds)
                if return_debug:
                    debug.append(dict(branch="coarse_shared", stages=(stage - 1, stage), **record))
            else:
                coarse_debug = debug if return_debug else None
                skip = self._coarse_update(*fine_state, stage - 1, coarse_debug)
                feature = self._coarse_update(feature, coord, batch, mass, stage, coarse_debug)
            records[-1] = (skip, records[-1][1])
        for s in reversed(range(len(records))):
            skip, inverse = records[s]
            feature = self.decoders[s](feature, skip, inverse)
        output = self.head(feature)
        return (output, debug) if return_debug else output
