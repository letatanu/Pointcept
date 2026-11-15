"""Neighbor-relative quaternion positional encoding (not an SO(3) network).

Builds e_ij from x_i-x_j, then mean-pools valid edges to e_i for compatibility
with 3D feature injection. Exposes encode_edges for explicit edge features.
No scene centroid or scene-wide radius normalization is used.
"""
import torch
from torch import nn


def apply_quat_rotation_to_features(feat, quat):
    """q*f*q* on scalar/vector groups. q must be unit length."""
    if feat.ndim != 2 or feat.shape[1] % 4 or quat.shape != (feat.shape[0], 4):
        raise ValueError("Expected feat=(N,4K), quat=(N,4)")
    f = feat.reshape(feat.shape[0], -1, 4)
    q = quat.to(feat.dtype)
    v = f[..., 1:]
    axis = q[:, None, 1:].expand_as(v)
    t = 2 * torch.cross(axis, v, dim=-1)
    rotated = v + q[:, None, :1] * t + torch.cross(axis, t, dim=-1)
    return torch.cat((f[..., :1], rotated), dim=-1).reshape_as(feat)


@torch.no_grad()
def knn_indices(coord, batch, k=16, backend="auto", chunk_size=512):
    """Return N,K GLOBAL indices, -1 for missing neighbors; never cross scenes.

    Self is excluded by index, not distance, so coincident distinct points remain
    eligible. Ties in distance have backend-dependent ordering. Graph selection
    is discrete/detached; gradients still flow through selected displacements.
    auto: SciPy cKDTree on CPU (small torch fallback), pointops on CUDA.
    torch: exact chunked brute force for testing; O(sum_s N_s^2) arithmetic.
    scipy: CPU search, including explicit transfers if coordinates are on CUDA.
    pointops: Pointcept's CUDA knn_query, including its quadratic search cost.
    """
    if not isinstance(k, int) or k < 1 or chunk_size < 1:
        raise ValueError("k and chunk_size must be positive integers")
    if coord.ndim != 2 or coord.shape[1] != 3 or batch.shape != (len(coord),):
        raise ValueError("Expected coord=(N,3), batch=(N,)")
    if batch.device != coord.device:
        raise ValueError("Coordinates and batch IDs must share a device")
    if backend not in ("auto", "torch", "scipy", "pointops"):
        raise ValueError("Unknown kNN backend")
    chosen = backend
    if chosen == "auto":
        if coord.is_cuda:
            chosen = "pointops"
        else:
            try:
                from scipy.spatial import cKDTree
                chosen = "scipy"
            except ImportError:
                if len(coord) > 4096:
                    raise ImportError("Install scipy for CPU kNN on large clouds; torch is a small-test fallback")
                chosen = "torch"
    if chosen == "scipy":
        from scipy.spatial import cKDTree
    if chosen == "pointops":
        if not coord.is_cuda:
            raise ValueError("pointops kNN requires CUDA coordinates")
        try:
            from pointops import knn_query
        except ImportError as exc:
            raise ImportError("Install Pointcept's pointops CUDA extension or explicitly select scipy kNN") from exc
        if k > 99:
            raise ValueError("pointops backend supports at most 99 neighbors here (one extra query excludes self)")
    neighbors = torch.full((len(coord), k), -1, dtype=torch.long, device=coord.device)
    for scene in torch.unique(batch, sorted=True):
        ids = torch.nonzero(batch == scene, as_tuple=False).flatten()
        n = len(ids)
        if n < 2:
            continue
        take, count = min(k, n-1), min(k+1, n)
        x = coord[ids].detach()
        if chosen == "scipy":
            _, result = cKDTree(x.cpu().double().numpy()).query(
                x.cpu().double().numpy(), k=count, workers=1)
            candidate = torch.as_tensor(result, dtype=torch.long, device=coord.device)
        elif chosen == "pointops":
            offset = torch.tensor([n], dtype=torch.int32, device=x.device)
            candidate, _ = knn_query(count, x.float().contiguous(), offset)
            candidate = candidate.long()
        else:
            # Bound the distance workspace in both dimensions; no N-by-N array.
            blocks = []
            x = x.float() if x.dtype != torch.float64 else x
            for begin in range(0, n, chunk_size):
                query = x[begin:begin+chunk_size]
                best_d = query.new_full((len(query), count), float("inf"))
                best_i = torch.zeros_like(best_d, dtype=torch.long)
                for ref_begin in range(0, n, chunk_size):
                    reference = x[ref_begin:ref_begin+chunk_size]
                    distances = torch.cdist(query, reference, compute_mode="donot_use_mm_for_euclid_dist")
                    ref_i = torch.arange(ref_begin, ref_begin+len(reference), device=x.device)
                    all_d = torch.cat((best_d, distances), dim=1)
                    all_i = torch.cat((best_i, ref_i.expand(len(query), -1)), dim=1)
                    best_d, at = all_d.topk(count, dim=1, largest=False, sorted=True)
                    best_i = all_i.gather(1, at)
                blocks.append(best_i)
            candidate = torch.cat(blocks)
        # Query K+1 candidates then remove self; this also handles zero-distance ties.
        positions = torch.arange(count, device=x.device).expand(n, -1)
        positions = positions.masked_fill(candidate == torch.arange(n, device=x.device)[:, None], count)
        candidate = candidate.gather(1, positions.argsort(dim=1)[:, :take])
        neighbors[ids, :take] = ids[candidate]
    return neighbors


class QuatRPE(nn.Module):
    """e_ij = sandwich(MLP([q_ij, sin(2^m*r_ij/ell_i), log1p(r_ij/s)]),q_ij).

    delta_ij=x_i-x_j; ell_i is the RMS valid-neighbor distance (clamped at epsilon).
    q_ij=normalize([1,delta_ij/ell_i]); s=distance_scale is a positive physical scale
    in the same units as coord (default 1). The extra log distance preserves size
    information removed by local normalization. Forward averages e_ij over j.

    This is local relative geometry injection, NOT per-edge attention-logit RPE.
    Ordinary MLPs and the full backbone remain unconstrained under SO(3).
    """
    def __init__(self, out_channels, num_freqs=8, k=16, backend="auto",
                 chunk_size=512, edge_chunk_size=2048, distance_scale=1.0):
        super().__init__()
        if out_channels % 4 or out_channels < 4:
            raise ValueError("QuatRPE channels must be a positive multiple of four")
        if k < 1 or edge_chunk_size < 1 or distance_scale <= 0:
            raise ValueError("k, edge_chunk_size and distance_scale must be positive")
        self.k, self.backend, self.chunk_size = k, backend, chunk_size
        self.edge_chunk_size, self.distance_scale = edge_chunk_size, distance_scale
        self.proj = nn.Sequential(nn.Linear(5 + num_freqs, out_channels),
                                  nn.LayerNorm(out_channels), nn.GELU(),
                                  nn.Linear(out_channels, out_channels))
        self.register_buffer("freqs", 2.0 ** torch.arange(num_freqs, dtype=torch.float32))

    @staticmethod
    def _build_quaternion(delta):
        raw = torch.cat((torch.ones_like(delta[..., :1]), delta), dim=-1)
        return raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True)

    def _neighbors(self, point):
        # Cache only topology, never autograd tensors. One Point object per stage.
        # New tensors, in-place coordinate/batch edits, or changed k invalidate it.
        coords, batch = point.coord, point.batch
        def version(t):
            try:
                return t._version
            except RuntimeError:  # inference tensors have no version counter
                return None
        cv, bv = version(coords), version(batch)
        key = (id(coords), id(batch), cv, bv, self.k, self.backend, self.chunk_size)
        cacheable = cv is not None and bv is not None
        cache = point.get("_neighbor_quatrpe_cache") if isinstance(point, dict) else getattr(point, "_neighbor_quatrpe_cache", None)
        if cacheable and cache is not None and cache[0] == key:
            return cache[1]
        idx = knn_indices(coords, batch, self.k, self.backend, self.chunk_size)
        if cacheable:
            if isinstance(point, dict):
                point["_neighbor_quatrpe_cache"] = (key, idx)
            else:
                point._neighbor_quatrpe_cache = (key, idx)
        return idx

    def encode_edges(self, coord, neighbor_index, query_index=None, batch=None):
        """Return (edge_features[M,K,C], valid[M,K]) for explicit neighbors.

        Indices refer to coord rows. -1 denotes padding; self edges are masked.
        Pass batch when supplying an external graph to reject cross-scene edges.
        Duplicate neighbor entries are caller error (internal kNN never adds them).
        """
        if neighbor_index.ndim != 2 or neighbor_index.dtype != torch.long:
            raise ValueError("neighbor_index must be an M,K int64 tensor")
        if query_index is None:
            query_index = torch.arange(len(coord), device=coord.device)
        if len(query_index) != len(neighbor_index):
            raise ValueError("query_index rows must match neighbor_index")
        if neighbor_index.numel() and bool(((neighbor_index < -1) | (neighbor_index >= len(coord))).any()):
            raise ValueError("Invalid neighbor index")
        valid = (neighbor_index >= 0) & (neighbor_index != query_index[:, None])
        safe = neighbor_index.clamp_min(0)
        if batch is not None and bool((valid & (batch[safe] != batch[query_index, None])).any()):
            raise ValueError("Neighbor edges must stay within their scene")
        dtype = self.proj[0].weight.dtype
        coords = coord.to(dtype)
        delta = coords[query_index, None, :] - coords[safe]  # x_i - x_j, not reverse
        delta = delta * valid[..., None]
        radius = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        degree = valid.sum(dim=1, keepdim=True).clamp_min(1)
        scale = (radius.square().sum(dim=1) / degree).clamp_min(1e-12).sqrt()
        quat = self._build_quaternion(delta / scale[:, None, :])
        radial = radius / scale[:, None, :]
        features = torch.cat((quat, torch.sin(radial * self.freqs),
                              torch.log1p(radius / self.distance_scale)), dim=-1)
        encoded = self.proj(features)
        c = encoded.shape[-1]
        rotated = apply_quat_rotation_to_features(encoded.reshape(-1,c), quat.reshape(-1,4))
        return rotated.reshape_as(encoded) * valid[..., None], valid

    def forward(self, point):
        if len(point.coord) == 0:
            return self.proj[0].weight.new_empty((0,self.proj[-1].out_features))
        idx = self._neighbors(point)
        result = []
        for start in range(0, len(idx), self.edge_chunk_size):
            queries = torch.arange(start, min(start+self.edge_chunk_size,len(idx)), device=idx.device)
            edges, valid = self.encode_edges(point.coord, idx[queries], queries, point.batch)
            result.append(edges.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1))
        return torch.cat(result)
