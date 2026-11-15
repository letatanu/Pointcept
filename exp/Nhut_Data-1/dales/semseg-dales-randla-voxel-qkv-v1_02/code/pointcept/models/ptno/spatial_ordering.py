"""Explicit 3D Morton, Hilbert and serpentine orderings; no Pointcept imports.

Curves are established ordering methods, not claimed as new algorithms. Ordering
is used for candidate retrieval, followed by Euclidean neighbor selection.
"""
from itertools import permutations
import torch


@torch.no_grad()
def serialized_patch_neighbors(coord, batch, patch_size=16, order='morton:xyz', bits=10):
    """Disjoint per-scene patches, no distance search. Padding/self use -1.

    Return N,K indices in original point order. Query chunking in attention
    avoids materializing all N,K,C edge activations in one operation.
    """
    if patch_size < 2:
        raise ValueError('patch_size must be at least two')
    graph=torch.full((len(coord),patch_size),-1,device=coord.device,dtype=torch.long)
    for scene in torch.unique(batch,sorted=True):
        ids=torch.nonzero(batch==scene,as_tuple=False).flatten()
        x=coord[ids]
        order_ids=_stable_spatial_sort(spatial_code(_quantize_scene(x,bits),order,bits),x)
        ordered=ids[order_ids]
        padded=torch.full(((len(ids)+patch_size-1)//patch_size*patch_size,),-1,
                          device=coord.device,dtype=torch.long)
        padded[:len(ids)]=ordered
        patches=padded.reshape(-1,patch_size)
        rows=patches.repeat_interleave(patch_size,dim=0)[:len(ids)]
        graph[ordered]=rows.masked_fill(rows==ordered[:,None],-1)
    return graph


@torch.no_grad()
def metric_morton_cells(coord,batch,cell_size,bits=20):
    """Quantize once, preserving the integer hierarchy through all poolings."""
    if cell_size <= 0:
        raise ValueError('base_voxel_size must be positive')
    cells=torch.empty_like(coord,dtype=torch.long)
    for scene in torch.unique(batch,sorted=True):
        ids=torch.nonzero(batch==scene,as_tuple=False).flatten()
        cells[ids]=((coord[ids]-coord[ids].amin(0))/cell_size).floor().long()
    # Validate overflow once; scene IDs are kept separate from spatial codes.
    spatial_code(cells,'morton:xyz',bits)
    return cells


def all_axis_orders(families=("morton", "hilbert", "serpentine")):
    return tuple(f"{family}:{''.join(p)}" for family in families for p in permutations("xyz"))


def _interleave(x, bits):
    code = torch.zeros(len(x), dtype=torch.long, device=x.device)
    for bit in range(bits):
        for axis in range(3):
            code |= ((x[:, axis] >> bit) & 1) << (3 * bit + 2 - axis)
    return code


def spatial_code(integer_coord, order="hilbert:xyz", bits=10):
    """Integer cube [0,2**bits)^3 -> nonnegative int64 spatial keys.

    Hilbert uses the transpose/Gray-code algorithm; tested against hilbertcurve.
    Serpentine is a finite-grid boustrophedon scan, not a new fractal SFC.
    At most 20 bits per axis (60-bit keys) are allowed.
    """
    if not isinstance(bits, int) or not 1 <= bits <= 20:
        raise ValueError("bits must be an integer in [1,20]")
    fields = order.split(":")
    if len(fields) > 2:
        raise ValueError("Use family:axes, e.g. hilbert:yzx")
    family, axes = fields[0], fields[1] if len(fields) == 2 else "xyz"
    if sorted(axes) != list("xyz") or family not in ("morton", "hilbert", "serpentine"):
        raise ValueError("Expected morton/hilbert/serpentine and a permutation of xyz")
    if integer_coord.ndim != 2 or integer_coord.shape[1] != 3 or integer_coord.dtype != torch.long:
        raise ValueError("integer_coord must be an N,3 int64 tensor")
    if bool(((integer_coord < 0) | (integer_coord >= 2**bits)).any()):
        raise ValueError("Coordinates outside the selected integer cube")
    x = integer_coord[:, ["xyz".index(a) for a in axes]].clone()
    if family == "morton":
        return _interleave(x, bits)
    if family == "serpentine":
        side = 2**bits
        row_y = torch.where((x[:,0] & 1).bool(), side-1-x[:,1], x[:,1])
        row = x[:,0] * side + row_y
        col = torch.where((row & 1).bool(), side-1-x[:,2], x[:,2])
        return row * side + col
    # Hilbert coordinates -> transpose: inverse exchange/invert, then Gray code.
    q = 1 << (bits-1)
    while q > 1:
        p = q-1
        for axis in range(3):
            first, current = x[:,0].clone(), x[:,axis].clone()
            flip = (current & q) != 0
            if axis == 0:
                x[:,0] = torch.where(flip, first ^ p, first)
            else:
                exchange = (first ^ current) & p
                x[:,0] = torch.where(flip, first ^ p, first ^ exchange)
                x[:,axis] = torch.where(flip, current, current ^ exchange)
        q >>= 1
    x[:,1] ^= x[:,0]
    x[:,2] ^= x[:,1]
    correction = torch.zeros(len(x), device=x.device, dtype=torch.long)
    q = 1 << (bits-1)
    while q > 1:
        correction ^= torch.where((x[:,2] & q) != 0, q-1, 0)
        q >>= 1
    return _interleave(x ^ correction[:,None], bits)


def _quantize_scene(coord, bits):
    # Common scale for all three axes preserves the scene's aspect ratio.
    centered = coord - coord.amin(0)
    extent = centered.amax().clamp_min(1e-12)
    return (centered / extent * (2**bits-1)).floor().clamp(0,2**bits-1).long()


def _stable_spatial_sort(code, coord):
    # Exact coordinate tie-breakers reduce sensitivity to collisions in the grid.
    # Exactly coincident points still have ambiguous ordering, as do tied kNNs.
    ids = torch.arange(len(coord), device=coord.device)
    for axis in (2,1,0):
        ids = ids[torch.argsort(coord[ids,axis], stable=True)]
    return ids[torch.argsort(code[ids], stable=True)]


@torch.no_grad()
def curve_neighbors(coord, batch, k=16,
                    orders=("hilbert:xyz", "morton:yzx", "serpentine:zxy"),
                    bits=10, window_radius=16, chunk_size=1024):
    """Approximate kNN from the union of adjacent ranks in several orderings.

    Excludes self, removes duplicate candidates, reranks in metric coordinates,
    and pads missing neighbors with -1. No cross-scene edges. No N-by-N matrix.
    A point may have fewer than k candidates even when the full scene has more.
    """
    if k < 1 or window_radius < 1 or chunk_size < 1 or not orders:
        raise ValueError("Positive k/window_radius/chunk_size and nonempty orders required")
    output = torch.full((len(coord),k), -1, dtype=torch.long, device=coord.device)
    for scene in torch.unique(batch, sorted=True):
        ids = torch.nonzero(batch == scene, as_tuple=False).flatten()
        n = len(ids)
        if n < 2:
            continue
        x = coord[ids]
        quantized = _quantize_scene(x, bits)
        sorted_ids, ranks = [], []
        for order in orders:
            sort = _stable_spatial_sort(spatial_code(quantized,order,bits), x)
            rank = torch.empty_like(sort)
            rank[sort] = torch.arange(n,device=x.device)
            sorted_ids.append(sort)
            ranks.append(rank)
        offsets = torch.cat((torch.arange(-window_radius,0,device=x.device),
                             torch.arange(1,window_radius+1,device=x.device)))
        for start in range(0,n,chunk_size):
            queries = torch.arange(start,min(start+chunk_size,n),device=x.device)
            candidates = []
            for sort, rank in zip(sorted_ids,ranks):
                positions = rank[queries,None] + offsets
                candidate = sort[positions.clamp(0,n-1)]
                candidates.append(candidate.masked_fill((positions<0)|(positions>=n), n))
            candidates = torch.cat(candidates,dim=1).sort(dim=1).values
            valid = candidates < n
            valid[:,1:] &= candidates[:,1:] != candidates[:,:-1]
            valid &= candidates != queries[:,None]
            distances = (x[queries,None,:] - x[candidates.clamp_max(n-1)]).square().sum(-1)
            distances = distances.masked_fill(~valid,float('inf'))
            take = min(k,distances.shape[1])
            values, at = distances.topk(take,dim=1,largest=False,sorted=True)
            selected = candidates.gather(1,at).clamp_max(n-1)
            output[ids[queries],:take] = ids[selected].masked_fill(~torch.isfinite(values),-1)
    return output


@torch.no_grad()
def exact_neighbors(coord, batch, k=16, backend="torch", chunk_size=512):
    """Exact reference kNN: torch chunked brute force or optional SciPy KD-tree."""
    if k < 1 or chunk_size < 1 or backend not in ("torch","scipy"):
        raise ValueError("Positive k/chunk_size and torch/scipy backend required")
    output = torch.full((len(coord),k),-1,dtype=torch.long,device=coord.device)
    for scene in torch.unique(batch,sorted=True):
        ids = torch.nonzero(batch==scene,as_tuple=False).flatten()
        n = len(ids)
        if n < 2:
            continue
        x, take = coord[ids], min(k,n-1)
        if backend == "scipy":
            from scipy.spatial import cKDTree
            array = x.cpu().double().numpy()
            _, result = cKDTree(array).query(array,k=take+1,workers=1)
            candidate = torch.as_tensor(result,device=x.device,dtype=torch.long)
            pos = torch.arange(take+1,device=x.device).expand(n,-1)
            pos = pos.masked_fill(candidate == torch.arange(n,device=x.device)[:,None],take+1)
            selected = candidate.gather(1,pos.argsort(dim=1)[:,:take])
            output[ids,:take] = ids[selected]
            continue
        for start in range(0,n,chunk_size):
            queries = torch.arange(start,min(start+chunk_size,n),device=x.device)
            best = x.new_full((len(queries),take),float('inf'))
            neighbors = torch.zeros_like(best,dtype=torch.long)
            for ref in range(0,n,chunk_size):
                refs = torch.arange(ref,min(ref+chunk_size,n),device=x.device)
                dist = torch.cdist(x[queries],x[refs],compute_mode="donot_use_mm_for_euclid_dist")
                dist.masked_fill_(queries[:,None]==refs[None,:],float('inf'))
                combined = torch.cat((best,dist),dim=1)
                candidate = torch.cat((neighbors,refs.expand(len(queries),-1)),dim=1)
                best, at = combined.topk(take,largest=False,sorted=True)
                neighbors = candidate.gather(1,at)
            output[ids[queries],:take] = ids[neighbors]
    return output
