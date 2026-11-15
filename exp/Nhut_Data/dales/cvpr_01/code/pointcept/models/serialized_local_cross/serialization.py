"""PTv3 order names with direct Pointcept and standalone reference backends."""
import torch

ORDERS=("z","z-trans","hilbert","hilbert-trans")


@torch.no_grad()
def reference_encode(grid,depth,order):
    """3D integer Morton/Skilling reference, matching Pointcept conventions.

    Reference Hilbert is intended for checks, not a GPU speed improvement.
    """
    grid=grid.long()
    if order.endswith("-trans"):
        grid=grid[:,[1,0,2]]
    if order.startswith("z"):
        code=grid.new_zeros(len(grid))
        for bit in range(depth):
            code |= ((grid[:,0]>>bit)&1) << (3*bit+2)
            code |= ((grid[:,1]>>bit)&1) << (3*bit+1)
            code |= ((grid[:,2]>>bit)&1) << (3*bit)
        return code
    # Apply Skilling's bit-plane exchanges, then decode the interleaved Gray code.
    shifts=torch.arange(depth-1,-1,-1,device=grid.device)
    planes=((grid[:,:,None]>>shifts)&1).bool()
    for bit in range(depth):
        for dim in range(3):
            active=planes[:,dim,bit].clone()
            planes[:,0,bit+1:] ^= active[:,None]
            exchange=(~active[:,None]) & (planes[:,0,bit+1:] ^ planes[:,dim,bit+1:])
            planes[:,dim,bit+1:] ^= exchange
            planes[:,0,bit+1:] ^= exchange
    gray=planes.transpose(1,2).reshape(len(grid),depth*3)
    binary=gray.long().cumsum(-1)&1
    weights=torch.arange(depth*3-1,-1,-1,device=grid.device)
    return (binary<<weights).sum(-1)


@torch.no_grad()
def serialized_order(coord,batch,resolution,order,backend="pointcept"):
    if order not in ORDERS:
        raise ValueError(f"Unknown serialization order: {order}")
    grid=torch.floor(coord/resolution).long()
    if bool((grid<0).any()):
        raise ValueError("Serialization needs nonnegative coordinates")
    # Match Point.serialization's adaptive depth convention.
    depth=max(1,int(grid.max()+1).bit_length())
    if depth>16 or depth*3+int(batch.max()+1).bit_length()>63:
        raise ValueError("PTv3 serialization exceeds its depth/int64 limit; increase serialization_resolution")
    if backend=="pointcept":
        try:
            from pointcept.models.utils.serialization import encode
        except ImportError as exc:
            raise ImportError("Pointcept serializer unavailable; use serialization_backend='reference' for standalone checks") from exc
        code=encode(grid,batch,depth=depth,order=order)
    elif backend=="reference":
        code=reference_encode(grid,depth,order) | (batch.long() << (depth*3))
    else:
        raise ValueError("serialization_backend must be pointcept or reference")
    # Upstream does not specify stable tie sorting. Stable ordering keeps raw
    # points sharing one grid coordinate reproducible without dropping them.
    return code.argsort(stable=True)
