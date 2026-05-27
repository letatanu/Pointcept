import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.utils.structure import Point
from pointcept.models.losses import build_criteria


class PCAAligner(nn.Module):
    """
    Geometrically aligns point clouds to their Principal Components.
    Includes sign disambiguation to guarantee 100% deterministic orientation.
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, coord, offset):
        aligned_coord = torch.zeros_like(coord)
        start = 0
        
        for end in offset:
            # 1. Extract and center the current point cloud
            c = coord[start:end]
            centered = c - c.mean(dim=0, keepdim=True)
            
            # 2. Compute SVD to find the principal axes
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            
            # 3. Project the points onto the principal axes
            proj = torch.matmul(centered, Vh.T)
            
            # 4. Disambiguate the axes (Majority-vote trick for stable orientation)
            pos_count = (proj > 0).sum(dim=0)
            neg_count = (proj < 0).sum(dim=0)
            
            signs = torch.where(
                pos_count >= neg_count, 
                torch.ones_like(pos_count, dtype=proj.dtype), 
                -torch.ones_like(pos_count, dtype=proj.dtype)
            )
            
            # 5. Apply the sign correction to our projected coordinates
            aligned_coord[start:end] = proj * signs.unsqueeze(0)
            
            start = end
            
        return aligned_coord


@MODELS.register_module("OPTNet")
class OPTNet(PointTransformerV3):
    def __init__(self, 
                 in_channels=6, 
                 align_pca=True,
                 **kwargs):
        
        super().__init__(in_channels=in_channels, **kwargs)
        
        self.align_pca = align_pca
        if self.align_pca:
            self.pca_aligner = PCAAligner()

    def forward(self, data_dict):
        point = Point(data_dict)

        if self.align_pca:
            # --- THE PCA ALIGNMENT WRAPPER ---
            
            # 1. Save the original grid coordinates
            original_grid_coord = point.grid_coord.clone()
            
            # 2. Get strictly aligned continuous coordinates
            aligned_coord = self.pca_aligner(point.coord, point.offset)
            
            # 3. Convert aligned coordinates back to a proxy grid for sorting
            # We normalize to [0, 1] and scale up by a large resolution (e.g., 2^10 = 1024 voxels)
            coord_min = aligned_coord.min(dim=0, keepdim=True)[0]
            coord_max = aligned_coord.max(dim=0, keepdim=True)[0]
            scale = coord_max - coord_min
            scale[scale < 1e-6] = 1.0 
            
            normalized_coord = (aligned_coord - coord_min) / scale
            proxy_grid_coord = (normalized_coord * 1024).int()
            
            # 4. Trick the Point object into using our aligned grid for serialization
            point.grid_coord = proxy_grid_coord

        # --- STANDARD PTv3 FLOW ---
        
        # 5. Serialize (Computes Hilbert/Z-order codes on the perfectly ALIGNED shape)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        
        if self.align_pca:
            # 6. Restore the original grid coordinates so pooling remains geometrically accurate
            point.grid_coord = original_grid_coord

        # 7. Sparsify (Voxelization)
        point.sparsify()

        # 8. Backbone Forward
        point = self.embedding(point)
        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)

        return point


@MODELS.register_module("OPTNetSegmentor")
class OPTNetSegmentor(nn.Module):
    def __init__(self, backbone, criteria, num_classes, backbone_out_channels=None):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        self.criteria = build_criteria(criteria)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, data_dict):
        point = self.backbone(data_dict)
        seg_logits = self.seg_head(point.feat)

        # 1. Training Mode (ordering_loss logic is completely removed)
        if self.training:
            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss, seg_loss=loss)
        
        # 2. Validation Mode (Not training, but has labels)
        elif "segment" in data_dict:
            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
            
        # 3. Test Mode (No labels)
        return dict(seg_logits=seg_logits)