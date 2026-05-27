import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.utils.structure import Point
from pointcept.models.losses import build_criteria
from pointcept.models.utils.serialization import encode
import pointops
import torch.nn.init as init
import torch.nn.functional as F

class PointSorter(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_orders=1, 
                 ordering_k=16, contrastive_k_near=16, contrastive_k_far=32,
                 temp_locality=0.1, temp_contrastive=0.5):
        super().__init__()
        self.num_orders = num_orders
        
        # Robust MLP with LayerNorm
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels), 
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_orders)
        )

        self._zorder_init(in_channels)
        
        self.ss_loss = SelfSupervisedOrderingLoss(
            ordering_k=ordering_k, 
            contrastive_k_near=contrastive_k_near,
            contrastive_k_far=contrastive_k_far, 
            temp_locality=temp_locality, 
            temp_contrastive=temp_contrastive
        )

    def _zorder_init(self, in_channels):
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
        
        for layer in self.mlp[:-1]:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None: init.zeros_(layer.bias)

        N = 4096
        depth = 10
        grid_scale = 1023
        dummy_grid = torch.randint(0, grid_scale + 1, (N, 3), device=device).int()
        dummy_batch = torch.zeros(N, device=device).long()
        dummy_coords = (dummy_grid.float() / grid_scale) * 2 - 1
        
        if in_channels > 3:
            dummy_feats = torch.zeros(N, in_channels - 3, device=device)
            dummy_inp = torch.cat([dummy_coords, dummy_feats], dim=1)
        else:
            dummy_inp = dummy_coords

        z_key = encode(dummy_grid, dummy_batch, depth, order="z")
        z_scores = (z_key.float() - z_key.float().min()) / (z_key.float().max() - z_key.float().min() + 1e-6)
        z_scores = z_scores * 0.98 + 0.01
        target_logit = torch.log(z_scores / (1 - z_scores)).unsqueeze(1)

        with torch.no_grad():
            feat = dummy_inp
            for layer in self.mlp[:-1]:
                feat = layer(feat)
            
            feat_mean = feat.mean(dim=0)
            target_mean = target_logit.mean(dim=0)
            feat_c = feat - feat_mean
            target_c = target_logit - target_mean
            
            I = torch.eye(feat_c.shape[1], device=device) * 1e-3
            weight = torch.linalg.solve(feat_c.T @ feat_c + I, feat_c.T @ target_c).T
            bias = target_mean - weight @ feat_mean

            last_layer = self.mlp[-1]
            out_dim = last_layer.weight.shape[0]
            last_layer.weight.copy_(weight.repeat(out_dim, 1))
            last_layer.bias.copy_(bias.repeat(out_dim))

    def forward(self, point):
        # 1. Prepare Input (Handle NaNs early)
        if point.feat is not None:
            inp = torch.cat([point.coord, point.feat], dim=1)
        else:
            inp = point.coord

        inp = torch.nan_to_num(inp, nan=0.0)
        inp = torch.clamp(inp, -10.0, 10.0)

        # 2. Predict Scores
        logits = self.mlp(inp)
        scores = torch.sigmoid(logits)
        scores = torch.clamp(scores, min=1e-5, max=1-1e-5)

        # 3. Generate Orders
        # Add batch offset to sort within batches
        batch_offset = point.batch.unsqueeze(1) * 2.0
        scores_for_sort = scores + batch_offset 

        orders_list = []
        inverses_list = []
        
        scores_t = scores_for_sort.transpose(0, 1)

        for i in range(scores_t.shape[0]):
            # STABILITY FIX: Add Noise to break ties
            noise = torch.rand_like(scores_t[i]) * 1e-6
            
            order = torch.argsort((scores_t[i] + noise).detach(), stable=True)
            inverse = torch.zeros_like(order)
            inverse[order] = torch.arange(len(order), device=order.device)
            orders_list.append(order)
            inverses_list.append(inverse)

        return scores, torch.stack(orders_list), torch.stack(inverses_list)

    def compute_loss(self, scores, coords, batch_ids, offset):
        return self.ss_loss(scores, coords, batch_ids, offset)


class SelfSupervisedOrderingLoss(nn.Module):
    def __init__(self, ordering_k=16, contrastive_k_near=16, contrastive_k_far=32, 
                 temp_locality=0.1, temp_contrastive=0.5):
        super().__init__()
        self.ordering_k = ordering_k
        self.contrastive_k_near = contrastive_k_near
        self.contrastive_k_far = contrastive_k_far
        self.temp_locality = temp_locality
        self.temp_contrastive = temp_contrastive
    
    def forward(self, scores, coords, batch_ids, offset=None):
        if scores.dim() > 1 and scores.shape[1] > 1:
            scores_mean = scores.mean(dim=1, keepdim=True)
        else:
            scores_mean = scores.view(-1, 1)
        
        scores_mean = torch.nan_to_num(scores_mean, nan=0.5)
        
        if offset is None:
            batch_ids_long = batch_ids.long()
            _, counts = torch.unique_consecutive(batch_ids_long, return_counts=True)
            offset = torch.cumsum(counts, dim=0).int()
        else:
            offset = offset.int()
        
        loss_locality = self._compute_locality_loss(scores_mean, coords, offset)
        loss_contrastive = self._compute_contrastive_loss(scores_mean, coords, offset)
        loss_distribution = self._compute_distribution_loss(scores_mean, batch_ids)
        
        if torch.isnan(loss_locality): loss_locality = torch.tensor(0.0, device=scores.device)
        if torch.isnan(loss_contrastive): loss_contrastive = torch.tensor(0.0, device=scores.device)
        if torch.isnan(loss_distribution): loss_distribution = torch.tensor(0.0, device=scores.device)

        total_loss = (1.0 * loss_locality + 
                      0.5 * loss_contrastive + 
                      1.0 * loss_distribution)
        
        return total_loss, {
            "locality": loss_locality.item(),
            "contrastive": loss_contrastive.item(),
            "distribution": loss_distribution.item()
        }
    
    def _compute_locality_loss(self, scores, coords, offset):
        N = scores.shape[0]
        if N < 2: return torch.tensor(0.0, device=scores.device)
        if torch.isnan(coords).any(): return torch.tensor(0.0, device=scores.device)

        k = min(self.ordering_k, scores.shape[0] - 1)
        idx = pointops.knn_query(k, coords.contiguous(), offset)[0].long()
        idx = torch.clamp(idx, min=0, max=scores.shape[0] - 1)
        
        neighbor_scores = scores[idx] 
        loss = F.mse_loss(scores.unsqueeze(1).expand_as(neighbor_scores), neighbor_scores)
        return loss

    def _compute_contrastive_loss(self, scores, coords, offset):
        if scores.shape[0] == 0: return torch.tensor(0.0, device=scores.device)
        var = torch.var(scores)
        return torch.relu(0.08 - var + 1e-6) 

    def _compute_distribution_loss(self, scores, batch_ids):
        batch_ids_long = batch_ids.long()
        num_batches = batch_ids_long.max().item() + 1
        total_loss = 0.0
        count = 0
        
        for b in range(num_batches):
            mask = (batch_ids_long == b)
            if mask.sum() < 2: continue
            
            scores_b = scores[mask].flatten()
            M = scores_b.shape[0]
            
            sorted_scores, _ = torch.sort(scores_b)
            target = torch.linspace(0, 1, M, device=scores.device)
            loss_b = F.l1_loss(sorted_scores, target)
            total_loss += loss_b
            count += 1
            
        return total_loss / max(count, 1)


@MODELS.register_module("OPTNet")
class OPTNet(PointTransformerV3):
    def __init__(self,
                 in_channels=6,
                 order=("z", "z-trans", "hilbert", "hilbert-trans"),
                 ordering_loss_weight=1.0,
                 ordering_k=16,
                 warmup_epoch=0,
                 contrastive_k_far=32,
                 contrastive_k_near=None,
                 contrastive_temp=0.5,
                 contrastive_weight=0.5,
                 enable_score_concat=True, # New flag to enable gradient flow
                 **kwargs):
        
        if isinstance(order, (list, tuple)):
            single_order = [order[0]]
        else:
            single_order = [order]
            
        self.enable_score_concat = enable_score_concat
        
        # [FEATURE CONCATENATION]: 
        # If enabled, we append the 1-dim score to the input features.
        # We must tell the backbone (PointTransformerV3) to expect 1 extra channel.
        backbone_in_channels = in_channels + 1 if enable_score_concat else in_channels
            
        super().__init__(in_channels=backbone_in_channels, order=single_order, **kwargs)
        
        self.ordering_loss_weight = ordering_loss_weight
        self.warmup_epoch = warmup_epoch
        
        if contrastive_k_near is None:
            contrastive_k_near = ordering_k

        # Sorter input is strictly coordinate + original features (not concatenated yet)
        self.sorter = PointSorter(
            in_channels=in_channels + 3,
            hidden_channels=64,
            num_orders=1,
            ordering_k=ordering_k,
            contrastive_k_near=contrastive_k_near,
            contrastive_k_far=contrastive_k_far,
            temp_contrastive=contrastive_temp
        )

    def forward(self, data_dict):
        point = Point(data_dict)
        current_epoch = data_dict.get("epoch", 0) if self.training else float('inf')
        
        need_standard_z = current_epoch < self.warmup_epoch

        point.serialization(order=self.order, 
                            shuffle_orders=self.shuffle_orders, 
                            compute_codes=True) 
        
        if "serialized_depth" not in point:
             point.serialized_depth = 10 
        
        # --- FIX: REMOVED EARLY SPARSIFY ---
        # point.sparsify() was here, causing the tensor to be built with old features
        
        # 2. Run OPTNet Sorter
        # scores: (N, 1) continuous values
        scores, learned_order, learned_inverse = self.sorter(point)
        
        # 3. Apply Order Strategy
        if not need_standard_z:
            point.serialized_order = learned_order
            point.serialized_inverse = learned_inverse
            primary_inverse = learned_inverse[0] 
            point.serialized_code = primary_inverse.unsqueeze(0).long() 
            
        # 4. Calculate Loss
        if self.training and self.ordering_loss_weight > 0:
            loss_ord, loss_dict = self.sorter.compute_loss(
                scores, 
                point.coord, 
                point.batch, 
                point.offset
            )
            loss_ord = torch.clamp(loss_ord, max=5.0) 
            point.ordering_loss = loss_ord * self.ordering_loss_weight

        # [FEATURE CONCATENATION]
        # Allow gradients to flow from backbone back to sorter via feature concatenation
        if self.enable_score_concat:
            if point.feat is None:
                point.feat = scores
            else:
                # Concatenate scores (N,1) to features (N, C)
                point.feat = torch.cat([point.feat, scores], dim=1)
        
        # --- FIX: CALL SPARSIFY HERE ---
        # Now point.feat has C+1 channels, so sparsify() will create 
        # a sparse tensor with C+1 channels, matching the backbone's expectation.
        point.sparsify() 
 
        # 5. Backbone Forward
        point = self.embedding(point) # Embedding layer now handles C+1 channels
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
        self.num_classes = num_classes

    def forward(self, data_dict):
        point = self.backbone(data_dict)
        seg_logits = self.seg_head(point.feat)
        
        if self.training:
            target = data_dict["segment"]
            
            if not target.dtype == torch.int64:
                target = target.long()

            valid_mask = (target >= 0) & (target < self.num_classes)
            if not valid_mask.all():
                target = target.clone()
                target[~valid_mask] = -1

            loss = self.criteria(seg_logits, target)
            return_dict = dict(seg_loss=loss)
            
            if "ordering_loss" in point:
                loss = loss + point["ordering_loss"]
                return_dict["ordering_loss"] = point["ordering_loss"]
            
            return_dict["loss"] = loss
            return return_dict
        
        elif "segment" in data_dict:
            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        
        return dict(seg_logits=seg_logits)