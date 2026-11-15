"""Upstream KPConv/KPFCNN adapted to Pointcept in one file.

Source: https://github.com/HuguesTHOMAS/KPConv-PyTorch
Upstream revision: d19c575d3fa9fcfd5a74845b5b27aac7e50472c7

Install: put this file in pointcept/models/kpconv_pointcept.py and import
    from .kpconv_pointcept import KPConvPointcept
in pointcept/models/__init__.py.

Example: use your existing Pointcept segmentor/criteria/data pipeline:
    model = dict(
        type="DefaultSegmentor", num_classes=8, backbone_out_channels=64,
        backbone=dict(type="KPConv-Pointcept-v1", in_channels=6,
                      head_channels=64, first_subsampling_dl=0.1),
        criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0,
                       ignore_index=-1)],
    )
Use the segmentor type available in your Pointcept version. in_channels must
match collected feat width. Coordinates and first_subsampling_dl use the same
units. This backbone returns full-resolution Point.feat [N,head_channels]; the
existing Pointcept head produces class logits and computes segmentation losses.
Input is a Point/dict with coord [N,3], feat [N,C], offset [B].

Preserved: upstream KPConv math, kernel optimization/init, rigid/deformable and
modulated operators, linear/constant/Gaussian influence, sum/closest aggregation,
residual bottlenecks, BatchNorm/bias + LeakyReLU, original KPFCNN encoder/decoder
block construction, nearest-neighbor upsampling and skip concatenation. The
upstream final class projection instead projects to backbone head_channels.

Adapter differences: builds a per-forward centroid voxel hierarchy in PyTorch;
uses Pointcept pointops capped kNN masked by radius rather than upstream C++
radius search; grid orientation is fixed rather than randomly rotated internally.
Finest input resolution is retained (no initial subsampling). Neighborhoods can
be truncated at neighbor_limits. Small packed clouds reduce k to the smallest
support cloud size; avoid highly uneven batch sizes. Kernel layouts are cached
in memory, not written as PLY. Supports 4..30 kernel points (upstream Lloyd
initialization for larger kernels is omitted). BatchNorm fixes upstream singleton
squeeze behavior. No dependence on the original repository at runtime.

Defaults are RIGID KPFCNN. For deformable blocks, supply architecture entries
such as 'resnetb_deformable' / 'resnetb_deformable_strided'. Then the training
wrapper MUST explicitly add backbone.deformation_loss() to its loss; Pointcept's
standard segmentation wrapper does not do so automatically. Keep rigid defaults
if using an unchanged standard wrapper. There are no dataset/training functions,
quaternion attention, SO(3) equivariance claims, or accuracy claims here.

Requires PyTorch, NumPy, Pointcept, and its CUDA pointops for GPU neighborhood
construction. CPU fallback is for small architecture/debug inputs only.
"""

# MIT License
# 
# Copyright (c) 2019 HuguesTHOMAS
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations
from functools import lru_cache
from types import SimpleNamespace
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.init import kaiming_uniform_
from torch.nn import functional as F
from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
try:
    import pointops
except ImportError:
    pointops = None

def _counts(offset):
    return torch.diff(F.pad(offset.long(), (1, 0)))


def _ranges(offset):
    ends = offset.tolist()
    return zip([0] + ends[:-1], ends)


@torch.no_grad()
def _knn(support, support_offset, query, query_offset, k, chunk_size=1024):
    k = min(k, int(_counts(support_offset).min().item()))
    if support.is_cuda:
        if pointops is None:
            raise RuntimeError('Install Pointcept pointops for CUDA execution.')
        idx, _ = pointops.knn_query(
            k, support.contiguous(), support_offset.int().contiguous(),
            query.contiguous(), query_offset.int().contiguous(),
        )
        if (idx < 0).any():
            raise RuntimeError('pointops returned invalid neighbors; check offsets.')
        return idx.long()
    indices = []
    for (s, e), (qs, qe) in zip(_ranges(support_offset), _ranges(query_offset)):
        for q in query[qs:qe].split(chunk_size):
            distances = torch.cdist(q.float(), support[s:e].float())
            indices.append(distances.topk(k, largest=False, sorted=True).indices + s)
    return torch.cat(indices)



def kernel_point_optimization_debug(radius, num_points, num_kernels=1, dimension=3, fixed='center', ratio=0.66, verbose=0):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """
    radius0 = 1
    diameter0 = 2
    moving_factor = 0.01
    continuous_moving_decay = 0.9995
    thresh = 1e-05
    clip = 0.05 * radius0
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while kernel_points.shape[0] < num_kernels * num_points:
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape((num_kernels, num_points, -1))
    if fixed == 'center':
        kernel_points[:, 0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3
    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    step = -1
    while step < 10000:
        step += 1
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 2) + 1e-06)
        inter_grads = np.sum(inter_grads, axis=1)
        circle_grads = 10 * kernel_points
        gradients = inter_grads + circle_grads
        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[step, :] = np.max(gradients_norms, axis=1)
        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)
        if fixed == 'center':
            moving_dists[:, 0] = 0
        if fixed == 'verticals':
            moving_dists[:, 0] = 0
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-06, -1)
        moving_factor *= continuous_moving_decay
    if step < 10000:
        saved_gradient_norms = saved_gradient_norms[:step + 1, :]
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])
    return (kernel_points * radius, saved_gradient_norms)

@lru_cache(maxsize=32)
def _base_kernel(num_kpoints, dimension, fixed):
    points, gradients = kernel_point_optimization_debug(
        1., num_kpoints, num_kernels=100, dimension=dimension, fixed=fixed, verbose=0)
    return points[np.argmin(gradients[-1])]


def load_kernels(radius, num_kpoints, dimension, fixed):
    # Upstream 3D default path: optimized layout + Gaussian jitter + Z rotation.
    points = _base_kernel(num_kpoints, dimension, fixed).copy()
    theta = np.random.rand() * 2 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    points += np.random.normal(scale=.01, size=points.shape)
    return ((points * radius) @ rotation).astype(np.float32)

def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig**2 + eps))


def closest_pool(x, inds):
    """
    Pools features from the closest neighbors. WARNING: this function assumes the neighbors are ordered.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] Only the first column is used for pooling
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get features for each pooling location [n2, d]
    return gather(x, inds[:, 0])


def max_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    max_features, _ = torch.max(pool_features, 1)
    return max_features


def global_average(x, batch_lengths):
    """
    Block performing a global average over batch pooling
    :param x: [N, D] input features
    :param batch_lengths: [B] list of batch lengths
    :return: [B, D] averaged features
    """

    # Loop over the clouds of the batch
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):

        # Average features for each batch cloud
        averaged_features.append(torch.mean(x[i0:i0 + length], dim=0))

        # Increment for next cloud
        i0 += length

    # Average features in each batch
    return torch.stack(averaged_features)


# ----------------------------------------------------------------------------------------------------------------------
#
#           KPConv class
#       \******************/
#


class KPConv(nn.Module):

    def __init__(self, kernel_size, p_dim, in_channels, out_channels, KP_extent, radius,
                 fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum',
                 deformable=False, modulated=False):
        """
        Initialize parameters for KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        :param deformable: choose deformable or not
        :param modulated: choose if kernel weights are modulated in addition to deformed
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated

        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        # Initiate weights for offsets
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(self.K,
                                      self.p_dim,
                                      self.in_channels,
                                      self.offset_dim,
                                      KP_extent,
                                      radius,
                                      fixed_kernel_points=fixed_kernel_points,
                                      KP_influence=KP_influence,
                                      aggregation_mode=aggregation_mode)
            self.offset_bias = Parameter(torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True)

        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def forward(self, q_pts, s_pts, neighb_inds0, x):

        ###################
        # Offset generation
        ###################

        if self.deformable:

            # Get offsets with a KPConv that only takes part of the features
            self.offset_features = self.offset_conv(q_pts, s_pts, neighb_inds0, x) + self.offset_bias

            if self.modulated:

                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features[:, :self.p_dim * self.K]
                unscaled_offsets = unscaled_offsets.view(-1, self.K, self.p_dim)

                # Get modulations
                modulations = 2 * torch.sigmoid(self.offset_features[:, self.p_dim * self.K:])

            else:

                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features.view(-1, self.K, self.p_dim)

                # No modulations
                modulations = None

            # Rescale offset for this layer
            offsets = unscaled_offsets * self.KP_extent

        else:
            offsets = None
            modulations = None

        ######################
        # Deformed convolution
        ######################

        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Use actual index of last point instead of -1
        neighb_inds = neighb_inds0.clone()
        neighb_inds[neighb_inds == -1] = s_pts.shape[0] - 1

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]

        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)

        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        if self.deformable:
            self.deformed_KP = offsets + self.kernel_points
            deformed_K_points = self.deformed_KP.unsqueeze(1)
        else:
            deformed_K_points = self.kernel_points

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        neighbors.unsqueeze_(2)
        differences = neighbors - deformed_K_points

        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences ** 2, dim=3)

        # Optimization by ignoring points outside a deformed KP range
        if self.deformable:

            # Save distances for loss
            self.min_d2, _ = torch.min(sq_distances, dim=1)

            # Boolean of the neighbors in range of a kernel point [n_points, n_neighbors]
            in_range = torch.any(sq_distances < self.KP_extent ** 2, dim=2).type(torch.int32)

            # New value of max neighbors
            new_max_neighb = torch.max(torch.sum(in_range, dim=1))

            # For each row of neighbors, indices of the ones that are in range [n_points, new_max_neighb]
            neighb_row_bool, neighb_row_inds = torch.topk(in_range, new_max_neighb.item(), dim=1)

            # Gather new neighbor indices [n_points, new_max_neighb]
            new_neighb_inds = neighb_inds.gather(1, neighb_row_inds, sparse_grad=False)

            # Gather new distances to KP [n_points, new_max_neighb, n_kpoints]
            neighb_row_inds.unsqueeze_(2)
            neighb_row_inds = neighb_row_inds.expand(-1, -1, self.K)
            sq_distances = sq_distances.gather(1, neighb_row_inds, sparse_grad=False)

            # New shadow neighbors have to point to the last shadow point
            new_neighb_inds *= neighb_row_bool
            new_neighb_inds -= (neighb_row_bool.type(torch.int64) - 1) * int(s_pts.shape[0] - 1)
        else:
            new_neighb_inds = neighb_inds

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 1, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=2)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)

        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, new_neighb_inds)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = torch.matmul(all_weights, neighb_x)

        # Apply modulations
        if self.deformable and self.modulated:
            weighted_features *= modulations.unsqueeze(2)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = weighted_features.permute((1, 0, 2))
        kernel_outputs = torch.matmul(weighted_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        return torch.sum(kernel_outputs, dim=0)

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Complex blocks
#       \********************/
#

def block_decider(block_name,
                  radius,
                  in_dim,
                  out_dim,
                  layer_ind,
                  config):

    if block_name == 'unary':
        return UnaryBlock(in_dim, out_dim, config.use_batch_norm, config.batch_norm_momentum)

    elif block_name in ['simple',
                        'simple_deformable',
                        'simple_invariant',
                        'simple_equivariant',
                        'simple_strided',
                        'simple_deformable_strided',
                        'simple_invariant_strided',
                        'simple_equivariant_strided']:
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name in ['resnetb',
                        'resnetb_invariant',
                        'resnetb_equivariant',
                        'resnetb_deformable',
                        'resnetb_strided',
                        'resnetb_deformable_strided',
                        'resnetb_equivariant_strided',
                        'resnetb_invariant_strided']:
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)

    elif block_name == 'global_average':
        return GlobalAverageBlock()

    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)


class BatchNormBlock(nn.Module):

    def __init__(self, in_dim, use_bn, bn_momentum):
        """
        Initialize a batch normalization block. If network does not use batch normalization, replace with biases.
        :param in_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            self.batch_norm = nn.BatchNorm1d(in_dim, momentum=bn_momentum)
            #self.batch_norm = nn.InstanceNorm1d(in_dim, momentum=bn_momentum)
        else:
            self.bias = Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.use_bn:

            x = x.unsqueeze(2)
            x = x.transpose(0, 2)
            x = self.batch_norm(x)
            x = x.transpose(0, 2)
            return x.squeeze(-1)
        else:
            return x + self.bias

    def __repr__(self):
        return 'BatchNormBlock(in_feat: {:d}, momentum: {:.3f}, only_bias: {:s})'.format(self.in_dim,
                                                                                         self.bn_momentum,
                                                                                         str(not self.use_bn))


class UnaryBlock(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """

        super(UnaryBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        return

    def forward(self, x, batch=None):
        x = self.mlp(x)
        x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(self.in_dim,
                                                                                        self.out_dim,
                                                                                        str(self.use_bn),
                                                                                        str(not self.no_relu))


class SimpleBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a simple convolution block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(SimpleBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Define the KPConv class
        self.KPConv = KPConv(config.num_kernel_points,
                             config.in_points_dim,
                             in_dim,
                             out_dim // 2,
                             current_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated)

        # Other opperations
        self.batch_norm = BatchNormBlock(out_dim // 2, self.use_bn, self.bn_momentum)
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, x, batch):

        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]

        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        return self.leaky_relu(self.batch_norm(x))


class ResnetBottleneckBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a resnet bottleneck block.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(ResnetBottleneckBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim

        # First downscaling mlp
        if in_dim != out_dim // 4:
            self.unary1 = UnaryBlock(in_dim, out_dim // 4, self.use_bn, self.bn_momentum)
        else:
            self.unary1 = nn.Identity()

        # KPConv block
        self.KPConv = KPConv(config.num_kernel_points,
                             config.in_points_dim,
                             out_dim // 4,
                             out_dim // 4,
                             current_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated)
        self.batch_norm_conv = BatchNormBlock(out_dim // 4, self.use_bn, self.bn_momentum)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(out_dim // 4, out_dim, self.use_bn, self.bn_momentum, no_relu=True)

        # Shortcut optional mpl
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim, out_dim, self.use_bn, self.bn_momentum, no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(self, features, batch):

        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]

        # First downscaling mlp
        x = self.unary1(features)

        # Convolution
        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        x = self.leaky_relu(self.batch_norm_conv(x))

        # Second upscaling mlp
        x = self.unary2(x)

        # Shortcut
        if 'strided' in self.block_name:
            shortcut = max_pool(features, neighb_inds)
        else:
            shortcut = features
        shortcut = self.unary_shortcut(shortcut)

        return self.leaky_relu(x + shortcut)


class GlobalAverageBlock(nn.Module):

    def __init__(self):
        """
        Initialize a global average block with its ReLU and BatchNorm.
        """
        super(GlobalAverageBlock, self).__init__()
        return

    def forward(self, x, batch):
        return global_average(x, batch.lengths[-1])


class NearestUpsampleBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        """
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return closest_pool(x, batch.upsamples[self.layer_ind - 1])

    def __repr__(self):
        return 'NearestUpsampleBlock(layer: {:d} -> {:d})'.format(self.layer_ind,
                                                                  self.layer_ind - 1)


class MaxPoolBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a max pooling block with its ReLU and BatchNorm.
        """
        super(MaxPoolBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return max_pool(x, batch.pools[self.layer_ind + 1])


def p2p_fitting_regularizer(net):

    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)



class _KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config):
        super(_KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = config.head_channels

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0, no_relu=True)

    def forward(self, batch, config):

        # Get input features
        x = batch.features

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x


_DEFAULT_ARCHITECTURE = (
    'simple', 'resnetb', 'resnetb_strided',
    'resnetb', 'resnetb', 'resnetb_strided',
    'resnetb', 'resnetb', 'resnetb_strided',
    'resnetb', 'resnetb', 'resnetb_strided',
    'resnetb', 'resnetb',
    'nearest_upsample', 'unary',
    'nearest_upsample', 'unary',
    'nearest_upsample', 'unary',
    'nearest_upsample', 'unary',
)


@torch.no_grad()
def _voxel_centroids(coord, offset, size):
    clouds, counts = [], []
    for start, end in _ranges(offset):
        pts = coord[start:end]
        # Translation before quantization improves georeferenced precision.
        keys = torch.floor((pts - pts.amin(0)) / size).long()
        _, inverse, count = torch.unique(keys, dim=0, sorted=True,
                                        return_inverse=True, return_counts=True)
        centers = pts.new_zeros((len(count), 3))
        centers.index_add_(0, inverse, pts)
        clouds.append(centers / count[:, None])
        counts.append(len(count))
    new_offset = torch.tensor(counts, device=offset.device, dtype=offset.dtype).cumsum(0)
    return torch.cat(clouds), new_offset


@torch.no_grad()
def _radius_neighbors(support, support_offset, query, query_offset, radius, limit):
    idx = _knn(support, support_offset, query, query_offset, limit)
    distance2 = (support[idx] - query[:, None]).square().sum(-1)
    return idx.masked_fill(distance2 > radius ** 2, len(support))


@MODELS.register_module('KPConv-Pointcept-v1')
class KPConvPointcept(nn.Module):
    def __init__(
        self, in_channels=6, head_channels=64, first_features_dim=128,
        first_subsampling_dl=.1, conv_radius=2.5, deform_radius=5.,
        num_kernel_points=15, KP_extent=1.2, KP_influence='linear',
        aggregation_mode='sum', fixed_kernel_points='center', modulated=False,
        use_batch_norm=True, batch_norm_momentum=.02,
        architecture=None, neighbor_limits=(32, 32, 32, 32, 32),
        deform_fitting_power=1., repulse_extent=1.2,
    ):
        super().__init__()
        architecture = tuple(_DEFAULT_ARCHITECTURE if architecture is None else architecture)
        allowed = {'simple', 'simple_deformable', 'resnetb', 'resnetb_deformable',
                   'resnetb_strided', 'resnetb_deformable_strided',
                   'nearest_upsample', 'unary'}
        if not architecture or any(b not in allowed for b in architecture):
            raise ValueError(f'architecture must contain only {sorted(allowed)}')
        if 'nearest_upsample' not in architecture:
            raise ValueError('KPFCNN requires a decoder with nearest_upsample blocks.')
        split = architecture.index('nearest_upsample')
        enc, dec = architecture[:split], architecture[split:]
        levels = 1 + sum('strided' in b for b in enc)
        if sum(b == 'nearest_upsample' for b in dec) != levels - 1:
            raise ValueError('Decoder must have one upsample for every encoder strided block.')
        if (any('strided' in b or b.startswith('simple') for b in dec)
                or dec[-1] == 'nearest_upsample'
                or any(a == b == 'nearest_upsample' for a, b in zip(dec, dec[1:]))):
            raise ValueError('Decoder must refine each upsample with unary/resnetb blocks.')
        if len(neighbor_limits) != levels or any(int(k) != k or k < 1 for k in neighbor_limits):
            raise ValueError('neighbor_limits needs one positive integer per encoder resolution.')
        if any(int(v) != v or v < 1 for v in (in_channels, head_channels, first_features_dim)):
            raise ValueError('Channel dimensions must be positive integers.')
        if first_features_dim < 4 or first_features_dim % 4:
            raise ValueError('first_features_dim must be a positive multiple of four.')
        if not 4 <= num_kernel_points <= 30 or int(num_kernel_points) != num_kernel_points:
            raise ValueError('num_kernel_points must be an integer in 4..30.')
        if fixed_kernel_points not in ('none', 'center', 'verticals'):
            raise ValueError('fixed_kernel_points must be none, center, or verticals.')
        if KP_influence not in ('constant', 'linear', 'gaussian') or aggregation_mode not in ('sum', 'closest'):
            raise ValueError('Unsupported KP influence or aggregation mode.')
        if any(not math.isfinite(v) or v <= 0 for v in
               (first_subsampling_dl, conv_radius, deform_radius, KP_extent, repulse_extent)):
            raise ValueError('Grid size, radii, extent, repulsion distance must be finite and positive.')
        if not math.isfinite(deform_fitting_power) or deform_fitting_power < 0:
            raise ValueError('deform_fitting_power must be finite and nonnegative.')
        self.in_channels = in_channels
        self.out_channels = self.head_channels = head_channels
        self.neighbor_limits = tuple(neighbor_limits)
        self.config = SimpleNamespace(
            architecture=architecture, in_features_dim=in_channels, in_points_dim=3,
            first_features_dim=first_features_dim, head_channels=head_channels,
            first_subsampling_dl=first_subsampling_dl, conv_radius=conv_radius,
            deform_radius=deform_radius, num_kernel_points=num_kernel_points,
            KP_extent=KP_extent, KP_influence=KP_influence,
            aggregation_mode=aggregation_mode, fixed_kernel_points=fixed_kernel_points,
            modulated=modulated, use_batch_norm=use_batch_norm,
            batch_norm_momentum=batch_norm_momentum,
        )
        self.network = _KPFCNN(self.config)
        self.network.deform_fitting_power = deform_fitting_power
        self.network.repulse_extent = repulse_extent
        self.network.l1 = nn.L1Loss()
        # Separate self-convolution and strided-convolution search radii, as upstream.
        self.layer_deformable = [False] * levels
        self.pool_deformable = [False] * (levels - 1)
        layer = 0
        for name in enc:
            if 'strided' in name:
                self.pool_deformable[layer] = 'deformable' in name
                layer += 1
            elif 'deformable' in name:
                self.layer_deformable[layer] = True
        layer = levels - 1
        for name in dec:
            if name == 'nearest_upsample':
                layer -= 1
            elif 'deformable' in name:
                self.layer_deformable[layer] = True

    def deformation_loss(self):
        """Call after forward and add to the training loss for deformable configs."""
        if not any(isinstance(m, KPConv) and m.deformable for m in self.network.modules()):
            return next(self.parameters()).sum() * 0.
        if any(isinstance(m, KPConv) and m.deformable and m.min_d2 is None
               for m in self.network.modules()):
            raise RuntimeError('Call forward before deformation_loss().')
        return p2p_fitting_regularizer(self.network)

    def _validate(self, point):
        coord, feat, offset = point.coord, point.feat, point.offset
        if coord.ndim != 2 or coord.shape[1] != 3 or not len(coord):
            raise ValueError('coord must be nonempty [N,3].')
        if feat.ndim != 2 or feat.shape != (len(coord), self.in_channels):
            raise ValueError(f'feat must be [N,{self.in_channels}].')
        if offset.ndim != 1 or not len(offset) or offset.dtype not in (torch.int32, torch.int64):
            raise ValueError('offset must be a nonempty cumulative integer vector.')
        if offset[-1].item() != len(coord) or (_counts(offset) <= 0).any():
            raise ValueError('offset must describe nonempty clouds and end at N.')
        if coord.device != feat.device or coord.device != offset.device:
            raise ValueError('coord, feat and offset must share a device.')

    def _hierarchy(self, point):
        config = self.config
        points, offsets = [point.coord.float()], [point.offset]
        for level in range(1, len(self.neighbor_limits)):
            coord, offset = _voxel_centroids(points[-1], offsets[-1],
                                             config.first_subsampling_dl * 2 ** level)
            points.append(coord)
            offsets.append(offset)
        neighbors, pools, upsamples = [], [], []
        for level, (coord, offset, limit) in enumerate(zip(points, offsets, self.neighbor_limits)):
            dl = config.first_subsampling_dl * 2 ** level
            radius = dl * (config.deform_radius if self.layer_deformable[level] else config.conv_radius)
            neighbors.append(_radius_neighbors(coord, offset, coord, offset, radius, limit))
            if level + 1 < len(points):
                pool_radius = dl * (config.deform_radius if self.pool_deformable[level] else config.conv_radius)
                pools.append(_radius_neighbors(coord, offset, points[level + 1], offsets[level + 1], pool_radius, limit))
                upsamples.append(_radius_neighbors(points[level + 1], offsets[level + 1],
                                                    coord, offset, 2 * pool_radius, 1))
        return SimpleNamespace(points=points, neighbors=neighbors, pools=pools,
                               upsamples=upsamples, features=point.feat)

    def forward(self, data_dict):
        point = Point(dict(data_dict))
        self._validate(point)
        batch = self._hierarchy(point)
        point.feat = self.network(batch, self.config)
        return point
