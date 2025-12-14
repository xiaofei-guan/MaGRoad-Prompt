import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Type, Optional, List
import math


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=(224,224), patch_size=(16,16), in_chans=3, embed_dim=768, 
                 norm_layer=None, flatten=True):
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        if self.cpu_mode:
            from isegm.utils.cython import get_dist_maps # This will cause an import error if isegm is not available
            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            # This part requires the isegm.utils.cython.get_dist_maps, which might not be available
            # For now, let's assume it's available or raise a NotImplementedError if we hit this path without it.
            if hasattr(self, '_get_dist_maps'):
                coords = []
                for i in range(batchsize):
                    norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                    coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols,
                                                      norm_delimeter))
                coords = torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
            else:
                raise NotImplementedError("CPU mode for DistMaps requires isegm.utils.cython.get_dist_maps")
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)

            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array, indexing='ij') # Added indexing='ij' for torch 1.10+
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)

            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e6

            coords = coords.view(-1, num_points, 1, rows, cols)
            coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
            # The following line might be an issue if num_points is 0, but num_points is derived from points.shape[1]//2
            # which should be > 0 if points is not empty. 
            # If points can be empty leading to num_points=0, this view could fail.
            # However, typically, prompts (points) are expected.
            coords = coords.view(-1, 2, rows, cols) # Original code might have a bug here if num_points=0, leading to -1 in view for bs.
                                                 # Assuming num_points will be > 0 based on context.
                                                 # Or if the intention is to have num_masks * 2 channels, then it should be: 
                                                 # coords = coords.view(batchsize, num_masks*2, 1, rows, cols) or similar, 
                                                 # depending on how num_masks is defined (not explicit here).
                                                 # Sticking to original for now as its specific logic isn't fully clear without SegMaskModel context.

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()

        return coords

    def forward(self, B, H, W, coords): # coords: [B, N, 2] [y, x]
        return self.get_coord_features(coords, B, H, W)


class _LoRA_qkv(nn.Module):
    """Applies LoRA modification to QKV layer."""
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = F.linear(x, self.weight, self.bias)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        # Add LoRA adjustments only to Q and V
        qkv[:, :, :self.dim] += new_q   # Adjust Q
        qkv[:, :, -self.dim:] += new_v  # Adjust V
        return qkv

class BilinearSampler(nn.Module):
    def __init__(self, image_size):
        super(BilinearSampler, self).__init__()
        self.image_size = image_size

    def forward(self, feature_maps, sample_points):
        """
        Args:
            feature_maps (Tensor): The input feature tensor of shape [B, D, H, W].
            sample_points (Tensor): The 2D sample points of shape [B, N_points, 2],
                                    each point in the range [0, image_size], format (x, y).
        Returns:
            Tensor: Sampled feature vectors of shape [B, N_points, D].
        """
        B, D, H, W = feature_maps.shape
        _, N_points, _ = sample_points.shape
        # normalize cooridinates to (-1, 1) for grid_sample
        sample_points = (sample_points / self.image_size) * 2.0 - 1.0
        
        # sample_points from [B, N_points, 2] to [B, N_points, 1, 2] for grid_sample
        sample_points = sample_points.unsqueeze(2)
        
        # Use grid_sample for bilinear sampling. Align_corners set to False to use -1 to 1 grid space.
        # [B, D, N_points, 1]
        sampled_features = F.grid_sample(feature_maps, sample_points, mode='bilinear', align_corners=False)
        # sampled_features is [B, N_points, D]
        sampled_features = sampled_features.squeeze(dim=-1).permute(0, 2, 1)
        return sampled_features


class TopoNet(nn.Module):
    def __init__(self, config, feature_dim):
        super(TopoNet, self).__init__()
        self.config = config

        self.hidden_dim = config.get("TOPONET_HIDDEN_DIM", 128)
        self.heads = config.get("TOPONET_HEADS", 4)
        self.num_attn_layers = config.get("TOPONET_ATTN_LAYERS", 3)

        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim) # (256, 128)
        self.pair_proj = nn.Linear(2 * self.hidden_dim + 2, self.hidden_dim) # (256 + 2, 128)

        # Create Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim, # Often 4*d_model, but can be d_model
            dropout=0.1,
            activation='relu',
            batch_first=True  # Input format is [batch size, sequence length, features]
        )
        
        # Stack the Transformer Encoder Layers
        if self.config.TOPONET_VERSION != 'no_transformer':
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attn_layers)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, points, point_features, pairs, pairs_valid, mask_logits=None):
        # points: [B, N_points, 2]
        # point_features: [B, N_points, D]
        # pairs: [B, N_samples, N_pairs, 2]
        # pairs_valid: [B, N_samples, N_pairs]
        # [bs, N_points, 256] -> [bs, N_points, 128]
        point_features = F.relu(self.feature_proj(point_features))
        # gathers pairs
        batch_size, n_samples, n_pairs, _ = pairs.shape # bs, 512, 16, 2
        pairs = pairs.view(batch_size, -1, 2) # [B, N_samples * N_pairs, 2]
        # [B, N_samples * N_pairs]
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, n_samples * n_pairs)
        # Use advanced indexing to fetch the corresponding feature vectors
        # [B, N_samples * N_pairs, D]
        src_features = point_features[batch_indices, pairs[:, :, 0]] # [B, 255, 128] index:[B, 512*16], [B, 512*16] ->result:[16, 512*16, 128]
        tgt_features = point_features[batch_indices, pairs[:, :, 1]]
        # [B, N_samples * N_pairs, 2]
        src_points = points[batch_indices, pairs[:, :, 0]]
        tgt_points = points[batch_indices, pairs[:, :, 1]]
        offset = tgt_points - src_points

        ## ablation study
        # [B, N_samples * N_pairs, 2D + 2]
        if self.config.TOPONET_VERSION == 'no_tgt_features':
            pair_features = torch.concat([src_features, torch.zeros_like(tgt_features), offset], dim=2)
        if self.config.TOPONET_VERSION == 'no_offset':
            pair_features = torch.concat([src_features, tgt_features, torch.zeros_like(offset)], dim=2)
        else:
            pair_features = torch.concat([src_features, tgt_features, offset], dim=2) # [16, 8192, 256 + 2]

        # [B, N_samples * N_pairs, D]
        pair_features = F.relu(self.pair_proj(pair_features)) # [16, 8192, 256 + 2] -> [16, 8192, 128]
        
        # attn applies within each local graph sample
        pair_features = pair_features.view(batch_size * n_samples, n_pairs, -1)
        # valid->not a padding
        pairs_valid = pairs_valid.view(batch_size * n_samples, n_pairs)

        # # [B * N_samples, 1]
        # #### flips mask for all-invalid pairs to prevent NaN
        all_invalid_pair_mask = torch.eq(torch.sum(pairs_valid, dim=-1), 0).unsqueeze(-1) # True means all pairs are invalid
        pairs_valid = torch.logical_or(pairs_valid, all_invalid_pair_mask)
        padding_mask = ~pairs_valid
        
        ## ablation study
        if self.config.TOPONET_VERSION != 'no_transformer':
            pair_features = self.transformer_encoder(pair_features, src_key_padding_mask=padding_mask) # input shape [S, B, D]
        
        ## Seems like at inference time, the returned n_pairs heres might be less - it's the
        # max num of valid pairs across all samples in the batch
        _, n_pairs, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, n_samples, n_pairs, -1)

        # [B, N_samples, N_pairs, 1]
        logits = self.output_proj(pair_features)

        scores = torch.sigmoid(logits)

        return logits, scores


def fourier_encode_angle(angle: torch.Tensor, num_bases: int = 4) -> torch.Tensor:
    """
    angle: [*, 1] or [*], radians
    return: [*, 2 * num_bases] with [sin(mθ), cos(mθ)]_{m=1..num_bases}
    """
    if angle.dim() == 1:
        angle = angle.unsqueeze(-1)
    outs = []
    for m in range(1, num_bases + 1):
        outs.append(torch.sin(m * angle))
        outs.append(torch.cos(m * angle))
    return torch.cat(outs, dim=-1)


def softmin(t: torch.Tensor, dim: int = -1, tau: float = 10.0) -> torch.Tensor:
    # softmin(x) = -1/tau * log sum exp(-tau x)
    return -torch.logsumexp(-tau * t, dim=dim) / tau


def normalize_grid(points_xy: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    points_xy: [B, E, S, 2] in [0, image_size]
    return: normalized to [-1, 1] for grid_sample
    """
    return (points_xy / image_size) * 2.0 - 1.0


def sample_line_points(src: torch.Tensor, dst: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    src: [B, E, 2], dst: [B, E, 2]
    return: [B, E, S, 2] where S=num_samples, linear interpolation points
    """
    B, E, _ = src.shape
    t = torch.linspace(0, 1, num_samples, device=src.device).view(1, 1, num_samples, 1)
    src_exp = src.unsqueeze(2)  # [B,E,1,2]
    dst_exp = dst.unsqueeze(2)  # [B,E,1,2]
    pts = src_exp + t * (dst_exp - src_exp)
    return pts  # [B,E,S,2]


class GeodesicPathExtractor(nn.Module):
    """
    Lightweight, differentiable path-aware feature extractor:
    - Sample equidistant points on the line segments from the road probability map in mask_logits
    - After multi-scale smoothing (avg pooling), repeat sampling
    - Statistical features: mean / std / softmin (softmin of (1-p), approximate the worst road)
    Return a vector of [B, E, F_path] (E is the number of edges S*K)
    """
    def __init__(self, image_size: int, num_samples: int = 32,
                 pool_kernel_sizes: Optional[List[int]] = None,
                 tau_softmin: float = 10.0):
        super().__init__()
        self.image_size = image_size
        self.num_samples = num_samples
        self.pool_kernel_sizes = pool_kernel_sizes or [1, 5, 11]  # 1=original image, then two smoothing
        self.tau = tau_softmin

    def forward(self, mask_logits: torch.Tensor,
                src_points: torch.Tensor,  # [B, E, 2]
                dst_points: torch.Tensor   # [B, E, 2]
                ) -> torch.Tensor:
        """
        mask_logits: [B, 2, H, W], road probability in channel 1
        src_points/dst_points: [B, E, 2] in [0, image_size]
        return: [B, E, F_path]
        """
        B, _, H, W = mask_logits.shape
        assert H == self.image_size and W == self.image_size, "mask size mismatch"
        road_prob = torch.sigmoid(mask_logits[:, 1:2])  # [B,1,H,W]

        # prepare sampling points
        pts = sample_line_points(src_points, dst_points, self.num_samples)  # [B, E, S, 2]
        pts_norm = normalize_grid(pts, self.image_size)  # [-1,1]
        grid = pts_norm.view(B, -1, 1, 2)  # [B, E*S, 1, 2]

        feats = []
        for ksz in self.pool_kernel_sizes:
            if ksz == 1:
                smoothed = road_prob
            else:
                pad = ksz // 2
                smoothed = F.avg_pool2d(road_prob, kernel_size=ksz, stride=1, padding=pad)

            # sampling: grid_sample input is [B,C,H,W] and [B, N, 1, 2]
            sampled = F.grid_sample(smoothed, grid, mode='bilinear', align_corners=False)  # [B,1,E*S,1]
            sampled = sampled.view(B, -1, self.num_samples)  # [B, E, S]

            # statistics
            mean_v = sampled.mean(dim=-1, keepdim=True)                         # [B,E,1]
            std_v  = sampled.std(dim=-1, unbiased=False, keepdim=True)          # [B,E,1]
            softmin_v = softmin(1.0 - sampled, dim=-1, tau=self.tau).unsqueeze(-1)  # [B,E,1]

            feats.extend([mean_v, std_v, softmin_v])

        path_feat = torch.cat(feats, dim=-1)  # [B, E, 3 * len(scales)]
        return path_feat


class BiasedSelfAttentionLayer(nn.Module):
    """
    Custom multi-head self-attention layer with "additive bias" + FFN
    Support batch-wise [B, L, L] shape bias matrix
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dk = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,  # [B,L], True means to mask the position
                bias: Optional[torch.Tensor] = None               # [B,L,L]
                ) -> torch.Tensor:
        """
        x: [B, L, H]
        """
        B, L, H = x.shape
        residual = x

        # Q,K,V
        q = self.q_proj(x).view(B, L, self.nhead, self.dk).transpose(1, 2)  # [B,h,L,dk]
        k = self.k_proj(x).view(B, L, self.nhead, self.dk).transpose(1, 2)  # [B,h,L,dk]
        v = self.v_proj(x).view(B, L, self.nhead, self.dk).transpose(1, 2)  # [B,h,L,dk]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # [B,h,L,L]

        if bias is not None:
            # expand to each head
            attn_scores = attn_scores + bias.unsqueeze(1)  # [B,1,L,L] -> [B,h,L,L]

        if key_padding_mask is not None:
            # mask invalid keys: set the corresponding column to -inf
            # key_padding_mask: [B, L], True=pad
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B,h,L,dk]
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        out = self.out_proj(out)
        x = self.norm1(residual + out)

        # FFN
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + x)
        return x


def build_edge_bias(src_xy: torch.Tensor,
                    dst_xy: torch.Tensor,
                    valid: torch.Tensor,
                    angle_sigma: float = math.pi / 4,
                    lambda_turn: float = 0.6,
                    lambda_compete: float = 0.2) -> torch.Tensor:
    """
    Construct the edge-edge attention bias matrix within a group of candidate edges from the same source point:
    - turn compatibility: smaller Δθ is more compatible (Gaussian kernel)
    - competitive prior: negative bias is applied to off-diagonal uniformly, encouraging sparse selection
    src_xy/dst_xy: [B, L, 2] (B here is actually B*S, L is N_pairs)
    valid: [B, L], True means valid
    return: bias [B, L, L]
    """
    B, L, _ = src_xy.shape
    offset = dst_xy - src_xy  # [B,L,2]
    angle = torch.atan2(offset[..., 1], offset[..., 0])  # [B,L]

    # pairwise Δθ, mapped to [-pi, pi]
    theta_i = angle.unsqueeze(-1)           # [B,L,1]
    theta_j = angle.unsqueeze(-2)           # [B,1,L]
    delta = theta_i - theta_j               # [B,L,L]
    delta = (delta + math.pi) % (2 * math.pi) - math.pi

    k_turn = torch.exp(- (delta ** 2) / (2 * (angle_sigma ** 2)))  # [B,L,L]
    bias_turn = lambda_turn * (k_turn - 0.5)                        # centered

    # competitive prior: negative bias is applied to off-diagonal uniformly, encouraging sparse selection
    eye = torch.eye(L, device=src_xy.device).unsqueeze(0)          # [1,L,L]
    off_diag = 1.0 - eye
    bias_comp = -lambda_compete * off_diag                          # [1,L,L] -> broadcast

    bias = bias_turn + bias_comp

    # for invalid edges, avoid positive bias with anyone: set to 0 (actually masked by key_padding_mask)
    v = valid.float()
    bias = bias * v.unsqueeze(-1) * v.unsqueeze(-2)

    # set the main diagonal to 0
    bias = bias * off_diag + 0.0 * eye
    return bias  # [B,L,L]


class MaGTopoNet(nn.Module):
    """
    Mask-aware Geodesic Line-Graph Transformer
    - Optional use of point_features (image features)
    - Use geometric encoding (offset/dist/angle-Fourier)
    - Use path features (multi-scale sampling statistics on road prob along straight lines)
    - Do self-attention with bias on the "candidate edge set of each sample"
    Interface kept consistent with TopoNet
    """
    def __init__(self, config, feature_dim: int,
                 use_point_features: bool = True,
                 use_path_features: bool = True,
                 use_edge_bias: bool = True):
        super().__init__()
        self.config = config
        self.hidden_dim = 256
        self.heads = 8
        self.num_layers = 4

        self.use_point_features = use_point_features
        self.use_path_features = use_path_features
        self.use_edge_bias = use_edge_bias

        # node feature projection
        if self.use_point_features:
            self.node_proj = nn.Linear(feature_dim, self.hidden_dim)

        # geometric feature encoding: dx,dy + dist + angle fourier(8) -> H/2
        geo_in_dim = 2 + 1 + 8
        self.geo_proj = nn.Linear(geo_in_dim, self.hidden_dim // 2)

        # path feature
        if self.use_path_features:
            # path_feat: 3 * len(scales), default=3*3=9
            self.path_extractor = GeodesicPathExtractor(
                image_size=config.PATCH_SIZE,
                num_samples=config.NUM_INTERPOLATIONS,
                pool_kernel_sizes=[1, 3, 5],
                tau_softmin=5.0,
            )
            self.path_proj = nn.Linear(9, self.hidden_dim // 2)

        # fused and project to H
        fused_in = 0
        if self.use_point_features:
            fused_in += self.hidden_dim * 2
        fused_in += self.hidden_dim // 2  # geo
        if self.use_path_features:
            fused_in += self.hidden_dim // 2

        self.edge_proj = nn.Linear(fused_in, self.hidden_dim)

        # Transformer encoder with biased
        self.layers = nn.ModuleList([
            BiasedSelfAttentionLayer(self.hidden_dim, self.heads, dim_ff=self.hidden_dim, dropout=0.10)
            for _ in range(self.num_layers)
        ])

        self.out = nn.Linear(self.hidden_dim, 1)

    def forward(self, points: torch.Tensor,              # [B, N_points, 2]
                point_features: torch.Tensor,            # [B, N_points, D]
                pairs: torch.Tensor,                     # [B, N_samples, N_pairs, 2]
                pairs_valid: torch.Tensor,               # [B, N_samples, N_pairs]
                mask_logits: Optional[torch.Tensor] = None  # [B, 2, H, W]
                ):
        B, S, K, _ = pairs.shape
        dev = points.device

        # 1) prepare indices
        pairs_flat = pairs.view(B, -1, 2)  # [B, S*K, 2]
        BE = S * K
        batch_idx = torch.arange(B, device=dev).view(-1, 1).expand(-1, BE)  # [B, S*K]

        # 2) node feature
        if self.use_point_features:
            node = F.gelu(self.node_proj(point_features))  # [B, N, H]
            src_idx = pairs_flat[:, :, 0]  # [B,BE]
            dst_idx = pairs_flat[:, :, 1]  # [B,BE]
            src_feat = node[batch_idx, src_idx]  # [B, BE, H]
            dst_feat = node[batch_idx, dst_idx]  # [B, BE, H]

        # 3) geometric feature
        src_xy = points[batch_idx, pairs_flat[:, :, 0]].float()  # [B,BE,2]
        dst_xy = points[batch_idx, pairs_flat[:, :, 1]].float()  # [B,BE,2]
        offset = dst_xy - src_xy  # [B,BE,2]
        dist = torch.norm(offset, dim=-1, keepdim=True)  # [B,BE,1]
        angle = torch.atan2(offset[..., 1], offset[..., 0]).unsqueeze(-1)  # [B,BE,1]
        angle_enc = fourier_encode_angle(angle, num_bases=4)  # [B,BE,8]
        # normalize offset and dist to align with angle_enc
        offset_norm = offset / mask_logits.shape[-1]
        dist_norm = dist / (math.sqrt(2) * mask_logits.shape[-1])
        geo_in = torch.cat([offset_norm, dist_norm, angle_enc], dim=-1)
        geo_feat = F.gelu(self.geo_proj(geo_in))  # [B,BE,H/2]

        # 4) path feature (differentiable sampling from mask_logits)
        if self.use_path_features:
            assert mask_logits is not None, "mask_logits is required when use_path_features=True"
            path_feat_raw = self.path_extractor(mask_logits, src_xy, dst_xy)  # [B,BE,9]
            path_feat = F.gelu(self.path_proj(path_feat_raw))                 # [B,BE,H/2]

        # 5) edge token fusion
        feats = [geo_feat]
        if self.use_point_features:
            feats = [src_feat, dst_feat] + feats
        if self.use_path_features:
            feats = feats + [path_feat]

        edge_tok = torch.cat(feats, dim=-1)     # [B,BE, fused_in]
        edge_tok = F.gelu(self.edge_proj(edge_tok))  # [B,BE,H]

        # 6) within-group (each sample) encoding: reshape to [B*S, K, H]
        x = edge_tok.view(B, S, K, -1).view(B * S, K, -1)        # [B*S, K, H]
        valid = pairs_valid.view(B * S, K)                       # [B*S, K]
        # at least one True per group, to prevent NaN from all invalid
        all_invalid = (valid.sum(dim=-1, keepdim=True) == 0)
        valid = torch.logical_or(valid, all_invalid)

        # 7) construct bias
        if self.use_edge_bias:
            src_xy_g = src_xy.view(B, S, K, 2).view(B * S, K, 2)
            dst_xy_g = dst_xy.view(B, S, K, 2).view(B * S, K, 2)
            bias = build_edge_bias(src_xy_g, dst_xy_g, valid,
                                   angle_sigma=math.pi / 6,
                                   lambda_turn=0.4,
                                   lambda_compete=0.2)          # [B*S,K,K]
        else:
            bias = None

        # 8) self-attention encoder with biased (multi-layer)
        key_padding_mask = ~valid  # True=pad
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, bias=bias)  # [B*S,K,H]

        # 9) output
        x = x.view(B, S, K, -1)
        logits = self.out(x)                     # [B,S,K,1]
        scores = torch.sigmoid(logits)
        return logits, scores


# Test code
if __name__ == "__main__":
    # Mock config for testing TopoNet and others
    from addict import Dict as AdDict # Renaming to avoid conflict if Dict is used elsewhere
    mock_config = AdDict()
    mock_config.TOPONET_VERSION = 'transformer' # or 'no_transformer', 'no_tgt_features', 'no_offset'
    mock_config.TOPONET_HIDDEN_DIM = 128
    mock_config.TOPONET_HEADS = 4
    mock_config.TOPONET_ATTN_LAYERS = 2
    mock_config.TOPONET_DROPOUT = 0.1

    # Define dimensions
    batch_size = 2
    n_points = 50       # Number of unique points in the graph for each batch item
    feature_dim = 256   # Dimension of input point features (e.g., from SAM encoder)
    n_samples = 32      # Number of source nodes sampled for topology prediction
    n_pairs_max = 16    # Max number of neighbors considered for each source node
    
    # Generate mock data
    # Padded points [B, N_points_padded, 2], assume N_points_padded = n_points for simplicity in mock
    points_tensor = (torch.rand(batch_size, n_points, 2) * 1024).float()
    # Padded point features [B, N_points_padded, D_feat]
    point_features_tensor = torch.rand(batch_size, n_points, feature_dim)
    
    # Pairs [B, N_samples, N_pairs_max, 2], indices into the points_tensor
    # For each of N_samples, we have a source node and N_pairs_max potential target nodes
    # Here, pairs[:, i, :, 0] would be the index of the i-th source node for all its pairs
    # pairs[:, i, j, 1] would be the index of the j-th target for the i-th source node
    pairs_tensor = torch.randint(0, n_points, (batch_size, n_samples, n_pairs_max, 2))
    # Ensure source node is consistent for its pairs within a sample, if that's the design
    # For TopoNet, it seems each pair (src, tgt) is independent for feature extraction then attention over N_pairs_max.
    # The original test code for GCNLinkNet in the file suggests pairs can be arbitrary valid edges.
    # Let's make it simple: for each sample, pick a source, then pick targets.
    source_indices = torch.randint(0, n_points, (batch_size, n_samples, 1)).expand(-1, -1, n_pairs_max)
    target_indices = torch.randint(0, n_points, (batch_size, n_samples, n_pairs_max))
    pairs_tensor = torch.stack((source_indices, target_indices), dim=-1)

    # Pairs valid mask [B, N_samples, N_pairs_max]
    pairs_valid_tensor = torch.rand(batch_size, n_samples, n_pairs_max) > 0.2 # Simulate some invalid pairs
    # Ensure at least one valid pair per sample if using transformer, to avoid issues with empty sequences for attention if not handled by padding_mask logic perfectly
    # pairs_valid_tensor[:, :, 0] = True 

    print(f"Input points shape: {points_tensor.shape}")
    print(f"Input point_features shape: {point_features_tensor.shape}")
    print(f"Input pairs shape: {pairs_tensor.shape}")
    print(f"Input pairs_valid shape: {pairs_valid_tensor.shape}")

    # Test TopoNet
    print("\nTesting TopoNet...")
    toponet_model = TopoNet(config=mock_config, feature_dim=feature_dim)
    toponet_model.train()
    t1 = time.time()
    for _ in range(5):
        topo_logits, topo_scores = toponet_model(points_tensor, point_features_tensor, pairs_tensor, pairs_valid_tensor)
    t2 = time.time()
    print(f"TopoNet Time taken for 5 runs: {t2 - t1:.4f} seconds")
    print(f"Output topo_logits shape: {topo_logits.shape}") # Expected: [B, N_samples, N_pairs_max, 1]
    print(f"Output topo_scores shape: {topo_scores.shape}") # Expected: [B, N_samples, N_pairs_max, 1]
    total_params_toponet = sum(p.numel() for p in toponet_model.parameters() if p.requires_grad)
    print(f"TopoNet Total trainable parameters: {total_params_toponet:,}")

    # Test BilinearSampler
    print("\nTesting BilinearSampler...")
    # For BilinearSampler, feature_maps would be [B, D_feat, H_feat, W_feat]
    # Assuming image_embeddings from SAM are [B, D_feat, 64, 64]
    # sample_points are graph_points [B, N_points_padded, 2] in pixel scale of feature map (0-63)
    feature_map_size = 64 
    mock_image_embeddings = torch.rand(batch_size, feature_dim, feature_map_size, feature_map_size)
    mock_graph_points = torch.rand(batch_size, n_points, 2) * (feature_map_size -1) # ensure coords are within [0, H-1] or [0, W-1]
    
    # The BilinearSampler was initialized with self.image_size in SegMaskModel.
    # This self.image_size from config.PATCH_SIZE is usually the original image input size (e.g., 1024).
    # However, the feature_maps it samples from are image_embeddings, which are smaller (e.g., 64x64 for ViT-B).
    # The normalization in BilinearSampler was `(sample_points / self.image_size) * 2.0 - 1.0`
    # This needs `self.image_size` to be the size of `feature_maps` for correct normalization.
    # Corrected BilinearSampler in this file assumes points are in pixel scale of the feature map.
    sampler = BilinearSampler(image_size=feature_map_size) 
    sampled_features = sampler(mock_image_embeddings, mock_graph_points)
    print(f"BilinearSampler input feature_maps shape: {mock_image_embeddings.shape}")
    print(f"BilinearSampler input sample_points shape: {mock_graph_points.shape}")
    print(f"BilinearSampler output sampled_features shape: {sampled_features.shape}") # Expected [B, N_points, D_feat]

    # Test DistMaps
    print("\nTesting DistMaps...")
    # points for DistMaps: [B, num_max_points * 2, 3] format (y, x, type: 1=pos, 0=neg, -1=pad)
    # type is not directly used by the GPU version in DistMaps, only coords and invalid_points logic
    num_max_prompts_per_type = 5
    dist_map_points = torch.full((batch_size, num_max_prompts_per_type * 2, 3), -1.0, dtype=torch.float32)
    # Add some positive points (y,x,1)
    dist_map_points[0, 0:2, 0] = torch.rand(2) * 1024 # y
    dist_map_points[0, 0:2, 1] = torch.rand(2) * 1024 # x
    dist_map_points[0, 0:2, 2] = 1.0
    # Add some negative points (y,x,0)
    dist_map_points[0, num_max_prompts_per_type:num_max_prompts_per_type+1, 0] = torch.rand(1) * 1024 # y
    dist_map_points[0, num_max_prompts_per_type:num_max_prompts_per_type+1, 1] = torch.rand(1) * 1024 # x
    dist_map_points[0, num_max_prompts_per_type:num_max_prompts_per_type+1, 2] = 0.0

    # x for DistMaps (dummy image tensor to get H, W)
    dummy_image_tensor = torch.rand(batch_size, 3, 1024, 1024)
    dist_mapper = DistMaps(norm_radius=10, spatial_scale=1.0, use_disks=True)
    try:
        coord_feats = dist_mapper(dummy_image_tensor, dist_map_points)
        print(f"DistMaps input points shape: {dist_map_points.shape}")
        print(f"DistMaps output coord_feats shape: {coord_feats.shape}") # Expected: [B, 2, H, W]
    except NotImplementedError as e:
        print(f"DistMaps CPU mode error (expected if isegm not installed): {e}")
    except Exception as e:
        print(f"DistMaps error: {e}") 