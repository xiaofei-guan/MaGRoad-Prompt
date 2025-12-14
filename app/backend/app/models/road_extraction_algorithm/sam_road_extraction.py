import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
import math
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import rtree
import time
import scipy
from collections import defaultdict

from ..segment_anything.modeling.image_encoder import ImageEncoderViT
from ..segment_anything.modeling.mask_decoder import MaskDecoderV2
from ..segment_anything.modeling.prompt_encoder import PromptEncoder
from ..segment_anything.modeling.transformer import TwoWayTransformer
from ..segment_anything.modeling.common import LayerNorm2d

from .module import (
    DistMaps, BilinearSampler, TopoNet, 
    PatchEmbed, MaGTopoNet
)
from .graph_extraction import extract_graph_points
from fvcore.nn import FlopCountAnalysis


class RoadExtractionModel(pl.LightningModule):
    """
    PyTorch Lightning module for training SAM for keypoint and road mask segmentation.
    Based on the segmentation part of SAMRoad.
    NOTE: This version is adapted for inference within the FastAPI backend.
          Training-specific parts like optimizer configuration, dataset loading for training,
          and detailed metric tracking/logging for epochs will be stubbed or removed if not directly usable.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config # This will be an addict.Dict loaded from YAML
        # self.save_hyperparameters(config) # Not saving hyperparameters in inference model

        self._validate_config()

        # --- Model Configuration ---
        encoder_embed_dim, encoder_depth, encoder_num_heads, encoder_global_attn_indexes = self._get_sam_config()
        prompt_embed_dim = 256
        image_size = config.PATCH_SIZE # Expect features to be precomputed for this size
        self.image_size = image_size
        vit_patch_size = 16 # Standard for SAM ViT
        image_embedding_size = image_size // vit_patch_size
        encoder_output_dim = prompt_embed_dim # Output of ImageEncoderViT neck

        # --- Buffers (Pixel mean/std for SAM if it were processing raw images) ---
        # For inference with precomputed embeddings, these are not directly used in forward pass
        # but are part of SAM's original structure if loading a full SAM checkpoint.
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

        # --- Image Encoder (Primarily for loading weights, actual encoding is precomputed) ---
        self.image_encoder = self._build_image_encoder(
            encoder_depth, encoder_embed_dim, image_size, encoder_num_heads,
            vit_patch_size, encoder_global_attn_indexes, prompt_embed_dim
        )

        # --- Prompt Encoder ---
        if not self.config.get('USE_SAM_DECODER', False): # Default to False if not specified
            self.dist_maps = DistMaps(norm_radius=config.DIST_MAP_NORM_R, spatial_scale=1.0, use_disks=config.USE_DISKS)
        else:
            self.dist_maps = None # Not needed if using SAM's prompt encoder path

        self.prompt_encoder = self._build_prompt_encoder(
            prompt_embed_dim, image_embedding_size, image_size
        )

        # --- Mask Decoder ---
        self.mask_decoder = self._build_mask_decoder(
            prompt_embed_dim, image_embedding_size, image_size, encoder_output_dim
        )

        # --- Topology components ---
        self.bilinear_sampler = BilinearSampler(image_size) # Samples from feature map
        
        toponet_type = self.config.get("TOPONET", "transformer")
        if toponet_type == "transformer":
            self.toponet = TopoNet(config, prompt_embed_dim)
        elif toponet_type == "magtoponet":
            self.toponet = MaGTopoNet(config, prompt_embed_dim,
                                      use_point_features=self.config.USE_POINT_FEATURES,
                                      use_path_features=self.config.USE_PATH_FEATURES,
                                      use_edge_bias=self.config.USE_EDGE_BIAS)
        else:
            raise ValueError(f"Invalid TOPONET type: {toponet_type}")

        # The main model checkpoint (containing weights for all parts including TopoNet)
        # will be loaded externally by the FastAPI app startup logic.

    def _validate_config(self):
        assert self.config.SAM_VERSION in {'vit_b', 'vit_l', 'vit_h'}, "Invalid SAM_VERSION"
        assert self.config.PATCH_SIZE > 0, "PATCH_SIZE must be positive"

    def _get_sam_config(self):
        if self.config.SAM_VERSION == 'vit_b':
            return 768, 12, 12, [2, 5, 8, 11]
        elif self.config.SAM_VERSION == 'vit_l':
            return 1024, 24, 16, [5, 11, 17, 23]
        elif self.config.SAM_VERSION == 'vit_h':
            return 1280, 32, 16, [7, 15, 23, 31]
        raise ValueError(f"Unknown SAM_VERSION: {self.config.SAM_VERSION}")

    def _build_image_encoder(self, depth, embed_dim, img_size, num_heads, patch_size, global_attn_indexes, out_chans):
        # print(f"Building SAM {self.config.SAM_VERSION} Encoder structure for weight loading")
        return ImageEncoderViT(
            depth=depth,
            embed_dim=embed_dim,
            img_size=img_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=num_heads,
            patch_size=patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=global_attn_indexes,
            window_size=14,
            out_chans=out_chans
        )
    
    def _build_prompt_encoder(self, prompt_embed_dim, image_embedding_size, image_size):
        if self.config.get('USE_SAM_DECODER', False):
            # print("Building SAM Prompt Encoder structure")
            prompt_encoder = PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16, # Default SAM value
            )
        else:
            # print("Building Naive Prompt Encoder (PatchEmbed for DistMaps)")
            prompt_encoder = PatchEmbed(
                img_size=(image_size, image_size), # DistMaps are image-sized
                patch_size=(16, 16), # This patch size should match subsequent feature map size if used like SAM
                in_chans=2, # For pos/neg dist maps
                embed_dim=prompt_embed_dim,
                flatten=False, # Output [B, D, h, w]
            )
        return prompt_encoder

    def _build_mask_decoder(self, prompt_embed_dim, image_embedding_size, image_size, encoder_output_dim):
        if self.config.get('USE_SAM_DECODER', False):
            # print("Building SAM Mask Decoder (V2) structure")
            mask_decoder = MaskDecoderV2(
                num_multimask_outputs=self.config.get('NUM_MULTIMASK_OUTPUTS', 2), # keypoint, road
                transformer=TwoWayTransformer(
                    depth=2, # SAM default
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048, # SAM default
                    num_heads=8, # SAM default
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3, # SAM default
                iou_head_hidden_dim=256, # SAM default
            )
        else:
            # print("Building Naive Convolutional Decoder structure")
            activation = nn.GELU
            mask_decoder = nn.Sequential(
                nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                activation(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(32, self.config.get('NUM_MULTIMASK_OUTPUTS', 2), kernel_size=2, stride=2), # Output channels for kp and road
            )
        return mask_decoder

    @staticmethod
    def resize_sam_pos_embed(state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        # This function is complex and highly specific to SAM's ViT structure.
        # For brevity and focus on inference flow, assuming it works as in original or not needed if image_size matches.
        # If issues arise, this function from the original training script needs careful review.
        # If pos_embed exists and size mismatch, log it. Actual resize logic is involved.
        if 'image_encoder.pos_embed' in state_dict:
            pos_embed = state_dict['image_encoder.pos_embed']
            original_token_size = int(math.sqrt(pos_embed.shape[1]))
            target_token_size = image_size // vit_patch_size
            if original_token_size != target_token_size:
                print(f"  Pos_embed size mismatch: original {original_token_size}, target {target_token_size}. Resize logic needed.")
                # Actual resizing logic from SAM utils or your training script would go here.
                # For now, returning state_dict as is if not implementing full resize here.
        return state_dict # Placeholder if full resize not implemented here

    def prepare_prompt_inputs(self, sampled_kp, kp_label):
        """
        Formats sampled keypoints and labels into the structure required by DistMaps.
        Args:
            sampled_kp (torch.Tensor): Padded keypoint coordinates (x, y). Shape: [B, N_seq, 2].
            kp_label (torch.Tensor): Padded keypoint labels (1=pos, 0=neg, -1=pad). Shape: [B, N_seq].
        Returns:
            torch.Tensor: Formatted points tensor for DistMaps.
                          Shape: [B, num_max_points * 2, 3], format (y, x, type)
                          where type is 1 for pos, 0 for neg. Padded with -1.
        """
        B, N_seq, _ = sampled_kp.shape
        device = sampled_kp.device
        dtype = sampled_kp.dtype

        batch_pos_points = []
        batch_neg_points = []
        max_pos_len = 0
        max_neg_len = 0

        for i in range(B):
            labels = kp_label[i]
            points = sampled_kp[i]
            pos_mask = (labels == 1)
            neg_mask = (labels == 0)
            pos_pts = points[pos_mask]
            neg_pts = points[neg_mask]
            batch_pos_points.append(pos_pts)
            batch_neg_points.append(neg_pts)
            max_pos_len = max(max_pos_len, pos_pts.shape[0])
            max_neg_len = max(max_neg_len, neg_pts.shape[0])

        num_max_points_type = max(1, max_pos_len, max_neg_len) # Max points for one type (pos or neg)

        # Output shape: [B, num_max_points_type * 2, 3] (y, x, type_label)
        output_points = torch.full((B, num_max_points_type * 2, 3), -1.0, dtype=dtype, device=device)

        for i in range(B):
            pos_pts = batch_pos_points[i] # [num_pos, 2] (x,y)
            neg_pts = batch_neg_points[i]
            num_pos = pos_pts.shape[0]
            num_neg = neg_pts.shape[0]

            if num_pos > 0:
                output_points[i, :num_pos, 0] = pos_pts[:, 1]  # y coordinate
                output_points[i, :num_pos, 1] = pos_pts[:, 0]  # x coordinate
                output_points[i, :num_pos, 2] = 1.0 # type for positive

            if num_neg > 0:
                start_idx = num_max_points_type
                end_idx = start_idx + num_neg
                output_points[i, start_idx:end_idx, 0] = neg_pts[:, 1] # y
                output_points[i, start_idx:end_idx, 1] = neg_pts[:, 0] # x
                output_points[i, start_idx:end_idx, 2] = 0.0 # type for negative
        return output_points
    
    def precompute_image_features(self, rgbs):
        """
        Precompute image features for given RGB images.
        """
        x = rgbs.permute(0, 3, 1, 2).float() # [B, C, H, W] float
        # Normalize
        x = (x - self.pixel_mean) / self.pixel_std

        # Image Encoder
        image_features = self.image_encoder(x) # [B, D, h, w] (e.g., [B, 256, 64, 64])
        flops = FlopCountAnalysis(self.image_encoder, x)
        print("precompute_image_features image encoder flops: ", flops.total())
        print("precompute_image_features image encoder flops by module: ", flops.by_module())
        self.image_features = image_features
        return image_features
    
    def infer_masks_and_features_from_prompts(
        self,
        image_features: torch.Tensor, 
        kp_coords: torch.Tensor, 
        kp_labels: torch.Tensor,
    ):
        """
        Inference function for mask prediction using precomputed image features and user prompts.
        Args:
            image_features (torch.Tensor): Precomputed image embeddings [B, D, h, w] (e.g., [1, 256, 64, 64])
            kp_coords (torch.Tensor): Keypoint coordinates [B, NumPrompts, 2] (x,y format, original image scale)
            kp_labels (torch.Tensor): Keypoint labels [B, NumPrompts] (1 for pos, 0 for neg)
        Returns:
            effective_image_embeddings (torch.Tensor): Embeddings used for mask decoder, potentially modified by prompts.
                                                        Can be used by TopoNet. [B, D, h, w]
            mask_scores (torch.Tensor): [B, 2, H, W] (keypoint, road)
            mask_logits (torch.Tensor): [B, 2, H, W] (keypoint, road)
        """
        # Determine input size for prompt transformation (model's expected input size)
        # This should align with how image_features were generated (e.g., 1024x1024)
        model_input_size_hw = (self.image_size, self.image_size) 
        effective_image_embeddings = image_features

        if self.config.get('USE_SAM_DECODER', False):

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(kp_coords, kp_labels), boxes=None, masks=None
            )
            effective_image_embeddings = image_features + dense_embeddings # SAM adds dense prompt embeds

            low_res_logits, _ = self.mask_decoder(
                image_embeddings=effective_image_embeddings, # Already includes dense prompt embeds
                image_pe=self.prompt_encoder.get_dense_pe(), # Positional encoding for image features
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=torch.zeros_like(dense_embeddings), # Pass zero as dense_embeddings already added to image_embeddings
                multimask_output=True 
            )
            # Upsample masks to original image size
            # SAM's postprocess_masks expects input_size to be the size of the network input (after ResizeLongestSide)
            # and original_size to be the true original image size.
            mask_logits = self.model.postprocess_masks(low_res_logits, model_input_size_hw, None)
        else: # Naive decoder path
            cost_time_in_generate_mask = {
                'total': 0,
                'prepare_prompt_inputs': 0,
                'dist_maps': 0,
                'prompt_encoder': 0,
                'mask_decoder': 0
            }
            start_time = time.time()
            # `prepare_prompt_inputs` expects B, N_seq, 2 (x,y) and B, N_seq labels
            points_for_distmap = self.prepare_prompt_inputs(kp_coords, kp_labels)
            end_time = time.time()
            cost_time_in_generate_mask['prepare_prompt_inputs'] = end_time - start_time

            start_time = time.time()
            distmap_shape = (image_features.shape[0], model_input_size_hw[0], model_input_size_hw[1])

            points_map = self.dist_maps(*distmap_shape, points_for_distmap) # [B, 2, model_H, model_W]
            end_time = time.time()
            cost_time_in_generate_mask['dist_maps'] = end_time - start_time
            start_time = time.time()
            coord_features = self.prompt_encoder(points_map) # PatchEmbed out: [B, D, h_feat, w_feat]
            end_time = time.time()
            cost_time_in_generate_mask['prompt_encoder'] = end_time - start_time
            start_time = time.time()
            assert coord_features.shape == image_features.shape, f"coord_features.shape {coord_features.shape} != image_features.shape {image_features.shape}"

            effective_image_embeddings = image_features + coord_features
            mask_logits = self.mask_decoder(effective_image_embeddings) # [B, 2, h, w]
            end_time = time.time()
            cost_time_in_generate_mask['mask_decoder'] = end_time - start_time
            cost_time_in_generate_mask['total'] = sum(cost_time_in_generate_mask.values())
            print(f"Cost time in infer_masks_and_features_from_prompts: {cost_time_in_generate_mask}")
            flops = FlopCountAnalysis(self.mask_decoder, effective_image_embeddings)
            print("infer_masks_and_features_from_prompts mask decoder flops: ", flops.total())
            print("infer_masks_and_features_from_prompts mask decoder flops by module: ", flops.by_module())
        # Apply sigmoid to get final scores
        mask_scores = torch.sigmoid(mask_logits) # [B, 2, H_orig, W_orig] or [B, 2, h, w]
        
        return effective_image_embeddings, mask_scores, mask_logits

    def infer_toponet(
        self, 
        effective_image_embeddings: torch.Tensor, 
        graph_points: torch.Tensor, # [B, N_graph_pts, 2] (x,y format, original image scale)
        pairs: torch.Tensor,        # [B, N_samples, N_pairs_max, 2] (indices into graph_points)
        valid: torch.Tensor,        # [B, N_samples, N_pairs_max]
        mask_logits: torch.Tensor    # [B, 2, H_orig, W_orig]
    ):
        """
        Infer topology connections using pre-calculated effective image embeddings and graph points.
        Args:
            effective_image_embeddings (torch.Tensor): [B, D, h_feat, w_feat]
            graph_points (torch.Tensor): Points from graph [B, N_graph_pts_padded, 2] in original image scale (x,y)
            pairs (torch.Tensor): Pairs of point indices [B, N_samples, N_pairs_max, 2]
            valid (torch.Tensor): Validity mask for pairs [B, N_samples, N_pairs_max]
        Returns:
            topo_scores (torch.Tensor): [B, N_samples, N_pairs_max, 1]
        """
        # bilinear_sampler expects points in original image scale. (eg 1024x1024)
        point_features_for_toponet = self.bilinear_sampler(effective_image_embeddings, graph_points)
        _, topo_scores = self.toponet(graph_points, point_features_for_toponet, pairs, valid, mask_logits)
        return topo_scores
    
    def inference_with_features(self, image_features, kp_coords, kp_labels, original_image_size_hw):
        """
        Inference function for road extraction.
        """
        res = {
            'pred_nodes': [],
            'pred_edges': [],
            'road_mask': [],
            'kp_mask': [],
            }
        total_inference_seconds = 0
        start_seconds = time.time()

        device = image_features.device
        effective_image_embeddings, mask_scores, mask_logits = self.infer_masks_and_features_from_prompts(
            image_features, kp_coords, kp_labels)
        
        # Keep full-size logits for TopoNet (expected to be [B, 2, image_size, image_size])
        full_size_mask_logits = mask_logits
        
        # Crop masks to original image size for graph extraction and output
        mask_scores = mask_scores[:, :, :original_image_size_hw[0], :original_image_size_hw[1]]
        mask_logits = mask_logits[:, :, :original_image_size_hw[0], :original_image_size_hw[1]]

        kp_mask, road_mask = mask_scores[:, 0, :, :], mask_scores[:, 1, :, :]
        kp_mask = kp_mask.cpu().numpy()  # float32, 0-1 
        road_mask = road_mask.cpu().numpy() # float32, 0-1
        cost_time_in_extract_graph_points = {
            'total': 0,
            'extract_graph_points': 0,
            'find_k_neighbors': 0,
            'prepare_pairs_and_valid': 0,
            'infer_toponet': 0,
            'filter_edges': 0
        }
        start_time = time.time()
        # extract graph points
        for i in range(image_features.shape[0]):
            graph_points, filtered_kp_mask, filtered_road_mask = extract_graph_points(kp_mask[i], road_mask[i], self.config)
            if graph_points.shape[0] == 0:
                print(f'index {i} has no predicted graph points')
                continue
            end_time = time.time()
            cost_time_in_extract_graph_points['extract_graph_points'] = end_time - start_time
            
            ## Pass 2: infer toponet to predict topology of points from stored img features
            edge_scores = defaultdict(float)
            edge_counts = defaultdict(float)
            start_time = time.time()
            # Extract point features
            batch_points = torch.tensor(graph_points, device=device, dtype=torch.float32).unsqueeze(0)
            
            # Prepare pairs for topology inference
            # Use KDTree for efficient neighbor search
            patch_kdtree = scipy.spatial.KDTree(graph_points)
            
            # Find k nearest neighbors for each point
            knn_d, knn_idx = patch_kdtree.query(
                graph_points, 
                k=self.config.MAX_NEIGHBOR_QUERIES + 1, 
                distance_upper_bound=self.config.NEIGHBOR_RADIUS
            )
            end_time = time.time()
            cost_time_in_extract_graph_points['find_k_neighbors'] = end_time - start_time

            start_time = time.time()
            # Remove self connections (first index is always self)
            knn_idx = knn_idx[:, 1:]
            
            patch_point_num = graph_points.shape[0]
            # Create source indices
            src_idx = np.tile(
                np.arange(patch_point_num)[:, np.newaxis],
                (1, self.config.MAX_NEIGHBOR_QUERIES)
            )
            
            # Create valid mask and target indices
            valid = knn_idx < patch_point_num
            tgt_idx = np.where(valid, knn_idx, src_idx)
            
            # Create pairs
            pairs = np.stack([src_idx, tgt_idx], axis=-1)
            
            # Convert to tensors for model input
            batch_pairs = torch.tensor(pairs, device=device).unsqueeze(0)
            batch_valid = torch.tensor(valid, device=device).unsqueeze(0)
            end_time = time.time()
            cost_time_in_extract_graph_points['prepare_pairs_and_valid'] = end_time - start_time

            start_time = time.time()
            # Get topology scores from the model
            topo_scores = self.infer_toponet(
                effective_image_embeddings[i:i+1], 
                batch_points, 
                batch_pairs, 
                batch_valid,
                full_size_mask_logits[i:i+1]
            )
            end_time = time.time()
            cost_time_in_extract_graph_points['infer_toponet'] = end_time - start_time

            start_time = time.time()
            # Handle NaNs and reshape
            topo_scores = torch.where(torch.isnan(topo_scores), -100.0, topo_scores).squeeze(-1).cpu().numpy()
            
            # Aggregate edge scores
            n_samples, n_pairs = topo_scores.shape[1:]
            for si in range(n_samples):
                for pi in range(n_pairs):
                    if not batch_valid[0, si, pi]:
                        continue
                    src_idx_all, tgt_idx_all = pairs[si, pi, :]
                    edge_score = topo_scores[0, si, pi]
                    # Ensure score is in valid range
                    if not (0.0 <= edge_score <= 1.0):
                        continue
                    edge_scores[(src_idx_all, tgt_idx_all)] += edge_score
                    edge_counts[(src_idx_all, tgt_idx_all)] += 1.0
            
            # Average edge scores and filter
            pred_edges = []
            for edge, score_sum in edge_scores.items():
                score = score_sum / edge_counts[edge]
                if score > self.config.TOPO_THRESHOLD:
                    pred_edges.append(edge)
            end_time = time.time()
            cost_time_in_extract_graph_points['filter_edges'] = end_time - start_time
            cost_time_in_extract_graph_points['total'] = sum(cost_time_in_extract_graph_points.values())
            print(f"Cost time in extract_graph_points: {cost_time_in_extract_graph_points}")
            pred_edges = np.array(pred_edges) if pred_edges else np.zeros((0, 2), dtype=np.int32)
            
            # Store results for visualization
            res['pred_nodes'].append(graph_points)
            res['pred_edges'].append(pred_edges)
            res['road_mask'].append(filtered_road_mask > self.config.ROAD_THRESHOLD) # save road mask as binary
            res['kp_mask'].append(filtered_kp_mask > self.config.ITSC_THRESHOLD) # save keypoint mask as binary
        end_seconds = time.time()
        total_inference_seconds += (end_seconds - start_seconds)
        print(f'total_inference_seconds: {total_inference_seconds}')

        return res
    
    def inference_with_imgs(self, imgs, kp_coords, kp_labels, original_image_size_hw):
        """
        Inference function for road extraction.
        """
        image_features = self.precompute_image_features(imgs)
        res = self.inference_with_features(image_features, kp_coords, kp_labels, original_image_size_hw)
        return res

    def forward(self, image_features, kp_coords, kp_labels, original_image_size_hw, input_features=True):
        """
        This forward is just for inference. No training should be done here.
        """
        if input_features:
            res = self.inference_with_features(image_features, kp_coords, kp_labels, original_image_size_hw)
        else:
            res = self.inference_with_imgs(image_features, kp_coords, kp_labels, original_image_size_hw)
        return res

# Helper to filter SAM checkpoint state_dict for specific components
def filter_state_dict(state_dict, component_prefix):
    filtered_sd = {}
    for k, v in state_dict.items():
        if k.startswith(component_prefix + '.'):
            filtered_sd[k[len(component_prefix)+1:]] = v
    return filtered_sd 