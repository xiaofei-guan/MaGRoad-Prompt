import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import math
from functools import partial
import pprint
import numpy as np
import matplotlib.pyplot as plt
from segment_anything.modeling.image_encoder import ImageEncoderViT
# from segment_anything.modeling.mask_decoder import MaskDecoderV2
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.modeling.common import LayerNorm2d
from module import (
    DistMaps, BilinearSampler, TopoNet, _LoRA_qkv, DiceLoss,
    PatchEmbed, MaGTopoNet
)


class MaGRoad(pl.LightningModule):
    """
    PyTorch Lightning module for training SAM for keypoint and road mask segmentation.
    Based on the segmentation part of SAMRoad.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config) # Log hyperparameters

        # --- Best metrics tracking ---
        self.best_metrics = {
            'train_loss': {'value': float('inf'), 'epoch': 0},
            'train_keypoint_iou': {'value': 0.0, 'epoch': 0},
            'train_road_iou': {'value': 0.0, 'epoch': 0},
            'val_loss': {'value': float('inf'), 'epoch': 0},
            'val_keypoint_iou': {'value': 0.0, 'epoch': 0},
            'val_road_iou': {'value': 0.0, 'epoch': 0},
            # Add topology metrics
            'train_topo_loss': {'value': float('inf'), 'epoch': 0},
            'train_topo_acc': {'value': 0.0, 'epoch': 0},
            'train_topo_f1': {'value': 0.0, 'epoch': 0},
            'val_topo_loss': {'value': float('inf'), 'epoch': 0},
            'val_topo_acc': {'value': 0.0, 'epoch': 0},
            'val_topo_f1': {'value': 0.0, 'epoch': 0},
        }

        self._validate_config()

        # --- Model Configuration ---
        encoder_embed_dim, encoder_depth, encoder_num_heads, encoder_global_attn_indexes = self._get_sam_config()
        prompt_embed_dim = 256
        image_size = config.PATCH_SIZE
        self.image_size = image_size
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        encoder_output_dim = prompt_embed_dim

        self.prev_mask = torch.zeros(1, 1, image_size, image_size)

        # --- Buffers ---
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

        # --- Image Encoder ---
        self.image_encoder = self._build_image_encoder(
            encoder_depth, encoder_embed_dim, image_size, encoder_num_heads,
            vit_patch_size, encoder_global_attn_indexes, prompt_embed_dim
        )

        # --- Prompt Encoder ---
        if not self.config.USE_SAM_DECODER:
            self.dist_maps = DistMaps(norm_radius=config.DIST_MAP_NORM_R, spatial_scale=1.0, use_disks=config.USE_DISKS)
        else:
            self.dist_maps = None

        self.prompt_encoder = self._build_prompt_encoder(
            prompt_embed_dim, image_embedding_size, image_size
        )

        # --- Mask Decoder ---
        self.mask_decoder = self._build_mask_decoder(
            prompt_embed_dim, image_embedding_size, image_size, encoder_output_dim
        )

        # --- Topology components ---
        # Add BilinearSampler for feature extraction
        self.bilinear_sampler = BilinearSampler(self.image_size)
        
        # Add TopoNet for topology prediction
        if self.config.TOPONET == "transformer":
            self.toponet = TopoNet(config, prompt_embed_dim)
        elif self.config.TOPONET == "magtoponet":
            self.toponet = MaGTopoNet(config, prompt_embed_dim,
                                      use_point_features=self.config.USE_POINT_FEATURES,
                                      use_path_features=self.config.USE_PATH_FEATURES,
                                      use_edge_bias=self.config.USE_EDGE_BIAS)
        else:
            raise ValueError(f"Invalid TOPONET: {self.config.TOPONET}")

        # --- LoRA Setup ---
        self.w_As, self.w_Bs = [], []
        if config.ENCODER_LORA:
            self._setup_lora()

        # --- Loss Functions ---
        self.mask_bce_criterion, self.mask_dice_criterion, self.topo_bce_criterion = self._setup_losses()
        
        # --- Metrics ---
        # Mask metrics
        self.train_keypoint_iou = torchmetrics.classification.BinaryJaccardIndex(threshold=0.5)
        self.train_road_iou = torchmetrics.classification.BinaryJaccardIndex(threshold=0.5)
        self.val_keypoint_iou = torchmetrics.classification.BinaryJaccardIndex(threshold=0.5)
        self.val_road_iou = torchmetrics.classification.BinaryJaccardIndex(threshold=0.5)
        
        # Topology metrics
        self.train_topo_acc = torchmetrics.classification.BinaryAccuracy(threshold=0.5)
        self.train_topo_f1 = torchmetrics.classification.BinaryF1Score(threshold=0.5)
        self.val_topo_acc = torchmetrics.classification.BinaryAccuracy(threshold=0.5)
        self.val_topo_f1 = torchmetrics.classification.BinaryF1Score(threshold=0.5)
        
        # Test-time metrics, not used in training
        self.keypoint_pr_curve = torchmetrics.classification.BinaryPrecisionRecallCurve(thresholds=torch.linspace(0.01, 0.99, 99))
        self.road_pr_curve = torchmetrics.classification.BinaryPrecisionRecallCurve(thresholds=torch.linspace(0.01, 0.99, 99))
        self.topo_pr_curve = torchmetrics.classification.BinaryPrecisionRecallCurve(thresholds=torch.linspace(0.01, 0.99, 99))

        # --- Load Checkpoint ---
        self.matched_param_names = set()
        if not self.config.NO_SAM and self.config.USE_PRETRAIN:
            self._load_sam_checkpoint()


    def _validate_config(self):
        assert self.config.SAM_VERSION in {'vit_b', 'vit_l', 'vit_h'}, "Invalid SAM_VERSION"

    def _get_sam_config(self):
        if self.config.SAM_VERSION == 'vit_b':
            return 768, 12, 12, [2, 5, 8, 11]
        elif self.config.SAM_VERSION == 'vit_l':
            return 1024, 24, 16, [5, 11, 17, 23]
        elif self.config.SAM_VERSION == 'vit_h':
            return 1280, 32, 16, [7, 15, 23, 31]

    def _build_image_encoder(self, depth, embed_dim, img_size, num_heads, patch_size, global_attn_indexes, out_chans):
        print(f"Using SAM {self.config.SAM_VERSION} Encoder")
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
        if self.config.USE_SAM_DECODER:
            print("Using SAM Prompt Encoder")
            prompt_encoder = PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            )
            if self.config.FREEZE_PROMPT_ENCODER:
                for param in prompt_encoder.parameters():
                    param.requires_grad = False
        else:
            print("Using Naive Prompt Encoder")
            prompt_encoder = PatchEmbed(
                img_size=(image_size, image_size),
                patch_size=(16, 16),
                in_chans=2,
                embed_dim=prompt_embed_dim,
                flatten=False,
            )
        return prompt_encoder

    def _build_mask_decoder(self, prompt_embed_dim, image_embedding_size, image_size, encoder_output_dim):
        if self.config.USE_SAM_DECODER:
            print("Using SAM Mask Decoder")
            mask_decoder = MaskDecoderV2(
                num_multimask_outputs=2, # keypoint, road
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            )
        else:
            print("Using Naive Convolutional Decoder")
            activation = nn.GELU
            # Output channels = 2 (keypoint, road)
            mask_decoder = nn.Sequential(
                nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                activation(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2),
            )
        return mask_decoder
    
    def prepare_prompt_inputs(self, sampled_kp, kp_label):
        """
        Formats sampled keypoints and labels into the structure required by DistMaps (GPU mode).

        Args:
            sampled_kp (torch.Tensor): Padded keypoint coordinates (x, y). Shape: [B, N_seq, 2].
            kp_label (torch.Tensor): Padded keypoint labels (1=pos, 0=neg, -1=pad). Shape: [B, N_seq].

        Returns:
            torch.Tensor: Formatted points tensor for DistMaps.
                          Shape: [B, num_max_points * 2, 3], format (x, y, 0) for valid points,
                          (-1, -1, -1) for padding. num_max_points is the max number of
                          pos/neg points across the batch.
        """
        B, N_seq, _ = sampled_kp.shape
        device = sampled_kp.device
        dtype = sampled_kp.dtype

        batch_pos_points = []
        batch_neg_points = []
        max_pos_len = 0
        max_neg_len = 0

        # 1. Separate positive and negative points for each batch item and find max lengths
        for i in range(B):
            labels = kp_label[i] # Shape: [N_seq]
            points = sampled_kp[i] # Shape: [N_seq, 2]

            pos_mask = (labels == 1)
            neg_mask = (labels == 0)

            pos_pts = points[pos_mask] # Shape: [num_pos, 2], format (x, y)
            neg_pts = points[neg_mask] # Shape: [num_neg, 2], format (x, y)

            batch_pos_points.append(pos_pts)
            batch_neg_points.append(neg_pts)

            max_pos_len = max(max_pos_len, pos_pts.shape[0])
            max_neg_len = max(max_neg_len, neg_pts.shape[0])

        # 2. Determine num_max_points
        num_max_points = max(1, max_pos_len, max_neg_len) # Ensure at least 1

        # 3. Create the output tensor initialized with padding value -1
        output_points = torch.full((B, num_max_points * 2, 3), -1.0, dtype=dtype, device=device)

        # 4. Fill the output tensor with formatted points
        for i in range(B):
            pos_pts = batch_pos_points[i]
            neg_pts = batch_neg_points[i]
            num_pos = pos_pts.shape[0]
            num_neg = neg_pts.shape[0]

            # Add positive points (y, x, 0)
            if num_pos > 0:
                output_points[i, :num_pos, 0] = pos_pts[:, 1] 
                output_points[i, :num_pos, 1] = pos_pts[:, 0]
                output_points[i, :num_pos, 2] = 1.0          

            # Add negative points (y, x, 0) starting after the positive section
            if num_neg > 0:
                start_idx = num_max_points
                end_idx = num_max_points + num_neg
                output_points[i, start_idx:end_idx, 0] = neg_pts[:, 1]
                output_points[i, start_idx:end_idx, 1] = neg_pts[:, 0]
                output_points[i, start_idx:end_idx, 2] = 0.0

        return output_points

    def _setup_lora(self):
        print(f"Setting up LoRA with rank {self.config.LORA_RANK}")
        r = self.config.LORA_RANK
        lora_layer_selection = list(range(len(self.image_encoder.blocks))) # Apply to all blocks

        # Freeze original encoder parameters
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # Apply LoRA surgery
        for t_layer_i, blk in enumerate(self.image_encoder.blocks):
            if t_layer_i not in lora_layer_selection:
                continue
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.extend([w_a_linear_q, w_a_linear_v])
            self.w_Bs.extend([w_b_linear_q, w_b_linear_v])
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v
            )

        # Initialize LoRA weights
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
        print(f"Applied LoRA to {len(lora_layer_selection)} encoder blocks.")

    def _setup_losses(self):
        # mask loss fn
        pos_weight_val = self.config.get('BCE_POS_WEIGHT', 1.0)
        print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight_val}")
        bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
        print("Using Dice Loss")
        dice_loss = DiceLoss(smooth=self.config.get('DICE_SMOOTH', 1.0)) # Allow configuring smooth factor
        # topo loss fn
        topo_pos_weight = self.config.get('TOPO_POS_WEIGHT', 1.0)
        print(f"Using BCEWithLogitsLoss with topo_pos_weight={topo_pos_weight} for topology prediction")
        topo_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([topo_pos_weight]))
        return bce_loss, dice_loss, topo_loss

    def _load_sam_checkpoint(self):
        print(f"Loading SAM checkpoint from: {self.config.SAM_CKPT_PATH}")
        try:
            with open(self.config.SAM_CKPT_PATH, "rb") as f:
                ckpt_state_dict = torch.load(f, map_location='cpu') # Load to CPU first

                # Resize position embeddings if necessary
                if self.image_size != 1024: # Default SAM image size
                     _, _, _, encoder_global_attn_indexes = self._get_sam_config()
                     vit_patch_size = 16
                     print(f"Resizing SAM positional embeddings for image size {self.image_size}")
                     ckpt_state_dict = self.resize_sam_pos_embed(
                         ckpt_state_dict, self.image_size, vit_patch_size, encoder_global_attn_indexes
                     )

                # Load matching parameters
                state_dict_to_load = {}
                mismatch_names = []
                model_params = dict(self.named_parameters())

                for k_ckpt, v_ckpt in ckpt_state_dict.items():
                    if k_ckpt in model_params and v_ckpt.shape == model_params[k_ckpt].shape:
                         # Check if the parameter requires grad (relevant for LoRA)
                         if model_params[k_ckpt].requires_grad or not self.config.ENCODER_LORA:
                              state_dict_to_load[k_ckpt] = v_ckpt
                              self.matched_param_names.add(k_ckpt)
  
                print("###### Loading Checkpoint ######")
                missing_keys, unexpected_keys = self.load_state_dict(state_dict_to_load, strict=False)

                print("--- Matched/Loaded Params ---")
                pprint.pprint(sorted(list(self.matched_param_names)))
                print("--- Missing Keys (Not in Checkpoint or Mismatched) ---")
                pprint.pprint(sorted(missing_keys))
                print("--- Unexpected Keys (In Checkpoint but not in Model State Dict) ---")
                pprint.pprint(sorted(unexpected_keys))
                    
        except FileNotFoundError:
            print(f"Warning: SAM checkpoint not found at {self.config.SAM_CKPT_PATH}. Model initialized randomly.")
        except Exception as e:
            print(f"Error loading SAM checkpoint: {e}")
            raise

    @staticmethod
    def resize_sam_pos_embed(state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        """
        Resizes the positional embeddings in the SAM state dictionary for a different image size.
        Static method as it only operates on the state dict.
        """
        new_state_dict = {k: v for k, v in state_dict.items()}
        pos_embed = new_state_dict.get('image_encoder.pos_embed', None)

        if pos_embed is None:
             print("Warning: 'image_encoder.pos_embed' not found in state_dict. Skipping resize.")
             return new_state_dict

        token_size = image_size // vit_patch_size
        original_token_size = int(math.sqrt(pos_embed.shape[1])) # Assuming square embedding grid

        if original_token_size != token_size:
            print(f"  Resizing pos_embed from {original_token_size}x{original_token_size} to {token_size}x{token_size}")
            # SAM format: [1, H*W, C] -> need to reshape to [1, C, H, W] for interpolate
            embed_dim = pos_embed.shape[-1]
            pos_embed_reshaped = pos_embed.view(1, original_token_size, original_token_size, embed_dim).permute(0, 3, 1, 2)
            pos_embed_resized = F.interpolate(pos_embed_reshaped, (token_size, token_size), mode='bilinear', align_corners=False)
            # Reshape back to [1, H*W, C]
            new_state_dict['image_encoder.pos_embed'] = pos_embed_resized.permute(0, 2, 3, 1).view(1, token_size * token_size, embed_dim)

            # Resize relative positional biases
            rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k and 'image_encoder.' in k]
            for k in rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                # Infer original size from shape, e.g., (2*H-1, HeadDim) or (2*H-1, 2*W-1)
                h_orig, w_orig = rel_pos_params.shape[:2] # Take first two dims
                h_new = 2 * token_size - 1
                w_new = w_orig # Assume width/head dim doesn't change or handle accordingly

                if h_orig != h_new :
                    print(f"  Resizing relative PE {k} from shape {rel_pos_params.shape} to ({h_new}, {w_new}, ...)")
                    # Add batch and channel dims for interpolation
                    rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                    # Handle different dimensions of rel_pos embeddings
                    if len(rel_pos_params.shape) == 4: # [1, 1, H, W]
                         rel_pos_params = F.interpolate(rel_pos_params, (h_new, w_new), mode='bilinear', align_corners=False)
                    elif len(rel_pos_params.shape) == 3: # [1, 1, H] - unlikely? check SAM format
                         # Might need specific handling if format is different
                         print(f"Warning: Skipping resize for rel_pos {k} with unexpected shape {rel_pos_params.shape[2:]}")
                         continue
                    else:
                         print(f"Warning: Skipping resize for rel_pos {k} with unexpected shape {rel_pos_params.shape}")
                         continue

                    new_state_dict[k] = rel_pos_params.squeeze(0).squeeze(0) # Remove added dims

        return new_state_dict
    
    def infer_masks_and_img_features(self, rgbs, kp_points, kp_labels):
        """
        Infer masks and image features from input images and keypoint points.
        
        Args:
            rgbs (torch.Tensor): Input images [B, H, W, C] uint8 [0, 255]
            kp_points (torch.Tensor): Keypoint points [B, N, 2]
            kp_labels (torch.Tensor): Keypoint labels [B, N]
        """
        x = rgbs.permute(0, 3, 1, 2).float() # [B, C, H, W] float
        # Normalize
        x = (x - self.pixel_mean) / self.pixel_std

        # Image Encoder
        image_embeddings = self.image_encoder(x) # [B, D, h, w] (e.g., [B, 256, 64, 64])
        points = self.prepare_prompt_inputs(kp_points, kp_labels)
        points_map = self.dist_maps(x, points) # [B, 2, H, W] [0-1]
        coord_features = self.prompt_encoder(points_map)
        # fuse prompt_features with image_embeddings
        image_embeddings = image_embeddings + coord_features
        mask_logits = self.mask_decoder(image_embeddings) # [B, 2, H, W]

        self.mask_logits = mask_logits
        mask_scores = torch.sigmoid(mask_logits)

        return image_embeddings, mask_scores, mask_logits

    def infer_toponet(self, image_embeddings, graph_points, pairs, valid, mask_logits):
        """
        Infer topology connections from image embeddings and graph points.
        """
        point_features = self.bilinear_sampler(image_embeddings, graph_points)
        topo_logits, topo_scores = self.toponet(graph_points, point_features, pairs, valid, mask_logits)

        return topo_scores
    
    def forward(self, rgb, sampled_kp, kp_label, graph_points, pairs, valid):
        """
        Forward pass of the model.
        
        Args:
            rgb (torch.Tensor): Input RGB image [B, H, W, C] uint8 [0, 255]
            sampled_kp (torch.Tensor, optional): Sampled keypoints [B, N, 2]
            kp_label (torch.Tensor, optional): Keypoint labels [B, N]
            graph_points (torch.Tensor, optional): Points from graph [B, N_points, 2]
            pairs (torch.Tensor, optional): Pairs of point indices [B, N_samples, N_pairs, 2]
            valid (torch.Tensor, optional): Validity mask for pairs [B, N_samples, N_pairs]
            
        Returns:
            tuple: (mask_logits, mask_scores, topo_logits)
                   mask_logits: [B, H, W, 2] - Raw logits for keypoint and road masks
                   mask_scores: [B, H, W, 2] - Sigmoid scores for masks
                   topo_logits: [B, N_samples, N_pairs] - Logits for topology connections
        """
        # rgb: [B, H, W, C] uint8 [0, 255]
        x = rgb.permute(0, 3, 1, 2).float() # [B, C, H, W] float
        # Normalize
        x = (x - self.pixel_mean) / self.pixel_std

        # Image Encoder
        image_embeddings = self.image_encoder(x) # [B, D, h, w] (e.g., [B, 256, 64, 64])

        # Mask Decoder
        if self.config.USE_SAM_DECODER:
            # Generate null prompt embeddings (as we are not using prompts here)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(sampled_kp, kp_label), boxes=None, masks=None
            )
            # Decode masks
            low_res_logits, _ = self.mask_decoder( # Ignore IoU predictions
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True # Get outputs for keypoint and road
            )
            # Upsample masks to original image size
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ) # [B, 2, H, W]
        else:
            # Use naive decoder
            points = self.prepare_prompt_inputs(sampled_kp, kp_label)
            points_map = self.dist_maps(x, points) # [B, 2, H, W] [0-1]
            coord_features = self.prompt_encoder(points_map)
            # fuse prompt_features with image_embeddings
            image_embeddings = image_embeddings + coord_features
            mask_logits = self.mask_decoder(image_embeddings) # [B, 2, H, W]

        mask_scores = torch.sigmoid(mask_logits)
        self.mask_logits = mask_logits
        # Permute to [B, H, W, 2] to match SAMRoad output convention
        mask_logits = mask_logits.permute(0, 2, 3, 1)
        mask_scores = mask_scores.permute(0, 2, 3, 1)
        
        # Extract point features using bilinear sampling
        point_features = self.bilinear_sampler(image_embeddings, graph_points)
        
        # Predict topology connections
        topo_logits, topo_scores = self.toponet(graph_points, point_features, pairs, valid, self.mask_logits)

        return mask_logits, mask_scores, topo_logits, topo_scores

    def _compute_mask_loss(self, mask_logits, keypoint_mask, road_mask):

        # Stack GT masks: [B, H, W, 2]
        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)

        bce_loss = self.mask_bce_criterion(mask_logits, gt_masks)
        dice_loss = self.mask_dice_criterion(mask_logits, gt_masks)
        mask_loss = bce_loss + dice_loss

        return mask_loss, bce_loss, dice_loss

    def _compute_topology_loss(self, topo_logits, connected, valid):
        """
        Compute topology loss for graph connection prediction.
        
        Args:
            topo_logits (torch.Tensor): Predicted topology logits [B, N_samples, N_pairs]
            connected (torch.Tensor): Ground truth connections [B, N_samples, N_pairs]
            valid (torch.Tensor): Validity mask [B, N_samples, N_pairs]
            
        Returns:
            torch.Tensor: Topology loss
        """
        # Flatten predictions and targets
        valid_flat = valid.view(-1)
        if valid_flat.sum() == 0:
            print(f"Warning: Epoch:{self.current_epoch}, Step: {self.global_step}, no valid pairs for topology loss computation")
            # If no valid pairs, return zero loss
            return torch.tensor(0.0, device=topo_logits.device)
            
        topo_logits_flat = topo_logits.view(-1)[valid_flat]
        connected_flat = connected.view(-1)[valid_flat].float()
        
        # Compute BCE loss
        topo_loss = self.topo_bce_criterion(topo_logits_flat, connected_flat)
        
        return topo_loss

    def _log_mask_metrics(self, prefix=""):
        if prefix.startswith("train"):
            keypoint_iou = self.train_keypoint_iou.compute()
            road_iou = self.train_road_iou.compute()
            self.train_keypoint_iou.reset()
            self.train_road_iou.reset()
        else:
            keypoint_iou = self.val_keypoint_iou.compute()
            road_iou = self.val_road_iou.compute()
            self.val_keypoint_iou.reset()
            self.val_road_iou.reset()

        self.log(f"{prefix}keypoint_iou", keypoint_iou, sync_dist=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}road_iou", road_iou, sync_dist=True, on_step=False, on_epoch=True)

    def _log_topo_metrics(self, prefix=""):
        """Log topology metrics and reset them."""
        if prefix.startswith("train"):
            topo_acc = self.train_topo_acc.compute()
            topo_f1 = self.train_topo_f1.compute()
            self.train_topo_acc.reset()
            self.train_topo_f1.reset()
        else:
            topo_acc = self.val_topo_acc.compute()
            topo_f1 = self.val_topo_f1.compute()
            self.val_topo_acc.reset()
            self.val_topo_f1.reset()

        self.log(f"{prefix}topo_acc", topo_acc, sync_dist=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}topo_f1", topo_f1, sync_dist=True, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        rgb = batch['rgb'] # [B, H, W, C] uint8 [0, 255]
        keypoint_mask = batch['keypoint_mask'] # [B, H, W] float [0, 1]
        road_mask = batch['road_mask']       # [B, H, W] float [0, 1]
        sampled_kp = batch['sampled_kp'] # [B, N, 2] float [0, 1024]
        kp_label = batch['kp_label'] # [B, N] int [0, 1]
        # Graph data
        graph_points = batch['graph_points'] # [B, N_points, 2]
        pairs = batch['pairs'] # [B, N_samples, N_pairs, 2]
        connected = batch['connected'] # [B, N_samples, N_pairs]
        valid = batch['valid'] # [B, N_samples, N_pairs]

        # Forward pass
        mask_logits, mask_scores, topo_logits, topo_scores = self(
            rgb, sampled_kp, kp_label, graph_points, pairs, valid
        )
        # Compute mask loss
        mask_loss, bce_loss, dice_loss = self._compute_mask_loss(mask_logits, keypoint_mask, road_mask)
        # Compute topology loss 
        topo_loss = self._compute_topology_loss(topo_logits, connected, valid)
        # Combined loss with weighting
        total_loss = mask_loss + topo_loss
        # Logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_mask_loss', mask_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_bce_loss', bce_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_dice_loss', dice_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_topo_loss', topo_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Update mask metrics
        self.train_keypoint_iou.update(mask_scores[..., 0], keypoint_mask)
        self.train_road_iou.update(mask_scores[..., 1], road_mask)
        # Update topology metrics
        valid_flat = valid.view(-1).bool()
        if valid_flat.sum() > 0:
            self.train_topo_acc.update(
                topo_scores.view(-1)[valid_flat], 
                connected.view(-1)[valid_flat]
            )
            self.train_topo_f1.update(
                topo_scores.view(-1)[valid_flat], 
                connected.view(-1)[valid_flat]
            )

        # Log images and topology periodically
        if batch_idx == 0:
            self._log_images(batch, mask_scores, prefix="train")

        return total_loss

    def on_train_epoch_end(self):
        self._log_mask_metrics(prefix="train_epoch_")
        self._log_topo_metrics(prefix="train_epoch_")
        
        # Update best train metrics
        train_loss = self.trainer.callback_metrics.get('train_loss')
        train_keypoint_iou = self.trainer.callback_metrics.get('train_epoch_keypoint_iou')
        train_road_iou = self.trainer.callback_metrics.get('train_epoch_road_iou')
        train_topo_loss = self.trainer.callback_metrics.get('train_topo_loss')
        train_topo_acc = self.trainer.callback_metrics.get('train_epoch_topo_acc')
        train_topo_f1 = self.trainer.callback_metrics.get('train_epoch_topo_f1')
        
        if train_loss < self.best_metrics['train_loss']['value']:
            self.best_metrics['train_loss']['value'] = train_loss.item()
            self.best_metrics['train_loss']['epoch'] = self.current_epoch
            
        if train_keypoint_iou > self.best_metrics['train_keypoint_iou']['value']:
            self.best_metrics['train_keypoint_iou']['value'] = train_keypoint_iou.item()
            self.best_metrics['train_keypoint_iou']['epoch'] = self.current_epoch
            
        if train_road_iou > self.best_metrics['train_road_iou']['value']:
            self.best_metrics['train_road_iou']['value'] = train_road_iou.item()
            self.best_metrics['train_road_iou']['epoch'] = self.current_epoch
            
        if train_topo_loss < self.best_metrics['train_topo_loss']['value']:
            self.best_metrics['train_topo_loss']['value'] = train_topo_loss.item()
            self.best_metrics['train_topo_loss']['epoch'] = self.current_epoch
            
        if train_topo_acc > self.best_metrics['train_topo_acc']['value']:
            self.best_metrics['train_topo_acc']['value'] = train_topo_acc.item()
            self.best_metrics['train_topo_acc']['epoch'] = self.current_epoch
            
        if train_topo_f1 > self.best_metrics['train_topo_f1']['value']:
            self.best_metrics['train_topo_f1']['value'] = train_topo_f1.item()
            self.best_metrics['train_topo_f1']['epoch'] = self.current_epoch

    def validation_step(self, batch, batch_idx):
        rgb = batch['rgb']
        keypoint_mask = batch['keypoint_mask']
        road_mask = batch['road_mask']
        sampled_kp = batch['sampled_kp']
        kp_label = batch['kp_label']
        
        # Graph data
        graph_points = batch['graph_points'] # [B, N_points, 2]
        pairs = batch['pairs'] # [B, N_samples, N_pairs, 2]
        connected = batch['connected'] # [B, N_samples, N_pairs]
        valid = batch['valid'] # [B, N_samples, N_pairs]

        # Forward pass
        mask_logits, mask_scores, topo_logits, topo_scores = self(
            rgb, sampled_kp, kp_label, graph_points, pairs, valid
        )
        # Compute mask loss
        mask_loss, bce_loss, dice_loss = self._compute_mask_loss(mask_logits, keypoint_mask, road_mask)
        # Compute topology loss if we have topology data
        topo_loss = self._compute_topology_loss(topo_logits, connected, valid)

        total_loss = mask_loss + topo_loss
        # Logging
        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mask_loss', mask_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_bce_loss', bce_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_dice_loss', dice_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_topo_loss', topo_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Update mask metrics
        self.val_keypoint_iou.update(mask_scores[..., 0], keypoint_mask)
        self.val_road_iou.update(mask_scores[..., 1], road_mask)
        # Update topology metrics
        valid_flat = valid.view(-1).bool()
        if valid_flat.sum() > 0:
            self.val_topo_acc.update(
                topo_scores.view(-1)[valid_flat], 
                connected.view(-1)[valid_flat]
            )
            self.val_topo_f1.update(
                topo_scores.view(-1)[valid_flat], 
                connected.view(-1)[valid_flat]
            )

        # Log first batch of validation images
        if batch_idx == 0:
            self._log_images(batch, mask_scores, prefix="val")

    def on_validation_epoch_end(self):
        self._log_mask_metrics(prefix="val_epoch_")
        self._log_topo_metrics(prefix="val_epoch_")
        
        # Update best validation metrics
        val_loss = self.trainer.callback_metrics.get('val_loss')
        val_keypoint_iou = self.trainer.callback_metrics.get('val_epoch_keypoint_iou')
        val_road_iou = self.trainer.callback_metrics.get('val_epoch_road_iou')
        val_topo_loss = self.trainer.callback_metrics.get('val_topo_loss')
        val_topo_acc = self.trainer.callback_metrics.get('val_epoch_topo_acc')
        val_topo_f1 = self.trainer.callback_metrics.get('val_epoch_topo_f1')
        
        if val_loss < self.best_metrics['val_loss']['value']:
            self.best_metrics['val_loss']['value'] = val_loss.item()
            self.best_metrics['val_loss']['epoch'] = self.current_epoch
            
        if val_keypoint_iou > self.best_metrics['val_keypoint_iou']['value']:
            self.best_metrics['val_keypoint_iou']['value'] = val_keypoint_iou.item()
            self.best_metrics['val_keypoint_iou']['epoch'] = self.current_epoch
            
        if val_road_iou > self.best_metrics['val_road_iou']['value']:
            self.best_metrics['val_road_iou']['value'] = val_road_iou.item()
            self.best_metrics['val_road_iou']['epoch'] = self.current_epoch
            
        if val_topo_loss < self.best_metrics['val_topo_loss']['value']:
            self.best_metrics['val_topo_loss']['value'] = val_topo_loss.item()
            self.best_metrics['val_topo_loss']['epoch'] = self.current_epoch
            
        if val_topo_acc > self.best_metrics['val_topo_acc']['value']:
            self.best_metrics['val_topo_acc']['value'] = val_topo_acc.item()
            self.best_metrics['val_topo_acc']['epoch'] = self.current_epoch
            
        if val_topo_f1 > self.best_metrics['val_topo_f1']['value']:
            self.best_metrics['val_topo_f1']['value'] = val_topo_f1.item()
            self.best_metrics['val_topo_f1']['epoch'] = self.current_epoch

    def test_step(self, batch, batch_idx):
        rgb = batch['rgb']
        keypoint_mask = batch['keypoint_mask']
        road_mask = batch['road_mask']
        sampled_kp = batch['sampled_kp']
        kp_label = batch['kp_label']
        
        # Graph data
        graph_points = batch['graph_points'] # [B, N_points, 2]
        pairs = batch['pairs'] # [B, N_samples, N_pairs, 2]
        connected = batch['connected'] # [B, N_samples, N_pairs]
        valid = batch['valid'] # [B, N_samples, N_pairs]

        # Forward pass
        mask_logits, mask_scores, topo_logits, topo_scores = self(
            rgb, sampled_kp, kp_label, graph_points, pairs, valid
        )
        # Update PR curve metrics for masks (no loss computation during test)
        self.keypoint_pr_curve.update(mask_scores[..., 0], keypoint_mask.to(torch.int32))
        self.road_pr_curve.update(mask_scores[..., 1], road_mask.to(torch.int32))
        
        # Update PR curve metrics for topology if available
        valid_flat = valid.view(-1).bool()
        if valid_flat.sum() > 0:
            self.topo_pr_curve.update(
                topo_scores.view(-1)[valid_flat], 
                connected.view(-1)[valid_flat].to(torch.int32)
            )

    def on_test_start(self):
        print("\n======= Test Results =======")
        self.keypoint_pr_curve.reset()
        self.road_pr_curve.reset()
        self.topo_pr_curve.reset()
        # load model ckpt
        ckpt_path = self.config.get("TEST_CKPT_PATH", None)
        if ckpt_path:
            self.load_state_dict(torch.load(ckpt_path)['state_dict'])
        else:
            raise ValueError("No test ckpt path provided")
        print(f"Loaded test model from {ckpt_path}")

    def on_test_end(self):
        print("\n======= Test Results =======")
        self._plot_and_log_pr_curve(self.keypoint_pr_curve, "Keypoint")
        self._plot_and_log_pr_curve(self.road_pr_curve, "Road")
        self._plot_and_log_pr_curve(self.topo_pr_curve, "Topology")
        print("===========================")

    def _plot_and_log_pr_curve(self, pr_curve_metric, category_name):
        try:
            precision, recall, thresholds = pr_curve_metric.compute()
            if len(thresholds) == len(precision) - 1: # Common case for PR curves
                 precision = precision[:-1]
                 recall = recall[:-1] 

            # Calculate F1 score for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6) # Add epsilon for stability

            # Find best threshold based on F1
            # Ensure threshold exists for argmax index
            valid_indices = ~torch.isnan(f1_scores) & ~torch.isinf(f1_scores)
            if valid_indices.sum() == 0:
                 print(f"Could not find valid F1 scores for {category_name}")
                 return

            best_f1_index = torch.argmax(f1_scores[valid_indices])
            # Map back to original index if needed (though argmax on filtered tensor should be okay)
            original_indices = torch.where(valid_indices)[0]
            best_threshold_index = original_indices[best_f1_index]

            best_threshold = thresholds[best_threshold_index]
            best_precision = precision[best_threshold_index]
            best_recall = recall[best_threshold_index]
            best_f1 = f1_scores[valid_indices][best_f1_index] # Use F1 from valid scores

            print(f"--- {category_name} ---")
            print(f"  Best Threshold (Max F1): {best_threshold:.4f}")
            print(f"  Precision at Best Threshold: {best_precision:.4f}")
            print(f"  Recall at Best Threshold:    {best_recall:.4f}")
            print(f"  F1 Score at Best Threshold:  {best_f1:.4f}")

            # Log best metrics
            if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'add_scalar'):
                self.logger.experiment.add_scalar(f"test/{category_name}_best_threshold", float(best_threshold), self.global_step)
                self.logger.experiment.add_scalar(f"test/{category_name}_best_precision", float(best_precision), self.global_step)
                self.logger.experiment.add_scalar(f"test/{category_name}_best_recall", float(best_recall), self.global_step)
                self.logger.experiment.add_scalar(f"test/{category_name}_best_f1", float(best_f1), self.global_step)

            # Plot PR Curve to TensorBoard
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot(recall.cpu(), precision.cpu(), label=f'{category_name} PR Curve')
            ax.scatter(best_recall.cpu(), best_precision.cpu(), color='red', label=f'Best F1 (Thresh={best_threshold:.3f})', zorder=5)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f'{category_name} Precision-Recall Curve')
            ax.legend()
            ax.grid(True)
            self.logger.experiment.add_figure(f'test_{category_name}_PR_Curve', fig, self.global_step)
            plt.close(fig)

        except Exception as e:
            print(f"Error processing PR curve for {category_name}: {e}")
        finally:
            pr_curve_metric.reset() # Reset metric

    def _log_images(self, batch, mask_scores, prefix="train", max_viz=4):
        """
        Log mask prediction visualizations to TensorBoard.
        """
        # Visualize only occasionally for training, always first batch for validation
        should_log = prefix == "val"  or \
                     (prefix == "train" and self.global_rank == 0 and self.current_epoch % self.config.get("VIZ_EPOCH_INTERVAL", 5) == 0)

        if not should_log:
             return

        # Log images to TensorBoard
        rgb = batch['rgb'][:max_viz]
        keypoint_mask = batch['keypoint_mask'][:max_viz]
        road_mask = batch['road_mask'][:max_viz]
        pred_keypoint = mask_scores[:max_viz, :, :, 0]
        pred_road = mask_scores[:max_viz, :, :, 1]

        num_viz = min(max_viz, rgb.size(0))
        for i in range(num_viz):
            fig, axs = plt.subplots(1, 5, figsize=(15, 3)) # Increased figsize

            # Ensure correct types and ranges for plotting
            rgb_img = rgb[i].cpu().numpy().astype(np.uint8)
            gt_kp_img = keypoint_mask[i].cpu().numpy().astype(float) # Keep as float [0,1]
            gt_rd_img = road_mask[i].cpu().numpy().astype(float)
            pred_kp_img = pred_keypoint[i].detach().cpu().numpy()
            pred_rd_img = pred_road[i].detach().cpu().numpy()

            axs[0].imshow(rgb_img)
            axs[0].set_title('RGB')
            axs[0].axis('off')

            axs[1].imshow(gt_kp_img, cmap='gray', vmin=0, vmax=1)
            axs[1].set_title('GT Keypoint')
            axs[1].axis('off')

            axs[2].imshow(gt_rd_img, cmap='gray', vmin=0, vmax=1)
            axs[2].set_title('GT Road')
            axs[2].axis('off')

            axs[3].imshow(pred_kp_img, cmap='gray', vmin=0, vmax=1)
            axs[3].set_title('Pred Keypoint')
            axs[3].axis('off')

            axs[4].imshow(pred_rd_img, cmap='gray', vmin=0, vmax=1)
            axs[4].set_title('Pred Road')
            axs[4].axis('off')

            plt.tight_layout()
            self.logger.experiment.add_figure(f'{prefix}_sample_{i}', fig, self.global_step)
            plt.close(fig)

    # on_validation_end 是在on_validation_epoch_end结束之后调用的，并不是所有epoch技术后调用的，所以使用on_fit_end
    def on_fit_end(self): 
        """Called at the end of training to display best metrics across all epochs."""
        print("\n" + "="*50)
        print("BEST METRICS SUMMARY")
        print("="*50)
        
        print("\nTraining Metrics:")
        print(f"  Best Train Loss: {self.best_metrics['train_loss']['value']:.4f} (Epoch {self.best_metrics['train_loss']['epoch']})")
        print(f"  Best Train Keypoint IoU: {self.best_metrics['train_keypoint_iou']['value']:.4f} (Epoch {self.best_metrics['train_keypoint_iou']['epoch']})")
        print(f"  Best Train Road IoU: {self.best_metrics['train_road_iou']['value']:.4f} (Epoch {self.best_metrics['train_road_iou']['epoch']})")
        print(f"  Best Train Topo Loss: {self.best_metrics['train_topo_loss']['value']:.4f} (Epoch {self.best_metrics['train_topo_loss']['epoch']})")
        print(f"  Best Train Topo Accuracy: {self.best_metrics['train_topo_acc']['value']:.4f} (Epoch {self.best_metrics['train_topo_acc']['epoch']})")
        print(f"  Best Train Topo F1: {self.best_metrics['train_topo_f1']['value']:.4f} (Epoch {self.best_metrics['train_topo_f1']['epoch']})")
        
        print("\nValidation Metrics:")
        print(f"  Best Val Loss: {self.best_metrics['val_loss']['value']:.4f} (Epoch {self.best_metrics['val_loss']['epoch']})")
        print(f"  Best Val Keypoint IoU: {self.best_metrics['val_keypoint_iou']['value']:.4f} (Epoch {self.best_metrics['val_keypoint_iou']['epoch']})")
        print(f"  Best Val Road IoU: {self.best_metrics['val_road_iou']['value']:.4f} (Epoch {self.best_metrics['val_road_iou']['epoch']})")
        print(f"  Best Val Topo Loss: {self.best_metrics['val_topo_loss']['value']:.4f} (Epoch {self.best_metrics['val_topo_loss']['epoch']})")
        print(f"  Best Val Topo Accuracy: {self.best_metrics['val_topo_acc']['value']:.4f} (Epoch {self.best_metrics['val_topo_acc']['epoch']})")
        print(f"  Best Val Topo F1: {self.best_metrics['val_topo_f1']['value']:.4f} (Epoch {self.best_metrics['val_topo_f1']['epoch']})")
        
        print("="*50)
        
        # Also log to TensorBoard as final metrics
        self.logger.experiment.add_scalar("best/train_loss", self.best_metrics['train_loss']['value'], self.best_metrics['train_loss']['epoch'])
        self.logger.experiment.add_scalar("best/train_keypoint_iou", self.best_metrics['train_keypoint_iou']['value'], self.best_metrics['train_keypoint_iou']['epoch'])
        self.logger.experiment.add_scalar("best/train_road_iou", self.best_metrics['train_road_iou']['value'], self.best_metrics['train_road_iou']['epoch'])
        self.logger.experiment.add_scalar("best/train_topo_loss", self.best_metrics['train_topo_loss']['value'], self.best_metrics['train_topo_loss']['epoch'])
        self.logger.experiment.add_scalar("best/train_topo_acc", self.best_metrics['train_topo_acc']['value'], self.best_metrics['train_topo_acc']['epoch'])
        self.logger.experiment.add_scalar("best/train_topo_f1", self.best_metrics['train_topo_f1']['value'], self.best_metrics['train_topo_f1']['epoch'])
        
        self.logger.experiment.add_scalar("best/val_loss", self.best_metrics['val_loss']['value'], self.best_metrics['val_loss']['epoch'])
        self.logger.experiment.add_scalar("best/val_keypoint_iou", self.best_metrics['val_keypoint_iou']['value'], self.best_metrics['val_keypoint_iou']['epoch'])
        self.logger.experiment.add_scalar("best/val_road_iou", self.best_metrics['val_road_iou']['value'], self.best_metrics['val_road_iou']['epoch'])
        self.logger.experiment.add_scalar("best/val_topo_loss", self.best_metrics['val_topo_loss']['value'], self.best_metrics['val_topo_loss']['epoch'])
        self.logger.experiment.add_scalar("best/val_topo_acc", self.best_metrics['val_topo_acc']['value'], self.best_metrics['val_topo_acc']['epoch'])
        self.logger.experiment.add_scalar("best/val_topo_f1", self.best_metrics['val_topo_f1']['value'], self.best_metrics['val_topo_f1']['epoch'])

    def configure_optimizers(self):
        param_dicts = []
        base_lr = self.config.BASE_LR
        encoder_lr_factor = self.config.get('ENCODER_LR_FACTOR', 0.1) 
        prompt_encoder_lr_factor = self.config.get('PROMPT_ENCODER_LR_FACTOR', 1.0)
        decoder_lr_factor = self.config.get('DECODER_LR_FACTOR', 1.0)
        topo_net_lr_factor = self.config.get('TOPO_NET_LR_FACTOR', 1.0)

        print("\n--- Optimizer Configuration ---")

        # --- Image Encoder Parameters ---
        if self.config.ENCODER_LORA:
            # Only optimize LoRA parameters
            lora_params = [p for name, p in self.named_parameters() if 'qkv.linear_' in name and p.requires_grad]
            if lora_params:
                 param_dicts.append({'params': lora_params, 'lr': base_lr})
                 print(f"Optimizing LoRA parameters ({sum(p.numel() for p in lora_params):,}) with LR: {base_lr}")
        elif not self.config.FREEZE_ENCODER:
            # Optimize all encoder parameters (distinguish pre-trained vs fresh if needed)
             encoder_params = [p for name, p in self.image_encoder.named_parameters() if p.requires_grad]
             if encoder_params:
                 param_dicts.append({'params': encoder_params, 'lr': base_lr * encoder_lr_factor})
                 print(f"Optimizing Image Encoder parameters ({sum(p.numel() for p in encoder_params):,}) with LR: {base_lr * encoder_lr_factor}")
        else:
             for p in self.image_encoder.parameters():
                 p.requires_grad = False
             print("Image Encoder is Frozen.")

        # --- Prompt Encoder Parameters ---
        prompt_encoder_params = [p for name, p in self.prompt_encoder.named_parameters() if p.requires_grad]
        if prompt_encoder_params:
            param_dicts.append({'params': prompt_encoder_params, 'lr': base_lr * prompt_encoder_lr_factor})
            print(f"Optimizing Prompt Encoder parameters ({sum(p.numel() for p in prompt_encoder_params):,}) with LR: {base_lr * prompt_encoder_lr_factor}")

        # --- Mask Decoder Parameters ---
        decoder_params = [p for name, p in self.mask_decoder.named_parameters() if p.requires_grad]
        if decoder_params:
            param_dicts.append({'params': decoder_params, 'lr': base_lr * decoder_lr_factor})
            print(f"Optimizing Mask Decoder parameters ({sum(p.numel() for p in decoder_params):,}) with LR: {base_lr * decoder_lr_factor}")
            
        # --- Topology Network Parameters ---
        topo_net_params = [p for name, p in self.toponet.named_parameters() if p.requires_grad]
            
        if topo_net_params:
            param_dicts.append({'params': topo_net_params, 'lr': base_lr * topo_net_lr_factor})
            print(f"Optimizing Topology Network parameters ({sum(p.numel() for p in topo_net_params):,}) with LR: {base_lr * topo_net_lr_factor}")

        # --- Sanity Check ---
        total_params = sum(sum(p.numel() for p in group['params']) for group in param_dicts)
        print(f"Total Optimizable Parameters: {total_params:,}")
        print("-----------------------------")

        if not param_dicts:
             raise ValueError("No parameters found to optimize. Check model freezing/LoRA settings.")

        # --- Optimizer ---
        # Allow configuring optimizer type and weight decay
        optimizer_name = self.config.get("OPTIMIZER", "AdamW").lower()
        weight_decay = self.config.get("WEIGHT_DECAY", 0.01)

        if optimizer_name == "adamw":
             print(f"Using AdamW optimizer with weight_decay={weight_decay}")
             optimizer = torch.optim.AdamW(param_dicts, lr=base_lr, weight_decay=weight_decay)
        elif optimizer_name == "adam":
             print("Using Adam optimizer")
             optimizer = torch.optim.Adam(param_dicts, lr=base_lr) # Adam usually doesn't use weight_decay arg directly
        else:
             raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # --- Scheduler ---
        # Allow configuring scheduler type, milestones, gamma
        scheduler_name = self.config.get("SCHEDULER", "MultiStepLR").lower()
        if scheduler_name == "multisteplr":
            # milestones = self.config.get("LR_MILESTONES", [int(self.trainer.max_epochs * 0.7), int(self.trainer.max_epochs * 0.9)])
            milestones = self.config.get("LR_MILESTONES", [int(self.trainer.max_epochs * 0.8), ])
            gamma = self.config.get("LR_GAMMA", 0.1)
            print(f"Using MultiStepLR scheduler with milestones={milestones}, gamma={gamma}")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_name == "cosine":
            print(f"Using CosineAnnealingLR scheduler")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs) # Adjust T_max as needed
        elif scheduler_name == "none":
            print("No LR scheduler used.")
            return optimizer # Return only optimizer if no scheduler
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch', # Adjust if needed ('step')
                'frequency': 1
            }
        }
