# Large Image Processing Functions for roadnet_service.py

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from fastapi import HTTPException
from omegaconf import DictConfig
import time
import json
import gzip
import base64
from scipy.spatial import KDTree
from collections import defaultdict
from rtree import index

# Import from parent modules
from app.models.road_extraction_algorithm import graph_extraction
from app.models.road_extraction_algorithm.sam_road_extraction import RoadExtractionModel
from app.core.config import settings
from app.services.mask_cache_service import mask_cache_service

logger = logging.getLogger(__name__)

# ========================================
# Mask Compression and Metadata Functions
# ========================================

def compress_mask_data(mask_array: np.ndarray) -> Optional[str]:
    """
    Compress mask data using gzip and base64 encoding for efficient transmission.
    This function is used by both regular and large image processing.
    
    Args:
        mask_array: Boolean numpy array to compress
        
    Returns:
        Base64 encoded compressed data or None if compression fails
    """
    try:
        if mask_array.size == 0:
            return None
            
        # Convert to uint8 for better compression
        mask_uint8 = mask_array.astype(np.uint8)
        
        # Compress using gzip
        compressed = gzip.compress(mask_uint8.tobytes())
        
        # Encode to base64 for JSON transmission
        encoded = base64.b64encode(compressed).decode('utf-8')
        
        logger.info(f"Mask compression: {mask_array.size} bytes -> {len(compressed)} bytes "
                   f"(ratio: {len(compressed)/mask_array.size:.3f})")
        
        return encoded
        
    except Exception as e:
        logger.error(f"Failed to compress mask data: {e}")
        return None

def create_mask_metadata(mask_array: np.ndarray) -> dict:
    """
    Create metadata for mask reconstruction.
    Used by both regular and large image processing.
    
    Args:
        mask_array: The mask array to create metadata for
        
    Returns:
        Dictionary containing mask metadata
    """
    return {
        'shape': mask_array.shape,
        'dtype': str(mask_array.dtype),
        'compressed': True
    }

def prepare_compressed_masks_response(
    road_mask_np: Optional[np.ndarray],
    kp_mask_np: Optional[np.ndarray],
    include_masks: bool = True
) -> Dict[str, Any]:
    """
    Prepare compressed mask data and metadata for API response.
    This function standardizes mask compression for both regular and large images,
    ensuring consistent transmission behavior and improved performance.
    
    Args:
        road_mask_np: Road mask numpy array (Binary)
        kp_mask_np: Keypoint mask numpy array (Binary)
        include_masks: Whether to include mask data in response
        
    Returns:
        Dictionary with compressed mask data and metadata
    """
    response_data = {
        "road_mask": None,
        "kp_mask": None,
        "road_mask_metadata": None,
        "kp_mask_metadata": None
    }
    
    if not include_masks:
        return response_data
    
    # Process road mask
    if road_mask_np is not None and road_mask_np.size > 0:
        road_mask_bool = road_mask_np.astype(bool)
        response_data["road_mask"] = compress_mask_data(road_mask_bool)
        response_data["road_mask_metadata"] = create_mask_metadata(road_mask_bool)
        logger.debug(f"Road mask processed: shape {road_mask_bool.shape}")
    
    # Process keypoint mask
    if kp_mask_np is not None and kp_mask_np.size > 0:
        kp_mask_bool = kp_mask_np.astype(bool)
        response_data["kp_mask"] = compress_mask_data(kp_mask_bool)
        response_data["kp_mask_metadata"] = create_mask_metadata(kp_mask_bool)
        logger.debug(f"Keypoint mask processed: shape {kp_mask_bool.shape}")
    
    return response_data

# ========================================
# Storage Compression Functions
# ========================================

def _group_prompts_by_patches(
    prompts_data: List[Dict[str, Any]], 
    metadata: Dict[str, Any]
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """
    Group prompts by the patches they belong to for efficient batch processing.
    
    Args:
        prompts_data: List of prompts with 'x', 'y' coordinates in original image scale
        metadata: Large image metadata containing grid info
        
    Returns:
        Dictionary mapping (row, col) tuples to lists of prompts in that patch
    """
    if not prompts_data:
        return {}
    
    patch_size = metadata.get('patch_size', settings.PATCH_SIZE)
    overlap = metadata.get('overlap', settings.PATCH_OVERLAP)
    grid_cols, grid_rows = metadata.get('grid_size_wh', [1, 1])
    
    stride = patch_size - overlap
    patch_prompts = {}
    
    for prompt in prompts_data:
        x, y = prompt['x'], prompt['y']
        
        # Find all patches that could contain this point
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Calculate patch boundaries
                patch_left = col * stride
                patch_top = row * stride
                patch_right = min(patch_left + patch_size, metadata['original_size_wh'][0])
                patch_bottom = min(patch_top + patch_size, metadata['original_size_wh'][1])
                
                # Check if point falls within this patch
                if (patch_left <= x <= patch_right and 
                    patch_top <= y <= patch_bottom):
                    
                    patch_key = (row, col)
                    if patch_key not in patch_prompts:
                        patch_prompts[patch_key] = []
                    
                    # Convert to local coordinates
                    local_prompt = {
                        'x': x - patch_left,
                        'y': y - patch_top,
                        'label': prompt['label'],
                        'global_x': x,  # Keep global coords for reference
                        'global_y': y
                    }
                    patch_prompts[patch_key].append(local_prompt)
    
    logger.info(f"Grouped {len(prompts_data)} prompts into {len(patch_prompts)} patches")
    return patch_prompts

def _build_graph_rtree_index(graph_points: np.ndarray) -> index.Index:
    """
    Build an rtree spatial index for graph points.
    
    Args:
        graph_points: [N, 2] numpy array with (x, y) coordinates
        
    Returns:
        rtree.index.Index object for spatial queries
    """
    graph_rtree = index.Index()
    
    for i, (x, y) in enumerate(graph_points):
        # Insert single points as (x, y, x, y) for rtree
        graph_rtree.insert(i, (x, y, x, y))
    
    logger.info(f"Built graph rtree index for {len(graph_points)} points")
    return graph_rtree

def _load_patch_features(patch_row: int, patch_col: int, features_dir: Path, image_id: str) -> Dict[str, Any]:
    """
    Load features for a specific patch.
    
    Args:
        patch_row: Row index of the patch
        patch_col: Column index of the patch  
        features_dir: Directory containing patch features
        image_id: Image identifier
        
    Returns:
        Dictionary containing patch features and metadata
    """
    patch_file = features_dir / image_id / f"patch_{patch_row}_{patch_col}.pt"
    
    if not patch_file.exists():
        raise FileNotFoundError(f"Patch features not found: {patch_file}")
    
    try:
        patch_data = torch.load(patch_file, map_location='cpu')
        return patch_data
    except Exception as e:
        raise ValueError(f"Failed to load patch features from {patch_file}: {e}")

def _generate_batch_masks(
    patch_data_list: List[Dict[str, Any]],
    model: RoadExtractionModel,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate masks for a batch of patches efficiently using model's batch processing capability.
    
    Args:
        patch_data_list: List of patch data dictionaries with keys:
            - 'patch_row', 'patch_col': patch indices
            - 'patch_features': loaded patch features tensor
            - 'local_prompts': list of prompts with local coordinates
            - 'actual_size': (height, width) of actual patch content
        model: Road extraction model
        device: Computation device
        
    Returns:
        List of (kp_mask, road_mask) tuples for each patch
    """
    if not patch_data_list:
        return []
    
    # Prepare batch data
    batch_features = []
    batch_coords = []
    batch_labels = []
    max_prompts = 0
    
    for patch_data in patch_data_list:
        local_prompts = patch_data['local_prompts']
        patch_features = patch_data['patch_features']
        
        if not local_prompts:
            # No prompts in this patch - use dummy data
            batch_features.append(patch_features)
            batch_coords.append([[0.0, 0.0]])  # Dummy coord
            batch_labels.append([0])  # Dummy label
            max_prompts = max(max_prompts, 1)
        else:
            # Extract coordinates and labels
            coords = [[p['x'], p['y']] for p in local_prompts]
            labels = [p['label'] for p in local_prompts]
            
            batch_features.append(patch_features)
            batch_coords.append(coords)
            batch_labels.append(labels)
            max_prompts = max(max_prompts, len(coords))
    
    # Pad prompt sequences to same length for batching
    padded_coords = []
    padded_labels = []
    
    for coords, labels in zip(batch_coords, batch_labels):
        # Pad with last coordinate if needed
        while len(coords) < max_prompts:
            coords.append(coords[-1] if coords else [0.0, 0.0])
            labels.append(-1)  # Padding label
        
        padded_coords.append(coords[:max_prompts])
        padded_labels.append(labels[:max_prompts])
    
    # Convert to tensors
    batch_features_tensor = torch.stack(batch_features).to(device)  # [B, D, H, W]
    batch_coords_tensor = torch.tensor(padded_coords, device=device, dtype=torch.float32)  # [B, N, 2]
    batch_labels_tensor = torch.tensor(padded_labels, device=device, dtype=torch.int64)  # [B, N]
    
    # Batch inference
    _, mask_scores, mask_logits = model.infer_masks_and_features_from_prompts(
        image_features=batch_features_tensor,
        kp_coords=batch_coords_tensor,
        kp_labels=batch_labels_tensor,
    )

    return mask_scores, mask_logits # [B, 2, H, W]

def _compress_mask_for_storage(mask_array: np.ndarray) -> bytes:
    """
    Compress mask data for efficient storage using gzip compression.
    Optimized for I/O performance with boolean arrays.
    
    Args:
        mask_array: Boolean numpy array to compress
        
    Returns:
        Compressed bytes data for storage
    """
    try:
        if mask_array.size == 0:
            return b""
            
        # Ensure boolean type for optimal compression
        mask_bool = mask_array.astype(np.bool_)
        
        # Use gzip with optimal compression level for I/O speed vs size balance
        # Level 3 provides good compression ratio with fast decompression
        compressed = gzip.compress(mask_bool.tobytes(), compresslevel=3)
        
        logger.debug(f"Storage compression: {mask_array.size} bytes -> {len(compressed)} bytes "
                    f"(ratio: {len(compressed)/mask_array.size:.3f})")
        
        return compressed
        
    except Exception as e:
        logger.error(f"Failed to compress mask for storage: {e}")
        raise

def _save_compressed_mask_file(mask_data: np.ndarray, file_path: Path) -> dict:
    """
    Save mask to compressed file with metadata for fast loading.
    Saves shape information in the file header for efficient loading.
    
    Args:
        mask_data: Mask array to save
        file_path: Path to save compressed file
        
    Returns:
        Metadata dictionary for the saved mask
    """
    try:
        # Create metadata
        metadata = {
            'shape': mask_data.shape,
            'dtype': str(mask_data.dtype),
            'compressed': True,
            'size_bytes': mask_data.size
        }
        
        # Compress mask data
        compressed_data = _compress_mask_for_storage(mask_data)
        
        # Create file header with shape information
        # Format: [shape_info_length(4 bytes)][shape_info(JSON)][compressed_data]
        shape_info = json.dumps(mask_data.shape).encode('utf-8')
        shape_info_length = len(shape_info).to_bytes(4, byteorder='little')
        
        # Save compressed data with header
        with open(file_path, 'wb') as f:
            f.write(shape_info_length)  # 4 bytes for length
            f.write(shape_info)         # Shape info as JSON
            f.write(compressed_data)    # Compressed mask data
        
        metadata['compressed_size_bytes'] = len(compressed_data) + len(shape_info) + 4
        metadata['compression_ratio'] = metadata['compressed_size_bytes'] / mask_data.size if mask_data.size > 0 else 0
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to save compressed mask file {file_path}: {e}")
        raise

def _load_compressed_mask_file(file_path: Path) -> np.ndarray:
    """
    Load mask from compressed file with shape header (new format only).
    
    Args:
        file_path: Path to compressed mask file
        
    Returns:
        Decompressed mask array
    """
    try:
        # Load file data
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        if len(file_data) < 4:
            logger.error(f"Invalid compressed file format: {file_path} (too small)")
            raise ValueError(f"Invalid compressed file format: {file_path}")
        
        # Read shape header (new format only)
        shape_info_length = int.from_bytes(file_data[:4], byteorder='little')
        
        # Sanity check: shape info length should be reasonable (< 1KB)
        if shape_info_length <= 0 or shape_info_length >= 1024:
            logger.error(f"Invalid shape info length in compressed file: {file_path} (length: {shape_info_length})")
            raise ValueError(f"Invalid compressed file format: {file_path}")
        
        if len(file_data) < 4 + shape_info_length:
            logger.error(f"Incomplete compressed file: {file_path}")
            raise ValueError(f"Incomplete compressed file: {file_path}")
        
        shape_info_bytes = file_data[4:4+shape_info_length]
        compressed_data = file_data[4+shape_info_length:]
        
        # Parse shape info
        shape = tuple(json.loads(shape_info_bytes.decode('utf-8')))
        
        # Decompress and reshape
        decompressed_bytes = gzip.decompress(compressed_data)
        mask_array = np.frombuffer(decompressed_bytes, dtype=np.bool_).reshape(shape)
        
        logger.debug(f"Loaded compressed mask: {file_path}, shape: {mask_array.shape}")
        return mask_array
        
    except Exception as e:
        logger.error(f"Failed to load compressed mask file {file_path}: {e}")
        raise

async def _generate_road_network_for_large_image(
    image_id: str,
    prompts_data: List[Dict[str, Any]],
    model: RoadExtractionModel,
    config: DictConfig,
    device: torch.device,
    features_dir: Path,
    metadata_path: Path,
    include_masks: bool = True  # Add option to exclude masks for faster response
) -> dict:
    """
    Generate road network for large images using tiled processing.
    Implements Phase 2: steps 2.1-2.2 up to mask fusion.
    
    Args:
        include_masks: Whether to include mask data in response (for faster responses without masks)
    """
    start_time = time.time()
    cost_time = {
        'load_metadata': 0,
        'identify_patches': 0, 
        'local_inference': 0,
        'mask_fusion': 0,
        'mask_compression': 0,
        'total': 0
    }
    
    try:
        # Load metadata
        step_start = time.time()
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if metadata.get('feature_status') != 'ready':
            raise ValueError(f"Large image features not ready. Status: {metadata.get('feature_status')}")
        
        original_width, original_height = metadata['original_size_wh']
        patch_size = metadata['patch_size']
        overlap = metadata['overlap']
        grid_cols, grid_rows = metadata['grid_size_wh']
        
        logger.info(f"Processing large image {original_width}x{original_height}, "
                   f"grid {grid_cols}x{grid_rows}, patch_size {patch_size}, overlap {overlap}")
        cost_time['load_metadata'] = time.time() - step_start
        
        model.eval()

        # Step A: Group prompts by patches
        step_start = time.time()
        patch_prompts = _group_prompts_by_patches(prompts_data, metadata)
        
        if not patch_prompts:
            logger.warning("No patches affected by prompts, returning empty result")
            return {
                "image_id": image_id,
                "geojson_data": {"type": "FeatureCollection", "features": []},
                "prompts": prompts_data,
                "road_mask": None,
                "kp_mask": None,
                "road_mask_metadata": None,
                "kp_mask_metadata": None
            }
        cost_time['identify_patches'] = time.time() - step_start
        
        # Step B: Efficient batch inference and mask fusion
        step_start = time.time()
        
        # Create global masks
        global_kp_mask = torch.zeros(original_height, original_width, device=device, dtype=torch.float32)
        global_road_mask = torch.zeros(original_height, original_width, device=device, dtype=torch.float32)
        global_weight_mask = torch.zeros(original_height, original_width, device=device, dtype=torch.float32)
        
        stride = patch_size - overlap
        
        # Prepare patch data for batch processing
        all_patch_coords = list(patch_prompts.keys())
        batch_size = settings.MASK_GENERATION_BATCH_SIZE  # From config

        # Initialize global data info to collect all patch information
        global_data_info = {
            'patch_row': [],
            'patch_col': [],
            'patch_features': [],
            'mask_logits': [],
            'local_prompts': [],
            'actual_size': [],
            'coors': []  # Global coordinates for each patch
        }
        
        for batch_start in range(0, len(all_patch_coords), batch_size):
            batch_end = min(batch_start + batch_size, len(all_patch_coords))
            batch_patch_coords = all_patch_coords[batch_start:batch_end]
            
            logger.info(f"Processing patch batch {batch_start//batch_size + 1}/{(len(all_patch_coords) + batch_size - 1)//batch_size}: "
                       f"patches {batch_start}-{batch_end-1}")
            
            # Prepare batch data
            batch_data = []
            batch_metadata = []
            
            for patch_row, patch_col in batch_patch_coords:
                try:
                    # Load patch features
                    patch_data = _load_patch_features(patch_row, patch_col, features_dir, image_id)
                    patch_features = patch_data['image_embedding'].to(device)
                    local_prompts = patch_prompts[(patch_row, patch_col)]
                    
                    # Calculate patch boundaries for fusion
                    patch_left = patch_col * stride
                    patch_top = patch_row * stride
                    patch_right = min(patch_left + patch_size, original_width)
                    patch_bottom = min(patch_top + patch_size, original_height)
                    
                    batch_data.append({
                        'patch_row': patch_row,
                        'patch_col': patch_col,
                        'patch_features': patch_features,
                        'local_prompts': local_prompts,
                        'actual_size': patch_data['actual_size']
                    })
                    
                    batch_metadata.append({
                        'patch_left': patch_left,
                        'patch_top': patch_top,
                        'patch_right': patch_right,
                        'patch_bottom': patch_bottom
                    })
                    
                except Exception as e:
                    logger.error(f"Error loading patch ({patch_row}, {patch_col}): {e}")
                    continue
            
            if not batch_data:
                continue
            
            # Collect patch information for global data
            for patch_info in batch_data:
                global_data_info['patch_row'].append(patch_info['patch_row'])
                global_data_info['patch_col'].append(patch_info['patch_col'])
                global_data_info['patch_features'].append(patch_info['patch_features'])
                global_data_info['local_prompts'].append(patch_info['local_prompts'])
                global_data_info['actual_size'].append(patch_info['actual_size'])

            for metadata in batch_metadata:
                global_data_info['coors'].append([metadata['patch_left'], metadata['patch_top'], metadata['patch_right'], metadata['patch_bottom']])

            # Batch inference
            try:
                with torch.no_grad():
                    mask_scores, mask_logits = _generate_batch_masks(batch_data, model, device)
                
                logger.debug(f"Batch processing completed. mask_scores shape: {mask_scores.shape}, "
                           f"batch_data length: {len(batch_data)}")
                
                # Fusion of batch results into global masks
                for i in range(len(batch_data)):  # Fix: use len(batch_data) instead of batch_size because batch_size is not always the same as the number of patches
                    try:
                        meta = batch_metadata[i]
                        patch_left = meta['patch_left']
                        patch_top = meta['patch_top']
                        patch_right = meta['patch_right']
                        patch_bottom = meta['patch_bottom']
                        
                        patch_row = batch_data[i]['patch_row']
                        patch_col = batch_data[i]['patch_col']
                        
                        local_kp_mask = mask_scores[i, 0] # [H, W]
                        local_road_mask = mask_scores[i, 1] # [H, W]
                        
                        # Fix: correctly access actual_size values
                        actual_size_dict = batch_data[i]['actual_size']
                        actual_height = actual_size_dict['height']
                        actual_width = actual_size_dict['width']
                        
                        # Calculate expected global region size
                        expected_height = patch_bottom - patch_top
                        expected_width = patch_right - patch_left
                        
                        # Safety check: ensure actual size matches the expected region size
                        if actual_height != expected_height or actual_width != expected_width:
                            logger.warning(f"Size mismatch for patch ({patch_row}, {patch_col}): "
                                         f"actual_size=({actual_height}, {actual_width}), "
                                         f"expected=({expected_height}, {expected_width})")
                            # Use the minimum dimensions to avoid index errors
                            crop_height = min(actual_height, expected_height, local_kp_mask.shape[0])
                            crop_width = min(actual_width, expected_width, local_kp_mask.shape[1])
                        else:
                            crop_height = actual_height
                            crop_width = actual_width
                        
                        # Crop local masks to actual size if needed
                        local_kp_crop = local_kp_mask[:crop_height, :crop_width]
                        local_road_crop = local_road_mask[:crop_height, :crop_width]
                        
                        # Add to global masks with proper fusion (weighted averaging for overlaps)
                        global_kp_mask[patch_top:patch_top+crop_height, patch_left:patch_left+crop_width] += local_kp_crop
                        global_road_mask[patch_top:patch_top+crop_height, patch_left:patch_left+crop_width] += local_road_crop
                        global_weight_mask[patch_top:patch_top+crop_height, patch_left:patch_left+crop_width] += 1.0
                        
                    except Exception as patch_error:
                        logger.error(f"Error processing individual patch ({batch_data[i]['patch_row']}, {batch_data[i]['patch_col']}) "
                                   f"in batch: {patch_error}")
                        continue
                    
            except Exception as e:
                logger.error(f"Error processing patch batch: {e}")
                # Log additional debug information
                logger.error(f"batch_data length: {len(batch_data) if batch_data else 'None'}")
                logger.error(f"batch_metadata length: {len(batch_metadata) if batch_metadata else 'None'}")
                continue

            for mask_logit in mask_logits:
                global_data_info['mask_logits'].append(mask_logit)
        
        if global_data_info['patch_features']:
            # Stack all patch features into [B, C, H, W] format
            global_data_info['patch_features'] = torch.stack(global_data_info['patch_features'], dim=0)
            global_data_info['mask_logits'] = torch.stack(global_data_info['mask_logits'], dim=0)
            logger.info(f"Global patch features shape: {global_data_info['patch_features'].shape}")
            logger.info(f"Global mask logits shape: {global_data_info['mask_logits'].shape}")
            
            # Validate that all keys have consistent lengths
            expected_length = len(global_data_info['patch_row'])
            for key, value in global_data_info.items():
                if key == 'patch_features':
                    actual_length = value.shape[0]
                else:
                    actual_length = len(value)
                
                if actual_length != expected_length:
                    logger.warning(f"Inconsistent length for key '{key}': expected {expected_length}, got {actual_length}")
        
        # Normalize by weights where overlaps occurred
        mask_nonzero = global_weight_mask > 0
        global_kp_mask[mask_nonzero] /= global_weight_mask[mask_nonzero]
        global_road_mask[mask_nonzero] /= global_weight_mask[mask_nonzero]
        
        cost_time['local_inference'] = time.time() - step_start
        
        # Step C: Global post-processing - extract graph points from fused masks  
        step_start = time.time()
        
        # Extract graph points from the fused global masks
        graph_points, filtered_global_kp_mask, filtered_global_road_mask = _extract_global_graph_points_from_fused_masks(
            global_kp_mask, global_road_mask, config, device
        )
        
        logger.info(f"Extracted {graph_points.shape[0] if graph_points.size > 0 else 0} graph points from global masks")
        
        cost_time['global_post_processing'] = time.time() - step_start
        
        # Step D: Topology inference
        step_start = time.time()
        
        pred_edges = []
        if graph_points.size > 0:  # Only proceed if we have valid graph points
            # Perform topology inference
            pred_edges = _perform_large_image_topology_inference(
                graph_points, model, config, device, global_data_info
            )
            
            logger.info(f"Topology inference completed: {len(pred_edges)} edges predicted")
        else:
            logger.warning("No graph points extracted, skipping topology inference")
            
        cost_time['topology_inference'] = time.time() - step_start
        
        # Step E: Convert to GeoJSON output format
        step_start = time.time()
        
        # Convert edges to numpy array format
        edges_np = np.array(pred_edges) if pred_edges else np.empty((0, 2), dtype=np.int32)
        
        # Generate GeoJSON using existing conversion function
        geojson_output = graph_extraction.convert_to_geojson(
            nodes_xy=graph_points,
            edges=edges_np,
            image_id=image_id,
        )
        
        logger.info(f"Large image road network generation completed successfully: "
                   f"{graph_points.shape[0] if graph_points.size > 0 else 0} nodes, {len(pred_edges)} edges")
        
        cost_time['geojson_conversion'] = time.time() - step_start
        
        # Step F: Optimize mask data for transmission
        step_start = time.time()
        
        response_data = {
            "image_id": image_id,
            "geojson_data": geojson_output,
            "prompts": prompts_data,
            "road_mask": None,
            "kp_mask": None,
            "road_mask_metadata": None,
            "kp_mask_metadata": None
        }
        
        # Cache and prepare masks for transmission
        if filtered_global_road_mask.size > 0 or filtered_global_kp_mask.size > 0:
            road_mask_np = filtered_global_road_mask if filtered_global_road_mask.size > 0 else None
            kp_mask_np = filtered_global_kp_mask if filtered_global_kp_mask.size > 0 else None
            
            # Cache masks in backend for future save operations
            mask_cache_service.cache_masks(
                image_id=image_id,
                road_mask=road_mask_np,
                kp_mask=kp_mask_np
            )
            
            mask_response_data = prepare_compressed_masks_response(
                road_mask_np=road_mask_np,
                kp_mask_np=kp_mask_np,
                include_masks=include_masks
            )
            
            # Update response with compressed mask data
            response_data.update(mask_response_data)
            
            if include_masks:
                logger.info(f"Large image masks compressed for transmission and cached: road_mask={road_mask_np is not None}, kp_mask={kp_mask_np is not None}")
        
        cost_time['mask_compression'] = time.time() - step_start
        cost_time['total'] = time.time() - start_time
        
        logger.info(f"Large image processing completed. Cost time: {cost_time}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in large image road network generation: {e}", exc_info=True)
        raise

def _extract_global_graph_points_from_fused_masks(
    global_kp_mask: torch.Tensor,
    global_road_mask: torch.Tensor, 
    config: DictConfig,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract graph points from globally fused masks for large images.
    
    Args:
        global_kp_mask: [H, W] tensor with keypoint predictions
        global_road_mask: [H, W] tensor with road predictions
        config: Model configuration with NMS and threshold parameters
        device: Computation device
        
    Returns:
        Tuple of (graph_points, filtered_kp_mask, filtered_road_mask)
        - graph_points: [N, 2] numpy array with (x, y) coordinates in global image scale
        - filtered_kp_mask: processed keypoint mask as numpy array
        - filtered_road_mask: processed road mask as numpy array
    """
    
    start_time = time.time()
    
    # Convert to numpy for graph_extraction processing
    kp_mask_np = global_kp_mask.cpu().numpy().astype(np.float32)
    road_mask_np = global_road_mask.cpu().numpy().astype(np.float32)
    
    logger.info(f"Extracting graph points from fused masks: "
               f"kp_mask {kp_mask_np.shape}, road_mask {road_mask_np.shape}")
    
    # Use existing graph extraction logic
    graph_points, filtered_kp_mask, filtered_road_mask = graph_extraction.extract_graph_points(
        kp_mask_np, road_mask_np, config
    )
    
    extraction_time = time.time() - start_time
    logger.info(f"Global graph point extraction completed in {extraction_time:.2f}s: "
               f"extracted {graph_points.shape[0] if graph_points.size > 0 else 0} points")
    
    return graph_points, filtered_kp_mask, filtered_road_mask

def _perform_large_image_topology_inference(
    graph_points: np.ndarray,
    model: RoadExtractionModel,
    config: DictConfig,
    device: torch.device,
    global_data_info: Dict[str, Any]
) -> List[Tuple[int, int]]:
    """
    Perform topology inference for large images using global patch data.
    
    Args:
        graph_points: [N, 2] numpy array with (x, y) coordinates in global scale
        model: RoadExtractionModel for topology inference
        config: Model configuration
        device: Computation device
        global_data_info: Dictionary containing patch information with keys:
                         - patch_row, patch_col, patch_features, mask_logits, local_prompts, actual_size, coors
        
    Returns:
        List of edge tuples (src_idx, tgt_idx) representing predicted connections
    """
    
    if graph_points.shape[0] == 0:
        logger.info("No graph points available for topology inference")
        return []
    
    start_time = time.time()
    topo_time_breakdown = {
        'graph_rtree_build': 0,
        'patch_processing': 0,
        'batch_preparation': 0,
        'model_inference': 0,
        'edge_aggregation': 0
    }
    
    logger.info(f"Starting topology inference for {graph_points.shape[0]} graph points")
    logger.info(f"Using global patch data with {len(global_data_info['patch_row'])} patches")
    
    if isinstance(global_data_info['patch_features'], torch.Tensor):
        logger.info(f"Global patch features tensor shape: {global_data_info['patch_features'].shape}")
    
    # Step 1: Build graph rtree index for all points
    step_start = time.time()
    graph_rtree = _build_graph_rtree_index(graph_points)
    topo_time_breakdown['graph_rtree_build'] = time.time() - step_start
    
    # Step 2: Process each patch to get patch-specific topology data
    step_start = time.time()
    
    # Get patch features and mask logits from global data
    patch_features = global_data_info['patch_features']  # [B, C, H, W]
    patch_mask_logits = global_data_info['mask_logits']  # [B, 2, H, W]
    all_patch_info = global_data_info['coors']  # List of [left, top, right, bottom]
    
    batch_size = settings.TOPO_CLASSIFICATION_BATCH_SIZE
    batch_num = (len(all_patch_info) + batch_size - 1) // batch_size
    
    topo_time_breakdown['patch_processing'] = time.time() - step_start
    
    # Step 3: Process batches
    step_start = time.time()
    
    edge_scores = defaultdict(float)
    edge_counts = defaultdict(float)
    
    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]
        
        topo_data = {
            'points': [],
            'pairs': [],
            'valid': [],
        }
        idx_maps = []
        
        # Prepare pairs queries for each patch in batch
        for patch_info in batch_patch_info:
            x0, y0, x1, y1 = patch_info
            
            # Find points in this patch using rtree
            patch_point_indices = list(graph_rtree.intersection((x0, y0, x1, y1)))
            idx_patch2all = {patch_idx: all_idx for patch_idx, all_idx in enumerate(patch_point_indices)}
            patch_point_num = len(patch_point_indices)
            
            if patch_point_num == 0:
                # No points in this patch, add empty data
                topo_data['points'].append(np.zeros((0, 2), dtype=np.float32))
                topo_data['pairs'].append(np.zeros((0, config.MAX_NEIGHBOR_QUERIES, 2), dtype=np.int32))
                topo_data['valid'].append(np.zeros((0, config.MAX_NEIGHBOR_QUERIES), dtype=bool))
                idx_maps.append({})
                continue
            
            # Normalize points into patch coordinates
            patch_points = graph_points[patch_point_indices, :] - np.array([[x0, y0]], dtype=graph_points.dtype)
            
            # Build KDTree for points in this patch
            patch_kdtree = KDTree(patch_points)
            
            # Find neighbors within patch
            # k+1 because the nearest one is always self
            knn_d, knn_idx = patch_kdtree.query(
                patch_points, 
                k=config.MAX_NEIGHBOR_QUERIES + 1, 
                distance_upper_bound=config.NEIGHBOR_RADIUS
            )
            
            # Remove self connections
            knn_idx = knn_idx[:, 1:]  # [patch_point_num, n_nbr]
            
            # Create pairs
            src_idx = np.tile(
                np.arange(patch_point_num)[:, np.newaxis],
                (1, config.MAX_NEIGHBOR_QUERIES)
            )
            valid = knn_idx < patch_point_num
            tgt_idx = np.where(valid, knn_idx, src_idx)
            pairs = np.stack([src_idx, tgt_idx], axis=-1)  # [patch_point_num, n_nbr, 2]
            
            topo_data['points'].append(patch_points)
            topo_data['pairs'].append(pairs)
            topo_data['valid'].append(valid)
            idx_maps.append(idx_patch2all)
        
        # Collate batch data
        collated = {}
        for key, x_list in topo_data.items():
            length = max([x.shape[0] for x in x_list])
            collated[key] = np.stack([
                np.pad(x, [(0, length - x.shape[0])] + [(0, 0)] * (len(x.shape) - 1))
                for x in x_list
            ], axis=0)
        
        # Skip this batch if there's no points
        if not collated or collated['points'].shape[1] == 0:
            continue
        
        topo_time_breakdown['batch_preparation'] += time.time() - step_start
        
        # Step 4: Model inference
        inference_start = time.time()
        
        # Get batch features and mask logits
        batch_features = patch_features[offset:offset+len(batch_patch_info)]  # [batch_size, C, H, W]
        batch_mask_logits = patch_mask_logits[offset:offset+len(batch_patch_info)]  # [batch_size, 2, H, W]
        
        # Prepare tensors for model input
        batch_points = torch.tensor(collated['points'], device=device, dtype=torch.float32)
        batch_pairs = torch.tensor(collated['pairs'], device=device)
        batch_valid = torch.tensor(collated['valid'], device=device)
        
        model.eval()
        with torch.no_grad():
            try:
                # Call model's infer_toponet method
                topo_scores = model.infer_toponet(
                    batch_features,  # [B, C, H, W]
                    batch_points,    # [B, N, 2]
                    batch_pairs,     # [B, N, max_neighbors, 2]
                    batch_valid,     # [B, N, max_neighbors]
                    batch_mask_logits  # [B, 2, H, W]
                )
                
                # Handle NaNs and convert to numpy
                topo_scores = torch.where(torch.isnan(topo_scores), -100.0, topo_scores).squeeze(-1).cpu().numpy()
                
                topo_time_breakdown['model_inference'] += time.time() - inference_start
                
                # Step 5: Collect edge scores
                aggregation_start = time.time()
                
                batch_size, n_samples, n_pairs = topo_scores.shape
                for bi in range(batch_size):
                    for si in range(n_samples):
                        for pi in range(n_pairs):
                            if not collated['valid'][bi, si, pi]:
                                continue
                            # idx to the full graph
                            src_idx_patch, tgt_idx_patch = collated['pairs'][bi, si, pi, :]
                            src_idx_all, tgt_idx_all = idx_maps[bi][src_idx_patch], idx_maps[bi][tgt_idx_patch]
                            edge_score = topo_scores[bi, si, pi]
                            if 0.0 <= edge_score <= 1.0:
                                edge_key = (min(src_idx_all, tgt_idx_all), max(src_idx_all, tgt_idx_all))
                                edge_scores[edge_key] += edge_score
                                edge_counts[edge_key] += 1.0

                topo_time_breakdown['edge_aggregation'] += time.time() - aggregation_start
                
            except Exception as e:
                logger.error(f"Error during model inference for batch {batch_index}: {e}")
                continue
        
        step_start = time.time()  # Reset for next batch
    
    # Step 6: Final edge filtering and aggregation
    step_start = time.time()
    
    pred_edges = []
    for edge_key, score_sum in edge_scores.items():
        avg_score = score_sum / edge_counts[edge_key]
        if avg_score > config.TOPO_THRESHOLD:
            pred_edges.append(edge_key)
    
    topo_time_breakdown['edge_aggregation'] += time.time() - step_start
    
    total_time = time.time() - start_time
    logger.info(f"Topology inference completed in {total_time:.2f}s: "
               f"found {len(pred_edges)} edges from {graph_points.shape[0]} points across {len(all_patch_info)} patches. "
               f"Time breakdown: {topo_time_breakdown}")
    
    return pred_edges 