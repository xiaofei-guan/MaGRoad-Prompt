import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from fastapi import HTTPException
from omegaconf import DictConfig
import time
import json

# Model and utility imports
from app.models.road_extraction_algorithm.sam_road_extraction import RoadExtractionModel
from app.models.road_extraction_algorithm import graph_extraction
from app.core.config import settings # For accessing storage paths
from app.schemas.roadnet import GeoJSONFeatureCollection
from app.services import image as image_service # Import image service for feature computation
from app.services.large_image_functions import (
    _generate_road_network_for_large_image, 
    _save_compressed_mask_file, 
    _load_compressed_mask_file,
    prepare_compressed_masks_response
)
from app.services.mask_cache_service import mask_cache_service

logger = logging.getLogger(__name__)

# Helper function to load features (can be made async if using async file I/O)
def _load_regular_image_features(image_id: str, features_dir: Path) -> Dict[str, Any]:
    """
    Load image features for the given image_id.
    This function now only handles regular images as large images are processed separately.
    """
    # Regular image processing
    feature_file = features_dir / f"{image_id}.pt"
    
    loaded_features = None
    if feature_file.exists():
        try:
            # Ensure loading to CPU first to avoid issues if called from a non-GPU process
            # The actual model inference will handle moving data to the correct device.
            loaded_features = torch.load(feature_file, map_location='cpu')
            logger.info(f"Successfully loaded features for image_id '{image_id}' from {feature_file}")
        except Exception as e:
            logger.warning(f"Could not load features from {feature_file} for image_id '{image_id}': {e}")
            # Fallthrough to raise FileNotFoundError if features couldn't be loaded
    else:
        logger.debug(f"Feature file {feature_file} not found for image_id '{image_id}'.")

    if loaded_features is None:
        err_msg = f"Precomputed features for image_id '{image_id}' not found at {feature_file}."
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    # Validate required keys in loaded features
    required_keys = ["image_embedding", "original_image_size"]
    for key in required_keys:
        if key not in loaded_features:
            err_msg = f"Feature file for image_id '{image_id}' ({feature_file}) is missing required key: '{key}'."
            logger.error(err_msg)
            raise ValueError(err_msg)
    
    # original_image_size should be a tuple (H, W)
    if not (isinstance(loaded_features["original_image_size"], (list, tuple)) and len(loaded_features["original_image_size"]) == 2):
        err_msg = f"Feature 'original_image_size' for image_id '{image_id}' ({feature_file}) must be a tuple/list of (height, width)."
        logger.error(err_msg)
        raise ValueError(err_msg)

    return loaded_features

# Helper function to check if features exist and compute them if needed
async def ensure_features_exist(
    image_id: str,
    model: RoadExtractionModel,
    device: torch.device,
    model_config: DictConfig,
    features_dir: Path
) -> bool:
    """
    Ensures that features exist for the given image ID.
    If features don't exist, computes them.
    Supports both regular images (.pt files) and large images (directory structure).

    Args:
        image_id: The ID of the image to check/compute features for
        model: The RoadExtractionModel instance
        device: The computation device
        model_config: The model configuration
        features_dir: Directory where features are stored

    Returns:
        True if features exist or were successfully computed, False otherwise
    """
    # Check for both regular image (.pt file) and large image (directory) formats
    regular_feature_file = features_dir / f"{image_id}.pt"
    large_image_dir = features_dir / image_id
    large_image_metadata = large_image_dir / "metadata.json"
    
    # Check if regular image features already exist
    if regular_feature_file.exists():
        try:
            # Try to load the features to verify they're valid
            loaded_features = torch.load(regular_feature_file, map_location='cpu')
            if "image_embedding" in loaded_features and "original_image_size" in loaded_features:
                logger.info(f"Regular image features already exist for image_id '{image_id}'")
                return True
        except Exception as e:
            logger.warning(f"Found regular feature file for {image_id} but it appears corrupted: {e}")
            # Continue to check large image format or compute new features
    
    # Check if large image features already exist
    if large_image_metadata.exists():
        try:
            with open(large_image_metadata, 'r') as f:
                metadata = json.load(f)
            
            # Check if computation is complete
            if metadata.get('feature_status') == 'ready':
                logger.info(f"Large image features already exist for image_id '{image_id}'")
                return True
            else:
                logger.info(f"Large image features exist but status is '{metadata.get('feature_status')}' for image_id '{image_id}'")
                # Status is not ready, may need to recompute or wait for completion
        except Exception as e:
            logger.warning(f"Found large image metadata for {image_id} but it appears corrupted: {e}")
            # Continue to compute new features
    
    # Features don't exist or are invalid, compute them
    try:
        # Find the image path
        image_path = image_service.find_image_path(image_id)
        if not image_path:
            logger.error(f"Image not found for image_id: {image_id}")
            return False
        
        # Compute features
        logger.info(f"Features not found for image_id: {image_id}. Computing them now...")
        await image_service._run_feature_computation_async(
            image_id=image_id,
            image_path=image_path,
            road_extraction_model=model,
            device=device,
            model_config=model_config
        )
        
        # Verify features were computed successfully (check both formats)
        if regular_feature_file.exists():
            logger.info(f"Successfully computed regular image features for image_id: {image_id}")
            return True
        elif large_image_metadata.exists():
            try:
                with open(large_image_metadata, 'r') as f:
                    metadata = json.load(f)
                if metadata.get('feature_status') == 'ready':
                    logger.info(f"Successfully computed large image features for image_id: {image_id}")
                    return True
                else:
                    logger.error(f"Large image feature computation completed but status is '{metadata.get('feature_status')}' for image_id: {image_id}")
                    return False
            except Exception as e:
                logger.error(f"Error reading large image metadata after computation for {image_id}: {e}")
                return False
        else:
            logger.error(f"Failed to compute features for image_id: {image_id} - no valid feature files found")
            return False
    except Exception as e:
        logger.exception(f"Error ensuring features exist for image_id {image_id}: {e}")
        return False

async def generate_road_network(
    image_id: str,
    prompts: List[Dict[str, Any]],
    device: torch.device,
    model: RoadExtractionModel,
    config: DictConfig,
    include_masks: bool = True
) -> Dict[str, Any]:
    """
    High-level function to generate road network with optional mask data.
    
    Args:
        image_id: The image ID to process
        prompts: List of prompt dictionaries with x, y, label
        device: Computation device
        model: Road extraction model
        config: Model configuration
        include_masks: Whether to include mask data in response
        
    Returns:
        Dictionary with road network data and optionally mask data
    """
    features_dir = Path(settings.FEATURES_DIR)
    
    # Call the existing processing function
    result = await process_road_network_generation(
        image_id=image_id,
        prompts_data=prompts,
        model=model,
        config=config,
        device=device,
        features_dir=features_dir,
        include_masks=include_masks
    )
    
    return result

async def process_road_network_generation(
    image_id: str,
    prompts_data: List[Dict[str, Any]], # List of {'x': float, 'y': float, 'label': int}
    model: RoadExtractionModel,
    config: DictConfig,
    device: torch.device,
    features_dir: Path,
    include_masks: bool = True
) -> dict:
    """
    Processes the road network generation request using DistMaps for prompts.
    Supports both regular images and large images with tiled processing.
    
    Will automatically compute image features if they don't exist.
    """
    logger.info(f"Starting road network generation for image_id: {image_id} with {len(prompts_data)} prompts using DistMaps path logic.")
    
    # Check if features exist, compute them if they don't
    features_exist = await ensure_features_exist(
        image_id=image_id,
        model=model,
        device=device,
        model_config=config,
        features_dir=features_dir
    )
    
    if not features_exist:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to compute or load features for image_id: {image_id}. Cannot generate road network."
        )

    # Check if this is a large image by looking for metadata.json
    large_image_dir = features_dir / image_id
    metadata_path = large_image_dir / "metadata.json"
    
    if metadata_path.exists():
        logger.info(f"Detected large image, using tiled processing for road network generation: {image_id}")
        return await _generate_road_network_for_large_image(
            image_id=image_id,
            prompts_data=prompts_data,
            model=model,
            config=config,
            device=device,
            features_dir=features_dir,
            metadata_path=metadata_path,
            include_masks=include_masks
        )
    else:
        logger.info(f"Detected regular image, using standard processing: {image_id}")
        return await _generate_road_network_for_regular_image(
            image_id=image_id,
            prompts_data=prompts_data,
            model=model,
            config=config,
            device=device,
            features_dir=features_dir,
            include_masks=include_masks
        )

async def _generate_road_network_for_regular_image(
    image_id: str,
    prompts_data: List[Dict[str, Any]],
    model: RoadExtractionModel,
    config: DictConfig,
    device: torch.device,
    features_dir: Path,
    include_masks: bool = True
) -> dict:
    """
    Original logic for regular image (size <= 1024), extracted into separate function.
    """
    try:
        res = {
            'pred_nodes': [],
            'pred_edges': [],
            'kp_mask': [],
            'road_mask': []
        }
        cost_time = {
            'load_and_scale_prompts': 0,
            'run_model': 0,
            'convert_to_geojson': 0,
            'total': 0
        }
        start_time = time.time()
        # 1. Load Precomputed Image Features
        image_features_data = _load_regular_image_features(image_id, features_dir)

        print(f"shape of image_features_data: {image_features_data['image_embedding'].shape}")

        image_embedding_from_file = image_features_data["image_embedding"].to(device) # Move to device
        original_image_size_hw = torch.tensor(image_features_data["original_image_size"]) # H, W
        
        logger.info(f"Original image size is {original_image_size_hw.shape} (H, W)")

        # 2. Prepare Prompts (scaled to model input size)
        point_coords_np = np.array([[p['x'], p['y']] for p in prompts_data], dtype=np.float32)
        point_labels_np = np.array([p['label'] for p in prompts_data], dtype=np.int32)

        batched_kp_coords = torch.tensor(point_coords_np, device=device).unsqueeze(0) # B, N, 2
        batched_kp_labels = torch.tensor(point_labels_np, device=device).unsqueeze(0)          # B, N

        model_input_size_hw = (model.image_size, model.image_size) # e.g., (1024, 1024)

        end_time = time.time()
        cost_time['load_and_scale_prompts'] = end_time - start_time

        model.eval()
        
        start_time = time.time()
        with torch.no_grad():
            res = model.forward(
                image_features=image_embedding_from_file,
                kp_coords=batched_kp_coords,
                kp_labels=batched_kp_labels,
                original_image_size_hw=original_image_size_hw,
                input_features=True
            )
        end_time = time.time()
        cost_time['run_model'] = end_time - start_time

        start_time = time.time()

        graph_nodes_xy_np = res['pred_nodes'][0] if res['pred_nodes'] else np.empty((0,2), dtype=np.float32)
        filtered_edges_np = res['pred_edges'][0] if res['pred_edges'] else np.empty((0,2), dtype=np.int32)

        if graph_nodes_xy_np.shape[0] == 0:
             graph_nodes_xy_np = np.empty((0,2), dtype=np.float32)

        geojson_output = graph_extraction.convert_to_geojson(
            nodes_xy=graph_nodes_xy_np,
            edges=filtered_edges_np,
            image_id=image_id,
        )
        logger.info(f"Successfully generated road network for image_id: {image_id}. Outputting GeoJSON.")
        
        # Simplified validation - only check essential keys for new format
        required_keys = ['type', 'nodes', 'edges', 'properties']
        for key in required_keys:
            if key not in geojson_output:
                logger.error(f"Invalid GeoJSON output: missing key '{key}'")
                raise HTTPException(status_code=500, detail=f"Invalid GeoJSON output structure: missing '{key}'")
        
        end_time = time.time()
        cost_time['convert_to_geojson'] = end_time - start_time
        cost_time['total'] = sum(cost_time.values())
        logger.info(f"Cost time: {cost_time}")
        
        # Prepare the response data (return dict for API layer processing)
        response_data = {
            "image_id": image_id,
            "geojson_data": geojson_output,  # Keep as dict for API layer processing
            "prompts": prompts_data, # Return the original prompts used for this generation
            "road_mask": None,
            "kp_mask": None,
            "road_mask_metadata": None,
            "kp_mask_metadata": None
        }
        
        # Extract and cache mask data for both transmission and future saves
        road_mask_np = res['road_mask'][0] if res['road_mask'] and len(res['road_mask']) > 0 else None
        kp_mask_np = res['kp_mask'][0] if res['kp_mask'] and len(res['kp_mask']) > 0 else None
        
        # Cache masks in backend for future save operations
        mask_cache_service.cache_masks(
            image_id=image_id,
            road_mask=road_mask_np,
            kp_mask=kp_mask_np
        )
        
        # Use unified compression approach for both regular and large images
        mask_response_data = prepare_compressed_masks_response(
            road_mask_np=road_mask_np,
            kp_mask_np=kp_mask_np,
            include_masks=include_masks
        )
        
        # Update response with compressed mask data
        response_data.update(mask_response_data)
        
        if include_masks and (road_mask_np is not None or kp_mask_np is not None):
            logger.info(f"Regular image masks compressed for transmission and cached: road_mask={road_mask_np is not None}, kp_mask={kp_mask_np is not None}")
        
        # Construct the full response including the input prompts for context
        return response_data

    except FileNotFoundError as e:
        logger.error(f"[Service] Error: File not found - {e}")
        raise  # Re-raise to be caught by API layer
    except ValueError as e:
        logger.error(f"[Service] Error: Value error - {e}")
        raise  # Re-raise
    except Exception as e:
        logger.exception(f"[Service] Unexpected service error during road network generation for image_id: {image_id}")
        # Consider raising a custom service layer exception if more specific handling is needed
        raise HTTPException(status_code=500, detail=f"Internal server error in road network generation service: {e}")


async def save_road_network_annotation(
    image_id: str, 
    geojson_data: GeoJSONFeatureCollection, # Expecting Pydantic model directly
    prompts: Optional[List[Dict[str, Any]]], # Prompts from the request
    road_mask: Optional[List[List[float]]] = None, # Road mask data (deprecated - will use cached data)
    kp_mask: Optional[List[List[float]]] = None, # Keypoint mask data (deprecated - will use cached data)
    annotations_dir: Path = settings.ANNOTATIONS_DIR
) -> str:
    """
    Saves the road network annotation with compressed .gz format for all masks.
    Now uses cached mask data instead of receiving masks from frontend.
    
    Performance Optimizations:
    - Uses gzip compression with shape headers for all masks
    - Separate compressed files for optimal I/O performance
    - Optimized compression level (3) for balance between size and speed
    - Metadata tracking for efficient loading
    - Backend caching eliminates mask data transmission overhead

    Args:
        image_id: The identifier of the image.
        geojson_data: The GeoJSON data (as a Pydantic model) to save.
        prompts: The list of prompts used for this generation.
        road_mask: (DEPRECATED) Use cached data instead
        kp_mask: (DEPRECATED) Use cached data instead
        annotations_dir: The directory where annotations should be saved. 
                         Defaults to settings.ANNOTATIONS_DIR.

    Returns:
        The path to the saved annotation file as a string.
    
    Raises:
        IOError: If there's an issue writing the file.
    """
    if not image_id:
        raise ValueError("image_id cannot be empty.")
    if not geojson_data:
        raise ValueError("geojson_data cannot be empty.")

    # Only store essential data in GeoJSON properties
    if geojson_data.properties is None:
        geojson_data.properties = {}
    
    # Only save prompts in GeoJSON properties (lightweight)
    if prompts:
        geojson_data.properties['prompts'] = prompts
    
    # Track saved files for cleanup on error
    saved_files = []
    storage_info = {}  # Track storage methods used
    
    try:
        # Save main GeoJSON file (compact format, no indentation)
        geojson_file_name = f"{image_id}_roadnetwork.geojson"
        geojson_file_path = annotations_dir / geojson_file_name
        
        with open(geojson_file_path, 'w') as f:
            json.dump(geojson_data.model_dump(exclude_none=True), f, separators=(',', ':'))
        
        saved_files.append(str(geojson_file_path))
        logger.info(f"GeoJSON saved to: {geojson_file_path}")
        
        # Get cached mask data instead of using frontend-transmitted data
        cached_road_mask, cached_kp_mask = mask_cache_service.get_cached_masks(image_id)
        
        # Process road mask with compression using cached data
        if cached_road_mask is not None:
            road_mask_path = annotations_dir / f"{image_id}_road_mask.gz"
            metadata = _save_compressed_mask_file(cached_road_mask, road_mask_path)
            storage_info['road_mask'] = {
                'method': 'compressed',
                'original_size': cached_road_mask.size,
                'compressed_size': metadata['compressed_size_bytes'],
                'compression_ratio': metadata['compression_ratio']
            }
            logger.info(f"Road mask (from cache) compressed and saved: {road_mask_path} "
                       f"(ratio: {metadata['compression_ratio']:.3f})")
            saved_files.append(str(road_mask_path))
        elif road_mask:
            # Fallback to frontend data if no cached data available (backwards compatibility)
            road_mask_array = np.array(road_mask, dtype=bool)
            road_mask_path = annotations_dir / f"{image_id}_road_mask.gz"
            metadata = _save_compressed_mask_file(road_mask_array, road_mask_path)
            storage_info['road_mask'] = {
                'method': 'compressed_fallback',
                'original_size': road_mask_array.size,
                'compressed_size': metadata['compressed_size_bytes'],
                'compression_ratio': metadata['compression_ratio']
            }
            logger.warning(f"Road mask saved from frontend (cache miss): {road_mask_path}")
            saved_files.append(str(road_mask_path))
        
        # Process keypoint mask with compression using cached data
        if cached_kp_mask is not None:
            kp_mask_path = annotations_dir / f"{image_id}_kp_mask.gz"
            metadata = _save_compressed_mask_file(cached_kp_mask, kp_mask_path)
            storage_info['kp_mask'] = {
                'method': 'compressed',
                'original_size': cached_kp_mask.size,
                'compressed_size': metadata['compressed_size_bytes'],
                'compression_ratio': metadata['compression_ratio']
            }
            logger.info(f"Keypoint mask (from cache) compressed and saved: {kp_mask_path} "
                       f"(ratio: {metadata['compression_ratio']:.3f})")
            saved_files.append(str(kp_mask_path))
        elif kp_mask:
            # Fallback to frontend data if no cached data available (backwards compatibility)
            kp_mask_array = np.array(kp_mask, dtype=bool)
            kp_mask_path = annotations_dir / f"{image_id}_kp_mask.gz"
            metadata = _save_compressed_mask_file(kp_mask_array, kp_mask_path)
            storage_info['kp_mask'] = {
                'method': 'compressed_fallback',
                'original_size': kp_mask_array.size,
                'compressed_size': metadata['compressed_size_bytes'],
                'compression_ratio': metadata['compression_ratio']
            }
            logger.warning(f"Keypoint mask saved from frontend (cache miss): {kp_mask_path}")
            saved_files.append(str(kp_mask_path))
        
        # Log performance summary
        total_masks = len([k for k in storage_info.keys() if 'mask' in k])
        
        logger.info(f"Road network annotation saved successfully for {image_id}. "
                   f"Files: {len(saved_files)}, Masks: {total_masks} (all compressed)")
        logger.debug(f"Storage info: {storage_info}")
        
        return str(geojson_file_path)  # Return main file path
        
    except IOError as e:
        logger.error(f"Failed to save road network annotation for {image_id}: {e}")
        # Cleanup partially saved files
        for file_path in saved_files:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving annotation for {image_id}: {e}")
        # Cleanup partially saved files
        for file_path in saved_files:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass
        raise IOError(f"Failed to save annotation due to an unexpected error: {e}")

async def load_road_network_annotation(
    image_id: str, 
    annotations_dir: Path = settings.ANNOTATIONS_DIR,
    include_masks: bool = True,
    config: Optional[DictConfig] = None
) -> Optional[Dict[str, Any]]: # Returns dict that can be used for RoadNetGenerationResponse
    """
    Loads a previously saved road network annotation with compressed .gz masks.
    
    Performance Optimizations:
    - Uses compressed .gz files with embedded shape headers
    - Fast decompression with shape information included in file
    - Optimized I/O with gzip compression
    - **NEW**: Compressed transmission using prepare_compressed_masks_response for faster frontend loading

    Args:
        image_id: The identifier of the image (filename).
        annotations_dir: Directory where annotations are stored.
        include_masks: Whether to include mask data in response (default: True).
        config: Model configuration for compression settings (optional).

    Returns:
        A dictionary containing the image_id, geojson_data, and prompts if found, otherwise None.
    """
    if not image_id:
        logger.warning("load_road_network_annotation called with empty image_id")
        return None

    # Check for new node-edge format file naming only
    file_path = annotations_dir / f"{image_id}_roadnetwork.geojson"
    
    if not file_path.exists():
        logger.info(f"Annotation file not found for image_id '{image_id}'")
        return None

    try:
        # Load main GeoJSON file
        with open(file_path, 'r') as f:
            loaded_data = json.load(f)
        
        # Extract prompts from GeoJSON properties
        prompts = None
        if 'properties' in loaded_data and 'prompts' in loaded_data['properties']:
            prompts = loaded_data['properties']['prompts']
        
        # Load masks (only .gz format with shape headers) and prepare for transmission
        road_mask_array = None
        kp_mask_array = None
        loading_stats = {}
        
        # Only load masks if requested
        if include_masks:
            # Load road mask (.gz format only)
            road_mask_path = annotations_dir / f"{image_id}_road_mask.gz"
            if road_mask_path.exists():
                try:
                    start_time = time.time()
                    road_mask_array = _load_compressed_mask_file(road_mask_path)
                    loading_stats['road_mask'] = {
                        'method': 'compressed',
                        'load_time': time.time() - start_time,
                        'shape': road_mask_array.shape
                    }
                    logger.debug(f"Loaded compressed road mask: {road_mask_path} in {loading_stats['road_mask']['load_time']:.3f}s")
                except Exception as e:
                    logger.error(f"Failed to load road mask from {road_mask_path}: {e}")
                    # Continue without road mask
            
            # Load keypoint mask (.gz format only)
            kp_mask_path = annotations_dir / f"{image_id}_kp_mask.gz"
            if kp_mask_path.exists():
                try:
                    start_time = time.time()
                    kp_mask_array = _load_compressed_mask_file(kp_mask_path)
                    loading_stats['kp_mask'] = {
                        'method': 'compressed',
                        'load_time': time.time() - start_time,
                        'shape': kp_mask_array.shape
                    }
                    logger.debug(f"Loaded compressed keypoint mask: {kp_mask_path} in {loading_stats['kp_mask']['load_time']:.3f}s")
                except Exception as e:
                    logger.error(f"Failed to load keypoint mask from {kp_mask_path}: {e}")
                    # Continue without keypoint mask
        
        # Clean up GeoJSON properties to avoid sending duplicate data
        if 'properties' in loaded_data and loaded_data['properties']:
            # Remove mask data from properties if it exists (to avoid duplication)
            loaded_data['properties'].pop('road_mask', None)
            loaded_data['properties'].pop('kp_mask', None)
        
        # Prepare base response data
        response_data = {
            "image_id": image_id,
            "geojson_data": loaded_data,  # Clean GeoJSON without embedded masks
            "prompts": prompts,
            "road_mask": None,
            "kp_mask": None,
            "road_mask_metadata": None,
            "kp_mask_metadata": None
        }
        
        # **🔥 PERFORMANCE OPTIMIZATION**: Use compressed transmission for masks
        if include_masks and (road_mask_array is not None or kp_mask_array is not None):
            try:
                # Use unified compression approach for transmission
                mask_response_data = prepare_compressed_masks_response(
                    road_mask_np=road_mask_array,
                    kp_mask_np=kp_mask_array,
                    include_masks=True
                )
                
                # Update response with compressed mask data
                response_data.update(mask_response_data)
                
                logger.info(f"Load operation: masks compressed for transmission (road_mask={road_mask_array is not None}, kp_mask={kp_mask_array is not None})")
                
            except Exception as e:
                logger.error(f"Failed to compress masks for transmission during load: {e}")
                # Fallback to direct transmission if compression fails
                if road_mask_array is not None:
                    response_data["road_mask"] = road_mask_array.tolist()
                if kp_mask_array is not None:
                    response_data["kp_mask"] = kp_mask_array.tolist()
                logger.warning(f"Falling back to direct mask transmission for load operation")
        
        # Log performance summary
        total_load_time = sum(stats.get('load_time', 0) for stats in loading_stats.values())
        masks_loaded = len(loading_stats)
        
        logger.info(f"Road network annotation loaded successfully for {image_id}. "
                   f"Masks: {masks_loaded} (all compressed), "
                   f"Total load time: {total_load_time:.3f}s")
        logger.debug(f"Loading stats: {loading_stats}")
        
        return response_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {file_path} for image_id '{image_id}': {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading annotation for {image_id}: {e}")
        return None
