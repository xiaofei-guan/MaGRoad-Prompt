# backend/app/services/image.py
import os
import time
import uuid
import shutil
import asyncio
import logging
import json
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Literal

import torch # Keep for feature computation later
from fastapi import UploadFile, HTTPException
from PIL import Image, ImageFile  # Keep for feature computation later
import numpy as np # Keep for feature computation later

from app.core.config import settings
from app.schemas.image import ImageInfo, FeatureStatusResponse
# Allow processing of ultra-large images without PIL's decompression bomb guard.
# We rely on our own tiled pipeline and batching to keep memory in check.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

logger = logging.getLogger(__name__)

def _generate_unique_id() -> str:
    """Generates a time-based unique ID."""
    return f"{int(time.time())}_{uuid.uuid4().hex[:6]}"

def _get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()

def _is_large_image(image_path: Path) -> bool:
    """Check if an image is considered large based on dimensions."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width > settings.LARGE_IMAGE_THRESHOLD or height > settings.LARGE_IMAGE_THRESHOLD
    except Exception as e:
        logger.warning(f"Could not determine image size for {image_path}: {e}")
        # Fail-safe: treat as large image so we route to the tiled pipeline
        return True

def save_uploaded_files(files: List[UploadFile]) -> Tuple[List[ImageInfo], List[str]]:
    """Saves uploaded files, validates them, and returns info."""
    saved_images = []
    errors = []
    settings.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for file in files:
        if not file.filename:
            errors.append("(Unnamed file)")
            continue

        ext = _get_file_extension(file.filename)
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(file.filename)
            continue

        try:
            image_id = _generate_unique_id()
            # Use original extension for saving
            save_filename = f"{image_id}{Path(file.filename).suffix}"
            save_path = settings.IMAGES_DIR / save_filename

            with save_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Check feature status (unified for both regular and large images)
            has_feature = _check_feature_exists(image_id)
            
            # Check if annotation exists
            annotation_path = settings.ANNOTATIONS_DIR / f"{image_id}_roadnetwork.geojson"
            is_annotated = annotation_path.exists()

            saved_images.append(
                ImageInfo(
                    id=image_id,
                    filename=save_filename, # Hashed filename for storage
                    original_filename=file.filename, # Original filename for display
                    has_feature=has_feature,
                    url=f"/api/images/{image_id}", # Construct API URL
                    is_annotated=is_annotated
                )
            )
        except Exception as e:
            print(f"Error saving {file.filename}: {e}")
            errors.append(file.filename)
        finally:
            file.file.close()

    return saved_images, errors

def _check_feature_exists(image_id: str) -> bool:
    """Check if features exist for an image (handles both regular and large images)."""
    # Check for regular image features
    regular_feature_path = settings.FEATURES_DIR / f"{image_id}.pt"
    if regular_feature_path.exists():
        return True
    
    # Check for large image features
    large_image_dir = settings.FEATURES_DIR / image_id
    metadata_path = large_image_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata.get('feature_status') == 'ready'
        except Exception:
            return False
    
    return False

def list_available_images() -> List[ImageInfo]:
    """Lists images from storage and checks their feature status."""
    images = []
    if not settings.IMAGES_DIR.exists():
        return []

    # Create a mapping of image_id to original_filename if available
    original_filenames = {}
    
    # Try to load original filenames from a metadata file if it exists
    metadata_path = settings.IMAGES_DIR / "original_filenames.txt"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and "," in line:
                        parts = line.split(",", 1)
                        if len(parts) == 2:
                            image_id, original_name = parts
                            original_filenames[image_id] = original_name
        except Exception as e:
            print(f"Error reading metadata file: {e}")

    for filename in os.listdir(settings.IMAGES_DIR):
        file_path = settings.IMAGES_DIR / filename
        if file_path.is_file() and _get_file_extension(filename) in ALLOWED_EXTENSIONS:
            image_id = file_path.stem # Get filename without extension
            
            # Use unified feature checking
            has_feature = _check_feature_exists(image_id)
            
            # Check if annotation exists - support both naming patterns
            annotation_path_roadnetwork = settings.ANNOTATIONS_DIR / f"{image_id}_roadnetwork.geojson"
            annotation_path_graph = settings.ANNOTATIONS_DIR / f"{image_id}_graph.geojson" 
            is_annotated = annotation_path_roadnetwork.exists() or annotation_path_graph.exists()
            
            # Log for debugging
            if is_annotated:
                if annotation_path_roadnetwork.exists():
                    print(f"Found annotation for {image_id} using _roadnetwork.geojson naming")
                elif annotation_path_graph.exists():
                    print(f"Found annotation for {image_id} using _graph.geojson naming")
            
            # Get original filename or use current filename if not found
            original_filename = original_filenames.get(image_id, filename)
            
            images.append(
                ImageInfo(
                    id=image_id,
                    filename=filename,
                    original_filename=original_filename,
                    has_feature=has_feature,
                    url=f"/api/images/{image_id}",
                    is_annotated=is_annotated
                )
            )
    
    # Sort by filename for consistent order
    images.sort(key=lambda img: img.filename)
    return images

# --- Deletion Logic ---
def _delete_single_image(image_id: str, delete_annotations: bool) -> Tuple[bool, str]:
    """Delete a single image by id. Returns (deleted, message)."""
    img_path = find_image_path(image_id)
    if not img_path or not img_path.exists():
        return False, f"Image {image_id} not found"

    # Delete image file
    try:
        img_path.unlink(missing_ok=True)
    except Exception as e:
        return False, f"Failed to delete image {image_id}: {e}"

    # Delete webp if exists
    try:
        webp_path = get_webp_path(image_id)
        if webp_path and webp_path.exists():
            webp_path.unlink(missing_ok=True)
    except Exception:
        pass

    # Delete features (regular or large-image dir)
    try:
        feature_file = settings.FEATURES_DIR / f"{image_id}.pt"
        progress_file = settings.FEATURES_DIR / f"{image_id}.progress"
        if feature_file.exists():
            feature_file.unlink(missing_ok=True)
        if progress_file.exists():
            progress_file.unlink(missing_ok=True)
        # Large-image directory
        large_dir = settings.FEATURES_DIR / image_id
        if large_dir.exists():
            shutil.rmtree(large_dir, ignore_errors=True)
    except Exception:
        pass

    # Delete annotations if requested
    if delete_annotations:
        try:
            ann1 = settings.ANNOTATIONS_DIR / f"{image_id}_roadnetwork.geojson"
            ann2 = settings.ANNOTATIONS_DIR / f"{image_id}_graph.geojson"
            if ann1.exists(): ann1.unlink(missing_ok=True)
            if ann2.exists(): ann2.unlink(missing_ok=True)
            # Compressed mask cache or files not handled here; add if needed.
        except Exception:
            pass

    return True, "deleted"

def delete_images(*, scope: Literal['current','all'], image_id: Optional[str], delete_annotations: bool) -> Dict[str, Any]:
    """Delete images by scope. Returns stats dict for API response."""
    deleted = 0
    skipped = 0

    if scope == 'current':
        if not image_id:
            raise HTTPException(status_code=400, detail="image_id is required for scope=current")
        ok, _ = _delete_single_image(image_id, delete_annotations)
        if ok: deleted += 1
        else: skipped += 1
    else:
        # all
        if not settings.IMAGES_DIR.exists():
            return {"deleted": 0, "skipped": 0, "message": "No images directory"}
        for filename in os.listdir(settings.IMAGES_DIR):
            file_path = settings.IMAGES_DIR / filename
            if file_path.is_file() and _get_file_extension(filename) in ALLOWED_EXTENSIONS:
                img_id = file_path.stem
                ok, _ = _delete_single_image(img_id, delete_annotations)
                if ok: deleted += 1
                else: skipped += 1

    return {"deleted": deleted, "skipped": skipped, "message": None}

def reset_features(image_id: str) -> None:
    """Remove existing feature artifacts for an image without deleting the image itself."""
    # Regular feature files
    try:
        feature_file = settings.FEATURES_DIR / f"{image_id}.pt"
        progress_file = settings.FEATURES_DIR / f"{image_id}.progress"
        if feature_file.exists():
            feature_file.unlink(missing_ok=True)
        if progress_file.exists():
            progress_file.unlink(missing_ok=True)
    except Exception:
        pass

    # Large image features directory
    try:
        large_dir = settings.FEATURES_DIR / image_id
        if large_dir.exists():
            shutil.rmtree(large_dir, ignore_errors=True)
    except Exception:
        pass

def find_image_path(image_id: str) -> Optional[Path]:
    """Finds the full path for an image ID, checking common extensions."""
    if not settings.IMAGES_DIR.exists():
        return None
    # Efficiently check for the image ID with allowed extensions
    for ext in ALLOWED_EXTENSIONS:
        potential_path = settings.IMAGES_DIR / f"{image_id}{ext}"
        if potential_path.is_file():
            return potential_path
    # Fallback: Check if filename already contains the ID pattern (less robust)
    for filename in os.listdir(settings.IMAGES_DIR):
         if filename.startswith(image_id) and Path(filename).stem == image_id:
              return settings.IMAGES_DIR / filename

    return None

def get_webp_path(image_id: str) -> Optional[Path]:
    """Get the path where WebP version should be stored."""
    webp_dir = settings.IMAGES_DIR / "webp"
    return webp_dir / f"{image_id}.webp"

def get_or_create_webp_version(image_id: str, original_path: Path) -> Optional[Path]:
    """Get existing WebP version or create it if it doesn't exist."""
    webp_path = get_webp_path(image_id)
    
    if webp_path and webp_path.exists():
        # Check if WebP is newer than original
        if webp_path.stat().st_mtime >= original_path.stat().st_mtime:
            return webp_path
    
    # Create WebP version
    try:
        # Ensure WebP directory exists
        webp_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to WebP with high quality but good compression
        with Image.open(original_path) as img:
            # Convert to RGB if needed (WebP doesn't support all modes)
            if img.mode in ('RGBA', 'LA', 'P'):
                # For images with transparency, keep it
                if img.mode == 'P' and 'transparency' in img.info:
                    img = img.convert('RGBA')
                elif img.mode == 'LA':
                    img = img.convert('RGBA')
            elif img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Save as WebP with optimized settings
            img.save(
                webp_path,
                'WEBP',
                quality=85,  # Good balance between quality and size
                method=6,    # Better compression (0-6, 6 is slowest but best)
                lossless=False  # Use lossy compression for smaller files
            )
        
        logger.info(f"Created WebP version for {image_id}: {webp_path}")
        return webp_path
        
    except Exception as e:
        logger.error(f"Failed to create WebP version for {image_id}: {e}")
        return None

def create_webp_versions_for_all_images():
    """Background task to create WebP versions for all existing images."""
    if not settings.IMAGES_DIR.exists():
        return
    
    logger.info("Starting batch WebP conversion for all images")
    converted_count = 0
    error_count = 0
    
    for filename in os.listdir(settings.IMAGES_DIR):
        file_path = settings.IMAGES_DIR / filename
        if file_path.is_file() and _get_file_extension(filename) in ALLOWED_EXTENSIONS:
            image_id = file_path.stem
            
            try:
                webp_path = get_or_create_webp_version(image_id, file_path)
                if webp_path:
                    converted_count += 1
                    # Log size savings
                    original_size = file_path.stat().st_size
                    webp_size = webp_path.stat().st_size
                    savings_percent = ((original_size - webp_size) / original_size) * 100
                    logger.info(f"WebP conversion for {image_id}: {original_size} -> {webp_size} bytes ({savings_percent:.1f}% savings)")
                else:
                    error_count += 1
            except Exception as e:
                logger.error(f"Error converting {filename} to WebP: {e}")
                error_count += 1
    
    logger.info(f"WebP batch conversion completed: {converted_count} converted, {error_count} errors")

# --- Feature Computation ---

 

def _calculate_patch_grid(image_size: Tuple[int, int], patch_size: int, overlap: int) -> Tuple[int, int]:
    """Calculate the number of patches needed for each dimension."""
    width, height = image_size
    stride = patch_size - overlap
    
    # Calculate grid size
    grid_cols = max(1, math.ceil((width - overlap) / stride))
    grid_rows = max(1, math.ceil((height - overlap) / stride))
    
    return grid_cols, grid_rows

def _extract_patch_coordinates(row: int, col: int, grid_cols: int, grid_rows: int, 
                              image_size: Tuple[int, int], patch_size: int, overlap: int) -> Dict[str, Any]:
    """Extract patch coordinates and metadata for a given row, col position."""
    width, height = image_size
    stride = patch_size - overlap
    
    # Calculate patch boundaries
    left = col * stride
    top = row * stride
    right = min(left + patch_size, width)
    bottom = min(top + patch_size, height)
    
    # Calculate actual patch size before padding
    actual_width = right - left
    actual_height = bottom - top
    
    # Determine if padding is needed
    needs_padding = actual_width < patch_size or actual_height < patch_size
    
    return {
        'left': left,
        'top': top,
        'right': right,
        'bottom': bottom,
        'actual_width': actual_width,
        'actual_height': actual_height,
        'needs_padding': needs_padding,
        'row': row,
        'col': col
    }

def _extract_and_pad_patch(image: Image.Image, patch_coords: Dict[str, Any]) -> Image.Image:
    """Extract a patch from the image and pad to patch_size if necessary."""
    # Extract the patch
    patch = image.crop((patch_coords['left'], patch_coords['top'], 
                       patch_coords['right'], patch_coords['bottom']))
    
    # Pad if necessary
    if patch_coords['needs_padding']:
        padded = Image.new('RGB', (settings.PATCH_SIZE, settings.PATCH_SIZE), (0, 0, 0))
        padded.paste(patch, (0, 0))
        return padded
    
    return patch

def _process_patches_in_batches(patches: List[Image.Image], model, device) -> List[torch.Tensor]:
    """Process patches in batches for efficient feature computation."""
    batch_size = settings.FEATURE_COMPUTATION_BATCH_SIZE
    all_features = []
    
    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i + batch_size]
        
        # Convert patches to tensors
        batch_tensors = []
        for patch in batch_patches:
            patch_np = np.array(patch).astype(np.float32)
            if patch_np.ndim == 2:  # Grayscale to RGB
                patch_np = np.stack((patch_np,)*3, axis=-1)
            batch_tensors.append(torch.from_numpy(patch_np))
        
        # Stack into batch tensor [B, H, W, C]
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # Pad batch if necessary
        if len(batch_tensors) < batch_size:
            padding_needed = batch_size - len(batch_tensors)
            padding_shape = (padding_needed,) + batch_tensor.shape[1:]
            padding = torch.zeros(padding_shape, device=device)
            batch_tensor = torch.cat([batch_tensor, padding], dim=0)
        
        # Process batch
        with torch.no_grad():
            batch_features = model.precompute_image_features(batch_tensor)
        
        # Extract only the valid features (not padding)
        valid_features = batch_features[:len(batch_patches)]
        all_features.extend([feat.cpu() for feat in valid_features])
    
    return all_features

def _create_large_image_metadata(image_id: str, image_path: Path, original_filename: str) -> Dict[str, Any]:
    """Create metadata for large image processing."""
    with Image.open(image_path) as img:
        width, height = img.size
    
    grid_cols, grid_rows = _calculate_patch_grid((width, height), settings.PATCH_SIZE, settings.PATCH_OVERLAP)
    total_patches = grid_cols * grid_rows
    
    metadata = {
        "image_id": image_id,
        "original_filename": original_filename,
        "original_size_wh": [width, height],
        "is_large_image": True,
        "feature_status": "computing",
        "patch_size": settings.PATCH_SIZE,
        "overlap": settings.PATCH_OVERLAP,
        "grid_size_wh": [grid_cols, grid_rows],
        "total_patches": total_patches,
        "completed_patches": 0,
        "batch_size": settings.FEATURE_COMPUTATION_BATCH_SIZE,
        "creation_time": time.time(),
        "patches": {}
    }
    
    return metadata

def _run_large_image_feature_computation(image_id: str, image_path: Path, 
                                       road_extraction_model, device, model_config):
    """Process large images with tiled feature computation and batch processing."""
    logger.info(f"[SERVICE] Starting large image feature computation for: {image_id}")
    
    features_dir = settings.FEATURES_DIR / image_id
    metadata_path = features_dir / "metadata.json"
    
    try:
        # Create directory for large image features
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Create initial metadata
        original_filename = image_path.name  # Get original filename from path
        metadata = _create_large_image_metadata(image_id, image_path, original_filename)
        
        # Save initial metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Load image without converting whole image to RGB to reduce peak memory usage
        with Image.open(image_path) as img:
            image_size = img.size  # (width, height)

            grid_cols, grid_rows = metadata["grid_size_wh"]

            # Collect all patch coordinates
            all_patch_coords = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    patch_coords = _extract_patch_coordinates(
                        row, col, grid_cols, grid_rows, image_size, 
                        settings.PATCH_SIZE, settings.PATCH_OVERLAP
                    )
                    all_patch_coords.append(patch_coords)

            # Process patches in batches
            batch_size = settings.FEATURE_COMPUTATION_BATCH_SIZE
            road_extraction_model.eval()

            for batch_start in range(0, len(all_patch_coords), batch_size):
                batch_coords = all_patch_coords[batch_start:batch_start + batch_size]

                # Extract patches for this batch
                batch_patches = []
                for patch_coords in batch_coords:
                    # Crop from source; convert each patch to RGB locally
                    patch = _extract_and_pad_patch(img, patch_coords)
                    if patch.mode != 'RGB':
                        patch = patch.convert('RGB')
                    batch_patches.append(patch)
                
                # Process batch
                batch_features = _process_patches_in_batches(batch_patches, road_extraction_model, device)
                
                # Save individual patch features
                for i, (patch_coords, features) in enumerate(zip(batch_coords, batch_features)):
                    row, col = patch_coords['row'], patch_coords['col']
                    
                    # Prepare patch data to save
                    patch_data = {
                        "image_embedding": features,
                        "patch_coordinates": {
                            "left": patch_coords['left'],
                            "top": patch_coords['top'],
                            "right": patch_coords['right'],
                            "bottom": patch_coords['bottom']
                        },
                        "patch_size": settings.PATCH_SIZE,
                        "actual_size": {
                            "width": patch_coords['actual_width'],
                            "height": patch_coords['actual_height']
                        },
                        "padded_size": {
                            "width": settings.PATCH_SIZE,
                            "height": settings.PATCH_SIZE
                        },
                        "needs_padding": patch_coords['needs_padding']
                    }
                    
                    # Save patch features
                    patch_filename = f"patch_{row}_{col}.pt"
                    torch.save(patch_data, features_dir / patch_filename)
                    
                    # Update metadata
                    metadata["completed_patches"] += 1
                    metadata["patches"][f"{row}_{col}"] = {
                        "filename": patch_filename,
                        "coordinates": patch_data["patch_coordinates"],
                        "needs_padding": patch_coords['needs_padding']
                    }
                
                # Update progress in metadata file
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"[SERVICE] Completed batch {batch_start//batch_size + 1}, "
                           f"processed {metadata['completed_patches']}/{metadata['total_patches']} patches")
        
        # Mark as completed
        metadata["feature_status"] = "ready"
        metadata["completion_time"] = time.time()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"[SERVICE] Large image feature computation completed for: {image_id}")
        
    except Exception as e:
        logger.error(f"[SERVICE] Error in large image feature computation for {image_id}: {e}")
        
        # Update metadata with error status
        try:
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata["feature_status"] = "error"
                metadata["error_message"] = str(e)
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        except Exception as meta_error:
            logger.error(f"[SERVICE] Failed to update metadata with error status: {meta_error}")

def _run_feature_computation(image_id: str, image_path: Path, road_extraction_model, device, model_config):
    """
    Internal function to run the actual feature computation.
    Now detects image size and routes to appropriate processing method.
    """
    logger.info(f"[SERVICE] Starting feature computation for: {image_id}")
    
    # Check if this is a large image
    if _is_large_image(image_path):
        logger.info(f"[SERVICE] Detected large image, using tiled processing: {image_id}")
        _run_large_image_feature_computation(image_id, image_path, road_extraction_model, device, model_config)
        return
    
    # Regular image processing (existing logic)
    progress_path = settings.FEATURES_DIR / f"{image_id}.progress"
    feature_path = settings.FEATURES_DIR / f"{image_id}.pt"
    
    try:
        settings.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        with progress_path.open("w") as f:
            f.write("computing")

        logger.info(f"[SERVICE] Processing regular size image: {image_id}")

        # 1. Load Image
        img_pil = Image.open(image_path).convert("RGB")

        # 2. Pad image to model input size (no resizing for small images)
        target_model_input_size = road_extraction_model.image_size
        original_size_hw = (img_pil.height, img_pil.width)
        if img_pil.size != (target_model_input_size, target_model_input_size):
            padded_img_pil = Image.new("RGB", (target_model_input_size, target_model_input_size), (0, 0, 0))
            padded_img_pil.paste(img_pil, (0, 0))
        else:
            padded_img_pil = img_pil
        
        # 3. Convert to Tensor and Preprocess
        img_np = np.array(padded_img_pil).astype(np.float32)
        if img_np.ndim == 2: # Grayscale, convert to RGB
            img_np = np.stack((img_np,)*3, axis=-1)
            logger.info(f"[SERVICE] Converted grayscale image to RGB. Shape: {img_np.shape}")
        
        # Add batch dimension and move to device before model call
        img_tensor_bhwc = torch.from_numpy(img_np).unsqueeze(0).to(device)
        
        logger.info(f"shape of img_tensor_bhwc: {img_tensor_bhwc.shape}")
        
        # 4. Run Model for Feature Extraction
        road_extraction_model.eval()
        with torch.no_grad():
            image_embedding = road_extraction_model.precompute_image_features(img_tensor_bhwc)
        
        # 5. Save Features (image_embedding and original_image_size)
        features_to_save = {
            "image_embedding": image_embedding.cpu(),
            "original_image_size": original_size_hw # Tuple (H, W)
        }
        torch.save(features_to_save, feature_path)
        
        logger.info(f"[SERVICE] Finished feature computation for: {image_id}. Features saved to {feature_path}")

        # Update progress file
        with progress_path.open("w") as f:
            f.write("ready") 

    except Exception as e:
        logger.error(f"[SERVICE] Error computing features for {image_id}: {e}")
        with progress_path.open("w") as f:
            f.write(f"error:{str(e)}")

async def _run_feature_computation_async(image_id: str, image_path: Path, road_extraction_model, device, model_config):
    """
    Asynchronous version of _run_feature_computation to be used from other services.
    Uses a thread pool to run the CPU/GPU-intensive feature computation without blocking.
    """
    # Create a loop if needed
    loop = asyncio.get_event_loop()
    # Run the synchronous function in a thread pool
    await loop.run_in_executor(
        None, 
        lambda: _run_feature_computation(
            image_id=image_id,
            image_path=image_path,
            road_extraction_model=road_extraction_model,
            device=device,
            model_config=model_config
        )
    )
    
    # Check if feature computation was successful
    success = _check_feature_exists(image_id)
    
    if success:
        logger.info(f"[SERVICE] Async feature computation completed for {image_id}")
        return True
    else:
        # Read error message if available
        error_message = "Unknown error during feature computation"
        progress_path = settings.FEATURES_DIR / f"{image_id}.progress"
        
        if progress_path.exists():
            try:
                with progress_path.open("r") as f:
                    status_content = f.read().strip()
                if status_content.startswith("error:"):
                    error_message = status_content.split(":", 1)[1]
            except Exception:
                pass
        
        # Check for large image error in metadata
        metadata_path = settings.FEATURES_DIR / image_id / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if metadata.get('feature_status') == 'error':
                    error_message = metadata.get('error_message', error_message)
            except Exception:
                pass
        
        logger.error(f"[SERVICE] Async feature computation failed for {image_id}: {error_message}")
        return False

def get_feature_status(image_id: str) -> FeatureStatusResponse:
    """
    Gets the status of feature computation for a given image ID.
    Handles both regular and large images.
    """
    # Check for regular image features
    feature_path = settings.FEATURES_DIR / f"{image_id}.pt"
    progress_path = settings.FEATURES_DIR / f"{image_id}.progress"

    if feature_path.exists():
        return FeatureStatusResponse(status="ready", message="feature computation completed")

    # Check for large image features
    large_image_dir = settings.FEATURES_DIR / image_id
    metadata_path = large_image_dir / "metadata.json"
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            status = metadata.get('feature_status', 'none')
            
            if status == "ready":
                return FeatureStatusResponse(status="ready", message="feature computation completed")
            elif status == "computing":
                completed = metadata.get('completed_patches', 0)
                total = metadata.get('total_patches', 0)
                progress_msg = f"feature computation in progress ({completed}/{total})"
                return FeatureStatusResponse(status="computing", message=progress_msg)
            elif status == "error":
                error_msg = metadata.get('error_message', 'unknown error')
                return FeatureStatusResponse(status="error", message=f"feature computation failed: {error_msg}")
            else:
                return FeatureStatusResponse(status="none", message="feature computation not started")
                
        except Exception as e:
            logger.error(f"Error reading large image metadata {metadata_path}: {e}")
    
    # Check for regular image progress
    if progress_path.exists():
        try:
            with progress_path.open("r") as f:
                status_content = f.read().strip()
            if status_content == "computing":
                return FeatureStatusResponse(status="computing", message="feature computation in progress")
            elif status_content.startswith("error:"):
                error_msg = status_content.split(":", 1)[1]
                return FeatureStatusResponse(status="error", message=f"feature computation failed: {error_msg}")
        except Exception as e:
            logger.error(f"Error reading progress file {progress_path}: {e}")

    return FeatureStatusResponse(status="none", message="feature computation not started") 