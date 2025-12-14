# backend/app/api/images.py
import os
import mimetypes
from typing import List, Optional, Literal

from pydantic import BaseModel
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    BackgroundTasks,
    HTTPException,
    Request, # Add Request
    Header
)
from fastapi.responses import FileResponse, Response

from app.schemas.image import (
    ImageListResponse,
    UploadResponse,
    FeatureStatusResponse,
    PrecomputeResponse
)
from app.core.config import settings
from app.services import image as image_service


router = APIRouter()

@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_images(files: List[UploadFile] = File(..., description="List of image files to upload")):
    """Uploads one or more image files."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    saved, errors = image_service.save_uploaded_files(files)

    if not saved and errors:
         raise HTTPException(status_code=400, detail=f"All uploads failed or were invalid types. Failed files: {errors}")

    # Save original filename mapping to a metadata file
    metadata_path = settings.IMAGES_DIR / "original_filenames.txt"
    
    try:
        # Create directory if it doesn't exist
        settings.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Append new entries to the file
        with open(metadata_path, "a") as f:
            for image in saved:
                f.write(f"{image.id},{image.original_filename}\n")
    except Exception as e:
        print(f"Error saving original filenames metadata: {e}")
        # Continue without failing the request, this is non-critical

    # Create an UploadResponse with the proper fields
    return UploadResponse(
        message=f"Successfully uploaded {len(saved)} files",
        image_ids=[img.id for img in saved],
        images=saved  # Pass the saved ImageInfo objects with all fields
    )

@router.get("/list", response_model=ImageListResponse)
async def list_images():
    """Lists all available images in the storage."""
    images = image_service.list_available_images()
    return ImageListResponse(images=images)

@router.get("/{image_id}",
            response_class=FileResponse,
            responses={
                200: {"content": {"image/*": {}}},
                206: {"description": "Partial Content"},
                404: {"description": "Image not found"}
            }
           )
async def get_image(
    image_id: str,
    request: Request,
    range_header: str = Header(None, alias="range")
):
    """Serves the image file with HTTP Range support for chunked downloads."""
    image_path = image_service.find_image_path(image_id)
    if not image_path or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    # Do not auto-switch to WebP based on Accept header.
    # Only serve WebP via explicit /webp route or query parameter.
    if request.query_params.get("format") == "webp":
        webp_path = image_service.get_or_create_webp_version(image_id, image_path)
        if webp_path and webp_path.exists():
            image_path = webp_path

    file_size = image_path.stat().st_size

    # Determine media type
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", 
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
        ".bmp": "image/bmp"
    }
    media_type = media_type_map.get(image_path.suffix.lower(), "application/octet-stream")

    # Handle Range requests for chunked downloads
    if range_header:
        try:
            # Parse Range header: "bytes=start-end"
            range_match = range_header.replace("bytes=", "").split("-")
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1
            
            # Ensure valid range
            start = max(0, min(start, file_size - 1))
            end = max(start, min(end, file_size - 1))
            content_length = end - start + 1

            # Read the specified range
            with open(image_path, "rb") as f:
                f.seek(start)
                content = f.read(content_length)

            # Return partial content with proper headers
            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
            }
            
            return Response(
                content=content,
                status_code=206,  # Partial Content
                headers=headers,
                media_type=media_type
            )

        except (ValueError, IndexError) as e:
            # Invalid range header, fall back to full file
            pass

    # Return full file with range support headers
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Cache-Control": "public, max-age=3600"
    }
    
    return FileResponse(
        image_path, 
        media_type=media_type, 
        filename=image_path.name,
        headers=headers
    )

@router.get("/{image_id}/webp",
            responses={
                200: {"content": {"image/webp": {}}},
                404: {"description": "Image not found"}
            }
           )
async def get_image_webp(
    image_id: str,
    request: Request,
    range_header: str = Header(None, alias="range")
):
    """Serves the WebP version of an image with Range support."""
    original_path = image_service.find_image_path(image_id)
    if not original_path or not original_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    # Get or create WebP version
    webp_path = image_service.get_or_create_webp_version(image_id, original_path)
    if not webp_path or not webp_path.exists():
        raise HTTPException(status_code=404, detail="WebP version not available")

    file_size = webp_path.stat().st_size

    # Handle Range requests
    if range_header:
        try:
            range_match = range_header.replace("bytes=", "").split("-")
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1
            
            start = max(0, min(start, file_size - 1))
            end = max(start, min(end, file_size - 1))
            content_length = end - start + 1

            with open(webp_path, "rb") as f:
                f.seek(start)
                content = f.read(content_length)

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Cache-Control": "public, max-age=3600"
            }
            
            return Response(
                content=content,
                status_code=206,
                headers=headers,
                media_type="image/webp"
            )

        except (ValueError, IndexError):
            pass

    # Return full WebP file
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Cache-Control": "public, max-age=3600"
    }
    
    return FileResponse(
        webp_path,
        media_type="image/webp",
        filename=f"{image_id}.webp",
        headers=headers
    )

@router.get("/{image_id}/info")
async def get_image_info(image_id: str):
    """Get image file information including size for chunked downloads."""
    image_path = image_service.find_image_path(image_id)
    if not image_path or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    file_size = image_path.stat().st_size
    
    # Check for WebP version
    webp_path = image_service.get_webp_path(image_id)
    webp_size = webp_path.stat().st_size if webp_path and webp_path.exists() else None

    return {
        "image_id": image_id,
        "original_size": file_size,
        "webp_size": webp_size,
        "webp_available": bool(webp_size),
        "supports_range": True,
        "recommended_chunk_size": 1024 * 1024,  # 1MB chunks
        "max_concurrent_chunks": 6
    }

# --- Feature Computation Endpoints (Stubbed) ---

class PrecomputeRequest(BaseModel):
     image_id: str

@router.post("/precompute-feature", response_model=PrecomputeResponse)
async def start_precompute_feature(
    request_body: PrecomputeRequest, # Use Pydantic model for request body
    background_tasks: BackgroundTasks,
    request: Request # Add Request to access app.state
):
    """
    Starts the pre-computation of image features in the background.
    """
    image_id = request_body.image_id
    if not image_id:
        raise HTTPException(status_code=400, detail="Missing image_id")

    image_path = image_service.find_image_path(image_id)
    if not image_path:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    # Check current status before starting
    current_status = image_service.get_feature_status(image_id)
    if current_status.status == 'ready':
        return PrecomputeResponse(status="ready", message="feature already exists")
    if current_status.status == 'computing':
         return PrecomputeResponse(status="computing", message="feature is already being computed")

    # Access model and device from app.state (populated in main.py lifespan)
    try:
        road_extraction_model = request.app.state.model_store["road_extraction_model"]
        device = request.app.state.model_store["device"]
        model_config = request.app.state.model_store["road_extraction_model_config"]
    except AttributeError:
        # This might happen if app.state.model_store is not initialized as expected
        # or if running outside a full FastAPI app context (e.g. direct script execution for testing)
        # For a running app, this should ideally not be hit if main.py setup is correct.
        raise HTTPException(status_code=500, detail="Model not loaded in application state. Cannot precompute features.")
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing expected key in model_store: {e}. Cannot precompute features.")

    if road_extraction_model is None:
        raise HTTPException(status_code=503, detail="RoadExtractionModel not available. Feature computation cannot proceed.")

    # Add task to background
    background_tasks.add_task(
        image_service._run_feature_computation,
        image_id=image_id,
        image_path=image_path,
        road_extraction_model=road_extraction_model, # Pass actual model instance
        device=device,                  # Pass actual device
        model_config=model_config       # Pass model_config
    )

    # Return status indicating computation has started
    # We don't wait for it here
    return PrecomputeResponse(status="computing", message="feature computation started")

@router.get("/feature-status/{image_id}", response_model=FeatureStatusResponse)
async def check_feature_status_endpoint(image_id: str):
    """
    Checks the current status of feature computation for an image.
    """
    status = image_service.get_feature_status(image_id)
    return status 

# --- Delete Images Endpoint ---

class DeleteRequest(BaseModel):
    scope: Literal['current', 'all']
    image_id: Optional[str] = None
    delete_annotations: bool = False

class DeleteResponse(BaseModel):
    deleted: int
    skipped: int
    message: Optional[str] = None

@router.post("/delete", response_model=DeleteResponse)
async def delete_images_endpoint(req: DeleteRequest):
    """Delete one image (scope=current) or all images (scope=all).
    Optionally delete annotations too.
    """
    try:
        result = image_service.delete_images(
            scope=req.scope,
            image_id=req.image_id,
            delete_annotations=req.delete_annotations
        )
        return DeleteResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

# --- Recompute Features Endpoint ---

class RecomputeRequest(BaseModel):
    image_id: str

@router.post("/recompute-feature", response_model=PrecomputeResponse)
async def recompute_feature_endpoint(req: RecomputeRequest, background_tasks: BackgroundTasks, request: Request):
    """Reset existing features (if any) and start a fresh computation for the given image."""
    image_id = req.image_id
    image_path = image_service.find_image_path(image_id)
    if not image_path:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    # Reset feature artifacts
    image_service.reset_features(image_id)

    # Start compute as in start_precompute_feature
    try:
        road_extraction_model = request.app.state.model_store["road_extraction_model"]
        device = request.app.state.model_store["device"]
        model_config = request.app.state.model_store["road_extraction_model_config"]
    except Exception:
        raise HTTPException(status_code=500, detail="Model not loaded in application state. Cannot (re)compute features.")

    background_tasks.add_task(
        image_service._run_feature_computation,
        image_id=image_id,
        image_path=image_path,
        road_extraction_model=road_extraction_model,
        device=device,
        model_config=model_config
    )

    return PrecomputeResponse(status="computing", message="recompute feature")

# (device endpoint removed by request)