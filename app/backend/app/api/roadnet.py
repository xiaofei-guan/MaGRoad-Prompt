from fastapi import APIRouter, HTTPException, Request, Body, Depends, Query
from typing import List, Dict, Any, Tuple, Optional
from pydantic import Field
import logging

# Service function import
from app.services import roadnet_service
from app.schemas.roadnet import (
    PointPrompt,
    RoadNetGenerationRequest,
    RoadNetGenerationResponse,
    SaveRoadNetworkRequest,
    SaveRoadNetworkResponse,
    GeoJSONFeatureCollection,
    ExportRequest,
    ExportResponse
)
from app.core.config import Settings # For type hinting get_settings
from fastapi.responses import StreamingResponse
from pathlib import Path
import io
import json
import zipfile

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Dependency Injection Helpers ---

def get_road_extraction_model(request: Request):
    model = request.app.state.model_store.get("road_extraction_model")
    if model is None:
        logger.error("Road extraction model not found in app.state.model_store.")
        raise HTTPException(status_code=500, detail="Road extraction model is not available. Check server startup logs.")
    return model

def get_road_extraction_model_config(request: Request):
    config = request.app.state.model_store.get("road_extraction_model_config")
    if config is None:
        logger.error("Road extraction model config not found in app.state.model_store.")
        raise HTTPException(status_code=500, detail="Road extraction model configuration is not available. Check server startup logs.")
    return config

def get_device(request: Request):
    device = request.app.state.model_store.get("device")
    if device is None:
        logger.error("Computation device not found in app.state.model_store.")
        raise HTTPException(status_code=500, detail="Computation device is not available. Check server startup logs.")
    return device

def get_settings(request: Request) -> Settings:
    app_settings = request.app.state.settings
    if app_settings is None:
        logger.error("Application settings not found in app.state.")
        raise HTTPException(status_code=500, detail="Application settings are not available. Check server startup logs.")
    return app_settings

# --- API Endpoints ---

@router.post(
    "/generate-from-prompts",
    response_model=RoadNetGenerationResponse,
    summary="Generate Road Network from Image ID and Prompts",
    description="Takes an image ID and user prompts to generate a road network in GeoJSON format."
)
async def generate_road_network(
    request: RoadNetGenerationRequest,
    include_masks: bool = Query(default=True, description="Whether to include mask data in response"),
    model: Any = Depends(get_road_extraction_model),
    config: Any = Depends(get_road_extraction_model_config),
    device: Any = Depends(get_device),
    settings: Settings = Depends(get_settings)
) -> RoadNetGenerationResponse:
    """
    Generate road network from prompts with optional mask data.
    
    Args:
        request: The road network generation request
        include_masks: Whether to include mask data (set to False for faster responses)
        model: Road extraction model
        config: Model configuration
        device: Computation device
        settings: Application settings
    """
    logger.info(f"Received request to generate road network for image ID: {request.image_id}")
    logger.info(f"Request prompts: {request.prompts}")
    logger.info(f"Include masks: {include_masks}")
    
    if not request.prompts:
        logger.warning("No prompts provided for road network generation")
        raise HTTPException(status_code=400, detail="At least one prompt is required")
    
    try:
        result = await roadnet_service.generate_road_network(
            image_id=request.image_id,
            prompts=[p.model_dump() if hasattr(p, 'model_dump') else p for p in request.prompts],  # Convert PointPrompt objects to dicts
            device=device,
            model=model,
            config=config,  # Pass the model config, not settings
            include_masks=include_masks
        )
        
        # Convert to Pydantic models for response (optimized for new format)
        try:
            geojson_data = result["geojson_data"]
            # Direct conversion - only support new node-edge format
            geojson_collection = GeoJSONFeatureCollection(**geojson_data)
            prompts_for_response = [PointPrompt(**p) for p in result["prompts"]]
            
        except Exception as e:
            logger.error(f"Error converting data to Pydantic models: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process road network data: {e}")
        
        logger.info(f"Road network generation completed successfully for {request.image_id}")
        
        return RoadNetGenerationResponse(
            image_id=result["image_id"],
            geojson_data=geojson_collection,
            prompts=prompts_for_response,
            road_mask=result.get("road_mask", None),
            kp_mask=result.get("kp_mask", None),
            road_mask_metadata=result.get("road_mask_metadata", None),
            kp_mask_metadata=result.get("kp_mask_metadata", None)
        )
    except Exception as e:
        logger.error(f"Error generating road network: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/save", 
    response_model=SaveRoadNetworkResponse,
    summary="Save Road Network Annotation",
    description="Saves the provided road network GeoJSON data to the server for a given image."
)
async def save_road_network(
    payload: SaveRoadNetworkRequest,
    settings: Settings = Depends(get_settings)
    # request: Request # Not strictly needed here unless accessing app.state for something specific
):
    """
    Saves a road network annotation (GeoJSON) to a file.

    - **image_id**: The ID of the image for which the annotation is being saved.
    - **geojson_data**: The road network data in GeoJSON format.
    """
    try:
        logger.info(f"Received request to save road network for image_id: {payload.image_id}")
        
        # Prioritize cached masks, only use transmitted masks as fallback
        road_mask_to_use = payload.road_mask if not payload.use_cached_masks else None
        kp_mask_to_use = payload.kp_mask if not payload.use_cached_masks else None
        
        saved_file_path = await roadnet_service.save_road_network_annotation(
            image_id=payload.image_id,
            geojson_data=payload.geojson_data,
            prompts= [p if isinstance(p, dict) else p.model_dump() for p in payload.prompts] if payload.prompts else None,
            road_mask=road_mask_to_use,  # Only used as fallback when cache miss
            kp_mask=kp_mask_to_use,      # Only used as fallback when cache miss
            annotations_dir=settings.ANNOTATIONS_DIR
        )
        
        return SaveRoadNetworkResponse(
            status="success",
            message="Road network annotation saved successfully.",
            file_path=saved_file_path
        )
    except ValueError as ve:
        logger.warning(f"Validation error while saving road network for {payload.image_id}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except IOError as ioe:
        logger.error(f"IOError while saving road network for {payload.image_id}: {ioe}")
        raise HTTPException(status_code=500, detail=f"Failed to save road network due to a storage error: {ioe}")
    except Exception as e:
        logger.exception(f"Unexpected error while saving road network for {payload.image_id}: {e}") # Use logger.exception to include stack trace
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")

@router.get("/{image_id}", response_model=RoadNetGenerationResponse)
async def load_road_network(
    image_id: str,
    include_masks: bool = Query(default=True, description="Whether to include mask data in response"),
    settings: Settings = Depends(get_settings),
    config: Any = Depends(get_road_extraction_model_config)
) -> RoadNetGenerationResponse:
    logger.info(f"Received request to load road network for image ID: {image_id}")
    logger.info(f"Include masks: {include_masks}")
    try:
        loaded_data = await roadnet_service.load_road_network_annotation(
            image_id=image_id,
            annotations_dir=settings.ANNOTATIONS_DIR,
            include_masks=include_masks,
            config=config
        )
        if loaded_data:
            # loaded_data is dict: {"image_id": ..., "geojson_data": ..., "prompts": ...}
            # Ensure prompts are converted to PointPrompt models if they exist
            prompts_for_response = None
            if loaded_data.get("prompts"):
                prompts_for_response = [PointPrompt(**p) for p in loaded_data["prompts"]]
            
            # Include road_mask and kp_mask in the response if available
            road_mask = loaded_data.get("road_mask", None)
            kp_mask = loaded_data.get("kp_mask", None)
            
            # Direct conversion - only support new node-edge format
            geojson_collection = GeoJSONFeatureCollection(**loaded_data["geojson_data"])
            
            return RoadNetGenerationResponse(
                image_id=loaded_data["image_id"],
                geojson_data=geojson_collection,
                prompts=prompts_for_response,
                road_mask=road_mask,
                kp_mask=kp_mask,
                road_mask_metadata=loaded_data.get("road_mask_metadata", None),
                kp_mask_metadata=loaded_data.get("kp_mask_metadata", None)
            )
        else:
            # When no data is found, return a clean 404 response
            logger.info(f"No annotation data found for image ID: {image_id}")
            raise HTTPException(status_code=404, detail=f"Annotation data not found for image ID: {image_id}")
    except HTTPException:
        # Let FastAPI HTTPExceptions pass through unchanged
        raise
    except Exception as e:
        # Only convert other exceptions to 500 errors
        logger.error(f"Error loading road network for {image_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Note: Remember to include this router in main.py
# from app.api import roadnet
# app.include_router(roadnet.router, prefix="/api/road-network", tags=["Road Network"]) 

# --- Export Endpoint ---

def _serialize_adjacency_pickle_bytes(geojson: dict, coord_format: str = 'xy') -> bytes:
    """Convert node-edge GeoJSON dict into adjacency list and pickle to bytes.
    Keys and values are tuples of ints. coordinate_format: 'xy' or 'rc' (row,col==y,x).
    """
    import pickle
    nodes = geojson.get('nodes', [])
    edges = geojson.get('edges', [])

    # id -> (x, y)
    id_to_xy = {int(n['id']): (int(round(n['x'])) if 'x' in n else int(round(n.get('x', 0))),
                                int(round(n['y'])) if 'y' in n else int(round(n.get('y', 0)))) for n in nodes}

    def format_point(xy: Tuple[int, int]) -> Tuple[int, int]:
        x, y = xy
        return (x, y) if coord_format == 'xy' else (y, x)

    adjacency: Dict[Tuple[int, int], set[Tuple[int, int]]] = {}
    for edge in edges:
        s = int(edge['source']); t = int(edge['target'])
        if s in id_to_xy and t in id_to_xy:
            ps = format_point(id_to_xy[s]); pt = format_point(id_to_xy[t])
            adjacency.setdefault(ps, set()).add(pt)
            adjacency.setdefault(pt, set()).add(ps)
    # Convert sets to lists for final structure
    adjacency_list: Dict[Tuple[int, int], List[Tuple[int, int]]] = {k: sorted(list(v)) for k, v in adjacency.items()}
    return pickle.dumps(adjacency_list, protocol=pickle.HIGHEST_PROTOCOL)

def _load_original_filename_map(images_dir: Path) -> Dict[str, str]:
    mapping_file = images_dir / 'original_filenames.txt'
    mapping: Dict[str, str] = {}
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    image_id = parts[0].strip()
                    original_name = ','.join(parts[1:]).strip()
                    mapping[image_id] = original_name
    return mapping

@router.post(
    "/export",
    response_model=ExportResponse,
    summary="Export road networks as adjacency list pickles",
)
async def export_road_networks(
    payload: ExportRequest = Body(...),
    settings: Settings = Depends(get_settings)
):
    images_dir = Path(settings.IMAGES_DIR)
    annotations_dir = Path(settings.ANNOTATIONS_DIR)
    original_map = _load_original_filename_map(images_dir)

    scope = payload.scope
    coord_format = payload.coordinate_format

    exported = 0
    missing = 0

    if scope == 'current':
        if not payload.image_id:
            raise HTTPException(status_code=400, detail="image_id is required when scope is 'current'")
        image_id = payload.image_id
        geojson_path = annotations_dir / f"{image_id}_roadnetwork.geojson"
        if not geojson_path.exists():
            missing = 1
            # Return an empty adjacency list pickle
            data_bytes = _serialize_adjacency_pickle_bytes({"nodes": [], "edges": []}, coord_format)
            filename = (original_map.get(image_id, image_id).rsplit('.', 1)[0]) + '.pickle'
            headers = {"Content-Disposition": f"attachment; filename=\"{filename}\"",
                       "X-Export-Stats": json.dumps({"exported": 0, "missing": 1, "filename": filename})}
            return StreamingResponse(io.BytesIO(data_bytes), media_type="application/octet-stream", headers=headers)

        with open(geojson_path, 'r') as f:
            geojson = json.load(f)
        data = _serialize_adjacency_pickle_bytes(geojson, coord_format)
        filename = (original_map.get(image_id, image_id).rsplit('.', 1)[0]) + '.pickle'
        headers = {"Content-Disposition": f"attachment; filename=\"{filename}\"",
                   "X-Export-Stats": json.dumps({"exported": 1, "missing": 0, "filename": filename})}
        return StreamingResponse(io.BytesIO(data), media_type="application/octet-stream", headers=headers)

    # scope == 'all'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        candidates = list(original_map.keys())
        if not candidates:
            for path in annotations_dir.glob('*_roadnetwork.geojson'):
                candidates.append(path.name.replace('_roadnetwork.geojson', ''))
        for image_id in candidates:
            geojson_path = annotations_dir / f"{image_id}_roadnetwork.geojson"
            if not geojson_path.exists():
                missing += 1
                continue
            try:
                with open(geojson_path, 'r') as f:
                    geojson = json.load(f)
                data = _serialize_adjacency_pickle_bytes(geojson, coord_format)
                base = original_map.get(image_id, image_id)
                base_no_ext = base.rsplit('.', 1)[0]
                zf.writestr(f"{base_no_ext}.pickle", data)
                exported += 1
            except Exception:
                missing += 1
                continue

    buffer.seek(0)
    filename = 'all.zip'
    headers = {"Content-Disposition": f"attachment; filename=\"{filename}\"",
               "X-Export-Stats": json.dumps({"exported": exported, "missing": missing, "filename": filename})}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)