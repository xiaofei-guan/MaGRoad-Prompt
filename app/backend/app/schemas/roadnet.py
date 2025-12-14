from typing import List, Dict, Any, Optional, Union, Tuple, Literal
from pydantic import BaseModel, Field
import json

# --- Node and Edge Schemas for optimized format ---

class RoadNode(BaseModel):
    """Schema for individual road network nodes."""
    id: int = Field(..., description="Unique node identifier")
    x: float = Field(..., description="X coordinate in image space")
    y: float = Field(..., description="Y coordinate in image space")

class RoadEdge(BaseModel):
    """Schema for individual road network edges."""
    id: int = Field(..., description="Unique edge identifier")
    source: int = Field(..., description="Source node ID")
    target: int = Field(..., description="Target node ID")

# --- Existing Schemas (ensure these are correctly placed or imported if they exist elsewhere) ---

class PointPrompt(BaseModel):
    """Schema for a single point prompt."""
    x: float = Field(..., description="X coordinate on the image")
    y: float = Field(..., description="Y coordinate on the image")
    label: int = Field(..., description="1 for positive prompt (road), 0 for negative prompt")
    id: Optional[str] = Field(None, description="Optional ID for the prompt, used for frontend tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "x": 100.5,
                "y": 200.75,
                "label": 1,
                "id": "prompt-123456"
            }
        }

class RoadNetGenerationRequest(BaseModel):
    """Request body for generating a road network."""
    image_id: str = Field(..., description="ID of the image to process")
    prompts: List[PointPrompt] = Field(..., description="List of point prompts")
    
    class Config:
        schema_extra = {
            "example": {
                "image_id": "image_001.png",
                "prompts": [
                    {"x": 100.5, "y": 200.75, "label": 1, "id": "prompt-123456"},
                    {"x": 150.0, "y": 250.0, "label": 0, "id": "prompt-789012"}
                ]
            }
        }

class GeoJSONGeometry(BaseModel):
    """Schema for a GeoJSON geometry object."""
    type: str = Field(..., description="Type of geometry ('Point', 'LineString', etc.)")
    coordinates: List[Any] = Field(..., description="Coordinates of the geometry")

class GeoJSONFeature(BaseModel):
    """Schema for a GeoJSON feature object."""
    type: str = Field("Feature", description="Must be 'Feature'")
    geometry: GeoJSONGeometry = Field(..., description="Geometry of the feature")
    properties: Optional[Dict[str, Any]] = Field(None, description="Properties of the feature")
    
class GeoJSONFeatureCollection(BaseModel):
    """Schema for a GeoJSON FeatureCollection object with node-edge format support."""
    type: str = Field("FeatureCollection", description="Must be 'FeatureCollection'")
    features: List[GeoJSONFeature] = Field(default_factory=list, description="List of features (kept for compatibility)")
    properties: Optional[Dict[str, Any]] = Field(None, description="Properties of the feature collection")
    
    # Node-edge format fields (v2.0+)
    nodes: List[RoadNode] = Field(..., description="List of road network nodes")
    edges: List[RoadEdge] = Field(..., description="List of road network edges")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[100.0, 0.0], [101.0, 1.0]]
                        },
                        "properties": {
                            "name": "Road segment 1"
                        }
                    }
                ],
                "properties": {
                    "creator": "auto_road_net_app"
                }
            }
        }

class RoadNetGenerationResponse(BaseModel):
    """Response model for road network generation with optimized mask transmission."""
    image_id: str
    geojson_data: GeoJSONFeatureCollection
    prompts: Optional[List[PointPrompt]] = None
    road_mask: Optional[Union[List[List[bool]], str]] = None  # List for small masks, compressed string for large masks
    kp_mask: Optional[Union[List[List[bool]], str]] = None    # List for small masks, compressed string for large masks
    road_mask_metadata: Optional[Dict[str, Any]] = None      # Metadata for decompression
    kp_mask_metadata: Optional[Dict[str, Any]] = None        # Metadata for decompression

# --- New Schemas for Saving Road Network ---

class SaveRoadNetworkRequest(BaseModel):
    """Request body for saving a road network."""
    image_id: str = Field(..., description="ID of the image")
    geojson_data: GeoJSONFeatureCollection = Field(..., description="Road network data as GeoJSON")
    prompts: Optional[List[Any]] = Field(None, description="Prompts used to generate this network")
    road_mask: Optional[List[List[float]]] = Field(None, description="Road mask data (DEPRECATED - uses cached data)")
    kp_mask: Optional[List[List[float]]] = Field(None, description="Keypoint mask data (DEPRECATED - uses cached data)")
    use_cached_masks: bool = Field(True, description="Whether to use cached mask data instead of transmitted masks")
    
    class Config:
        schema_extra = {
            "example": {
                "image_id": "image_001.png",
                "geojson_data": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[100.0, 0.0], [101.0, 1.0]]
                            },
                            "properties": {
                                "name": "Road segment 1"
                            }
                        }
                    ]
                }
            }
        }

class SaveRoadNetworkResponse(BaseModel):
    """Response body for a road network save request."""
    message: str = Field(..., description="Success/error message")
    file_path: Optional[str] = Field(None, description="Path where the annotation was saved") 


# --- Export Schemas ---

class ExportRequest(BaseModel):
    """Request body for exporting road network annotations as adjacency list pickles.

    scope: 'current' exports only the currently selected image (requires image_id).
           'all' exports all available images found in original filename mapping.
    coordinate_format: 'xy' means (x, y); 'rc' means (row, col) == (y, x).
    """
    scope: Literal['current', 'all'] = Field(..., description="Export scope: 'current' or 'all'")
    image_id: Optional[str] = Field(None, description="Image ID to export when scope is 'current'")
    coordinate_format: Literal['xy', 'rc'] = Field('xy', description="Coordinate format: 'xy' or 'rc'")

class ExportResponse(BaseModel):
    """Metadata response for export operations (not used for file streaming body)."""
    exported: int = Field(..., description="Number of exported annotations")
    missing: int = Field(0, description="Number of images without annotations")
    filename: str = Field(..., description="Suggested filename for the download")