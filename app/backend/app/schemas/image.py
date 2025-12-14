from pydantic import BaseModel
from typing import List

class ImageInfo(BaseModel):
    id: str
    filename: str
    original_filename: str  # Original filename before hashing
    has_feature: bool # Use snake_case for Python/Pydantic
    url: str # URL to access the image via API
    is_annotated: bool = False  # Whether this image has been annotated

class ImageListResponse(BaseModel):
    images: List[ImageInfo]

class UploadResponse(BaseModel):
    message: str
    image_ids: List[str]
    images: List[ImageInfo]

class FeatureStatusResponse(BaseModel):
    status: str # 'none', 'computing', 'ready', 'error'
    message: str

# Reuse FeatureStatusResponse for precompute start response
class PrecomputeResponse(BaseModel):
    status: str # 'computing', 'ready', 'error'
    message: str 