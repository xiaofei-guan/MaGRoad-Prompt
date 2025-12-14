import os
from pathlib import Path

# Project Root Directory (assuming config.py is in app/core)
# Adjust if your structure is different
BASE_DIR = Path(__file__).resolve().parent.parent.parent # auto_road_net_app/backend
STORAGE_DIR = BASE_DIR / "storage"
IMAGES_DIR = STORAGE_DIR / "images"
FEATURES_DIR = STORAGE_DIR / "features"
ANNOTATIONS_DIR = STORAGE_DIR / "annotations"

# Large Image Processing Configuration
LARGE_IMAGE_THRESHOLD = 1024  # If width or height > this, treat as large image
PATCH_SIZE = 1024  # Size of each patch for large images
PATCH_OVERLAP = 128  # Overlap between patches to avoid edge artifacts

# Batch Processing Configuration for Performance Optimization
FEATURE_COMPUTATION_BATCH_SIZE = 16  # Process patches in batches for efficiency
MASK_GENERATION_BATCH_SIZE = 16      # Batch size for mask generation (typically smaller due to memory)
TOPO_CLASSIFICATION_BATCH_SIZE = 16  # Batch size for node relationship classification

# Memory Management Configuration
MAX_PATCHES_IN_MEMORY = 64  # Maximum number of patches to keep in memory
ENABLE_PATCH_CACHING = True  # Enable caching of frequently accessed patches

# GPU Device Configuration
# DEVICE_TYPE options:
#   - "auto": automatically select (use GPU if available, otherwise use CPU)
#   - "cpu": force use CPU
#   - "cuda": force use GPU (if available)
DEVICE_TYPE = "auto"  
GPU_DEVICE_ID = 0     # GPU device ID (0, 1, 2, ...)

# Road Extraction Algorithm Paths
ROAD_EXTRACTION_MODEL_CONFIG_PATH = BASE_DIR / "app" / "models" / "road_extraction_algorithm" / "config_road_extraction.yaml"
# ROAD_EXTRACTION_MODEL_CKPT_PATH will be set in the config YAML file

# Cache settings
MASK_CACHE_EXPIRY_HOURS = 24  # Cache expiry time in hours
MASK_CACHE_MAX_SIZE = 100     # Maximum number of cached mask sets
MASK_CACHE_CLEANUP_INTERVAL = 3600  # Cleanup interval in seconds (1 hour)

class Settings:
    PROJECT_NAME: str = "Auto Road Network Annotation"
    VERSION: str = "0.1.0"

    # Storage Paths
    IMAGES_DIR = IMAGES_DIR
    FEATURES_DIR = FEATURES_DIR
    ANNOTATIONS_DIR = ANNOTATIONS_DIR

    # Road Extraction Algorithm Paths
    ROAD_EXTRACTION_MODEL_CONFIG_PATH = ROAD_EXTRACTION_MODEL_CONFIG_PATH

    # Large image processing settings
    LARGE_IMAGE_THRESHOLD = LARGE_IMAGE_THRESHOLD
    PATCH_SIZE = PATCH_SIZE
    PATCH_OVERLAP = PATCH_OVERLAP
    
    # Batch processing settings for performance optimization
    FEATURE_COMPUTATION_BATCH_SIZE = FEATURE_COMPUTATION_BATCH_SIZE
    MASK_GENERATION_BATCH_SIZE = MASK_GENERATION_BATCH_SIZE
    TOPO_CLASSIFICATION_BATCH_SIZE = TOPO_CLASSIFICATION_BATCH_SIZE
    
    # Memory management settings
    MAX_PATCHES_IN_MEMORY = MAX_PATCHES_IN_MEMORY
    ENABLE_PATCH_CACHING = ENABLE_PATCH_CACHING

    # GPU Device Configuration
    DEVICE_TYPE = DEVICE_TYPE
    GPU_DEVICE_ID = GPU_DEVICE_ID

    # Cache settings
    MASK_CACHE_EXPIRY_HOURS = MASK_CACHE_EXPIRY_HOURS
    MASK_CACHE_MAX_SIZE = MASK_CACHE_MAX_SIZE
    MASK_CACHE_CLEANUP_INTERVAL = MASK_CACHE_CLEANUP_INTERVAL

    def __init__(self):
        # Create storage directories if they don't exist
        self.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        self.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        self.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        # Optional: Create a placeholder weights directory if it doesn't exist
        # (Path(self.IMAGE_ENCODER_PATH).parent).mkdir(parents=True, exist_ok=True)

settings = Settings()

# You could also use Pydantic's BaseSettings for environment variable loading
# from pydantic_settings import BaseSettings
# class Settings(BaseSettings):
#     # ... fields ...
#     class Config:
#         env_file = '.env' 