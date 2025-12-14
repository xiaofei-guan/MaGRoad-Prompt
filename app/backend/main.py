from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
from contextlib import asynccontextmanager
from omegaconf import OmegaConf
import logging
import sys # Added for explicit path modification if needed
import os  # Added for path joining

from app.api import images
from app.api import roadnet
from app.core.config import settings
from app.models.road_extraction_algorithm.sam_road_extraction import RoadExtractionModel
from app.services.mask_cache_service import mask_cache_service


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_device_from_config(settings) -> torch.device:
    """
    according to the configuration to determine the computing device
    
    Args:
        settings: application configuration object
        
    Returns:
        torch.device: computing device
    """
    device_type = settings.DEVICE_TYPE.lower()
    
    if device_type == "cpu":
        logger.info("Device configuration: CPU")
        return torch.device("cpu")
    
    elif device_type == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        
        # verify if the GPU device ID is valid
        gpu_count = torch.cuda.device_count()
        if settings.GPU_DEVICE_ID >= gpu_count:
            logger.warning(f"GPU device {settings.GPU_DEVICE_ID} not available (only {gpu_count} GPUs found), using GPU 0")
            device_id = 0
        else:
            device_id = settings.GPU_DEVICE_ID
        
        device = torch.device(f"cuda:{device_id}")
        logger.info(f"Device configuration: CUDA GPU {device_id}")
        
        # print GPU information
        try:
            gpu_name = torch.cuda.get_device_name(device_id)
            gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            logger.info(f"Using GPU: {gpu_name} (Memory: {gpu_memory:.1f} GB)")
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
        
        return device
    
    elif device_type == "auto":
        if torch.cuda.is_available():
            # automatically select the GPU device ID configured in auto mode
            gpu_count = torch.cuda.device_count()
            if settings.GPU_DEVICE_ID >= gpu_count:
                logger.warning(f"GPU device {settings.GPU_DEVICE_ID} not available (only {gpu_count} GPUs found), using GPU 0")
                device_id = 0
            else:
                device_id = settings.GPU_DEVICE_ID
            
            device = torch.device(f"cuda:{device_id}")
            logger.info(f"Device configuration: Auto - selected CUDA GPU {device_id}")
            
            # print GPU information
            try:
                gpu_name = torch.cuda.get_device_name(device_id)
                gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
                logger.info(f"Using GPU: {gpu_name} (Memory: {gpu_memory:.1f} GB)")
            except Exception as e:
                logger.warning(f"Could not get GPU info: {e}")
            
            return device
        else:
            logger.info("Device configuration: Auto - CUDA not available, using CPU")
            return torch.device("cpu")
    
    else:
        logger.warning(f"Unknown device type '{device_type}', falling back to auto mode")
        return _get_device_from_config(type('Settings', (), {'DEVICE_TYPE': 'auto', 'GPU_DEVICE_ID': settings.GPU_DEVICE_ID})())

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    """
    # --- Startup ---
    # Initialize model_store directly on app.state
    app.state.model_store = {}
    app.state.settings = settings # keep settings assignment

    # Determine compute device based on configuration
    device = _get_device_from_config(settings)
    # Store device in app.state.model_store
    app.state.model_store["device"] = device
    logger.info(f"Application startup. Using device: {app.state.model_store['device']}")

    # Initialize mask cache service
    try:
        mask_cache_service  # Ensure singleton is initialized
        logger.info("Mask cache service initialized successfully")
        logger.info(f"Cache settings: max_size={settings.MASK_CACHE_MAX_SIZE}, expiry={settings.MASK_CACHE_EXPIRY_HOURS}h")
    except Exception as e:
        logger.error(f"Failed to initialize mask cache service: {e}")

    # --- Model Loading Logic ---
    try:
        logger.info("Loading Road Extraction Model...")
        # Load Road Extraction Model Configuration
        cfg = OmegaConf.load(settings.ROAD_EXTRACTION_MODEL_CONFIG_PATH)
        # Store config in app.state.model_store
        app.state.model_store["road_extraction_model_config"] = cfg
        logger.info(f"Loaded road extraction model config from: {settings.ROAD_EXTRACTION_MODEL_CONFIG_PATH}")

        # Initialize RoadExtractionModel
        road_extraction_model = RoadExtractionModel(cfg)
        logger.info("RoadExtractionModel initialized.")

        # Load Checkpoint
        logger.info(f"Loading checkpoint from: {cfg.ROAD_EXTRACTION_MODEL_CKPT_PATH}")
        # Using map_location='cpu' initially
        ckpt = torch.load(cfg.ROAD_EXTRACTION_MODEL_CKPT_PATH, map_location='cpu')

        # Load State Dictionary
        state_dict_key = 'state_dict'
        if state_dict_key not in ckpt:
            logger.warning(f"Checkpoint does not contain key '{state_dict_key}'. Attempting to load entire checkpoint as state_dict.")
            model_state_dict = ckpt
        else:
            model_state_dict = ckpt[state_dict_key]

        road_extraction_model.load_state_dict(model_state_dict, strict=False)
        logger.info("Model state_dict loaded successfully.")
        # Removed verbose model logging for brevity
        # logger.info(f"Model: {road_extraction_model}")

        # Set to evaluation mode and move to device
        road_extraction_model.eval()
        road_extraction_model.to(device)
        # Store model in app.state.model_store
        app.state.model_store["road_extraction_model"] = road_extraction_model
        logger.info(f"Road Extraction Model loaded, set to eval mode, and moved to device: {device}.")

    except FileNotFoundError as e:
        logger.error(f"Error loading road extraction model: Model file not found at {e.filename}. Please check paths in config.py.")
        app.state.model_store["road_extraction_model"] = None
        app.state.model_store["road_extraction_model_config"] = None
    except Exception as e:
        logger.error(f"An unexpected error occurred during road extraction model loading: {e}")
        logger.exception("Detailed traceback:")
        app.state.model_store["road_extraction_model"] = None
        app.state.model_store["road_extraction_model_config"] = None

    yield # The application runs while the context manager is active

    # --- Shutdown ---
    logger.info("Application shutdown.")
    
    # Clear mask cache
    try:
        mask_cache_service.clear_cache()
        logger.info("Mask cache cleared on shutdown")
    except Exception as e:
        logger.error(f"Error clearing mask cache on shutdown: {e}")
    
    # Clear the model store from app.state
    if hasattr(app.state, 'model_store'):
         del app.state.model_store
    logger.info("Model store cleared from app.state.")


# Initialize FastAPI app with the lifespan context manager
app = FastAPI(title="Auto Road Network Annotation API", lifespan=lifespan)

# CORS Middleware Configuration (Allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["Content-Disposition", "X-Export-Stats"],  # Expose download filename and export stats
)

# --- API Routers ---
# Register Routers
app.include_router(images.router, prefix="/api/images", tags=["Images"])
app.include_router(roadnet.router, prefix="/api/road-network", tags=["Road Network"])

# --- Example Health Check Endpoint ---
@app.get("/api/health")
async def health_check(request: Request): # Added Request to access app state
    """Basic health check endpoint."""
    # Access model_store safely from app.state
    device_info = "Device not initialized"
    if hasattr(request.app.state, 'model_store') and request.app.state.model_store:
        device_info = str(request.app.state.model_store.get("device", "Device key missing"))
    return {"status": "healthy", "device": device_info}

# --- Main Execution Block (for running with uvicorn directly) ---
if __name__ == "__main__":
    import uvicorn
    # Ensure reload works correctly with lifespan (uvicorn handles this typically)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
