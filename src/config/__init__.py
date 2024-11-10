from .general_config import *
from .models_config import *

# Version info
__version__ = '1.0.0'

# Validate configurations
def validate_config():
    """Validate critical configuration values."""
    required_dirs = [DATA_DIR, PROFILE_DIR, CACHE_DIR]
    for directory in required_dirs:
        if not directory.exists():
            raise ValueError(f"Required directory does not exist: {directory}")
            
    if CAMERA_INDEX < 0:
        raise ValueError("Invalid camera index")
        
    if FRAME_SCALE_FACTOR <= 0 or FRAME_SCALE_FACTOR > 1:
        raise ValueError("Frame scale factor must be between 0 and 1")
        
    if not all(ext.startswith('.') for ext in SUPPORTED_IMAGE_EXTENSIONS):
        raise ValueError("Image extensions must start with '.'")

# Run validation when config is imported
validate_config()
