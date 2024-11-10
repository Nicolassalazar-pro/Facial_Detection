# Import main components for easier access
from .services import (
    VideoService,
    RecognitionService,
    ClusteringService,
    ProfileWatcherService
)

from .utils import (
    cleanup_profile_images,
    FileManager,
    CacheManager
)

from .config import (
    DATA_DIR,
    PROFILE_DIR,
    CACHE_DIR,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    WINDOW_NAME
)

# Version info
__version__ = '1.0.0'

__all__ = [
    'VideoService',
    'RecognitionService',
    'ClusteringService',
    'ProfileWatcherService',
    'cleanup_profile_images',
    'FileManager',
    'CacheManager',
    'DATA_DIR',
    'PROFILE_DIR',
    'CACHE_DIR',
    'CAMERA_WIDTH',
    'CAMERA_HEIGHT',
    'WINDOW_NAME'
]
