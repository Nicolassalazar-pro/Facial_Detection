from pathlib import Path

# Directory Configuration
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
PROFILE_DIR = DATA_DIR / "profiles"
CACHE_DIR = DATA_DIR / "cache"

# Video Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_INDEX = 0  # Default webcam

# Buffer Configuration
FRAME_BUFFER_SIZE = 2
OVERLAY_BUFFER_SIZE = 1

# Display Configuration
WINDOW_NAME = "Spot's Live Feed"
DISPLAY_FPS = 30  # Target FPS for display

# File Processing
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
