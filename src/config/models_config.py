# Recognition Model Configuration
FACE_DETECTION_MODEL = 'hog'  # or 'cnn' for GPU systems
RECOGNITION_TOLERANCE = 0.6
NUM_JITTERS = 1
RECOGNITION_MODEL = 'small'  # or 'large' for more accuracy

# Real-time Processing Configuration
FRAME_SCALE_FACTOR = 0.25  # Scale down frames for faster processing

# Clustering Configuration
CLUSTERING_EPS = 0.5  # Maximum distance between samples
CLUSTERING_MIN_SAMPLES = 2  # Minimum cluster size
CLUSTERING_METRIC = 'euclidean'

# Image Processing Configuration
TARGET_FACE_SIZE = (216, 216)  # Size for processed face images
IMAGE_QUALITY_THRESHOLD = 50  # Minimum image quality score (0-100)

# Colors (BGR format)
KNOWN_FACE_COLOR = (0, 255, 0)     # Green
UNKNOWN_FACE_COLOR = (255, 0, 0)    # Blue
TEXT_COLOR = (255, 255, 255)        # White

# Recognition Performance
BATCH_SIZE = 128  # Batch size for face encoding
MAX_CONCURRENT_PROCESSES = 4  # For parallel processing
