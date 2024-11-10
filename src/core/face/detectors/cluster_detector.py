import face_recognition
import numpy as np
from typing import List, Optional
import cv2

from .base_detector import BaseFaceDetector
from models.face_model import FaceLocation
from config.models_config import (
    RECOGNITION_MODEL,
    NUM_JITTERS
)

class ClusterFaceDetector(BaseFaceDetector):
    """High-accuracy face detector for clustering operations."""
    
    def __init__(self,
                model_type: str = RECOGNITION_MODEL,
                num_jitters: int = NUM_JITTERS * 2):  # More jitters for accuracy
        """
        Initialize the clustering face detector.
        
        Args:
            model_type: Face recognition model type ('small' or 'large')
            num_jitters: Number of times to sample face during encoding
        """
        self.model_type = model_type
        self.num_jitters = num_jitters
        self.tolerance = 0.4  # Stricter tolerance for clustering
    
    def detect(self, image: np.ndarray) -> List[FaceLocation]:
        """
        Detect faces using CNN detector for higher accuracy.
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use CNN model for more accurate detection
        face_locations = face_recognition.face_locations(
            rgb_image,
            model="cnn"  # Always use CNN for clustering
        )
        
        return [FaceLocation.from_tuple(loc) for loc in face_locations]
    
    def get_encodings(self, image: np.ndarray, locations: Optional[List[FaceLocation]] = None) -> List[np.ndarray]:
        """
        Get high-accuracy face encodings for clustering.
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert locations to tuples if provided
        face_locations = None
        if locations is not None:
            face_locations = [loc.to_tuple() for loc in locations]
        
        # Get encodings with more jitters for accuracy
        encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=face_locations,
            num_jitters=self.num_jitters,
            model=self.model_type
        )
        
        return encodings
    
    def compare_faces(self, face_encodings: List[np.ndarray], face_to_compare: np.ndarray) -> List[bool]:
        """
        Compare faces with stricter tolerance for clustering.
        """
        return face_recognition.compare_faces(
            face_encodings,
            face_to_compare,
            tolerance=self.tolerance
        )
    
    def face_distance(self, face_encodings: List[np.ndarray], face_to_compare: np.ndarray) -> np.ndarray:
        """
        Compute face distances for clustering.
        """
        return face_recognition.face_distance(face_encodings, face_to_compare)
    
    def batch_encode(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Batch encode multiple images for efficient clustering.
        """
        encodings = []
        for image in images:
            locs = self.detect(image)
            if locs:
                encs = self.get_encodings(image, [locs[0]])  # Take first face only
                if encs:
                    encodings.append(encs[0])
        return encodings
