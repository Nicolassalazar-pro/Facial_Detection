import face_recognition
import numpy as np
from typing import List, Optional
import cv2

from .base_detector import BaseFaceDetector
from models.face_model import FaceLocation
from config.models_config import (
    FACE_DETECTION_MODEL,
    RECOGNITION_TOLERANCE,
    NUM_JITTERS,
    RECOGNITION_MODEL
)

class RealtimeFaceDetector(BaseFaceDetector):
    """Optimized face detector for real-time video processing."""
    
    def __init__(self, 
                model: str = FACE_DETECTION_MODEL,
                num_jitters: int = NUM_JITTERS,
                model_type: str = RECOGNITION_MODEL):
        """
        Initialize the realtime face detector.
        
        Args:
            model: Face detection model ('hog' or 'cnn')
            num_jitters: Number of times to sample face during encoding
            model_type: Face recognition model type ('small' or 'large')
        """
        self.model = model
        self.num_jitters = num_jitters
        self.model_type = model_type
        self.tolerance = RECOGNITION_TOLERANCE
    
    def detect(self, image: np.ndarray) -> List[FaceLocation]:
        """
        Detect faces in the image using HOG-based detector for speed.
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get face locations
        face_locations = face_recognition.face_locations(
            rgb_image,
            model=self.model
        )
        
        # Convert to our FaceLocation type
        return [FaceLocation.from_tuple(loc) for loc in face_locations]
    
    def get_encodings(self, image: np.ndarray, locations: Optional[List[FaceLocation]] = None) -> List[np.ndarray]:
        """
        Get face encodings optimized for real-time processing.
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert locations to tuples if provided
        face_locations = None
        if locations is not None:
            face_locations = [loc.to_tuple() for loc in locations]
        
        # Get encodings
        encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=face_locations,
            num_jitters=self.num_jitters,
            model=self.model_type
        )
        
        return encodings
    
    def compare_faces(self, face_encodings: List[np.ndarray], face_to_compare: np.ndarray) -> List[bool]:
        """
        Compare faces with optimized tolerance for real-time matching.
        """
        return face_recognition.compare_faces(
            face_encodings,
            face_to_compare,
            tolerance=self.tolerance
        )
    
    def face_distance(self, face_encodings: List[np.ndarray], face_to_compare: np.ndarray) -> np.ndarray:
        """
        Compute optimized face distances for real-time processing.
        """
        return face_recognition.face_distance(face_encodings, face_to_compare)
