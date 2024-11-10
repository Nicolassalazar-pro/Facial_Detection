from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional

from models.face_model import FaceLocation

class BaseFaceDetector(ABC):
    """Abstract base class for face detectors."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceLocation]:
        """
        Detect faces in the given image.
        
        Args:
            image: numpy array of image data (BGR format)
            
        Returns:
            List of FaceLocation objects
        """
        pass
    
    @abstractmethod
    def get_encodings(self, image: np.ndarray, locations: Optional[List[FaceLocation]] = None) -> List[np.ndarray]:
        """
        Get face encodings from the image.
        
        Args:
            image: numpy array of image data (BGR format)
            locations: Optional list of face locations to encode
            
        Returns:
            List of face encodings as numpy arrays
        """
        pass
    
    def detect_and_encode(self, image: np.ndarray) -> Tuple[List[FaceLocation], List[np.ndarray]]:
        """
        Detect faces and get their encodings in one pass.
        
        Args:
            image: numpy array of image data (BGR format)
            
        Returns:
            Tuple of (face locations, face encodings)
        """
        locations = self.detect(image)
        encodings = self.get_encodings(image, locations)
        return locations, encodings
    
    @abstractmethod
    def compare_faces(self, face_encodings: List[np.ndarray], face_to_compare: np.ndarray) -> List[bool]:
        """
        Compare face encodings against a known face.
        
        Args:
            face_encodings: List of known face encodings
            face_to_compare: Face encoding to compare against
            
        Returns:
            List of boolean matches
        """
        pass
    
    @abstractmethod
    def face_distance(self, face_encodings: List[np.ndarray], face_to_compare: np.ndarray) -> np.ndarray:
        """
        Compute distance between face encodings.
        
        Args:
            face_encodings: List of known face encodings
            face_to_compare: Face encoding to compare against
            
        Returns:
            Array of distances for each face
        """
        pass
