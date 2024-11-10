from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from models.face_model import FaceEncoding, FaceLocation
from ..detectors.base_detector import BaseFaceDetector

class BaseFaceEncoder(ABC):
    """Abstract base class for face encoders."""
    
    def __init__(self, detector: BaseFaceDetector):
        self.detector = detector
    
    @abstractmethod
    def encode_face(self, 
                   image: np.ndarray, 
                   face_location: Optional[FaceLocation] = None) -> Optional[np.ndarray]:
        """
        Encode a single face from an image.
        
        Args:
            image: numpy array of image data
            face_location: Optional known face location
            
        Returns:
            Face encoding as numpy array, or None if no face found
        """
        pass
    
    @abstractmethod
    def encode_faces(self, 
                    image: np.ndarray,
                    face_locations: Optional[List[FaceLocation]] = None) -> List[np.ndarray]:
        """
        Encode all faces in an image.
        
        Args:
            image: numpy array of image data
            face_locations: Optional list of known face locations
            
        Returns:
            List of face encodings
        """
        pass
    
    @abstractmethod
    def encode_image_file(self, 
                         image_path: Path,
                         face_location: Optional[FaceLocation] = None) -> Optional[FaceEncoding]:
        """
        Load and encode a face from an image file.
        
        Args:
            image_path: Path to image file
            face_location: Optional known face location
            
        Returns:
            FaceEncoding object, or None if no face found
        """
        pass
    
    @abstractmethod
    def batch_encode_files(self,
                          image_paths: List[Path],
                          batch_size: int = 32) -> List[Tuple[Path, Optional[FaceEncoding]]]:
        """
        Batch encode multiple image files.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to process at once
            
        Returns:
            List of tuples containing (image_path, face_encoding)
        """
        pass
    
    def compute_distance(self,
                        encoding1: np.ndarray,
                        encoding2: np.ndarray) -> float:
        """
        Compute distance between two face encodings.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            Distance between encodings
        """
        return np.linalg.norm(encoding1 - encoding2)
