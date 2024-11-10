from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional

from models.face_model import FaceEncoding, RecognitionResult
from config.models_config import RECOGNITION_TOLERANCE

class BaseFaceMatcher(ABC):
    """Abstract base class for face matching operations."""
    
    def __init__(self, tolerance: float = RECOGNITION_TOLERANCE):
        self.tolerance = tolerance
    
    @abstractmethod
    def match(self, 
              unknown_encoding: np.ndarray,
              known_encodings: List[FaceEncoding]) -> Optional[RecognitionResult]:
        """
        Match an unknown face encoding against known face encodings.
        
        Args:
            unknown_encoding: Face encoding to match
            known_encodings: List of known face encodings to match against
            
        Returns:
            RecognitionResult if match found, None otherwise
        """
        pass
    
    @abstractmethod
    def batch_match(self,
                   unknown_encodings: List[np.ndarray],
                   known_encodings: List[FaceEncoding]) -> List[Optional[RecognitionResult]]:
        """
        Match multiple unknown face encodings against known faces.
        
        Args:
            unknown_encodings: List of face encodings to match
            known_encodings: List of known face encodings to match against
            
        Returns:
            List of RecognitionResult, None for no matches
        """
        pass
    
    @abstractmethod
    def compute_similarity(self,
                         encoding1: np.ndarray,
                         encoding2: np.ndarray) -> float:
        """
        Compute similarity score between two face encodings.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            Similarity score (0-1), higher is more similar
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the matcher implementation."""
        pass
