import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime

from .base_matcher import BaseFaceMatcher
from models.face_model import FaceEncoding, RecognitionResult
from config.models_config import RECOGNITION_TOLERANCE

class EuclideanFaceMatcher(BaseFaceMatcher):
    """Face matcher using Euclidean distance metrics."""
    
    def __init__(self, tolerance: float = RECOGNITION_TOLERANCE):
        super().__init__(tolerance)
        self._name = "euclidean"
    
    def match(self, 
              unknown_encoding: np.ndarray,
              known_encodings: List[FaceEncoding]) -> Optional[RecognitionResult]:
        """
        Match using Euclidean distance.
        """
        if not known_encodings:
            return None
            
        # Extract numpy arrays from FaceEncoding objects
        known_arrays = [enc.encoding for enc in known_encodings]
        
        # Calculate distances
        distances = np.linalg.norm(np.array(known_arrays) - unknown_encoding, axis=1)
        
        # Find best match
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        # Check if match is within tolerance
        if min_distance <= self.tolerance:
            confidence = 1 - (min_distance / self.tolerance)
            matched_encoding = known_encodings[min_distance_idx]
            
            return RecognitionResult(
                location=None,  # Location not needed for matching only
                name=matched_encoding.name,
                confidence=float(confidence),
                encoding=unknown_encoding,
                timestamp=datetime.now()
            )
            
        return None
    
    def batch_match(self,
                   unknown_encodings: List[np.ndarray],
                   known_encodings: List[FaceEncoding]) -> List[Optional[RecognitionResult]]:
        """
        Batch match using vectorized operations.
        """
        if not unknown_encodings or not known_encodings:
            return [None] * len(unknown_encodings)
            
        # Extract numpy arrays from FaceEncoding objects
        known_arrays = np.array([enc.encoding for enc in known_encodings])
        unknown_arrays = np.array(unknown_encodings)
        
        # Calculate all distances at once
        distances = np.linalg.norm(
            unknown_arrays[:, np.newaxis] - known_arrays, 
            axis=2
        )
        
        results = []
        for i, dist_row in enumerate(distances):
            min_idx = np.argmin(dist_row)
            min_dist = dist_row[min_idx]
            
            if min_dist <= self.tolerance:
                confidence = 1 - (min_dist / self.tolerance)
                matched_encoding = known_encodings[min_idx]
                
                result = RecognitionResult(
                    location=None,
                    name=matched_encoding.name,
                    confidence=float(confidence),
                    encoding=unknown_encodings[i],
                    timestamp=datetime.now()
                )
            else:
                result = None
                
            results.append(result)
            
        return results
    
    def compute_similarity(self,
                         encoding1: np.ndarray,
                         encoding2: np.ndarray) -> float:
        """
        Compute similarity score using Euclidean distance.
        """
        distance = np.linalg.norm(encoding1 - encoding2)
        
        # Convert distance to similarity score (0-1)
        similarity = max(0, 1 - (distance / self.tolerance))
        return float(similarity)
    
    @property
    def name(self) -> str:
        return self._name
