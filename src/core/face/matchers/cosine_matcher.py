import numpy as np
from typing import List, Optional
from datetime import datetime
from scipy.spatial.distance import cosine

from .base_matcher import BaseFaceMatcher
from models.face_model import FaceEncoding, RecognitionResult
from config.models_config import RECOGNITION_TOLERANCE

class CosineFaceMatcher(BaseFaceMatcher):
    """Face matcher using Cosine similarity metrics."""
    
    def __init__(self, tolerance: float = RECOGNITION_TOLERANCE):
        super().__init__(tolerance)
        self._name = "cosine"
    
    def match(self, 
              unknown_encoding: np.ndarray,
              known_encodings: List[FaceEncoding]) -> Optional[RecognitionResult]:
        """
        Match using Cosine similarity.
        """
        if not known_encodings:
            return None
            
        # Extract numpy arrays from FaceEncoding objects
        known_arrays = [enc.encoding for enc in known_encodings]
        
        # Calculate similarities
        similarities = [1 - cosine(unknown_encoding, enc) for enc in known_arrays]
        
        # Find best match
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
        
        # Check if match is within tolerance
        if max_similarity >= (1 - self.tolerance):
            matched_encoding = known_encodings[max_similarity_idx]
            
            return RecognitionResult(
                location=None,
                name=matched_encoding.name,
                confidence=float(max_similarity),
                encoding=unknown_encoding,
                timestamp=datetime.now()
            )
            
        return None
    
    def batch_match(self,
                   unknown_encodings: List[np.ndarray],
                   known_encodings: List[FaceEncoding]) -> List[Optional[RecognitionResult]]:
        """
        Batch match using vectorized cosine similarity.
        """
        if not unknown_encodings or not known_encodings:
            return [None] * len(unknown_encodings)
            
        # Extract numpy arrays from FaceEncoding objects
        known_arrays = np.array([enc.encoding for enc in known_encodings])
        unknown_arrays = np.array(unknown_encodings)
        
        # Normalize vectors for faster cosine similarity
        known_norm = known_arrays / np.linalg.norm(known_arrays, axis=1)[:, np.newaxis]
        unknown_norm = unknown_arrays / np.linalg.norm(unknown_arrays, axis=1)[:, np.newaxis]
        
        # Calculate all similarities at once
        similarities = np.dot(unknown_norm, known_norm.T)
        
        results = []
        for i, sim_row in enumerate(similarities):
            max_idx = np.argmax(sim_row)
            max_sim = sim_row[max_idx]
            
            if max_sim >= (1 - self.tolerance):
                matched_encoding = known_encodings[max_idx]
                
                result = RecognitionResult(
                    location=None,
                    name=matched_encoding.name,
                    confidence=float(max_sim),
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
        Compute similarity score using Cosine similarity.
        """
        similarity = 1 - cosine(encoding1, encoding2)
        return float(similarity)
    
    @property
    def name(self) -> str:
        return self._name
