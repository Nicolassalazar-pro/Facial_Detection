import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

from .base_encoder import BaseFaceEncoder
from models.face_model import FaceEncoding, FaceLocation
from ..detectors.cluster_detector import ClusterFaceDetector
from config.models_config import MAX_CONCURRENT_PROCESSES
from utils.image_processor import smart_crop_and_resize

class ClusterFaceEncoder(BaseFaceEncoder):
    """Face encoder optimized for clustering operations."""
    
    def __init__(self, detector: Optional[ClusterFaceDetector] = None):
        super().__init__(detector or ClusterFaceDetector())
        
    def encode_face(self,
                   image: np.ndarray,
                   face_location: Optional[FaceLocation] = None) -> Optional[np.ndarray]:
        """Encode a single face with high accuracy for clustering."""
        # Get face location if not provided
        if not face_location:
            locations = self.detector.detect(image)
            if not locations:
                return None
            face_location = locations[0]
        
        # Crop and standardize face region
        top, right, bottom, left = face_location.to_tuple()
        face_image = image[top:bottom, left:right]
        face_image = smart_crop_and_resize(face_image)
        
        # Get encoding with high accuracy settings
        encodings = self.detector.get_encodings(face_image)
        return encodings[0] if encodings else None
    
    def encode_faces(self,
                    image: np.ndarray,
                    face_locations: Optional[List[FaceLocation]] = None) -> List[np.ndarray]:
        """Encode all faces with high accuracy for clustering."""
        # Get face locations if not provided
        if not face_locations:
            face_locations = self.detector.detect(image)
        
        encodings = []
        for location in face_locations:
            encoding = self.encode_face(image, location)
            if encoding is not None:
                encodings.append(encoding)
        
        return encodings
    
    def encode_image_file(self,
                         image_path: Path,
                         face_location: Optional[FaceLocation] = None) -> Optional[FaceEncoding]:
        """Encode face from image file with high accuracy."""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Get encoding
            encoding = self.encode_face(image, face_location)
            if encoding is None:
                return None
            
            return FaceEncoding(
                encoding=encoding,
                name=image_path.stem
            )
        except Exception:
            return None
    
    def _process_batch(self,
                      batch_paths: List[Path]) -> List[Tuple[Path, Optional[FaceEncoding]]]:
        """Process a batch of images for parallel encoding."""
        results = []
        for path in batch_paths:
            encoding = self.encode_image_file(path)
            results.append((path, encoding))
        return results
    
    def batch_encode_files(self,
                          image_paths: List[Path],
                          batch_size: int = 32) -> List[Tuple[Path, Optional[FaceEncoding]]]:
        """Batch encode multiple files with parallel processing."""
        results = []
        
        # Split into batches
        batches = [image_paths[i:i + batch_size] 
                  for i in range(0, len(image_paths), batch_size)]
        
        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROCESSES) as executor:
            futures = [executor.submit(self._process_batch, batch) 
                      for batch in batches]
            
            # Show progress
            for future in tqdm(concurrent.futures.as_completed(futures),
                             total=len(futures),
                             desc="Encoding faces"):
                batch_results = future.result()
                results.extend(batch_results)
        
        return results
    
    def compute_encoding_similarity(self,
                                  encoding1: np.ndarray,
                                  encoding2: np.ndarray) -> float:
        """
        Compute similarity score between two encodings.
        Returns value between 0 and 1 where 1 is most similar.
        """
        distance = self.compute_distance(encoding1, encoding2)
        return 1 / (1 + distance)
