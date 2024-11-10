import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from .base_encoder import BaseFaceEncoder
from models.face_model import FaceEncoding, FaceLocation
from ..detectors.realtime_detector import RealtimeFaceDetector
from config.models_config import FRAME_SCALE_FACTOR
from utils.image_processor import smart_crop_and_resize

class RealtimeFaceEncoder(BaseFaceEncoder):
    """Face encoder optimized for real-time video processing."""
    
    def __init__(self, detector: Optional[RealtimeFaceDetector] = None):
        super().__init__(detector or RealtimeFaceDetector())
    
    def encode_face(self,
                   image: np.ndarray,
                   face_location: Optional[FaceLocation] = None) -> Optional[np.ndarray]:
        """Encode a single face with real-time optimizations."""
        # Scale down image for faster processing
        if FRAME_SCALE_FACTOR != 1.0:
            height, width = image.shape[:2]
            new_size = (int(width * FRAME_SCALE_FACTOR), 
                       int(height * FRAME_SCALE_FACTOR))
            image = cv2.resize(image, new_size)
            
            if face_location:
                # Scale face location
                face_location = FaceLocation(
                    top=int(face_location.top * FRAME_SCALE_FACTOR),
                    right=int(face_location.right * FRAME_SCALE_FACTOR),
                    bottom=int(face_location.bottom * FRAME_SCALE_FACTOR),
                    left=int(face_location.left * FRAME_SCALE_FACTOR)
                )
        
        # Get face location if not provided
        if not face_location:
            locations = self.detector.detect(image)
            if not locations:
                return None
            face_location = locations[0]
        
        # Get encoding
        encodings = self.detector.get_encodings(image, [face_location])
        return encodings[0] if encodings else None
    
    def encode_faces(self,
                    image: np.ndarray,
                    face_locations: Optional[List[FaceLocation]] = None) -> List[np.ndarray]:
        """Encode all faces in frame with real-time optimizations."""
        # Scale down image
        if FRAME_SCALE_FACTOR != 1.0:
            height, width = image.shape[:2]
            new_size = (int(width * FRAME_SCALE_FACTOR), 
                       int(height * FRAME_SCALE_FACTOR))
            image = cv2.resize(image, new_size)
            
            if face_locations:
                # Scale face locations
                face_locations = [
                    FaceLocation(
                        top=int(loc.top * FRAME_SCALE_FACTOR),
                        right=int(loc.right * FRAME_SCALE_FACTOR),
                        bottom=int(loc.bottom * FRAME_SCALE_FACTOR),
                        left=int(loc.left * FRAME_SCALE_FACTOR)
                    )
                    for loc in face_locations
                ]
        
        # Get face locations if not provided
        if not face_locations:
            face_locations = self.detector.detect(image)
        
        # Get encodings
        return self.detector.get_encodings(image, face_locations)
    
    def encode_image_file(self,
                         image_path: Path,
                         face_location: Optional[FaceLocation] = None) -> Optional[FaceEncoding]:
        """Encode face from image file with optimizations."""
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
    
    def batch_encode_files(self,
                          image_paths: List[Path],
                          batch_size: int = 32) -> List[Tuple[Path, Optional[FaceEncoding]]]:
        """Batch encode multiple files with real-time optimizations."""
        results = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_results = []
            
            for path in batch_paths:
                encoding = self.encode_image_file(path)
                batch_results.append((path, encoding))
            
            results.extend(batch_results)
        
        return results
