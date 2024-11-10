import cv2
import imagehash
from PIL import Image
import face_recognition
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import os

from config.general_config import SUPPORTED_IMAGE_EXTENSIONS
from config.models_config import (
    TARGET_FACE_SIZE,
    IMAGE_QUALITY_THRESHOLD
)

def is_valid_image(file_path: str) -> bool:
    """Check if file is a valid image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_image_hash(image_path: str) -> str:
    """Generate a hash for the image for duplicate detection."""
    try:
        with Image.open(image_path) as img:
            return str(imagehash.average_hash(img))
    except Exception:
        return ""

def smart_crop_and_resize(image: Image.Image, size: Tuple[int, int] = TARGET_FACE_SIZE) -> Image.Image:
    """Crop and resize image while maintaining aspect ratio."""
    width, height = image.size
    crop_size = min(width, height)
    
    # Calculate crop coordinates
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    # Crop and resize
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize(size, Image.LANCZOS)

def assess_image_quality(image_path: str) -> float:
    """
    Assess image quality based on multiple factors.
    Returns a score from 0 to 100.
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return 0.0
        
        # Convert to grayscale for calculations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()  # Blur detection
        brightness = np.mean(gray)  # Average brightness
        contrast = np.std(gray)  # Contrast
        
        # Normalize and combine scores
        blur_score = min(blur_score / 500.0, 1.0) * 40  # Max 40 points
        brightness_score = (1 - abs(brightness - 127) / 127) * 30  # Max 30 points
        contrast_score = min(contrast / 80.0, 1.0) * 30  # Max 30 points
        
        return blur_score + brightness_score + contrast_score
        
    except Exception:
        return 0.0

def cleanup_profile_images(folder_path: str, remove_duplicates: bool = True) -> List[str]:
    """
    Clean up profile images folder by removing problematic images.
    Returns list of removed files.
    """
    folder_path = Path(folder_path)
    removed_files = []
    image_hashes = {}
    
    # Get all image files
    image_files = [
        f for f in folder_path.iterdir()
        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    
    for img_path in image_files:
        try:
            # Check if image is valid
            if not is_valid_image(str(img_path)):
                img_path.unlink()
                removed_files.append(img_path.name)
                continue
            
            # Check image quality
            quality_score = assess_image_quality(str(img_path))
            if quality_score < IMAGE_QUALITY_THRESHOLD:
                img_path.unlink()
                removed_files.append(img_path.name)
                continue
            
            # Handle duplicates
            if remove_duplicates:
                img_hash = get_image_hash(str(img_path))
                if img_hash in image_hashes:
                    img_path.unlink()
                    removed_files.append(img_path.name)
                    continue
                image_hashes[img_hash] = img_path.name
            
            # Verify face detection
            image = face_recognition.load_image_file(str(img_path))
            if not face_recognition.face_locations(image):
                img_path.unlink()
                removed_files.append(img_path.name)
                continue
                
        except Exception as e:
            # Remove problematic files
            try:
                img_path.unlink()
                removed_files.append(img_path.name)
            except:
                pass
                
    return removed_files
