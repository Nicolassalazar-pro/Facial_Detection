import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from typing import Optional

from config.general_config import PROFILE_DIR, SUPPORTED_IMAGE_EXTENSIONS
from services.recognition_service import RecognitionService
from services.clustering_service import ClusteringService
from core.face.encoders.realtime_encoder import RealtimeFaceEncoder
from utils.image_processor import cleanup_profile_images

class ProfileChangeHandler(FileSystemEventHandler):
    def __init__(self, recognition_service: RecognitionService, clustering_service: ClusteringService):
        self.recognition_service = recognition_service
        self.clustering_service = clustering_service
        self.processing_lock = threading.Lock()
        self._processing_thread: Optional[threading.Thread] = None
        self.encoder = RealtimeFaceEncoder()

    def on_any_event(self, event):
        """Handle any change in the profile directory."""
        if event.is_directory:
            return
            
        # Check if file is an image
        if not any(event.src_path.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS):
            return
            
        # Debounce multiple rapid changes
        with self.processing_lock:
            if self._processing_thread is None or not self._processing_thread.is_alive():
                self._processing_thread = threading.Thread(target=self._process_changes)
                self._processing_thread.start()

    def _process_changes(self):
        """Process changes in profile directory."""
        try:
            # Run cleanup
            cleanup_profile_images(PROFILE_DIR)
            
            # Get all image files in profile directory
            image_paths = [
                Path(f) for f in Path(PROFILE_DIR).iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]
            
            # Encode all faces
            encodings = []
            names = []
            for image_path in image_paths:
                face_encoding = self.encoder.encode_image_file(image_path)
                if face_encoding is not None:
                    encodings.append(face_encoding.encoding)
                    names.append(face_encoding.name)
            
            # Update recognition service
            if encodings and names:
                self.recognition_service.update_known_faces(encodings, names)
                # Also update clustering
                self.clustering_service.update_clusters(encodings)
            
        except Exception as e:
            print(f"Error processing profile changes: {e}")

class ProfileWatcherService:
    def __init__(self, 
                 stop_event: threading.Event,
                 recognition_service: RecognitionService,
                 clustering_service: ClusteringService):
        self.stop_event = stop_event
        self.observer = Observer()
        self.event_handler = ProfileChangeHandler(recognition_service, clustering_service)
        
        # State management
        self.is_running = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        """Start watching the profile directory."""
        with self._lock:
            if self.is_running.is_set():
                return
            
            self.observer.schedule(self.event_handler, str(PROFILE_DIR), recursive=False)
            self.observer.start()
            self.is_running.set()
            print("Profile watcher service started")

    def stop(self):
        """Stop watching the profile directory."""
        with self._lock:
            if self.is_running.is_set():
                self.observer.stop()
                self.observer.join()
                self.is_running.clear()
                print("Profile watcher service stopped")
