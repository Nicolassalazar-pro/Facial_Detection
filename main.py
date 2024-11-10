import threading
from pathlib import Path
from queue import Queue
import cv2
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from services.video_service import VideoService
from services.recognition_service import RecognitionService
from services.clustering_service import ClusteringService
from services.watcher_service import ProfileWatcherService
from utils.file_manager import FileManager
from config.general_config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    WINDOW_NAME
)

def main():
    # Initialize stop event for graceful shutdown
    stop_event = threading.Event()
    
    try:
        # Ensure required directories exist
        FileManager.ensure_directories()
        
        # Initialize services
        video_service = VideoService(stop_event)
        recognition_service = RecognitionService(
            frame_buffer=video_service.frame_buffer,
            overlay_buffer=video_service.overlay_buffer,
            stop_event=stop_event
        )
        clustering_service = ClusteringService(stop_event)
        watcher_service = ProfileWatcherService(
            stop_event=stop_event,
            recognition_service=recognition_service,
            clustering_service=clustering_service
        )
        
        # Start all services
        clustering_service.start()
        recognition_service.start()
        watcher_service.start()
        
        # Start video processing
        video_service.capture_thread.start()
        video_service.display_thread.start()
        
        print(f"System initialized. Press 'q' to quit.")
        
        # Wait for quit signal
        while not stop_event.is_set():
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
                
    except Exception as e:
        print(f"Error during execution: {e}")
        
    finally:
        # Cleanup
        print("\nShutting down...")
        stop_event.set()
        
        # Stop all services
        if 'video_service' in locals():
            video_service.stop()
        if 'recognition_service' in locals():
            recognition_service.stop()
        if 'clustering_service' in locals():
            clustering_service.stop()
        if 'watcher_service' in locals():
            watcher_service.stop()
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
