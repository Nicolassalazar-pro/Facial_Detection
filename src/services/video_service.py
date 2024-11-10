import cv2
import threading
from queue import Queue, Empty
import numpy as np
from typing import Optional

from config.general_config import (
    CAMERA_WIDTH, 
    CAMERA_HEIGHT, 
    CAMERA_INDEX,
    WINDOW_NAME
)

class VideoService:
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.capture: Optional[cv2.VideoCapture] = None
        
        # Increased buffer sizes
        self.frame_buffer = Queue(maxsize=10)  # Increased from 2
        self.overlay_buffer = Queue(maxsize=5)  # Increased from 1
        
        # Threading
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        
        # State management
        self.is_running = threading.Event()
        self._lock = threading.Lock()
        
        # FPS tracking
        self.fps_time = cv2.getTickCount()
        self.fps = 0
        
        # Initialize video capture
        self.capture = cv2.VideoCapture(CAMERA_INDEX)
        if not self.capture.isOpened():
            raise RuntimeError("Failed to open camera")
            
        # Set camera properties
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.capture.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS if possible
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize camera buffering

    def _capture_loop(self):
        """Continuous capture loop running in its own thread."""
        while self.is_running.is_set() and not self.stop_event.is_set():
            if self.capture is None or not self.capture.isOpened():
                break
                
            ret, frame = self.capture.read()
            if not ret:
                continue
            
            # Keep only latest frame if buffer is full
            if self.frame_buffer.full():
                try:
                    while self.frame_buffer.qsize() > 1:  # Keep clearing until only one old frame remains
                        self.frame_buffer.get_nowait()
                except Empty:
                    pass
                    
            try:
                self.frame_buffer.put_nowait(frame)
            except:
                continue

    def _display_loop(self):
        """Display loop running in its own thread."""
        cv2.namedWindow(WINDOW_NAME)
        
        while self.is_running.is_set() and not self.stop_event.is_set():
            try:
                # Get latest frame
                frame = self.frame_buffer.get(timeout=0.1)
                
                # Update FPS counter
                current_time = cv2.getTickCount()
                self.fps = cv2.getTickFrequency() / (current_time - self.fps_time)
                self.fps_time = current_time
                
                # Apply latest overlay if available
                try:
                    # Clear old overlays if multiple are queued
                    while self.overlay_buffer.qsize() > 1:
                        self.overlay_buffer.get_nowait()
                    
                    if not self.overlay_buffer.empty():
                        overlay = self.overlay_buffer.get_nowait()
                        if overlay is not None:
                            frame = cv2.addWeighted(frame, 1, overlay, 0.5, 0)
                except Empty:
                    pass
                
                # Draw FPS counter
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow(WINDOW_NAME, frame)
                
                # Check for quit command with shorter wait time
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break
                    
            except Queue.Empty:
                continue
            except Exception as e:
                print(f"Error in display loop: {e}")
                continue
            
            # Check if window was closed
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self.stop_event.set()
                break

    @property
    def is_active(self) -> bool:
        """Check if the video service is currently active."""
        return self.is_running.is_set() and self.capture is not None and self.capture.isOpened()
