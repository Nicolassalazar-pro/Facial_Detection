class RecognitionService:
    def _process_loop(self):
        """Main processing loop for face recognition."""
        while self.is_running.is_set() and not self.stop_event.is_set():
            try:
                # Skip frames if we're falling behind
                frames_to_skip = self.frame_buffer.qsize() - 1
                for _ in range(frames_to_skip):
                    try:
                        self.frame_buffer.get_nowait()
                    except Empty:
                        break
                
                # Get latest frame
                frame = self.frame_buffer.get(timeout=0.1)
                
                # Process frame
                overlay = self._process_frame(frame)
                
                # Update overlay, dropping old ones if we're behind
                if self.overlay_buffer.full():
                    try:
                        while self.overlay_buffer.qsize() > 1:
                            self.overlay_buffer.get_nowait()
                    except Empty:
                        pass
                        
                self.overlay_buffer.put_nowait(overlay)
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in recognition loop: {e}")
                continue
