import threading
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, List
import face_recognition

from config.models_config import (
    CLUSTERING_EPS,
    CLUSTERING_MIN_SAMPLES,
    CLUSTERING_METRIC
)
from utils.cache_manager import CacheManager
from core.face.encoders.cluster_encoder import ClusterFaceEncoder

class ClusteringService:
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.encoder = ClusterFaceEncoder()
        self.cache_manager = CacheManager()
        self.clusters: Dict[str, List[np.ndarray]] = {}
        
        # Threading
        self.process_thread = threading.Thread(target=self._cluster_loop, daemon=True)
        
        # State management
        self.is_running = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        """Start the clustering service."""
        with self._lock:
            if self.is_running.is_set():
                return
            
            self.is_running.set()
            self.process_thread.start()
            print("Clustering service started")

    def stop(self):
        """Stop the clustering service."""
        with self._lock:
            self.is_running.clear()
            print("Clustering service stopped")

    def _cluster_faces(self, encodings: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
        """Cluster face encodings and assign group names."""
        if not encodings:
            return {}
            
        # Perform clustering
        clustering = DBSCAN(
            eps=CLUSTERING_EPS,
            min_samples=CLUSTERING_MIN_SAMPLES,
            metric=CLUSTERING_METRIC
        ).fit(encodings)
        
        # Group encodings by cluster
        clusters = {}
        for label, encoding in zip(clustering.labels_, encodings):
            if label == -1:
                group_name = f"Group_Single_{len(clusters)}"
            else:
                group_name = f"Group_{label}"
            
            if group_name not in clusters:
                clusters[group_name] = []
            clusters[group_name].append(encoding)
        
        return clusters

    def update_clusters(self, encodings: List[np.ndarray]):
        """Update face clusters."""
        with self._lock:
            self.clusters = self._cluster_faces(encodings)
            self.cache_manager.set('face_groups', self.clusters)

    def get_group_name(self, encoding: np.ndarray) -> str:
        """Get the group name for a face encoding."""
        with self._lock:
            if not self.clusters:
                return "Unknown"
            
            # Find the closest matching group
            min_distance = float('inf')
            best_group = "Unknown"
            
            for group_name, group_encodings in self.clusters.items():
                for group_encoding in group_encodings:
                    distance = np.linalg.norm(encoding - group_encoding)
                    if distance < min_distance and distance < 0.6:
                        min_distance = distance
                        best_group = group_name
            
            return best_group

    def _cluster_loop(self):
        """Main clustering loop."""
        while self.is_running.is_set() and not self.stop_event.is_set():
            try:
                # Load cached clusters
                cached_clusters = self.cache_manager.get('face_groups')
                if cached_clusters:
                    with self._lock:
                        self.clusters = cached_clusters
                
                # Sleep for a while before next check
                self.stop_event.wait(60.0)  # Check every minute
                
            except Exception as e:
                print(f"Error in clustering loop: {e}")
                continue
