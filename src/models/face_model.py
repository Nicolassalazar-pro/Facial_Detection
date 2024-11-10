from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from datetime import datetime

@dataclass
class FaceEncoding:
    """Represents a face encoding with its metadata."""
    encoding: np.ndarray
    name: str
    timestamp: datetime = datetime.now()

@dataclass
class FaceLocation:
    """Represents the location of a face in an image."""
    top: int
    right: int
    bottom: int
    left: int
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.top, self.right, self.bottom, self.left)
    
    @classmethod
    def from_tuple(cls, location: Tuple[int, int, int, int]) -> 'FaceLocation':
        return cls(location[0], location[1], location[2], location[3])

@dataclass
class RecognitionResult:
    """Represents the result of face recognition on a single face."""
    location: FaceLocation
    name: str
    confidence: float
    encoding: Optional[np.ndarray] = None
    timestamp: datetime = datetime.now()

@dataclass
class ClusterGroup:
    """Represents a group of similar faces."""
    cluster_id: int
    face_files: List[str]
    representative_encoding: Optional[np.ndarray] = None
    
    def __len__(self) -> int:
        return len(self.face_files)

class FaceDatabase:
    """Manages the collection of known face encodings."""
    def __init__(self):
        self._encodings: Dict[str, FaceEncoding] = {}
        self._clusters: List[ClusterGroup] = []
        self.last_updated: datetime = datetime.now()

    def add_face(self, name: str, encoding: np.ndarray) -> None:
        """Add a new face encoding to the database."""
        self._encodings[name] = FaceEncoding(encoding=encoding, name=name)
        self.last_updated = datetime.now()

    def get_face(self, name: str) -> Optional[FaceEncoding]:
        """Retrieve a face encoding by name."""
        return self._encodings.get(name)

    def remove_face(self, name: str) -> None:
        """Remove a face encoding from the database."""
        self._encodings.pop(name, None)
        self.last_updated = datetime.now()

    def update_clusters(self, clusters: List[ClusterGroup]) -> None:
        """Update the face clusters."""
        self._clusters = clusters
        self.last_updated = datetime.now()

    def get_all_encodings(self) -> List[Tuple[str, np.ndarray]]:
        """Get all face encodings with their names."""
        return [(name, face.encoding) for name, face in self._encodings.items()]

    def get_clusters(self) -> List[ClusterGroup]:
        """Get all face clusters."""
        return self._clusters.copy()

    def clear(self) -> None:
        """Clear all data from the database."""
        self._encodings.clear()
        self._clusters.clear()
        self.last_updated = datetime.now()

    @property
    def size(self) -> int:
        """Get the number of faces in the database."""
        return len(self._encodings)
