from .base_matcher import BaseFaceMatcher
from .euclidean_matcher import EuclideanFaceMatcher
from .cosine_matcher import CosineFaceMatcher

# Default matcher to use
DEFAULT_MATCHER = EuclideanFaceMatcher

__all__ = [
    'BaseFaceMatcher',
    'EuclideanFaceMatcher',
    'CosineFaceMatcher',
    'DEFAULT_MATCHER'
]
