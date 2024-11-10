import pickle
from pathlib import Path
from typing import Any, Optional, Dict, List
import time
from datetime import datetime, timedelta
import threading

from config.general_config import CACHE_DIR
from .file_manager import FileManager

class CacheManager:
    """Manages caching of face recognition data."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_lock = threading.Lock()
        self._memory_cache: Dict[str, Dict] = {}
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache. Checks memory first, then disk.
        """
        # Check memory cache first
        with self._cache_lock:
            if key in self._memory_cache:
                cache_data = self._memory_cache[key]
                if not self._is_expired(cache_data):
                    return cache_data['value']
                else:
                    del self._memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            try:
                data = FileManager.load_pickle(cache_file)
                if data and not self._is_expired(data):
                    # Update memory cache
                    with self._cache_lock:
                        self._memory_cache[key] = data
                    return data['value']
            except Exception:
                pass
        
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache with optional TTL (in seconds).
        """
        cache_data = {
            'value': value,
            'timestamp': datetime.now(),
            'ttl': ttl
        }
        
        # Update memory cache
        with self._cache_lock:
            self._memory_cache[key] = cache_data
        
        # Update disk cache
        cache_file = self.cache_dir / f"{key}.cache"
        FileManager.save_pickle(cache_data, cache_file)
    
    def delete(self, key: str) -> None:
        """
        Delete item from cache.
        """
        # Remove from memory cache
        with self._cache_lock:
            self._memory_cache.pop(key, None)
        
        # Remove from disk cache
        cache_file = self.cache_dir / f"{key}.cache"
        try:
            cache_file.unlink(missing_ok=True)
        except Exception:
            pass
    
    def clear(self) -> None:
        """
        Clear all cache data.
        """
        # Clear memory cache
        with self._cache_lock:
            self._memory_cache.clear()
        
        # Clear disk cache
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
        except Exception:
            pass
    
    def cleanup_expired(self) -> None:
        """
        Remove expired items from cache.
        """
        # Clean memory cache
        with self._cache_lock:
            expired_keys = [
                key for key, data in self._memory_cache.items()
                if self._is_expired(data)
            ]
            for key in expired_keys:
                del self._memory_cache[key]
        
        # Clean disk cache
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    data = FileManager.load_pickle(cache_file)
                    if data and self._is_expired(data):
                        cache_file.unlink()
                except Exception:
                    # Remove corrupt cache files
                    cache_file.unlink()
        except Exception:
            pass
    
    @staticmethod
    def _is_expired(cache_data: Dict) -> bool:
        """
        Check if cache data is expired.
        """
        if not cache_data.get('ttl'):
            return False
        
        age = datetime.now() - cache_data['timestamp']
        return age.total_seconds() > cache_data['ttl']
