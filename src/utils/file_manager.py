import os
import shutil
from pathlib import Path
from typing import List, Set, Optional
import pickle
from datetime import datetime

from config.general_config import (
    CACHE_DIR,
    PROFILE_DIR,
    SUPPORTED_IMAGE_EXTENSIONS
)

class FileManager:
    """Manages file operations for the face recognition system."""
    
    @staticmethod
    def ensure_directories() -> None:
        """Ensure all required directories exist."""
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_image_files(directory: Path) -> List[Path]:
        """Get all valid image files from directory."""
        return [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

    @staticmethod
    def save_pickle(data: any, filepath: Path) -> None:
        """Save data to pickle file."""
        temp_file = filepath.with_suffix('.tmp')
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f)
            # Atomic replace
            temp_file.replace(filepath)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    @staticmethod
    def load_pickle(filepath: Path, default: any = None) -> any:
        """Load data from pickle file."""
        try:
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
        return default

    @staticmethod
    def save_text_file(data: List[str], filepath: Path) -> None:
        """Save list of strings to text file."""
        temp_file = filepath.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                f.write('\n'.join(data))
            # Atomic replace
            temp_file.replace(filepath)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    @staticmethod
    def backup_directory(src_dir: Path, backup_name: Optional[str] = None) -> Path:
        """Create backup of directory."""
        if backup_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f'backup_{timestamp}'
        
        backup_dir = src_dir.parent / backup_name
        shutil.copytree(src_dir, backup_dir)
        return backup_dir

    @staticmethod
    def cleanup_old_backups(backup_dir: Path, max_backups: int = 5) -> None:
        """Remove old backups keeping only the most recent ones."""
        backups = sorted(
            [d for d in backup_dir.iterdir() if d.is_dir() and d.name.startswith('backup_')],
            key=lambda x: x.stat().st_mtime
        )
        
        # Remove oldest backups if we have too many
        while len(backups) > max_backups:
            oldest = backups.pop(0)
            shutil.rmtree(oldest)

    @staticmethod
    def is_file_in_use(filepath: Path) -> bool:
        """Check if file is currently in use."""
        try:
            with open(filepath, 'r+b'):
                return False
        except IOError:
            return True
