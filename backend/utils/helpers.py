"""
Utility helper functions
"""
import os
import time
import logging
from pathlib import Path
from typing import List

from config import Config

logger = logging.getLogger(__name__)

def cleanup_old_files():
    """Clean up old uploaded files and plots"""
    try:
        current_time = time.time()
        max_age = Config.FILE_MAX_AGE
        
        # Clean uploads
        if Config.UPLOAD_FOLDER.exists():
            for file_path in Config.UPLOAD_FOLDER.iterdir():
                if file_path.is_file():
                    if current_time - file_path.stat().st_ctime > max_age:
                        try:
                            file_path.unlink()
                            logger.info(f"🧹 Cleaned old upload: {file_path.name}")
                        except Exception as e:
                            logger.error(f"Cleanup error for {file_path}: {str(e)}")
        
        # Clean plots
        if Config.PLOTS_FOLDER.exists():
            for file_path in Config.PLOTS_FOLDER.iterdir():
                if file_path.is_file():
                    if current_time - file_path.stat().st_ctime > max_age:
                        try:
                            file_path.unlink()
                            logger.info(f"🧹 Cleaned old plot: {file_path.name}")
                        except Exception as e:
                            logger.error(f"Cleanup error for {file_path}: {str(e)}")
                            
        logger.info("✓ Cleanup completed")
                            
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def is_text_file(filepath: Path) -> bool:
    """Check if file is a text file"""
    text_extensions = {'.txt', '.md', '.csv', '.json', '.xml'}
    return filepath.suffix.lower() in text_extensions

def get_file_extension(filename: str) -> str:
    """Get file extension safely"""
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower()
    return ''

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    # Remove any characters that aren't alphanumeric, dash, underscore, or dot
    filename = re.sub(r'[^\w\s.-]', '', filename)
    # Replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename)
    return filename

def create_directory_structure(base_path: Path, subdirs: List[str]):
    """Create directory structure"""
    for subdir in subdirs:
        dir_path = base_path / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {dir_path}")

def get_directory_size(directory: Path) -> int:
    """Get total size of directory in bytes"""
    total = 0
    try:
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total += file_path.stat().st_size
    except Exception as e:
        logger.error(f"Error calculating directory size: {str(e)}")
    return total

def count_files_by_type(directory: Path) -> dict:
    """Count files by extension"""
    counts = {}
    try:
        for file_path in directory.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                counts[ext] = counts.get(ext, 0) + 1
    except Exception as e:
        logger.error(f"Error counting files: {str(e)}")
    return counts