"""
Custom logger for AURA
Provides consistent logging across all modules
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config.settings import settings

def setup_logger(name: str = "aura") -> logging.Logger:
    """
    Set up logger with file and console handlers
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Get log level from config
    log_level = settings.get('logging.level', 'INFO')
    logger.setLevel(getattr(logging, log_level))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    log_format = settings.get(
        'logging.format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = Path(settings.get('logging.file', 'logs/aura.log'))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=settings.get('logging.max_bytes', 10485760),  # 10MB
        backupCount=settings.get('logging.backup_count', 5)
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logger()
