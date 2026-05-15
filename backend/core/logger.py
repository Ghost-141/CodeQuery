import logging
import sys
from rich.logging import RichHandler

def setup_logger(name: str) -> logging.Logger:
    """
    Creates and returns a centrally configured logger.
    """
    logger = logging.getLogger(name)
    
    # Prevent adding handlers multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Rich console handler for beautiful terminal output
        console_handler = RichHandler(rich_tracebacks=True, markup=True)
        console_handler.setLevel(logging.INFO)
        # Using a simpler format for rich because it provides its own visual formatting
        console_format = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_format)
        
        # File handler for persistent debugging
        file_handler = logging.FileHandler("/tmp/code_qa_backend.log")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        # Ensure we don't propagate to the root logger and duplicate logs
        logger.propagate = False
        
    return logger
