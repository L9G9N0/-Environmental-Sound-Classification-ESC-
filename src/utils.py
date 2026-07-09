import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Generator

def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """Sets up unified logging to console and file with standardized formatters."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "data_pipeline.log")
    
    logger = logging.getLogger("ESC_Pipeline")
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers if logger is re-initialized
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # File Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    
    logger.info("Logging initialized. Output redirecting to console and: %s", log_file)
    return logger

@contextmanager
def time_execution(name: str) -> Generator[None, None, None]:
    """Context manager to measure and log execution duration of a code block."""
    logger = logging.getLogger("ESC_Pipeline")
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info("Execution of [%s] completed in %.4f seconds.", name, duration)
