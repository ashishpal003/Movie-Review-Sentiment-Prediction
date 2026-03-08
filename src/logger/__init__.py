import logging
import os
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from logging import StreamHandler
from datetime import datetime
import sys

# Constants for log configuration
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5*1024*1024 # 5 MB
BACKUP_COUNT = 3 # Numbe of backup log files to keep

# Construct log file path
root_dir = Path(__file__).parent.parent.parent
log_dir_path = root_dir / LOG_DIR
log_dir_path.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir_path / LOG_FILE

def configure_logger():
    """
    Configure logging with a rotating file handler and a console handler.
    """
    # Create a custom logger
    logger = logging.getLogger()
    if logger.hasHandlers():
        return # Prevent duplicate logs

    logger.setLevel(logging.DEBUG)

    # Define formatter
    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
    
    # File handler with rotation
    file_handler = RotatingFileHandler(filename=log_file_path,
                                       maxBytes=MAX_LOG_SIZE,
                                       backupCount=BACKUP_COUNT,
                                       encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # console stream handler
    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)


    # Add handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Configure the logger
configure_logger()

if __name__ == "__main__":
    logging.info("Logger is live and kicking!")