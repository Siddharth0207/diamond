import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)  # Create logs directory, not the file

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging with both file and console output
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

# Create a logger instance and force immediate write
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {LOG_FILE_PATH}")

# Force flush to ensure logs are written immediately
for handler in logging.getLogger().handlers:
    if hasattr(handler, 'flush'):
        handler.flush()