import logging
import sys

__version__ = "0.0.1"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    force=True,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(funcName)s: %(message)s",
)

# Suppress specific loggers to reduce noise
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("s3transfer").setLevel(logging.WARNING)
