"""
Logging configuration for the RAG application.
"""
import logging
import sys


def setup_logging(debug: bool = False) -> None:
    """
    Configure application-wide structured logging.

    Args:
        debug: If True, set the root log level to DEBUG, otherwise INFO.
    """
    log_level = logging.DEBUG if debug else logging.INFO

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    # Silence noisy third-party loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

