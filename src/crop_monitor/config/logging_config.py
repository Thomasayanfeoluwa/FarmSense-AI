# src/crop_monitor/config/logging_config.py
import logging

def setup_logging(level=logging.INFO, logger_name="ai_crop_monitor"):
    """
    Configure application-wide logging.
    Can be reused across all modules and services.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    return logging.getLogger(logger_name)
