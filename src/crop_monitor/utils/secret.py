# src/crop_monitor/utils/secrets.py
import os
from typing import Optional

def get_secret(secret_name: str, default: Optional[str] = None) -> str:
    """
    Get a secret from environment variables.
    In production, this could be extended to use cloud provider secret managers.
    """
    value = os.getenv(secret_name)
    if value is None and default is None:
        raise ValueError(f"Secret {secret_name} not found and no default provided")
    return value or default