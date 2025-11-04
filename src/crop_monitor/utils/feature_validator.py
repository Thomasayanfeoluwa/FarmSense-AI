import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def validate_features(df: pd.DataFrame, json_path: str) -> pd.DataFrame:
    """
    Validate DataFrame columns against model expected features.
    Reorders, checks missing, and warns for unexpected features.

    Args:
        df (pd.DataFrame): Input features DataFrame
        json_path (str): Path to soil_features.json

    Returns:
        pd.DataFrame: Cleaned DataFrame aligned with model expectations
    """
    # Load expected features
    with open(json_path, "r") as f:
        config = json.load(f)
    expected_features = config["model_expected_features"]

    # Find missing features
    missing = [f for f in expected_features if f not in df.columns]
    extra = [f for f in df.columns if f not in expected_features]

    if missing:
        logger.error(f"❌ Missing features: {missing}")
        raise ValueError(f"Missing required features: {missing}")

    if extra:
        logger.warning(f"⚠️ Extra features found (will be ignored): {extra}")

    # Keep only expected features and enforce correct order
    df_aligned = df[expected_features]

    logger.info("✅ Features validated and aligned successfully.")
    return df_aligned
