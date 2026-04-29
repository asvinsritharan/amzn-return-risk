from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from return_risk.data.schema import raw_schema

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def load_raw(validate: bool = True) -> pd.DataFrame:
    """Read the raw CSV. Pass validate=False to skip Pandera validation."""
    cfg = load_config()
    raw_file = PROJECT_ROOT / cfg["data"]["raw_file"]
    if not raw_file.exists():
        raise FileNotFoundError(
            f"{raw_file} not found. Run `uv run python scripts/download_data.py` first."
        )
    df = pd.read_csv(raw_file)
    if validate:
        df = raw_schema.validate(df, lazy=True)
    return df


def drop_leakage_and_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID and leakage columns listed in config['features']['drop_cols']."""
    cfg = load_config()
    drop_cols = cfg["features"]["drop_cols"]
    missing = [c for c in drop_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Drop list references columns not in the dataframe: {missing}")
    return df.drop(columns=drop_cols)
