from pathlib import Path

import kaggle
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _authenticate_kaggle_token():
    kaggle.api.authenticate()


def load_config() -> dict:
    with CONFIG_PATH.open("r") as file:
        return yaml.safe_load(file)


def download_dataset():
    _authenticate_kaggle_token()
    config = load_config()
    raw_path = PROJECT_ROOT / config["data"]["raw_dir"]
    kaggle_slug = config["data"]["kaggle_dataset"]
    try:
        kaggle.api.dataset_download_files(kaggle_slug, path=raw_path, unzip=True)
    except Exception as e:
        raise RuntimeError(
            f"Kaggle download failed for '{kaggle_slug}'." f"Original Error: {e}"
        ) from e
    files = raw_path.glob("*.csv")
    if not files:
        raise RuntimeError("No csvs found")


if __name__ == "__main__":
    download_dataset()
