"""Configuration loader for FlightMoE v2."""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "flightmoe_v2.yaml"


def load_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load YAML config and return nested dict.

    Args:
        config_path: path to YAML config file.

    Returns:
        Nested dict of configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return cfg


def get_data_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg["data"]


def get_model_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg["model"]


def get_training_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg["training"]


def get_perturbation_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg["perturbation"]


def get_logging_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg["logging"]


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded successfully.")
    print(f"Data paths: {get_data_cfg(cfg)}")
    print(f"Model encoder dims: {get_model_cfg(cfg)['encoder']['temporal_embed_dim']}")
