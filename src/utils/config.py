"""Configuration management for loading YAML config files."""

from pathlib import Path
from typing import Any, Dict
import yaml


class Config:
    """Configuration object with attribute-style access."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary.

        Args:
            config_dict: Dictionary of configuration values
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return hasattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self.to_dict()})"


def load_config(config_path: str | Path) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with attribute-style access

    Example:
        >>> config = load_config("configs/circle_baseline.yaml")
        >>> print(config.path.radius)
        0.3
        >>> print(config.ppo.learning_rate)
        0.0003
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config: Config, config_path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Config object to save
        config_path: Path to save YAML file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
