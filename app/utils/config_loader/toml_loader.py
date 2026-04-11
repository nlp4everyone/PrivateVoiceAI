from typing import Dict, Any
import os, toml

class TomlConfigLoader:
    """Handles loading and accessing TOML configuration files."""

    def __init__(self, config_path: str = None):
        """
        Initialize the TOML config loader.

        Args:
            config_path: Path to the TOML config file. If None, uses default path.
        """
        if config_path is None:
            # Default path: project root/config/config.toml
            # Navigate from app/utils/config_loader/ to project root
            current_dir = os.path.dirname(__file__)
            project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
            config_path = os.path.join(project_root, "config", "config.toml")

        self.config_path = config_path
        self._config = None
        self._load_config()

    def _load_config(self) -> None:
        """Load the TOML configuration file."""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = toml.load(file)
        except toml.TomlDecodeError as e:
            raise ValueError(f"Invalid TOML syntax in {self.config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.config_path}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        if self._config is None:
            return default

        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Section name

        Returns:
            Dictionary containing the section configuration
        """
        return self.get(section, {})

    def reload(self) -> None:
        """Reload the configuration file."""
        self._load_config()

    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self._config.copy() if self._config is not None else {}

    def exists(self, key: str) -> bool:
        """
        Check if a configuration key exists.

        Args:
            key: Configuration key (supports dot notation for nested keys)

        Returns:
            True if key exists, False otherwise
        """
        return self.get(key) is not None

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator usage."""
        return self.exists(key)


# Global instance for easy access
_toml_loader = TomlConfigLoader()


def get_toml_config() -> TomlConfigLoader:
    """Get the global TOML config loader instance."""
    return _toml_loader


def reload_toml_config() -> None:
    """Reload the global TOML configuration."""
    _toml_loader.reload()