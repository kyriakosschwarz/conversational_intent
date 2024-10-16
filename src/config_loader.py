# src/config_loader.py

import toml
from pathlib import Path

class ConfigLoader:
    _config = None

    @classmethod
    def load_config(cls, config_file: str = 'config.toml'):
        """
        Loads the configuration file (TOML format) and caches it.
        """
        if cls._config is None:
            # Get the absolute path to the config file
            config_path = Path(__file__).resolve().parent.parent / config_file
            with open(config_path, 'r') as f:
                cls._config = toml.load(f)
        return cls._config

    @classmethod
    def get(cls, key: str, default=None):
        """
        Access a specific configuration value using dot notation.
        Example: 'paths.data_file'
        """
        keys = key.split('.')
        value = cls.load_config()
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value

