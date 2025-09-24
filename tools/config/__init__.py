"""Configuration package exposing shared configuration helpers."""

from .config import Config, get_config

# Expose a module-level reference for convenience
config = get_config()

__all__ = ["Config", "config", "get_config"]
