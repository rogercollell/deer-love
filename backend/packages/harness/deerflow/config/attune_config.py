# backend/packages/harness/deerflow/config/attune_config.py
"""Configuration for attune upstream wisdom framing."""

from pydantic import BaseModel, Field

from deerflow.attune.models import Domain


class AttuneConfig(BaseModel):
    """Configuration for attune upstream wisdom framing."""

    enabled: bool = Field(default=False, description="Whether to enable attune upstream wisdom framing")
    model_name: str | None = Field(default=None, description="Model name for framing (None = use default model)")
    domain: Domain = Field(default=Domain.general, description="Framing domain")


# Global configuration instance
_attune_config: AttuneConfig = AttuneConfig()


def get_attune_config() -> AttuneConfig:
    """Get the current attune configuration."""
    return _attune_config


def set_attune_config(config: AttuneConfig) -> None:
    """Set the attune configuration."""
    global _attune_config
    _attune_config = config


def load_attune_config_from_dict(config_dict: dict) -> None:
    """Load attune configuration from a dictionary."""
    global _attune_config
    _attune_config = AttuneConfig(**config_dict)
