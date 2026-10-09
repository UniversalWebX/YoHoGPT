"""YoHoGPT: a coding-only LLM and autonomous software-engineering agent."""

__version__ = "0.1.0"

from .config import ModelConfig, TrainConfig
from .model import YohoGPT

__all__ = ["ModelConfig", "TrainConfig", "YohoGPT", "__version__"]
