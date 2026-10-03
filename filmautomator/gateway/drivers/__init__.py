"""Model driver adapters.

Each module here wraps one kind of backend. Drivers are never imported by
agents — only by the gateway.
"""

from .mock import MockDriver
from .openai_compat import OpenAICompatDriver

__all__ = ["MockDriver", "OpenAICompatDriver"]
