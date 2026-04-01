"""
Agent registry — adding a new agent = one line here.
"""

from .base_agent   import BaseAgent
from .standard_dqn import StandardDQNAgent
from .complex_dqn  import ComplexDQNAgent

AGENT_REGISTRY = {
    "standard": StandardDQNAgent,
    "complex":  ComplexDQNAgent,
}

__all__ = ["AGENT_REGISTRY", "BaseAgent", "StandardDQNAgent", "ComplexDQNAgent"]
