"""
A2A World Platform - XAI (Explainable AI) Agents Module

This module contains agents responsible for generating explainable AI explanations,
narrative-driven interpretations, and multimodal content for pattern discovery
and validation results.
"""

from .narrative_xai_agent import NarrativeXAIAgent, NarrativeXAIConfig

__all__ = [
    'NarrativeXAIAgent',
    'NarrativeXAIConfig'
]

# Module metadata
__version__ = "1.0.0"
__description__ = "Explainable AI agents for narrative-driven explanations"
__author__ = "A2A World Platform"