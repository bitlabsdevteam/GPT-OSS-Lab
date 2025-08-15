# Components package for GPT-OSS
# This package contains modular components for the GPT-OSS architecture

__version__ = "1.0.0"
__author__ = "GPT-OSS Team"

# Import all components for easy access
from .normalization import LayerNorm, RMSNorm
from .attention import MultiHeadAttention, RotaryPositionalEmbedding
from .feedforward import FeedForward, SwiGLU
from .moe import MixtureOfExperts, Router, Expert

__all__ = [
    'LayerNorm',
    'RMSNorm', 
    'MultiHeadAttention',
    'RotaryPositionalEmbedding',
    'FeedForward',
    'SwiGLU',
    'MixtureOfExperts',
    'Router',
    'Expert'
]