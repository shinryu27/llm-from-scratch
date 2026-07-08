"""Model configuration.

Stage 3 support module: hyperparameters that define the GPT architecture.
The default preset reproduces the 124M-parameter GPT-2 "small" layout.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257      # GPT-2 BPE vocabulary size
    context_length: int = 1024   # maximum sequence length
    emb_dim: int = 768           # embedding (model) dimension
    n_heads: int = 12            # attention heads per block
    n_layers: int = 12           # number of transformer blocks
    drop_rate: float = 0.1       # dropout probability
    qkv_bias: bool = False       # bias terms in the Q/K/V projections

    def __post_init__(self) -> None:
        if self.emb_dim % self.n_heads != 0:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )


PRESETS = {
    "gpt2-small (124M)": GPTConfig(emb_dim=768, n_layers=12, n_heads=12),
    "gpt2-medium (355M)": GPTConfig(emb_dim=1024, n_layers=24, n_heads=16),
    "gpt2-large (774M)": GPTConfig(emb_dim=1280, n_layers=36, n_heads=20),
    "gpt2-xl (1558M)": GPTConfig(emb_dim=1600, n_layers=48, n_heads=25),
}
