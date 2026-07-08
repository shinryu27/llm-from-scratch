"""GPT-style LLM implemented from scratch in PyTorch.

Follows the methodology of Sebastian Raschka's
"Build a Large Language Model (From Scratch)"
(https://github.com/rasbt/LLMs-from-scratch).

Imports are lazy so torch-free utilities (e.g. the BPE tokenizer)
work without PyTorch installed.
"""

__all__ = ["GPTConfig", "PRESETS", "GPTModel"]
__version__ = "1.0.0"


def __getattr__(name):
    if name in ("GPTConfig", "PRESETS"):
        from llm import config

        return getattr(config, name)
    if name == "GPTModel":
        from llm.model import GPTModel

        return GPTModel
    raise AttributeError(f"module 'llm' has no attribute {name!r}")
