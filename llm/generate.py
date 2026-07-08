"""Stage 4b — Text generation.

Autoregressive decoding: repeatedly feed the (cropped) context through the
model, turn the last position's logits into a next-token choice, append it,
and repeat. Supports greedy decoding, temperature scaling, and top-k
filtering, with <|endoftext|> as an optional stop token.
"""

from __future__ import annotations

import argparse

import torch

from llm.config import GPTConfig
from llm.model import GPTModel


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    try:  # tiktoken needs explicit permission for the special token
        ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    except TypeError:
        ids = tokenizer.encode(text)
    return torch.tensor(ids).unsqueeze(0)  # add batch dimension


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(token_ids.squeeze(0).tolist())


@torch.no_grad()
def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Generate ``max_new_tokens`` continuation tokens for each row of ``idx``."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]          # crop to context window
        logits = model(idx_cond)[:, -1, :]         # last time step only

        if top_k is not None:                      # top-k filtering
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.full_like(logits, float("-inf")), logits
            )

        if temperature > 0.0:                      # probabilistic sampling
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:                                      # greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Every effort moves you")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**payload["config"])
    model = GPTModel(cfg)
    model.load_state_dict(payload["model_state_dict"])

    from llm.tokenizer import get_gpt2_tokenizer

    tokenizer = get_gpt2_tokenizer()
    out = generate(
        model,
        text_to_token_ids(args.prompt, tokenizer),
        max_new_tokens=args.max_new_tokens,
        context_size=cfg.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(token_ids_to_text(out, tokenizer))


if __name__ == "__main__":
    main()
