"""End-to-end smoke test with a tiny config — verifies all six stages wire up.

Run:  python -m llm.smoke_test
"""

from __future__ import annotations

import torch

from llm.attention import MultiHeadAttention
from llm.config import GPTConfig
from llm.generate import generate
from llm.model import GPTModel
from llm.tokenizer import BPETokenizer


def main() -> None:
    torch.manual_seed(123)
    cfg = GPTConfig(
        vocab_size=200, context_length=32, emb_dim=32, n_heads=4,
        n_layers=2, drop_rate=0.1,
    )

    # Stage 1: tokenizer trained from scratch
    text = ("Every effort moves you forward. The quick brown fox jumps over "
            "the lazy dog. ") * 20
    tok = BPETokenizer()
    tok.train(text, vocab_size=150)
    ids = tok.encode("Every effort moves you")
    assert tok.decode(ids).startswith("Every"), "BPE round-trip failed"
    print(f"[1] BPE tokenizer: {len(tok.vocab)} tokens, round-trip OK")

    # Stage 2: attention shape check
    mha = MultiHeadAttention(d_in=32, d_out=32, context_length=32,
                             dropout=0.0, num_heads=4)
    out = mha(torch.randn(2, 8, 32))
    assert out.shape == (2, 8, 32)
    print(f"[2] Multi-head attention: output {tuple(out.shape)} OK")

    # Stage 3: full model forward pass
    model = GPTModel(cfg)
    batch = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(batch)
    assert logits.shape == (2, 16, cfg.vocab_size)
    print(f"[3] GPTModel: {model.num_params():,} params, "
          f"logits {tuple(logits.shape)} OK")

    # Stage 4: one training step + generation
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_before = torch.nn.functional.cross_entropy(
        model(batch).flatten(0, 1), targets.flatten()
    )
    loss_before.backward()
    optimizer.step()
    gen = generate(model, batch[:, :4], max_new_tokens=5,
                   context_size=cfg.context_length, temperature=1.0, top_k=10)
    assert gen.shape[1] == 9
    print(f"[4] Training step (loss {loss_before.item():.3f}) + generation OK")

    # Stage 5: classifier head surgery
    model.out_head = torch.nn.Linear(cfg.emb_dim, 2)
    cls_logits = model(batch)[:, -1, :]
    assert cls_logits.shape == (2, 2)
    print("[5] Classification head: last-token logits OK")

    # Stage 6: instruction collation
    from llm.finetune_instruction import collate_fn, format_input

    entry = {"instruction": "Add 2+2", "input": "", "output": "4"}
    assert "### Instruction:" in format_input(entry)
    inp, tgt = collate_fn([[1, 2, 3], [1, 2, 3, 4, 5]], pad_token_id=0)
    assert inp.shape == tgt.shape and (tgt == -100).any()
    print("[6] Instruction formatting + loss masking OK")

    print("\nAll six stages passed.")


if __name__ == "__main__":
    main()
