"""Stage 6 — Instruction finetuning.

Supervised finetuning (SFT) on instruction–response pairs using Alpaca-style
prompt formatting. The custom collate function pads each batch dynamically
and masks both the padding and (optionally) the prompt portion of the targets
with ignore_index=-100, so the loss is computed only where we want the model
to learn.

Expected data: a JSON list of {"instruction": ..., "input": ..., "output": ...}.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from llm.config import GPTConfig
from llm.model import GPTModel

IGNORE_INDEX = -100
PAD_TOKEN_ID = 50256  # <|endoftext|> doubles as the pad token


def format_input(entry: dict) -> str:
    """Alpaca-style prompt."""
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer) -> None:
        self.encoded = []
        for entry in data:
            full_text = (
                format_input(entry) + f"\n\n### Response:\n{entry['output']}"
            )
            self.encoded.append(tokenizer.encode(full_text))

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> list[int]:
        return self.encoded[idx]


def collate_fn(batch, pad_token_id=PAD_TOKEN_ID, ignore_index=IGNORE_INDEX,
               allowed_max_length=None, device="cpu"):
    batch_max = max(len(item) + 1 for item in batch)
    inputs, targets = [], []

    for item in batch:
        padded = item + [pad_token_id] * (batch_max - len(item))
        inp = torch.tensor(padded[:-1])
        tgt = torch.tensor(padded[1:])

        # Mask all but the first pad token in the targets.
        mask = tgt == pad_token_id
        indices = torch.nonzero(mask).squeeze(-1)
        if indices.numel() > 1:
            tgt[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inp = inp[:allowed_max_length]
            tgt = tgt[:allowed_max_length]

        inputs.append(inp)
        targets.append(tgt)

    return torch.stack(inputs).to(device), torch.stack(targets).to(device)


def train_instruction_model(model, train_loader, val_loader, device,
                            num_epochs=2, lr=5e-5) -> None:
    from llm.train import calc_loss_batch, evaluate_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.to(device)
    step = -1
    for epoch in range(num_epochs):
        model.train()
        for inp, tgt in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(inp, tgt, model, device)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 10 == 0:
                tr, va = evaluate_model(model, train_loader, val_loader,
                                        device, eval_iter=5)
                print(f"Ep {epoch + 1} step {step:05d}: "
                      f"train {tr:.3f}, val {va:.3f}")


@torch.no_grad()
def respond(model, tokenizer, entry: dict, device, max_new_tokens=128) -> str:
    """Generate a response for one instruction entry (for evaluation)."""
    from llm.generate import generate, text_to_token_ids, token_ids_to_text

    prompt = format_input(entry) + "\n\n### Response:\n"
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    out = generate(model, idx, max_new_tokens=max_new_tokens,
                   context_size=model.cfg.context_length,
                   temperature=0.0, eos_id=PAD_TOKEN_ID)
    full = token_ids_to_text(out, tokenizer)
    return full[len(prompt):].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Instruction-finetune a GPT")
    parser.add_argument("--data", required=True,
                        help="JSON list of instruction/input/output dicts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default="checkpoints/gpt_instruct.pt")
    args = parser.parse_args()

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from llm.tokenizer import get_gpt2_tokenizer

    tokenizer = get_gpt2_tokenizer()
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    split = int(0.9 * len(data))

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**payload["config"])
    model = GPTModel(cfg)
    model.load_state_dict(payload["model_state_dict"])

    collate = partial(collate_fn, device=device,
                      allowed_max_length=cfg.context_length)
    train_loader = DataLoader(InstructionDataset(data[:split], tokenizer),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate)
    val_loader = DataLoader(InstructionDataset(data[split:], tokenizer),
                            batch_size=args.batch_size, collate_fn=collate)

    train_instruction_model(model, train_loader, val_loader, device,
                            num_epochs=args.epochs)

    # Show a few generated responses for manual evaluation.
    for entry in data[split:][:3]:
        print("\n### Instruction:", entry["instruction"])
        print("Model:", respond(model, tokenizer, entry, device))
        print("Reference:", entry["output"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "config": cfg.__dict__}, out_path)
    print(f"\nInstruction-tuned model saved to {out_path}")


if __name__ == "__main__":
    main()
