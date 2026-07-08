"""Stage 5 — Classification finetuning.

Turn the pretrained GPT into a binary text classifier (e.g. spam vs. ham):

1. Replace the 50,257-way LM head with a small ``num_classes`` head.
2. Freeze most of the network; train only the new head, the last transformer
   block, and the final LayerNorm (a cheap, effective recipe).
3. Classify using the logits of the LAST token — the only position that has
   attended to the entire sequence under the causal mask.

Expected data: a CSV with columns ``label`` (ham/spam or 0/1) and ``text``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from llm.config import GPTConfig
from llm.model import GPTModel


class ClassificationDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=None, pad_token_id=50256):
        self.texts = [r[1] for r in rows]
        self.labels = [r[0] for r in rows]
        self.encoded = [tokenizer.encode(t) for t in self.texts]

        if max_length is None:
            max_length = max(len(e) for e in self.encoded)
        self.max_length = max_length

        self.encoded = [
            e[:max_length] + [pad_token_id] * (max_length - len(e[:max_length]))
            for e in self.encoded
        ]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded[idx]),
            torch.tensor(self.labels[idx]),
        )


def load_csv(path: str | Path) -> list[tuple[int, str]]:
    label_map = {"ham": 0, "spam": 1, "0": 0, "1": 1}
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((label_map[row["label"].strip().lower()], row["text"]))
    return rows


def build_classifier(checkpoint: str | Path, num_classes: int = 2) -> GPTModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**payload["config"])
    model = GPTModel(cfg)
    model.load_state_dict(payload["model_state_dict"])

    # Freeze everything ...
    for p in model.parameters():
        p.requires_grad = False
    # ... then unfreeze the new head, last block, and final norm.
    model.out_head = nn.Linear(cfg.emb_dim, num_classes)
    for p in model.trf_blocks[-1].parameters():
        p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True
    return model


def last_token_logits(model: GPTModel, input_batch: torch.Tensor) -> torch.Tensor:
    return model(input_batch)[:, -1, :]  # (batch, num_classes)


@torch.no_grad()
def calc_accuracy(loader, model, device, num_batches=None) -> float:
    model.eval()
    correct, total = 0, 0
    num_batches = len(loader) if num_batches is None else min(num_batches, len(loader))
    for i, (inp, tgt) in enumerate(loader):
        if i >= num_batches:
            break
        inp, tgt = inp.to(device), tgt.to(device)
        preds = torch.argmax(last_token_logits(model, inp), dim=-1)
        correct += (preds == tgt).sum().item()
        total += tgt.numel()
    return correct / total


def train_classifier(model, train_loader, val_loader, device,
                     num_epochs=5, lr=5e-5) -> None:
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=0.1,
    )
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                last_token_logits(model, inp), tgt
            )
            loss.backward()
            optimizer.step()
        train_acc = calc_accuracy(train_loader, model, device, num_batches=20)
        val_acc = calc_accuracy(val_loader, model, device, num_batches=20)
        print(f"Epoch {epoch + 1}: train acc {train_acc:.2%}, val acc {val_acc:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune GPT as a classifier")
    parser.add_argument("--data", required=True, help="CSV with label,text columns")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default="checkpoints/gpt_classifier.pt")
    args = parser.parse_args()

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from llm.tokenizer import get_gpt2_tokenizer

    tokenizer = get_gpt2_tokenizer()
    rows = load_csv(args.data)
    split = int(0.85 * len(rows))
    train_ds = ClassificationDataset(rows[:split], tokenizer)
    val_ds = ClassificationDataset(
        rows[split:], tokenizer, max_length=train_ds.max_length
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = build_classifier(args.checkpoint)
    train_classifier(model, train_loader, val_loader, device,
                     num_epochs=args.epochs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "config": model.cfg.__dict__}, out_path)
    print(f"Classifier saved to {out_path}")


if __name__ == "__main__":
    main()
