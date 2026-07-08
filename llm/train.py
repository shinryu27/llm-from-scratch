"""Stage 4a — Pretraining.

Next-token language modeling: minimize the cross-entropy between the model's
logits at every position and the target token (the input shifted by one).
This is equivalent to maximizing the log-likelihood of the corpus, and
exp(mean loss) is the model's perplexity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm.config import GPTConfig
from llm.dataset import create_dataloader
from llm.generate import generate, text_to_token_ids, token_ids_to_text
from llm.model import GPTModel


def calc_loss_batch(input_batch, target_batch, model, device) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_loss_loader(data_loader, model, device, num_batches=None) -> float:
    total, count = 0.0, 0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(
        num_batches, len(data_loader)
    )
    for i, (inp, tgt) in enumerate(data_loader):
        if i >= num_batches:
            break
        total += calc_loss_batch(inp, tgt, model, device).item()
        count += 1
    return total / count


@torch.no_grad()
def evaluate_model(model, train_loader, val_loader, device, eval_iter) -> tuple:
    model.eval()
    train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def generate_sample(model, tokenizer, device, start_context, max_new_tokens=30) -> str:
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    out = generate(
        model,
        encoded,
        max_new_tokens=max_new_tokens,
        context_size=model.cfg.context_length,
        temperature=0.0,
    )
    model.train()
    return token_ids_to_text(out, tokenizer).replace("\n", " ")


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs: int,
    eval_freq: int = 5,
    eval_iter: int = 5,
    start_context: str = "Every effort moves you",
    tokenizer=None,
):
    train_losses, val_losses, tokens_seen_track = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                tr, va = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(tr)
                val_losses.append(va)
                tokens_seen_track.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (step {global_step:06d}): "
                    f"train {tr:.3f}, val {va:.3f}"
                )

        if tokenizer is not None:
            print("  sample:", generate_sample(model, tokenizer, device, start_context))

    return train_losses, val_losses, tokens_seen_track


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a small GPT")
    parser.add_argument("--data", required=True, help="plain-text training file")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--emb", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--out", default="checkpoints/gpt_pretrained.pt")
    args = parser.parse_args()

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = Path(args.data).read_text(encoding="utf-8")

    from llm.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(train_text=text)
    try:
        vocab_size = tokenizer.n_vocab          # tiktoken
    except AttributeError:
        vocab_size = len(tokenizer.vocab)       # fallback BPE

    cfg = GPTConfig(
        vocab_size=vocab_size,
        context_length=args.context,
        emb_dim=args.emb,
        n_heads=args.heads,
        n_layers=args.layers,
    )
    model = GPTModel(cfg).to(device)
    print(f"Parameters: {model.num_params():,}")

    split = int(0.9 * len(text))
    train_loader = create_dataloader(
        text[:split], tokenizer, batch_size=args.batch_size,
        max_length=cfg.context_length, stride=cfg.context_length,
    )
    val_loader = create_dataloader(
        text[split:], tokenizer, batch_size=args.batch_size,
        max_length=cfg.context_length, stride=cfg.context_length,
        shuffle=False, drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs, tokenizer=tokenizer,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": cfg.__dict__}, out_path
    )
    print(f"Checkpoint saved to {out_path}")


if __name__ == "__main__":
    main()
