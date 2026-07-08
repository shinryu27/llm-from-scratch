"""Stage 1b — Sliding-window dataset for next-token prediction.

Each sample is a pair ``(input_ids, target_ids)`` where the targets are the
inputs shifted one position to the left:

    text:    "In the heart of the city"
    input:   [In, the, heart, of]
    target:  [the, heart, of, the]

The model therefore learns, at every position, to predict the next token.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    def __init__(self, text: str, tokenizer, max_length: int, stride: int) -> None:
        token_ids = tokenizer.encode(text)
        if len(token_ids) <= max_length:
            raise ValueError(
                f"Corpus has only {len(token_ids)} tokens; need more than "
                f"max_length={max_length}."
            )

        self.inputs: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        for i in range(0, len(token_ids) - max_length, stride):
            chunk = token_ids[i : i + max_length + 1]
            self.inputs.append(torch.tensor(chunk[:-1]))
            self.targets.append(torch.tensor(chunk[1:]))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


def create_dataloader(
    text: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
