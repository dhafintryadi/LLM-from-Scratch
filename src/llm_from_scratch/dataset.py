from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    def __init__(self, text: str, tokenizer, max_length: int, stride: int) -> None:
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        if len(token_ids) <= max_length:
            raise ValueError("Text is too short for the requested context length.")

        self.input_ids = []
        self.target_ids = []
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int):
        return self.input_ids[index], self.target_ids[index]


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def create_dataloader(
    text: str,
    tokenizer,
    batch_size: int = 2,
    max_length: int = 64,
    stride: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    dataset = GPTDataset(text, tokenizer, max_length=max_length, stride=stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
