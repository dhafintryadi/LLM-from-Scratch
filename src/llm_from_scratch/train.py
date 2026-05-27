from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm_from_scratch.dataset import create_dataloader, read_text
from llm_from_scratch.generate import generate
from llm_from_scratch.model import GPTConfig, GPTModel
from llm_from_scratch.tokenizer import get_tokenizer, text_to_token_ids, token_ids_to_text


def calc_loss_batch(input_batch, target_batch, model, device) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def train_one_epoch(model, train_loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(train_loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny GPT-style model.")
    parser.add_argument("--data-path", default="data/sample/tiny_corpus.txt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--prompt", default="Every effort")
    args = parser.parse_args()

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    text = read_text(Path(args.data_path))

    cfg = GPTConfig(vocab_size=tokenizer.n_vocab, context_length=args.context_length)
    model = GPTModel(cfg).to(device)
    train_loader = create_dataloader(
        text,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.context_length,
        stride=args.stride,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"epoch={epoch + 1} train_loss={loss:.4f}")

    token_ids = text_to_token_ids(args.prompt, tokenizer).to(device)
    generated = generate(model, token_ids, max_new_tokens=8, context_size=cfg.context_length)
    print(token_ids_to_text(generated.cpu(), tokenizer))


if __name__ == "__main__":
    main()
