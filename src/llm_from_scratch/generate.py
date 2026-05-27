from __future__ import annotations

import argparse

import torch

from llm_from_scratch.model import GPTConfig, GPTModel
from llm_from_scratch.tokenizer import get_tokenizer, text_to_token_ids, token_ids_to_text


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf"), device=logits.device), logits)

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with an untrained tiny GPT model.")
    parser.add_argument("--prompt", default="Every effort", help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(123)
    tokenizer = get_tokenizer()
    cfg = GPTConfig(vocab_size=tokenizer.n_vocab)
    model = GPTModel(cfg)
    token_ids = text_to_token_ids(args.prompt, tokenizer)
    generated = generate(
        model,
        token_ids,
        max_new_tokens=args.max_new_tokens,
        context_size=cfg.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(token_ids_to_text(generated, tokenizer))


if __name__ == "__main__":
    main()
