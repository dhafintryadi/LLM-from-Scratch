from __future__ import annotations

import tiktoken


class ByteTokenizer:
    """Small offline tokenizer used when GPT-2 BPE files are unavailable."""

    n_vocab = 256

    def encode(self, text: str, allowed_special=None) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        valid_bytes = bytes(token_id for token_id in token_ids if 0 <= token_id < 256)
        return valid_bytes.decode("utf-8", errors="ignore")


def get_tokenizer(encoding_name: str = "gpt2", allow_fallback: bool = True):
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        if not allow_fallback:
            raise
        return ByteTokenizer()


def text_to_token_ids(text: str, tokenizer):
    import torch

    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer) -> str:
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)
