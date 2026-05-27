import torch

from llm_from_scratch.generate import generate
from llm_from_scratch.model import GPTConfig, GPTModel


def test_gpt_forward_shape():
    cfg = GPTConfig(vocab_size=100, context_length=8, emb_dim=16, n_heads=4, n_layers=1)
    model = GPTModel(cfg)
    token_ids = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))

    logits = model(token_ids)

    assert logits.shape == (2, cfg.context_length, cfg.vocab_size)


def test_generate_appends_tokens():
    cfg = GPTConfig(vocab_size=100, context_length=8, emb_dim=16, n_heads=4, n_layers=1)
    model = GPTModel(cfg)
    token_ids = torch.tensor([[1, 2, 3]])

    generated = generate(model, token_ids, max_new_tokens=2, context_size=cfg.context_length)

    assert generated.shape == (1, 5)
