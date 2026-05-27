import pytest

from llm_from_scratch.dataset import create_dataloader
from llm_from_scratch.tokenizer import get_tokenizer


def test_create_dataloader_returns_expected_shapes():
    tokenizer = get_tokenizer()
    text = "Every effort moves you forward. " * 20

    loader = create_dataloader(
        text,
        tokenizer,
        batch_size=2,
        max_length=8,
        stride=4,
        shuffle=False,
    )
    inputs, targets = next(iter(loader))

    assert inputs.shape == (2, 8)
    assert targets.shape == (2, 8)


def test_create_dataloader_rejects_too_short_text():
    tokenizer = get_tokenizer()

    with pytest.raises(ValueError):
        create_dataloader("short", tokenizer, max_length=32)
