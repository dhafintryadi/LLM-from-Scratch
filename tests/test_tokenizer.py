from llm_from_scratch.tokenizer import get_tokenizer, text_to_token_ids, token_ids_to_text


def test_tokenizer_round_trip():
    tokenizer = get_tokenizer()
    text = "Every effort moves you"

    token_ids = text_to_token_ids(text, tokenizer)
    decoded = token_ids_to_text(token_ids, tokenizer)

    assert decoded == text
