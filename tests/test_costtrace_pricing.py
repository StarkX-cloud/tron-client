from tron_costtrace.pricing import cost_usd, is_known_model


def test_known_model_computes_cost():
    cost = cost_usd("gpt-4o-mini", input_tokens=1000, output_tokens=1000)
    assert cost == (1000 / 1000.0) * 0.00015 + (1000 / 1000.0) * 0.0006


def test_unknown_model_returns_none_not_zero():
    assert cost_usd("some-made-up-model", 1000, 1000) is None


def test_missing_token_counts_returns_none():
    assert cost_usd("gpt-4o-mini", None, 1000) is None
    assert cost_usd("gpt-4o-mini", 1000, None) is None


def test_is_known_model():
    assert is_known_model("gpt-4o-mini") is True
    assert is_known_model("not-a-real-model") is False
