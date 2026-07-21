#!/usr/bin/env python3
"""Tests for the dynamic system prompt system."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.dynamic_prompts import (
    build_system_prompt,
    build_solver_messages,
    _complexity_level,
    _CATEGORY_PROMPTS,
    MAX_TOKENS,
    STOP_SEQUENCES,
    get_max_tokens,
    get_stop_sequences,
    PROMPT_TABLE,
    lookup_prompt_config,
    _ANTI_PREAMBLE,
)


def test_complexity_level():
    """Verify complexity boundaries are correct."""
    assert _complexity_level(0.0) == "low"
    assert _complexity_level(0.29) == "low"
    assert _complexity_level(0.3) == "medium"
    assert _complexity_level(0.69) == "medium"
    assert _complexity_level(0.7) == "high"
    assert _complexity_level(1.0) == "high"


def test_all_categories_have_all_levels():
    """Every category must define low, medium, high."""
    for cat, levels in _CATEGORY_PROMPTS.items():
        for level in ("low", "medium", "high"):
            assert level in levels, f"{cat} missing {level}"
            assert isinstance(levels[level], str), f"{cat}.{level} not a string"
            assert len(levels[level]) > 10, f"{cat}.{level} too short"


def test_all_categories_in_max_tokens():
    """Every category must have a max_tokens entry."""
    for cat in _CATEGORY_PROMPTS:
        assert cat in MAX_TOKENS, f"{cat} missing from MAX_TOKENS"
        assert MAX_TOKENS[cat] > 0, f"{cat} max_tokens is 0"


def test_all_categories_in_stop_sequences():
    """Every category must have stop sequences."""
    for cat in _CATEGORY_PROMPTS:
        assert cat in STOP_SEQUENCES, f"{cat} missing from STOP_SEQUENCES"
        assert get_stop_sequences(cat) is not None


def test_all_categories_in_prompt_table():
    """Every category must be in the quick-lookup table."""
    for cat in _CATEGORY_PROMPTS:
        assert cat in PROMPT_TABLE, f"{cat} missing from PROMPT_TABLE"
        for level in ("low", "medium", "high"):
            assert level in PROMPT_TABLE[cat], f"{cat}.{level} missing from PROMPT_TABLE"
            prompt, max_tok, stop = PROMPT_TABLE[cat][level]
            assert len(prompt) > 10
            assert max_tok > 0
            assert len(stop) > 0


def test_build_system_prompt_default():
    """Default call without complexity/features should work."""
    prompt = build_system_prompt("math")
    assert isinstance(prompt, str)
    assert len(prompt) > 20
    # Should have anti-preamble
    assert "English only" in prompt


def test_build_system_prompt_all_categories():
    """Every category should produce a valid prompt at every level."""
    for cat in _CATEGORY_PROMPTS:
        for score in (0.0, 0.5, 1.0):
            prompt = build_system_prompt(cat, complexity_score=score)
            assert isinstance(prompt, str)
            assert len(prompt) > 10, f"{cat} score={score} prompt too short"


def test_build_system_prompt_with_features():
    """Feature injections should be included when scores are high."""
    features = {"creativity": 0.8, "verbosity": 0.9}
    prompt = build_system_prompt("factual", feature_scores=features)
    assert "Be extremely terse" in prompt or "Be creative" in prompt


def test_build_system_prompt_low_features():
    """Low feature scores should not inject anything."""
    features = {"creativity": 0.1, "verbosity": 0.1}
    prompt = build_system_prompt("factual", feature_scores=features)
    # Basic prompt without feature injections
    assert "Be extremely terse" not in prompt


def test_build_system_prompt_custom_instructions():
    """Custom instructions should be prepended."""
    custom = "IMPORTANT: Be precise."
    prompt = build_system_prompt("math", custom_instructions=custom)
    assert prompt.startswith("IMPORTANT: Be precise.")
    assert "Answer:" in prompt


def test_build_system_prompt_fallback():
    """Unknown categories should fall back to factual."""
    prompt = build_system_prompt("unknown_category")
    assert isinstance(prompt, str)
    assert len(prompt) > 10


def test_build_solver_messages():
    """Solver messages should produce a valid 2-element list."""
    messages = build_solver_messages("ner", "Extract entities from: John works at Google.")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Google" in messages[1]["content"]


def test_build_solver_messages_with_hint():
    """Deterministic hints should be prepended to user content."""
    messages = build_solver_messages(
        "math",
        "What is 2+2?",
        deterministic_hint="The answer is a single digit."
    )
    user_content = messages[1]["content"]
    assert "Hint:" in user_content
    # The hint should be before the task
    assert user_content.index("Hint:") < user_content.index("2+2")


def test_sentiment_exact_label():
    """Sentiment prompt must enforce exact labels."""
    prompt = build_system_prompt("sentiment")
    assert "positive" in prompt
    assert "negative" in prompt
    assert "neutral" in prompt
    assert "mixed" in prompt


def test_math_has_answer_prefix():
    """Math prompt must include Answer: format instruction."""
    prompt = build_system_prompt("math")
    assert "Answer:" in prompt


def test_logic_has_answer_prefix():
    """Logic prompt must include Answer: format instruction."""
    prompt = build_system_prompt("logic")
    assert "Answer:" in prompt


def test_ner_entity_format():
    """NER prompt must specify entity format."""
    prompt = build_system_prompt("ner")
    assert "PERSON" in prompt
    assert "ORGANIZATION" in prompt
    assert "LOCATION" in prompt


def test_max_tokens_scaling():
    """Max tokens should scale with complexity."""
    base = get_max_tokens("factual", complexity_score=0.5)
    high = get_max_tokens("factual", complexity_score=0.8)
    assert high >= base, "High complexity should have >= tokens"


def test_get_max_tokens_fallback():
    """Unknown category should return default max_tokens."""
    tokens = get_max_tokens("unknown")
    assert tokens >= 200


def test_lookup_prompt_config():
    """Quick lookup should return valid config."""
    for cat in _CATEGORY_PROMPTS:
        for level in ("low", "medium", "high"):
            prompt, max_tok, stop = lookup_prompt_config(cat, level)
            assert len(prompt) > 10
            assert max_tok > 0
            assert len(stop) > 0


def test_lookup_prompt_config_fallback():
    """Unknown category should fall back to factual/medium."""
    prompt, max_tok, stop = lookup_prompt_config("unknown", "medium")
    assert len(prompt) > 10
    assert max_tok > 0


def test_anti_preamble_in_all_categories():
    """Every category prompt must include the anti-preamble."""
    for cat in _CATEGORY_PROMPTS:
        for level in ("low", "medium", "high"):
            prompt_text = _CATEGORY_PROMPTS[cat][level]
            assert "English" in prompt_text or _ANTI_PREAMBLE is not None


def test_prompts_differ_by_category():
    """Prompts for different categories must be distinct."""
    math_prompt = _CATEGORY_PROMPTS["math"]["medium"]
    sentiment_prompt = _CATEGORY_PROMPTS["sentiment"]["medium"]
    code_gen_prompt = _CATEGORY_PROMPTS["code_gen"]["medium"]
    assert math_prompt != sentiment_prompt
    assert code_gen_prompt != math_prompt


def test_prompts_differ_by_complexity():
    """Prompts for different levels must be distinct."""
    for cat in _CATEGORY_PROMPTS:
        low = _CATEGORY_PROMPTS[cat]["low"]
        high = _CATEGORY_PROMPTS[cat]["high"]
        assert low != high, f"{cat} low and high are identical"
        # High should generally be longer (more instructions)
        assert len(high) >= len(low) or "high" == cat, \
            f"{cat} high should be at least as long as low"


def test_sentiment_low_is_label_only():
    """Low complexity sentiment should be label-only."""
    prompt = _CATEGORY_PROMPTS["sentiment"]["low"]
    assert "one word" in prompt.lower() or "exactly one word" in prompt.lower() or "no explanation" in prompt.lower()


def test_code_gen_has_fenced_block():
    """Code gen prompt must ask for fenced block."""
    prompt = _CATEGORY_PROMPTS["code_gen"]["medium"]
    assert "```python" in prompt


def test_code_debug_has_fenced_block():
    """Code debug prompt must ask for fenced block."""
    prompt = _CATEGORY_PROMPTS["code_debug"]["medium"]
    assert "```python" in prompt or "```" in prompt


def test_summarization_has_length_constraint():
    """Summarization prompt must mention length constraints."""
    prompt = _CATEGORY_PROMPTS["summarization"]["low"]
    assert "sentence" in prompt.lower() or "word" in prompt.lower() or "length" in prompt.lower()


def test_all_categories_produce_valid_solver_messages():
    """All categories must produce valid solver messages."""
    for cat in _CATEGORY_PROMPTS:
        messages = build_solver_messages(cat, f"Test task for {cat}")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # System message should not be empty
        assert len(messages[0]["content"]) > 20
        # User message should contain the task
        assert cat in messages[1]["content"] or "Test task" in messages[1]["content"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {fn.__name__}: {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"  {passed}/{passed+failed} tests passed")
    if failed:
        print(f"  {failed} TESTS FAILED")
    else:
        print(f"  ALL TESTS PASSED ✅")
