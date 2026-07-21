"""Functional test for ReadyJudge voting logic.

Tests majority vote grouping and judgment strategies with synthetic data.
Does not require spacy, Fireworks, or any external deps.
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from staging.ready_config import ReadyConfig
from staging.ready_judge import ReadyJudge, fuzzy_match_answers


def test_fuzzy_match():
    """Test the fuzzy_match_answers cascade."""
    cases = [
        # (a, b, expected)
        ("42", "42", True),                          # exact match
        ("  hello  ", "hello", True),                # whitespace-normalized
        ("Paris", "paris", True),                    # case-insensitive
        ("15.0", "15", True),                        # numeric tolerance
        ("The capital is Paris.", "Paris is capital", True),  # token overlap
        ("42", "43", False),                         # different numbers
        ("Positive", "Negative", False),             # different words
    ]
    passed = 0
    for a, b, expected in cases:
        result = fuzzy_match_answers(a, b)
        status = "PASS" if result == expected else "FAIL"
        if result == expected:
            passed += 1
        print(f"  {status}: fuzzy_match({a!r}, {b!r}) = {result} (expected {expected})")
    return passed, len(cases)


def test_judge_majority_3plus():
    """Test: 3 out of 5 answers agree → majority_3plus."""
    config = ReadyConfig()
    judge = ReadyJudge(config)

    for i in range(5):
        answer_idx = 0 if i < 3 else 1  # 3 "Paris", 2 "London"
        text = "Paris" if answer_idx == 0 else "London"
        judge.add_answer({
            "worker_id": f"w{i}", "task_id": "t1",
            "answer": text, "timing_ms": 100,
            "prompt": "Capital of France?", "category": "factual",
        })

    assert judge.ready_to_judge("t1"), "Should have 5 answers"
    answer, meta = judge.judge("t1")
    assert answer == "Paris", f"Expected Paris, got {answer}"
    assert meta["strategy"] == "majority_3plus", f"Expected majority_3plus, got {meta['strategy']}"
    print(f"  PASS: majority_3plus → answer={answer}, strategy={meta['strategy']}")
    return 1, 1


def test_judge_all_different():
    """Test: all 5 answers different → escalate_fireworks (returns empty without API key)."""
    config = ReadyConfig()
    config.fw_api_key = ""  # No API key → escalation returns ""
    judge = ReadyJudge(config)

    answers = ["Red", "Blue", "Green", "Yellow", "Purple"]
    for i, text in enumerate(answers):
        judge.add_answer({
            "worker_id": f"w{i}", "task_id": "t2",
            "answer": text, "timing_ms": 100,
            "prompt": "What color?", "category": "factual",
        })

    answer, meta = judge.judge("t2")
    # Without Fireworks key, escalation returns "" → falls back to best available
    assert meta["strategy"] == "escalate_fireworks" or meta["strategy"] == "fallback_best", \
        f"Expected escalate_fireworks or fallback_best, got {meta['strategy']}"
    print(f"  PASS: all_different → strategy={meta['strategy']}, answer={answer!r}")
    return 1, 1


def test_judge_2of3_majority():
    """Test: 2 agree, 1 disagrees out of 3 → majority_2of3."""
    config = ReadyConfig()
    config.judgment_votes = 3
    judge = ReadyJudge(config)

    for i in range(3):
        text = "Positive" if i < 2 else "Negative"
        judge.add_answer({
            "worker_id": f"w{i}", "task_id": "t3",
            "answer": text, "timing_ms": 100,
            "prompt": "Good movie?", "category": "sentiment",
        })

    answer, meta = judge.judge("t3")
    assert answer.lower() == "positive", f"Expected positive, got {answer}"
    assert meta["strategy"] == "majority_2of3", f"Expected majority_2of3, got {meta['strategy']}"
    print(f"  PASS: majority_2of3 → answer={answer}, strategy={meta['strategy']}")
    return 1, 1


def test_judge_all_answers():
    """Full flow: judge_all returns correct task_id/answer format."""
    config = ReadyConfig()
    config.judgment_votes = 3
    judge = ReadyJudge(config)

    # Task 1: 3/3 "42"
    for i in range(3):
        judge.add_answer({
            "worker_id": f"w{i}", "task_id": "t1",
            "answer": "42", "timing_ms": 50,
            "prompt": "6*7?", "category": "math",
        })
    a1, m1 = judge.judge("t1")
    print(f"  → judge(t1): answer={a1}, strategy={m1['strategy']}, judged={judge.total_judged}")

    # Task 2: 2/3 "Positive"
    for i in range(3):
        judge.add_answer({
            "worker_id": f"w{i}", "task_id": "t2",
            "answer": "Positive" if i < 2 else "Negative",
            "timing_ms": 50,
            "prompt": "Good?", "category": "sentiment",
        })
    a2, m2 = judge.judge("t2")
    print(f"  → judge(t2): answer={a2}, strategy={m2['strategy']}, judged={judge.total_judged}")

    print(f"  → task_answers keys: {list(judge._task_answers.keys())}")
    print(f"  → _judged keys: {list(judge._judged.keys())}")

    results = judge.judge_all()
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0]["task_id"] in ("t1", "t2")
    assert "answer" in results[0]
    print(f"  PASS: judge_all returned {len(results)} results with correct format")
    return 1, 1


def main():
    tests = [
        ("fuzzy_match", test_fuzzy_match),
        ("majority_3plus", test_judge_majority_3plus),
        ("all_different", test_judge_all_different),
        ("majority_2of3", test_judge_2of3_majority),
        ("judge_all", test_judge_all_answers),
    ]

    total = 0
    passed = 0
    failed = []

    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            p, t = fn()
            passed += p
            total += t
        except Exception as e:
            failed.append(name)
            total += 1
            traceback.print_exc()
            print(f"  FAIL: {e}")

    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
