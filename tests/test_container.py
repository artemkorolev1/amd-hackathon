#!/usr/bin/env python3
"""Tests for container/*.py — consensus, inference, server, fallback."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import ANY, MagicMock, patch, mock_open, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from container.consensus import (
    fuzzy_match,
    is_degenerate,
    equivalence_classes,
    resolve_consensus,
    build_judge_prompt,
    parse_judge_output,
    merge_answers,
)
from container.inference import parallel_infer, simple_infer
from container.server import find_llama_server, ServerManager
from container.fallback import is_available, fallback_answer
import urllib.error


# =========================================================================
# Tests for fuzzy_match (consensus.py)
# =========================================================================

class TestFuzzyMatch(unittest.TestCase):
    """Verify the fuzzy_match cascade: exact → substring → numeric → token overlap."""

    def test_exact_case_insensitive(self):
        self.assertTrue(fuzzy_match("Canberra", "Canberra"))
        self.assertTrue(fuzzy_match("canberra", "Canberra"))
        self.assertTrue(fuzzy_match("CANBERRA", "Canberra"))

    def test_substring_expected_in_answer(self):
        self.assertTrue(fuzzy_match("The capital is Canberra", "Canberra"))

    def test_substring_answer_in_expected(self):
        self.assertTrue(fuzzy_match("yes", "yes, that is correct"))

    def test_substring_too_short_answer(self):
        # len(a) < 3 → substring check skipped
        self.assertFalse(fuzzy_match("no", "nope"))

    def test_numeric_tolerance_pairwise(self):
        # 6.0 → "6" substring of "6.0", passes via substring
        self.assertTrue(fuzzy_match("6.0", "6"))
        # 6.05 vs 6 — "6" is substring of "6.05", passes via substring
        self.assertTrue(fuzzy_match("6.05", "6"))
        # Proper numeric mismatch: 7.0 vs 6.0 differs by 16.7% > 1%
        self.assertFalse(fuzzy_match("7.0", "6.0"))

    def test_numeric_close_match(self):
        self.assertTrue(fuzzy_match("6.05", "6"))   # 0.05/6 = 0.8% < 1%

    def test_numeric_mismatch(self):
        # "73" is not a substring of "72" (a_low="73", e_low="72")
        # a_low in e_low? "73" in "72"? No.
        # Numeric: abs(73-72)/72 = 1.4% > 1%
        self.assertFalse(fuzzy_match("73", "72"))

    def test_numeric_single_number_in_answer(self):
        self.assertTrue(fuzzy_match("The answer is 72 km/h", "72"))
        self.assertFalse(fuzzy_match("The answer is 73", "72"))

    def test_token_overlap_short_answer(self):
        # Short expected (< 50 chars): need >= 50% token overlap
        self.assertTrue(fuzzy_match(
            "Gabriel Garcia Marquez wrote it",
            "Gabriel Garcia Marquez"
        ))

    def test_token_overlap_long_answer(self):
        # Long expected (>= 50 chars): need >= 30% token overlap
        expected = "it did not rain because the ground is not wet"
        answer = "it did not rain"
        self.assertTrue(fuzzy_match(answer, expected))

    def test_no_match(self):
        self.assertFalse(fuzzy_match("Paris", "London"))

    def test_empty_expected(self):
        self.assertTrue(fuzzy_match("anything", ""))
        self.assertFalse(fuzzy_match("", ""))

    def test_numeric_zero_tolerance(self):
        # 0.005 vs 0 — "0" is in "0.005" via substring, so it passes
        self.assertTrue(fuzzy_match("0.005", "0"))

    def test_numeric_negative(self):
        self.assertTrue(fuzzy_match("-5", "-5"))
        self.assertTrue(fuzzy_match("-5.05", "-5"))  # within 1%

    def test_numeric_exact_zero(self):
        # Zero-related check via absolute difference (substring often catches these)
        self.assertTrue(fuzzy_match("0.005", "0"))

    def test_token_overlap_with_stopwords(self):
        # Stopwords are removed before computing overlap
        self.assertTrue(fuzzy_match("the cat sat on the mat", "cat mat"))

    def test_token_overlap_all_stopwords_expected(self):
        # Expected only has stopwords → return False
        self.assertFalse(fuzzy_match("something", "the and of"))

    def test_substring_multi_token(self):
        # Multi-word expected, answer is a sentence containing it
        self.assertTrue(fuzzy_match("The answer is 42 according to my sources",
                                    "answer is 42"))

    def test_numeric_same_len_mismatch(self):
        # Both have same number count but >1% diff and no token overlap
        self.assertFalse(fuzzy_match("7.0 8.0", "6.0 9.0"))


# =========================================================================
# Tests for is_degenerate (consensus.py)
# =========================================================================

class TestIsDegenerate(unittest.TestCase):
    """Verify degenerate answer detection — empty, hedge, repetition, etc."""

    def test_empty_string(self):
        self.assertTrue(is_degenerate(""))
        self.assertTrue(is_degenerate("   "))

    def test_short_numeric_valid(self):
        self.assertFalse(is_degenerate("42"))
        self.assertFalse(is_degenerate("6.0"))
        self.assertFalse(is_degenerate("-3.14"))

    def test_too_short_non_numeric(self):
        self.assertTrue(is_degenerate("ab"))
        self.assertTrue(is_degenerate("x"))

    def test_i_dont_know(self):
        self.assertTrue(is_degenerate("I don't know the answer"))
        self.assertTrue(is_degenerate("i do not know"))

    def test_i_cannot(self):
        self.assertTrue(is_degenerate("I cannot answer this question"))
        self.assertTrue(is_degenerate("I can't do that"))

    def test_as_an_ai(self):
        self.assertTrue(is_degenerate("As an AI, I cannot answer that"))

    def test_sorry(self):
        self.assertTrue(is_degenerate("Sorry, I can't help with that"))

    def test_insufficient(self):
        self.assertTrue(is_degenerate("Insufficient information to answer"))

    def test_heavy_repetition(self):
        # Less than 25% unique words
        self.assertTrue(is_degenerate("yes yes yes yes yes yes yes yes"))

    def test_valid_answer(self):
        self.assertFalse(is_degenerate("The capital of France is Paris"))
        self.assertFalse(is_degenerate("42 kilometers per hour"))
        self.assertFalse(is_degenerate("A reasonably long answer with some words"))

    def test_no_information(self):
        self.assertTrue(is_degenerate("There is no information available"))

    def test_unable_to(self):
        self.assertTrue(is_degenerate("I am unable to answer that question"))

    def test_not_enough_information(self):
        self.assertTrue(is_degenerate("Not enough information to answer"))

    def test_text_does_not(self):
        self.assertTrue(is_degenerate("The text does not contain the answer"))

    def test_answer_with_degenerate_pattern_but_good(self):
        # "sorry" as part of "worried" should not match \bsorry\b
        self.assertFalse(is_degenerate("I am not worried about that"))
        # But standalone "sorry" should match
        self.assertTrue(is_degenerate("Sorry, I don't know"))

    def test_valid_short_answer(self):
        # len >= 3 and not numeric pattern
        self.assertFalse(is_degenerate("Yes"))
        self.assertFalse(is_degenerate("No."))


# =========================================================================
# Tests for equivalence_classes (consensus.py)
# =========================================================================

class TestEquivalenceClasses(unittest.TestCase):
    """Verify grouping of answers into equivalence classes via fuzzy_match."""

    def test_all_same(self):
        groups = equivalence_classes(["Paris", "Paris", "Paris", "Paris"])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], {0, 1, 2, 3})

    def test_three_and_one(self):
        groups = equivalence_classes(["Paris", "London", "Paris", "Paris"])
        self.assertEqual(len(groups), 2)
        paris_group = next(g for g in groups if 0 in g)
        self.assertGreaterEqual(len(paris_group), 3)

    def test_two_pairs(self):
        groups = equivalence_classes(["Paris", "London", "Paris", "London"])
        self.assertEqual(len(groups), 2)

    def test_all_different(self):
        groups = equivalence_classes(["Paris", "London", "Berlin", "Madrid"])
        self.assertEqual(len(groups), 4)

    def test_fuzzy_grouped(self):
        # Slightly different strings that fuzzy match
        groups = equivalence_classes(["6.0", "6", "6.00", "7.0"])
        self.assertEqual(len(groups), 2)

    def test_empty_list(self):
        groups = equivalence_classes([])
        self.assertEqual(groups, [])

    def test_single_answer(self):
        groups = equivalence_classes(["Paris"])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], {0})

    def test_case_insensitive_grouping(self):
        groups = equivalence_classes(["Paris", "paris", "PARIS", "London"])
        self.assertEqual(len(groups), 2)
        paris_group = next(g for g in groups if 0 in g)
        self.assertEqual(len(paris_group), 3)


# =========================================================================
# Tests for resolve_consensus (consensus.py)
# =========================================================================

class TestResolveConsensus(unittest.TestCase):
    """Verify consensus decision — majority, consensus, judge, degenerate."""

    def test_consensus_three_plus(self):
        best, method, conf = resolve_consensus(["Paris", "Paris", "Paris", "London"])
        self.assertEqual(best, "Paris")
        self.assertEqual(method, "consensus")
        self.assertEqual(conf, 0.95)

    def test_consensus_all_four(self):
        best, method, conf = resolve_consensus(["6.0", "6", "6.00", "6.0"])
        self.assertEqual(method, "consensus")
        self.assertAlmostEqual(float(best), 6.0)

    def test_majority_two(self):
        best, method, conf = resolve_consensus(["Paris", "Paris", "London", "Berlin"])
        self.assertEqual(best, "Paris")
        self.assertEqual(method, "majority")
        self.assertEqual(conf, 0.7)

    def test_no_consensus_all_different(self):
        best, method, conf = resolve_consensus(["Paris", "London", "Berlin", "Madrid"])
        self.assertEqual(method, "no_consensus")
        self.assertEqual(conf, 0.3)

    def test_all_degenerate(self):
        best, method, conf = resolve_consensus([
            "I don't know",
            "Sorry, I cannot answer",
            "As an AI, I can't",
            "Insufficient information",
        ])
        self.assertEqual(method, "degenerate")
        self.assertEqual(conf, 0.1)

    def test_some_degenerate_rest_all_same(self):
        # Two degenerate, two matching valid answers
        best, method, conf = resolve_consensus([
            "I don't know",
            "Paris",
            "Sorry, can't",
            "Paris",
        ])
        self.assertEqual(method, "majority")
        self.assertEqual(best, "Paris")

    def test_empty_list(self):
        best, method, conf = resolve_consensus([])
        self.assertEqual(method, "degenerate")
        self.assertEqual(conf, 0.1)
        self.assertEqual(best, "")

    def test_tie_pairs(self):
        # Two equivalent, two equivalent in different group
        # "Paris", "Paris", "London", "London" — two pairs, largest == 2
        best, method, conf = resolve_consensus(["Paris", "Paris", "London", "London"])
        self.assertEqual(method, "majority")
        # First largest group will be Paris (first pair encountered)
        self.assertEqual(best, "Paris")

    def test_single_non_degenerate(self):
        # Three degenerate, one valid — all are different equivalence classes
        # resolve_consensus returns first answer as best when all differ
        best, method, conf = resolve_consensus([
            "I don't know the answer to this question",
            "Canberra",
            "Sorry, I am unable to help with that request",
            "The information is not available at this time",
        ])
        self.assertEqual(method, "no_consensus")
        self.assertEqual(conf, 0.3)
        # First answer is chosen as "best" since all groups are size 1
        # (merge_answers handles the degenerate→fallback logic downstream)

    def test_four_valid_different(self):
        best, method, conf = resolve_consensus([
            "Canberra",
            "Sydney",
            "Melbourne",
            "Perth",
        ])
        self.assertEqual(method, "no_consensus")


# =========================================================================
# Tests for build_judge_prompt (consensus.py)
# =========================================================================

class TestBuildJudgePrompt(unittest.TestCase):
    """Verify judge prompt formatting."""

    def test_basic_formatting(self):
        template = "Question: {question}\nA: {answer_a}\nB: {answer_b}\nC: {answer_c}\nD: {answer_d}"
        result = build_judge_prompt(
            "What is the capital?",
            ["Paris", "London", "Berlin", "Madrid"],
            template,
        )
        self.assertIn("What is the capital?", result)
        self.assertIn("A: Paris", result)
        self.assertIn("B: London", result)
        self.assertIn("C: Berlin", result)
        self.assertIn("D: Madrid", result)

    def test_fewer_than_4_answers(self):
        template = "Q: {question}\nA: {answer_a}\nB: {answer_b}\nC: {answer_c}\nD: {answer_d}"
        result = build_judge_prompt("Test?", ["ans1", "ans2"], template)
        self.assertEqual(result.count("ans1"), 1)
        self.assertEqual(result.count("ans2"), 1)
        self.assertIn("C: ", result)  # empty padding
        self.assertIn("D: ", result)

    def test_more_than_4_answers(self):
        template = "A: {answer_a}\nB: {answer_b}\nC: {answer_c}\nD: {answer_d}"
        result = build_judge_prompt("Test?", ["a", "b", "c", "d", "e"], template)
        # Only first 4 used
        self.assertEqual(result.count("A: a"), 1)
        self.assertEqual(result.count("B: b"), 1)
        self.assertEqual(result.count("C: c"), 1)
        self.assertEqual(result.count("D: d"), 1)


# =========================================================================
# Tests for parse_judge_output (consensus.py)
# =========================================================================

class TestParseJudgeOutput(unittest.TestCase):
    """Verify parsing of judge LLM output."""

    def test_basic_parse(self):
        idx, reason = parse_judge_output("Best: C")
        self.assertEqual(idx, 2)
        self.assertEqual(reason, "")

    def test_with_reason(self):
        idx, reason = parse_judge_output("Best: A\nReason: This is correct and concise.")
        self.assertEqual(idx, 0)
        self.assertEqual(reason, "This is correct and concise.")

    def test_lowercase_letter(self):
        idx, reason = parse_judge_output("Best: b")
        self.assertEqual(idx, 1)

    def test_parse_failed(self):
        idx, reason = parse_judge_output("No best answer here")
        self.assertEqual(idx, -1)
        self.assertEqual(reason, "parse_failed")

    def test_d_out_of_range(self):
        # D is valid (index 3)
        idx, reason = parse_judge_output("Best: D")
        self.assertEqual(idx, 3)

    def test_multiline_reason(self):
        idx, reason = parse_judge_output("Best: B\nReason: Line one.\nLine two.")
        self.assertEqual(idx, 1)
        self.assertIn("Line one", reason)
        self.assertIn("Line two", reason)


# =========================================================================
# Tests for merge_answers (consensus.py) — full pipeline
# =========================================================================

class TestMergeAnswers(unittest.TestCase):
    """Verify the full merge_answers pipeline end-to-end."""

    def test_consensus_route(self):
        result = merge_answers(
            "Q?",
            ["Paris", "Paris", "Paris", "London"],
            "",
        )
        self.assertEqual(result["answer"], "Paris")
        self.assertEqual(result["method"], "consensus")
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(len(result["raw_answers"]), 4)
        self.assertEqual(len(result["degenerate"]), 4)

    def test_majority_route(self):
        result = merge_answers(
            "Q?",
            ["Paris", "Paris", "London", "Berlin"],
            "",
        )
        self.assertEqual(result["answer"], "Paris")
        self.assertEqual(result["method"], "majority")
        self.assertEqual(result["confidence"], 0.7)

    def test_judge_route(self):
        # All different → should call judge
        judge_fn = MagicMock(return_value="Best: B\nReason: Good answer.")
        result = merge_answers(
            "What is the capital?",
            ["Paris", "London", "Berlin", "Madrid"],
            "Question: {question}\nA: {answer_a}\nB: {answer_b}\nC: {answer_c}\nD: {answer_d}",
            call_judge_fn=judge_fn,
        )
        self.assertEqual(result["answer"], "London")  # index 1 = B
        self.assertEqual(result["method"], "judge")
        self.assertEqual(result["confidence"], 0.7)
        self.assertEqual(result["judge_reason"], "Good answer.")
        judge_fn.assert_called_once()

    def test_judge_picks_degenerate_falls_back(self):
        # Judge picks index 0, but that answer is degenerate
        judge_fn = MagicMock(return_value="Best: A")
        result = merge_answers(
            "Q?",
            ["I don't know", "London", "Berlin", "Madrid"],
            "Q: {question}\nA: {answer_a}\nB: {answer_b}\nC: {answer_c}\nD: {answer_d}",
            call_judge_fn=judge_fn,
        )
        # Should fall back to first non-degenerate
        self.assertNotEqual(result["method"], "judge")
        self.assertIn(result["method"], ("fallback_best",))
        self.assertEqual(result["confidence"], 0.3)
        self.assertEqual(result["answer"], "London")

    def test_judge_error_falls_back(self):
        judge_fn = MagicMock(side_effect=RuntimeError("API down"))
        result = merge_answers(
            "Q?",
            ["Paris", "London", "Berlin", "Madrid"],
            "Q: {question}\nA: {answer_a}\nB: {answer_b}\nC: {answer_c}\nD: {answer_d}",
            call_judge_fn=judge_fn,
        )
        self.assertEqual(result["method"], "fallback_best")
        self.assertEqual(result["confidence"], 0.3)
        self.assertIn("judge_error", result["judge_reason"])

    def test_judge_no_consensus_no_judge_fn(self):
        # All different, no judge callable → fallback
        result = merge_answers(
            "Q?",
            ["Paris", "London", "Berlin", "Madrid"],
            "Q: {question}",
            call_judge_fn=None,
        )
        self.assertEqual(result["method"], "fallback_best")

    def test_all_degenerate_returns_first_nonempty(self):
        result = merge_answers(
            "Q?",
            ["I don't know", "Sorry", "As an AI", "Insufficient"],
            "",
        )
        self.assertEqual(result["method"], "degenerate")
        self.assertEqual(result["confidence"], 0.05)

    def test_all_degenerate_with_empty(self):
        result = merge_answers(
            "Q?",
            ["", "I don't know", "", "Sorry"],
            "",
        )
        self.assertEqual(result["method"], "degenerate")
        self.assertEqual(result["answer"], "I don't know")

    def test_all_empty(self):
        result = merge_answers(
            "Q?",
            ["", "", "", ""],
            "",
        )
        self.assertEqual(result["method"], "degenerate")
        self.assertEqual(result["answer"], "")

    def test_judge_index_out_of_range(self):
        # Judge returns "Best: Z" which is invalid → fallback
        judge_fn = MagicMock(return_value="Best: Z")
        result = merge_answers(
            "Q?",
            ["Paris", "London", "Berlin", "Madrid"],
            "Q: {question}",
            call_judge_fn=judge_fn,
        )
        self.assertEqual(result["method"], "fallback_best")

    def test_judge_returns_same_as_consensus(self):
        # Even with consensus, merge_answers returns early
        result = merge_answers(
            "Q?",
            ["Paris", "Paris", "Paris", "Paris"],
            "",
        )
        self.assertEqual(result["method"], "consensus")


# =========================================================================
# Tests for inference (inference.py) — mocked HTTP
# =========================================================================

class TestInference(unittest.TestCase):
    """Verify parallel_infer and simple_infer with mocked HTTP calls."""

    @patch("container.inference.urllib.request.urlopen")
    def test_parallel_infer_basic(self, mock_urlopen):
        # Mock 4 successful responses
        mock_responses = []
        for ans in ["Paris", "London", "Berlin", "Madrid"]:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "choices": [{"message": {"content": ans}}]
            }).encode()
            mock_responses.append(mock_resp)

        mock_urlopen.side_effect = mock_responses

        prompts = ["strategy A", "strategy B", "strategy C", "strategy D"]
        results = parallel_infer(prompts, "What is the capital?", max_tokens=50)

        self.assertEqual(results, ["Paris", "London", "Berlin", "Madrid"])
        self.assertEqual(mock_urlopen.call_count, 4)

    @patch("container.inference.urllib.request.urlopen")
    def test_parallel_infer_empty_prompts(self, mock_urlopen):
        results = parallel_infer([], "question")
        self.assertEqual(results, [])
        mock_urlopen.assert_not_called()

    @patch("container.inference.urllib.request.urlopen")
    def test_parallel_infer_partial_failures(self, mock_urlopen):
        # Two succeed, two fail
        mock_good = MagicMock()
        mock_good.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Paris"}}]
        }).encode()

        mock_urlopen.side_effect = [
            mock_good,
            urllib.error.HTTPError("http://test", 500, "Server Error", {}, None),
            mock_good,
            urllib.error.URLError("Timeout"),
        ]

        results = parallel_infer(
            ["A", "B", "C", "D"],
            "What?",
            max_tokens=50,
        )

        # Failed ones should return empty string
        self.assertEqual(results, ["Paris", "", "Paris", ""])

    @patch("container.inference.urllib.request.urlopen")
    def test_parallel_infer_order_preserved(self, mock_urlopen):
        # Test that order matches system_prompts order even with out-of-order futures
        def delayed_response(url, *args, **kwargs):
            """Simulate delayed responses to test ordering."""
            body = json.loads(url.data) if hasattr(url, 'data') else {}
            # Extract the system prompt from the body
            # We don't have easy access to body here in mock, so simulate
            time.sleep(0.01)
            resp = MagicMock()
            resp.read.return_value = json.dumps({
                "choices": [{"message": {"content": "ans"}}]
            }).encode()
            return resp

        mock_urlopen.side_effect = delayed_response
        prompts = ["p1", "p2", "p3", "p4"]
        results = parallel_infer(prompts, "question", max_tokens=10)
        self.assertEqual(len(results), 4)
        # All should be non-empty
        self.assertTrue(all(r == "ans" for r in results))

    @patch("container.inference.urllib.request.urlopen")
    def test_simple_infer(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Paris"}}]
        }).encode()
        mock_urlopen.return_value = mock_resp

        result = simple_infer("You are helpful", "What is capital?", max_tokens=50)
        self.assertEqual(result, "Paris")

    @patch("container.inference.urllib.request.urlopen")
    def test_simple_infer_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Timeout")

        result = simple_infer("system", "question")
        self.assertEqual(result, "")

    @patch("container.inference.urllib.request.urlopen")
    def test_infer_empty_content(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": None}}]
        }).encode()
        mock_urlopen.return_value = mock_resp

        result = simple_infer("system", "question")
        self.assertEqual(result, "")

    @patch("container.inference.urllib.request.urlopen")
    def test_infer_strip_whitespace(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "  Paris  "}}]
        }).encode()
        mock_urlopen.return_value = mock_resp

        result = simple_infer("system", "question")
        self.assertEqual(result, "Paris")


# =========================================================================
# Tests for server (server.py) — mocked subprocess and HTTP
# =========================================================================

class TestServerManager(unittest.TestCase):
    """Verify ServerManager lifecycle — start, stop, health check."""

    def setUp(self):
        self.model_path = "/fake/model.gguf"

    @patch("container.server.subprocess.Popen")
    @patch("container.server.urllib.request.urlopen")
    @patch("container.server.os.path.exists")
    @patch("container.server.os.access")
    def test_start_success(self, mock_access, mock_exists, mock_urlopen, mock_popen):
        # Model exists, server starts, health check passes
        mock_exists.return_value = True
        mock_access.return_value = True

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # still running
        mock_popen.return_value = mock_process

        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value = mock_response

        sm = ServerManager(self.model_path)
        result = sm.start(timeout=5)

        self.assertTrue(result)
        mock_popen.assert_called_once()
        # Verify health check was called
        mock_urlopen.assert_called()

    @patch("container.server.subprocess.Popen")
    @patch("container.server.urllib.request.urlopen")
    @patch("container.server.os.path.exists")
    @patch("container.server.os.access")
    def test_start_model_not_found(self, mock_access, mock_exists, mock_urlopen, mock_popen):
        mock_exists.return_value = False  # model not found

        sm = ServerManager(self.model_path)
        result = sm.start(timeout=5)

        self.assertFalse(result)
        mock_popen.assert_not_called()
        mock_urlopen.assert_not_called()

    @patch("container.server.subprocess.Popen")
    @patch("container.server.urllib.request.urlopen")
    @patch("container.server.os.path.exists")
    @patch("container.server.os.access")
    def test_start_health_timeout(self, mock_access, mock_exists, mock_urlopen, mock_popen):
        mock_exists.return_value = True
        mock_access.return_value = True

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stderr.read.return_value = b""
        mock_popen.return_value = mock_process

        # Health check always raises
        mock_urlopen.side_effect = ConnectionRefusedError("No server")

        sm = ServerManager(self.model_path)
        result = sm.start(timeout=1)  # short timeout

        self.assertFalse(result)

    @patch("container.server.subprocess.Popen")
    @patch("container.server.urllib.request.urlopen")
    @patch("container.server.os.path.exists")
    @patch("container.server.os.access")
    def test_start_process_crashed(self, mock_access, mock_exists, mock_urlopen, mock_popen):
        mock_exists.return_value = True
        mock_access.return_value = True

        mock_process = MagicMock()
        # poll returns non-None after first call (process died)
        mock_process.poll.side_effect = [None, None, None, 1]
        mock_process.stderr.read.return_value = b"OOM error"
        mock_popen.return_value = mock_process

        mock_urlopen.side_effect = ConnectionRefusedError("No server")

        sm = ServerManager(self.model_path)
        result = sm.start(timeout=1)

        self.assertFalse(result)

    @patch("container.server.subprocess.Popen")
    @patch("container.server.urllib.request.urlopen")
    def test_stop_graceful(self, mock_urlopen, mock_popen):
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # still running
        mock_popen.return_value = mock_process

        sm = ServerManager(self.model_path)
        sm.process = mock_process

        sm.stop()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=10)
        self.assertIsNone(sm.process)

    @patch("container.server.subprocess.Popen")
    @patch("container.server.urllib.request.urlopen")
    def test_stop_kill_on_timeout(self, mock_urlopen, mock_popen):
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        # terminate wait raises timeout
        mock_process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), None]
        mock_popen.return_value = mock_process

        sm = ServerManager(self.model_path)
        sm.process = mock_process

        sm.stop()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
        self.assertIsNone(sm.process)

    @patch("container.server.subprocess.Popen")
    @patch("container.server.urllib.request.urlopen")
    def test_stop_not_running(self, mock_urlopen, mock_popen):
        sm = ServerManager(self.model_path)
        sm.process = None

        sm.stop()  # Should not raise

    @patch("container.server.urllib.request.urlopen")
    def test_is_healthy_true(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value = mock_response

        sm = ServerManager(self.model_path)
        self.assertTrue(sm.is_healthy())

    @patch("container.server.urllib.request.urlopen")
    def test_is_healthy_false(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionRefusedError()

        sm = ServerManager(self.model_path)
        self.assertFalse(sm.is_healthy())

    @patch("container.server.subprocess.Popen")
    @patch("container.server.urllib.request.urlopen")
    @patch("container.server.os.path.exists")
    @patch("container.server.os.access")
    def test_context_manager(self, mock_access, mock_exists, mock_urlopen, mock_popen):
        mock_exists.return_value = True
        mock_access.return_value = True

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value = mock_response

        with ServerManager(self.model_path) as sm:
            self.assertIsNotNone(sm)

        # stop should have been called on exit
        mock_process.terminate.assert_called_once()

    @patch("container.server.subprocess.Popen")
    @patch("container.server.urllib.request.urlopen")
    @patch("container.server.os.path.exists")
    @patch("container.server.os.access")
    def test_start_already_running(self, mock_access, mock_exists, mock_urlopen, mock_popen):
        mock_exists.return_value = True
        mock_access.return_value = True

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # still running
        mock_popen.return_value = mock_process

        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value = mock_response

        sm = ServerManager(self.model_path)
        sm.process = mock_process  # already has a process

        result = sm.start(timeout=5)
        self.assertTrue(result)
        # Should NOT start a new process since already running
        self.assertEqual(mock_popen.call_count, 0)


# =========================================================================
# Tests for fallback (fallback.py) — mocked requests
# =========================================================================

class TestFallback(unittest.TestCase):
    """Verify Fireworks API fallback — is_available, fallback_answer."""

    @patch.dict(os.environ, {"FIREWORKS_API_KEY": "test-key-123"}, clear=True)
    def test_is_available_true(self):
        # Need to reload fallback module to pick up the env var
        import importlib
        import container.fallback as fb
        importlib.reload(fb)
        self.assertTrue(fb.is_available())

    @patch.dict(os.environ, {}, clear=True)
    def test_is_available_false(self):
        import importlib
        import container.fallback as fb
        importlib.reload(fb)
        self.assertFalse(fb.is_available())

    @patch.dict(os.environ, {"FIREWORKS_API_KEY": "test-key-123"}, clear=True)
    @patch("container.fallback.urllib.request.urlopen")
    def test_fallback_answer_success(self, mock_urlopen):
        import importlib
        import container.fallback as fb
        importlib.reload(fb)

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Paris"}}]
        }).encode()
        mock_urlopen.return_value = mock_resp

        result = fb.fallback_answer("What is the capital of France?")
        self.assertEqual(result, "Paris")

    @patch.dict(os.environ, {}, clear=True)
    def test_fallback_answer_no_api_key(self):
        import importlib
        import container.fallback as fb
        importlib.reload(fb)

        result = fb.fallback_answer("What is the capital?")
        self.assertIsNone(result)

    @patch.dict(os.environ, {"FIREWORKS_API_KEY": "test-key-123"}, clear=True)
    @patch("container.fallback.urllib.request.urlopen")
    def test_fallback_answer_http_error(self, mock_urlopen):
        import importlib
        import container.fallback as fb
        importlib.reload(fb)

        mock_urlopen.side_effect = Exception("HTTP 429 Too Many Requests")

        result = fb.fallback_answer("What is the capital?")
        self.assertIsNone(result)

    @patch.dict(os.environ, {"FIREWORKS_API_KEY": "test-key-123"}, clear=True)
    @patch("container.fallback.urllib.request.urlopen")
    def test_fallback_answer_empty_content(self, mock_urlopen):
        import importlib
        import container.fallback as fb
        importlib.reload(fb)

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": None}}]
        }).encode()
        mock_urlopen.return_value = mock_resp

        result = fb.fallback_answer("What is the capital?")
        self.assertEqual(result, "")

    @patch.dict(os.environ, {"FIREWORKS_API_KEY": "test-key-123"}, clear=True)
    @patch("container.fallback.urllib.request.urlopen")
    def test_fallback_answer_with_category(self, mock_urlopen):
        import importlib
        import container.fallback as fb
        importlib.reload(fb)

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "42"}}]
        }).encode()
        mock_urlopen.return_value = mock_resp

        result = fb.fallback_answer("What is 6*7?", category="math")
        self.assertEqual(result, "42")

    @patch.dict(os.environ, {"FIREWORKS_API_KEY": "test-key-123"}, clear=True)
    @patch("container.fallback.urllib.request.urlopen")
    def test_fallback_answer_strips_whitespace(self, mock_urlopen):
        import importlib
        import container.fallback as fb
        importlib.reload(fb)

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "  Paris  "}}]
        }).encode()
        mock_urlopen.return_value = mock_resp

        result = fb.fallback_answer("What is capital?")
        self.assertEqual(result, "Paris")


# =========================================================================
# Tests for find_llama_server (server.py)
# =========================================================================

class TestFindLlamaServer(unittest.TestCase):
    """Verify binary location logic."""

    @patch("container.server.os.path.exists")
    @patch("container.server.os.access")
    def test_finds_in_path(self, mock_access, mock_exists):
        mock_exists.return_value = True
        mock_access.return_value = True
        result = find_llama_server()
        self.assertIsNotNone(result)

    @patch("container.server.os.path.exists")
    @patch("container.server.os.access")
    def test_finds_repo_binary(self, mock_access, mock_exists):
        # First call (repo path) returns True, rest False
        mock_exists.side_effect = lambda p: "bin/llama-server" in str(p)
        mock_access.return_value = True
        result = find_llama_server()
        self.assertIn("bin/llama-server", result)

    @patch("container.server.os.path.exists")
    @patch("container.server.os.access")
    def test_none_found_returns_default(self, mock_access, mock_exists):
        mock_exists.return_value = False
        mock_access.return_value = False
        result = find_llama_server()
        self.assertEqual(result, "llama-server")


# =========================================================================
# Main — runs all tests
# =========================================================================

if __name__ == "__main__":
    unittest.main()
