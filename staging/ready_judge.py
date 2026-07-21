"""ReadyJudge — Collects 5 answers per task, applies majority vote.

PIPELINE:
  1. Worker sends 5 answers (different prompts/temps) per task
  2. Judge groups answers by fuzzy-match similarity
  3. Large group (≥3) → majority vote winner → done
  4. Weak/no majority → escalate to Fireworks API (best model per fw_router)
  5. If Fireworks unavailable → tiebreaker fallback

This mirrors the existing container/consensus.py pattern but simplifies
the judge step: instead of a model-based judge, we go straight to
Fireworks when consensus is weak.
"""

import logging
import re
import time
from collections import defaultdict
from typing import Optional

from staging.ready_config import ReadyConfig

logger = logging.getLogger(__name__)


# ── Answer normalization for vote grouping ──

def _normalize_answer(answer: str) -> str:
    """Canonical form for vote counting."""
    text = answer.strip()
    if not text:
        return ""
    text = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return re.sub(r"\s+", " ", text)


def _token_overlap(a: str, b: str) -> float:
    """Fraction of tokens in common (0.0-1.0)."""
    tokens_a = set(_normalize_answer(a).split())
    tokens_b = set(_normalize_answer(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))


def fuzzy_match_answers(a: str, b: str) -> bool:
    """Cascade: exact → normalized → numeric tolerance → token overlap."""
    if a.strip() == b.strip():
        return True
    na, nb = _normalize_answer(a), _normalize_answer(b)
    if na == nb:
        return True
    if not na or not nb:
        return False
    # Numeric tolerance (1%)
    nums_a = re.findall(r"-?\d+(?:\.\d+)?", a)
    nums_b = re.findall(r"-?\d+(?:\.\d+)?", b)
    if nums_a and nums_b and len(nums_a) == len(nums_b):
        try:
            if all(
                abs(float(va) - float(vb)) / max(abs(float(vb)), 1) <= 0.01
                for va, vb in zip(nums_a, nums_b)
            ):
                return True
        except (ValueError, ZeroDivisionError):
            pass
    # Token overlap ≥ 50%
    if _token_overlap(a, b) >= 0.5:
        return True
    return False


def _is_degenerate(answer: str) -> bool:
    """Check if answer is empty/hedge/unhelpful.

    Short numeric answers (e.g. '42') are valid — exempt from length check.
    """
    t = answer.strip()
    if not t:
        return True
    # Short numeric answers are valid (math, counts, etc.)
    if re.match(r"^-?\d+(?:\.\d+)?$", t):
        return False
    if len(t) < 3:
        return True
    low = t.lower()
    patterns = [
        r"\bi don'?t know\b", r"\bi cannot\b", r"\bas an ai\b",
        r"\bsorry\b", r"\bno information\b", r"\bnot enough information\b",
        r"\binsufficient\b",
    ]
    return any(re.search(p, low) for p in patterns)


class ReadyJudge:
    """Collects answers per task, applies voting, escalates to Fireworks when needed."""

    def __init__(self, config: ReadyConfig):
        self.config = config
        self._task_answers: dict[str, list[dict]] = defaultdict(list)
        self._judged: dict[str, dict] = {}
        # Track when first answer was received per task (for timeout fallback)
        self._task_first_answer_time: dict[str, float] = {}
        # Dynamic tracking — not hardcoded
        self._active_worker_types: set[str] = set()
        self._worker_timing: dict[str, list[float]] = {}  # worker_type → [timing_ms, ...]
        self.total_expected_answers = config.judgment_votes
        self.deadline_emergency = None  # Set externally by monitor
        # Fireworks solver (lazy-loaded if escalation needed)
        self._fw_solver = None

    # ── Answer collection ──

    def add_answer(self, answer: dict) -> None:
        """Record one worker's answer for a task.

        answer dict: {worker_id, worker_type, task_id, answer, timing_ms}
        """
        tid = answer.get("task_id")
        if not tid or "answer" not in answer:
            logger.warning("[judge] Dropping malformed result: missing task_id or answer: %s", answer)
            return
        self._task_answers[tid].append(answer)
        # Track worker type dynamically
        wt = self._get_worker_type(answer)
        self._active_worker_types.add(wt)
        # Track per-worker-type timing
        timing_ms = answer.get("timing_ms", 0) or 0
        if wt not in self._worker_timing:
            self._worker_timing[wt] = []
        self._worker_timing[wt].append(timing_ms)
        # Track when first answer was received (for timeout fallback)
        if tid not in self._task_first_answer_time:
            self._task_first_answer_time[tid] = time.monotonic()

    def count_answers(self, task_id: str) -> int:
        return len(self._task_answers.get(task_id, []))

    def get_answer_details(self, task_id: str) -> list[dict]:
        """Return raw answers for a task including worker_id, model_id, timing, etc."""
        return self._task_answers.get(task_id, [])

    def get_timing_summary(self) -> dict:
        """Return per-worker-type aggregate timing."""
        summary = {}
        for wtype, times in self._worker_timing.items():
            if not times:
                continue
            total_s = sum(times) / 1000.0
            avg_ms = sum(times) / len(times)
            sorted_t = sorted(times)
            p95 = sorted_t[int(len(sorted_t) * 0.95)] if len(sorted_t) > 1 else (sorted_t[0] if sorted_t else 0)
            summary[wtype] = {
                "total_s": round(total_s, 1),
                "count": len(times),
                "avg_ms": round(avg_ms, 1),
                "p95_ms": round(p95, 1),
            }
        return summary

    def _get_worker_type(self, answer: dict) -> str:
        """Extract worker type from an answer dict.

        Preferred: use the ``worker_type`` field set by `ReadyWorker._push_results`.
        Fallback: infer from ``worker_id`` prefix for backward compatibility.
        """
        wt = answer.get("worker_type", "")
        if wt:
            return wt
        wid = answer.get("worker_id", "")
        if wid.startswith("det_"):
            return "deterministic"
        elif wid.startswith("loc_"):
            return "local"
        elif wid.startswith("fw_"):
            return "fireworks"
        return wid  # last-resort: use worker_id itself as type

    def ready_to_judge(self, task_id: str) -> bool:
        """Check if enough votes collected to judge.

        Returns True when EITHER:
          a) Enough non-degenerate answers (threshold=judgment_votes,
             halved under deadline_emergency), AND at least one answer
             from a 'local' worker type (prevents fast DetWorkers from
             dominating judgment before LocWorker GPU results arrive), OR
          b) All answers are empty AND every active worker type has tried, OR
          c) Timeout elapsed (30s normally, 15s under deadline_emergency).
        """
        answers = self._task_answers.get(task_id, [])
        count = len(answers)
        if count < self.config.judgment_votes:
            return False

        non_degenerate = [
            a for a in answers
            if not _is_degenerate(a.get("answer", ""))
        ]

        # Determine effective threshold (halved under deadline emergency)
        threshold = self.config.judgment_votes
        if self.deadline_emergency and hasattr(self.deadline_emergency, 'value') and self.deadline_emergency.value:
            threshold = max(1, threshold // 2)

        # Primary: enough non-degenerate answers
        if len(non_degenerate) >= threshold:
            # ── Judge Speed Dominance Fix ──────────────────────────
            # Require at least one answer from a 'local' worker type to
            # prevent fast DetWorkers (microsecond answers) from judging
            # before LocWorkers (~2.5s GPU inference) can contribute.
            # Without this, LocWorker answers arrive after judgment
            # and never influence the vote.
            has_local = any(
                self._get_worker_type(a) == "local"
                for a in answers
            )
            if has_local or self.config.loc_workers == 0:
                return True
            logger.debug(
                "[judge] Waiting for local worker answer on %s "
                "(have %d/%d non-degenerate, %d total) — "
                "will fall back to timeout if local never arrives",
                task_id, len(non_degenerate), threshold, count,
            )
            # Fall through to timeout / all-empty checks below

        # Secondary: all active worker types contributed (all failed)
        task_types = set(self._get_worker_type(a) for a in answers)
        all_empty = all(
            _is_degenerate(a.get("answer", ""))
            for a in answers
        )
        if all_empty and task_types == self._active_worker_types:
            logger.warning(
                "[judge] All answers degenerate for %s — forcing judgment "
                "(types=%s, votes=%d)",
                task_id, task_types, count,
            )
            return True

        # Timeout fallback (reduced under deadline emergency)
        timeout = 15.0 if (self.deadline_emergency and
                           hasattr(self.deadline_emergency, 'value') and
                           self.deadline_emergency.value) else self.config.judge_timeout_s
        first_time = self._task_first_answer_time.get(task_id)
        if first_time is not None and (time.monotonic() - first_time) >= timeout:
            logger.warning(
                "[judge] Timeout for %s — forcing judgment after %.0fs "
                "(types=%s, votes=%d)",
                task_id, timeout, task_types, count,
            )
            return True

        return False

    # ── Judgment ──

    def judge(self, task_id: str) -> tuple[str, dict]:
        """Apply voting + optional Fireworks escalation.

        Returns (final_answer, metadata).
        """
        answers = self._task_answers.get(task_id, [])
        texts = [a.get("answer", "") for a in answers]
        non_empty = [t for t in texts if t.strip() and not _is_degenerate(t)]

        if not non_empty:
            meta = {"strategy": "all_failed", "answer": "", "votes_cast": len(answers), "total": self.config.judgment_votes, "largest_group": 0, "num_groups": 0}
            self._judged[task_id] = meta
            return "", meta

        # ── Step 1: Group answers by similarity ──
        groups = self._group_answers(texts)
        if not groups:
            meta = {"strategy": "all_failed", "answer": "", "votes_cast": len(answers), "total": self.config.judgment_votes, "largest_group": 0, "num_groups": 0}
            self._judged[task_id] = meta
            return "", meta

        sorted_groups = sorted(groups.items(), key=lambda g: len(g[1]), reverse=True)
        largest = sorted_groups[0]
        largest_size = len(largest[1])
        largest_answer = largest[0]
        total = len(texts)

        # ── Step 2: Routing decision ──
        strategy = ""

        if largest_size >= 3:
            # Strong majority → accept
            strategy = "majority_3plus"
            final_answer = largest_answer

        elif largest_size == 2 and total >= 4:
            # Moderate majority → accept with medium confidence
            strategy = "majority_2plus"
            final_answer = largest_answer

        elif largest_size == 2 and total == 3:
            # 2 agree, 1 disagrees → majority wins
            strategy = "majority_2of3"
            final_answer = largest_answer

        elif largest_size == 2 and total == 2:
            # 50/50 split → escalate to Fireworks
            strategy = "escalate_fireworks"
            final_answer = self._escalate_to_fireworks(task_id, answers, texts)

        else:
            # All different → escalate to Fireworks
            strategy = "escalate_fireworks"
            final_answer = self._escalate_to_fireworks(task_id, answers, texts)

        # Re-check: if escalation returned empty, fallback to best available
        if not final_answer.strip():
            strategy = "fallback_best"
            best = next((t for t in non_empty if not _is_degenerate(t)), texts[0])
            final_answer = best

        meta = {
            "task_id": task_id,
            "answer": final_answer,
            "strategy": strategy,
            "votes_cast": total,
            "largest_group": largest_size,
            "num_groups": len(groups),
            "group_sizes": {answer: len(indices) for answer, indices in groups.items()},
            "vote_distribution": [
                {"worker_id": a.get("worker_id", ""), "answer": a.get("answer", "")}
                for a in answers
            ],
        }
        self._judged[task_id] = meta
        return final_answer, meta

    def judge_all(self) -> list[dict]:
        """Judge all pending tasks and return final results list."""
        for tid in list(self._task_answers.keys()):
            if tid not in self._judged:
                self.judge(tid)

        results = []
        for tid, meta in self._judged.items():
            results.append({
                "task_id": tid,
                "answer": meta["answer"],
                "_judgment": {
                    "strategy": meta["strategy"],
                    "votes": meta["votes_cast"],
                    "largest_group": meta["largest_group"],
                    "num_groups": meta["num_groups"],
                    "group_sizes": meta.get("group_sizes", {}),
                    "vote_distribution": meta.get("vote_distribution", []),
                },
            })
        return results

    def is_judged(self, task_id: str) -> bool:
        return task_id in self._judged

    @property
    def total_judged(self) -> int:
        return len(self._judged)

    @property
    def pending_tasks(self) -> list[str]:
        return [tid for tid in self._task_answers if tid not in self._judged]

    def ingest_results(self, results_queue) -> int:
        """Pull available results from the shared queue. Returns count ingested."""
        import queue as _queue
        count = 0
        try:
            while True:
                result = results_queue.get_nowait()
                self.add_answer(result)
                count += 1
        except _queue.Empty:
            pass
        return count

    def consume_loop(self, results_queue, deadline_emergency=None,
                     stop_event=None) -> None:
        """Autonomous loop: pull results, judge, repeat.

        Runs in its own daemon thread. Decoupled from pool.
        """
        self.deadline_emergency = deadline_emergency

        while not (stop_event and stop_event.is_set()):
            ingested = self.ingest_results(results_queue)

            # Try to judge any ready tasks
            for tid in list(self.pending_tasks):
                if self.ready_to_judge(tid):
                    self.judge(tid)

            # Sleep briefly if no new results
            if ingested == 0:
                timeout = 0.05 if not (deadline_emergency and
                                        deadline_emergency.value) else 0.02
                time.sleep(timeout)

    # ── Internal helpers ──

    def _group_answers(self, texts: list[str]) -> dict[str, list[int]]:
        """Group answer texts by fuzzy-match similarity.

        Returns dict[canonical_answer → list of indices].
        """
        groups: list[tuple[str, list[int]]] = []

        for idx, text in enumerate(texts):
            if not text.strip() or _is_degenerate(text):
                continue
            matched = False
            for i, (canonical, indices) in enumerate(groups):
                if fuzzy_match_answers(text, canonical):
                    groups[i][1].append(idx)
                    matched = True
                    break
            if not matched:
                groups.append((text, [idx]))

        return {c: indices for c, indices in groups}

    def _escalate_to_fireworks(self, task_id: str, answers: list[dict], texts: list[str]) -> str:
        """Escalate to Fireworks API when no consensus.

        Uses fw_router for optimal model selection, then calls Fireworks.
        """
        if not self.config.fw_api_key:
            logger.warning("[judge] No Fireworks API key — skipping escalation for %s", task_id)
            return ""

        try:
            from agent.solvers.fireworks import FireworksSolver
            from agent.solvers.fw_router import route

            # Get category and prompt from the first answer (attached by worker)
            category = answers[0].get("category", "") if answers else ""
            prompt = answers[0].get("prompt", texts[0]) if answers else texts[0]

            if not prompt:
                return ""

            # Use fw_router to get optimal config
            cfg = route(category, prompt, 0.5)

            solver = FireworksSolver(api_key=self.config.fw_api_key)
            logger.info("[judge] Escalating %s to Fireworks (model=%s, category=%s)",
                        task_id, cfg.model_id, category)

            result = solver.solve(
                cfg.model_id,
                cfg.system_prompt,
                prompt,
                max_tokens=cfg.max_tokens,
                temperature=0.0,
                prefill=cfg.prefill,
                task_type=category,
                timeout=int(self.config.worker_timeout_s),
            )
            return result

        except ImportError as exc:
            logger.warning("[judge] Fireworks solver not available: %s", exc)
            return ""
        except Exception as exc:
            logger.warning("[judge] Fireworks escalation failed for %s: %s", task_id, exc)
            return ""

    def _fallback_best(self, texts: list[str]) -> str:
        """Pick the best available answer when all else fails."""
        non_degen = [t for t in texts if not _is_degenerate(t)]
        if non_degen:
            return non_degen[0]
        non_empty = [t for t in texts if t.strip()]
        return non_empty[0] if non_empty else ""
