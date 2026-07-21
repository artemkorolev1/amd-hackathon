#!/usr/bin/env python3
"""Mutation agent — applies genetic operators to cells to create the next generation.

Operators are task-aware: different prompt styles and mutations work better
for different task types.  The agent can also accept analysis tags to bias
mutations (e.g. "too verbose" → bias toward shorter prompts).
"""

from __future__ import annotations

import random
import re
from typing import Optional

from agent.cell import Cell, DecodingConfig, StepConfig, TASK_LABELS


# ── Parameter pools for mutation ────────────────────────────────────────────
TOP_P_OPTIONS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
TOP_K_OPTIONS = [10, 20, 40, 80, 100]
MIN_P_OPTIONS = [0.0, 0.01, 0.02, 0.05, 0.1]
REPEAT_PENALTY_OPTIONS = [1.0, 1.05, 1.1, 1.15, 1.2]


# ── Prompt component pools (task-specific) ──────────────────────────────────


# Prefixes that work across tasks
PREFIXES = {
    "factual": ["", "Fact:", "Answer:", "Answer directly:"],
    "math": ["", "Math:", "Calculate:", "Answer:"],
    "sentiment": ["", "Sentiment:", "Classify:"],
    "summarization": ["", "Summary:", "TL;DR:", "Brief:"],
    "ner": ["", "Entities:", "NER:", "Extract:"],
    # Default fallback
    "__default__": ["", "Answer:", "Direct:"],
}

# Instruction snippets keyed by task
INSTRUCTIONS = {
    "factual": [
        "Answer the question directly.",
        "Use exact names, dates, and numbers.",
        "Be concise.",
        "Be precise.",
        "No preamble.",
    ],
    "math": [
        "Solve step by step.",
        "Output only the final answer.",
        "Show your working briefly.",
        "Use standard decimal format.",
    ],
    "sentiment": [
        "Classify as positive, negative, neutral, or mixed.",
        "Watch for sarcasm and hedging.",
        "Output exactly one word.",
    ],
    "summarization": [
        "Summarize in 1-2 sentences.",
        "Use exact names and numbers.",
        "Capture the core event.",
        "Obey any length constraints.",
    ],
    "ner": [
        "Extract all named entities.",
        "Label as PERSON, ORG, LOC, DATE.",
        "Only include entities explicitly in the text.",
    ],
}

# Constraint pool (shared across tasks)
CONSTRAINT_POOL = [
    "No preamble.",
    "No explanation.",
    "No commentary.",
    "Keep under 15 words.",
    "Keep under 5 words.",
    "Be specific.",
    "Don't hedge.",
    "If unsure, give your best guess.",
    "Output only the answer.",
    "Address all parts of the question.",
]

# Rephrase thesaurus (shared)
THESAURUS = {
    "Answer the question directly.": [
        "Respond directly to the question.",
        "Give a direct answer.",
        "Answer straight.",
        "Answer clearly.",
    ],
    "Be concise.": [
        "Keep it brief.",
        "Be brief.",
        "Answer concisely.",
    ],
    "Use exact names, dates, and numbers.": [
        "Give precise names, dates, and numbers.",
        "Use exact facts and figures.",
        "Provide specific details.",
    ],
}

# Verbosity switch pools
SHORT_STYLES = ["Be concise.", "Be brief.", "Keep it short."]
MEDIUM_STYLES = ["Answer clearly.", "Provide a clear answer.", "Respond with the answer."]

# Temperature options
TEMP_OPTIONS = [0.0, 0.1, 0.2]

MAX_PROMPT_CHARS = 200


# ── Helpers ─────────────────────────────────────────────────────────────────

def _truncate_prompt(text: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    if not truncated.endswith("."):
        truncated += "."
    return truncated


def _task_prefixes(cell: Cell) -> list[str]:
    """Return the prefix pool for this cell's task."""
    cat = cell.task_label
    return PREFIXES.get(cat, PREFIXES["__default__"])


def _task_instructions(cell: Cell) -> list[str]:
    cat = cell.task_label
    return INSTRUCTIONS.get(cat, INSTRUCTIONS["factual"])


# ── Workflow templates registry ─────────────────────────────────────────────

WORKFLOW_TEMPLATES = {
    "single_shot": None,  # No steps (default)
    "plan_solve": [
        StepConfig(name="plan", system_prompt="Analyze the problem.", input_from="_input"),
        StepConfig(name="solve", system_prompt="Output the answer concisely.", input_from="plan"),
    ],
    "analyze_answer": [
        StepConfig(name="analyze", system_prompt="Reason step by step.", input_from="_input"),
        StepConfig(name="answer", system_prompt="Based on your analysis, output the final answer only.", input_from="analyze"),
    ],
    "verify": [
        StepConfig(name="draft", system_prompt="Answer the question.", input_from="_input"),
        StepConfig(name="verify", system_prompt="Verify the answer above. Fix any mistakes and output the corrected final answer.", input_from="draft"),
    ],
}


# ── Task-specific prompts for workflow operators ────────────────────────────

# Answer prompts for the second step of operator 15 (split into steps)
SPLIT_ANSWER_PROMPTS = {
    "factual": "Based on your analysis, answer the question directly.",
    "math": "Based on your analysis, output the final answer only.",
    "sentiment": "Based on your analysis, output exactly one word: positive, negative, neutral, or mixed.",
    "summarization": "Based on your analysis, produce the summary concisely.",
    "ner": "Based on your analysis, extract all named entities.",
    "code_debug": "Based on your analysis, output the corrected code.",
    "code_gen": "Based on your analysis, output the generated code.",
    "logic": "Based on your analysis, output the logical conclusion.",
}

# Verify prompts for operator 16 (add verification step)
VERIFY_PROMPTS = {
    "factual": "Verify the answer above. Is it factually correct? Correct any mistakes and output the final answer.",
    "math": "Verify the calculation above. Check for arithmetic errors and output the corrected answer in \\boxed{}.",
    "sentiment": "Verify the sentiment. Given the analysis above, is the sentiment positive, negative, neutral, or mixed? Reply with one word.",
    "summarization": "Verify the summary above. Does it capture all key points? Correct any omissions and output the final summary.",
    "ner": "Verify the entities above. Remove any not explicitly in the text and fix incorrect labels.",
    "code_debug": "Verify the fix above. Check for remaining bugs and output the completely corrected code.",
    "code_gen": "Verify the generated code above. Check for errors and output the corrected version.",
    "logic": "Verify the reasoning above. Check for logical fallacies and output the corrected conclusion.",
}


# ── Mutation Agent ──────────────────────────────────────────────────────────

class MutationAgent:
    """Creates new cells from existing ones via genetic operators.

    Operators:
        0  Rephrase: replace a phrase with a synonym
        1  Add constraint: append a random constraint
        2  Remove last sentence
        3  Swap verbosity (short ↔ medium)
        4  Change prefix
        5  Add word limit (keep under N words)
        6  Add preamble guard
        7  Change temperature
        8  Swap model (to a different model key)
        9  Copy decoding params from another cell
       10  Change top_p
       11  Change top_k
       12  Change min_p
       13  Change repeat_penalty
       14  All-params jitter
       15  Split into steps (single-shot → 2-step workflow)
       16  Add verification step
       17  Remove step (merge workflow → single-shot)
       18  Reorder steps (swap adjacent steps)
       19  Swap model per step
    """

    def __init__(self, model_keys: Optional[list[str]] = None, seed: Optional[int] = None):
        self.model_keys = model_keys or []
        if seed is not None:
            random.seed(seed)

    # ── Public API ──────────────────────────────────────────────────────────

    def evolve(
        self,
        parents: list[Cell],
        tags: Optional[list[str]] = None,
        target_size: Optional[int] = None,
        elite_count: int = 2,
        crossover_fraction: float = 0.6,
    ) -> list[Cell]:
        """Create the next generation from a population.

        Args:
            parents: current population.
            tags: optional analysis tags to bias mutation direction.
            target_size: desired population size (default: same as parents).
            elite_count: number of elites to carry over unchanged.
            crossover_fraction: fraction of non-elite slots via crossover (rest mutate).

        Returns:
            New population of cells (includes elites unchanged).
        """
        if target_size is None:
            target_size = len(parents)
        if target_size < elite_count + 1:
            target_size = elite_count + 1

        # 1. Sort parents by accuracy (descending) for elite selection
        sorted_parents = sorted(
            parents,
            key=lambda c: c.metadata.get("accuracy", 0.0) if c.metadata else 0.0,
            reverse=True,
        )

        # 2. Elites (carry over best cells unchanged)
        elites = sorted_parents[:elite_count]
        children = [self._clone_elite(c) for c in elites]

        # 3. Fill remaining slots
        crossover_target = int((target_size - len(children)) * crossover_fraction)
        while len(children) < target_size:
            if len(children) < elite_count + crossover_target:
                # Crossover: pick two parents via tournament
                p1 = self._tournament_select(sorted_parents)
                p2 = self._tournament_select(sorted_parents)
                child = self._crossover(p1, p2, tags)
                # Apply mutation to crossover child
                child = self._mutate(child, tags)
            else:
                # Pure mutation
                parent = self._tournament_select(sorted_parents)
                child = self._mutate(parent, tags)

            # Avoid duplicates
            if not any(c.eq_content(child) for c in children + elites):
                children.append(child)

        # Ensure at least one fresh random cell
        if not any(c.name.startswith("fresh_") for c in children):
            fresh_task = random.choice(sorted(set(c.task_id for c in parents)))
            fresh_model = random.choice(self.model_keys) if self.model_keys else parents[0].model_key
            children.append(self._random_cell(fresh_task, fresh_model))

        return children[:target_size]

    def _mutate(self, parent: Cell, tags: Optional[list[str]] = None) -> Cell:
        """Apply a randomly-selected mutation operator."""
        child = parent.clone(new_name=f"{parent.name}_mut", generation=parent.generation + 1)
        text = child.system_prompt

        # Bias toward certain operators based on tags
        if tags and "verbose" in tags and random.random() < 0.4:
            op = random.choice([2, 3, 5])  # shorten ops
        elif tags and "imprecise" in tags and random.random() < 0.4:
            op = random.choice([0, 4, 6])  # precision ops
        elif tags and "params" in tags and random.random() < 0.4:
            op = random.choice([7, 10, 11, 12, 13, 14])  # parameter ops
        elif tags and "workflow" in tags and random.random() < 0.4:
            op = random.choice([15, 16, 17, 18, 19])  # workflow ops
        else:
            op = random.randint(0, 19)

        # Number of ops to apply (1-2)
        num_ops = 2 if random.random() < 0.3 else 1

        for _ in range(num_ops):
            op_actual = op if _ == 0 else random.randint(0, 19)
            text = self._apply_operator(op_actual, text, child)

        # Clean up
        text = re.sub(r"\s+", " ", text).strip()
        text = _truncate_prompt(text)
        child.system_prompt = text
        return child

    def _crossover(
        self, p1: Cell, p2: Cell, tags: Optional[list[str]] = None
    ) -> Cell:
        """Sentence-level uniform crossover between two parents."""
        s1 = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p1.system_prompt) if s.strip()]
        s2 = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p2.system_prompt) if s.strip()]

        if not s1 and not s2:
            child_text = ""
        elif not s1:
            child_text = " ".join(s2)
        elif not s2:
            child_text = " ".join(s1)
        else:
            child = []
            i = j = 0
            while i < len(s1) and j < len(s2):
                if random.random() < 0.5:
                    child.append(s1[i])
                    i += 1
                else:
                    child.append(s2[j])
                    j += 1
            remaining = s1[i:] if random.random() < 0.5 else s2[j:]
            child.extend(remaining)
            cleaned = [s.rstrip(".!?") for s in child]
            child_text = ". ".join(cleaned)
            if child and not child_text.endswith("."):
                child_text += "."

        # Cross over decoding params
        temp = p1.decoding.temperature if random.random() < 0.5 else p2.decoding.temperature
        mt = p1.decoding.max_tokens if random.random() < 0.5 else p2.decoding.max_tokens
        top_p = p1.decoding.top_p if random.random() < 0.5 else p2.decoding.top_p
        top_k = p1.decoding.top_k if random.random() < 0.5 else p2.decoding.top_k
        min_p = p1.decoding.min_p if random.random() < 0.5 else p2.decoding.min_p
        rp = p1.decoding.repeat_penalty if random.random() < 0.5 else p2.decoding.repeat_penalty

        # Cross over aggregation strategy
        agg = p1.aggregation if random.random() < 0.5 else p2.aggregation

        # Cross over workflow steps
        if p1.steps and p2.steps:
            child_steps = p1.steps if random.random() < 0.5 else p2.steps
        elif p1.steps:
            child_steps = p1.steps
        elif p2.steps:
            child_steps = p2.steps
        else:
            child_steps = None
        # Ensure aggregation matches steps
        if child_steps and agg != "workflow":
            agg = "workflow"
        elif not child_steps and agg == "workflow":
            agg = "single"

        # Cross over model
        mk = p1.model_key if random.random() < 0.8 else p2.model_key

        # Cross over task_id (pick the more relevant one)
        tid = p1.task_id if random.random() < 0.5 else p2.task_id

        return Cell(
            task_id=tid,
            model_key=mk,
            system_prompt=child_text,
            decoding=DecodingConfig(temperature=temp, max_tokens=mt,
                                     top_p=top_p, top_k=top_k, min_p=min_p, repeat_penalty=rp),
            aggregation=agg,
            steps=child_steps,
            name=f"xover_{p1.name}_{p2.name}",
            parent=f"{p1.name}+{p2.name}",
            generation=max(p1.generation, p2.generation) + 1,
        )

    def _random_cell(self, task_id: str, model_key: str) -> Cell:
        """Generate a random cell for a given task and model."""
        cat = TASK_LABELS.get(task_id, "factual")
        prefixes = PREFIXES.get(cat, PREFIXES["__default__"])
        instrs = INSTRUCTIONS.get(cat, INSTRUCTIONS["factual"])

        prefix = random.choice(prefixes)
        parts = [prefix] if prefix else []

        num_instr = random.randint(0, 2)
        if num_instr > 0:
            parts.append(". ".join(random.sample(instrs, min(num_instr, len(instrs)))))

        num_constraints = random.randint(0, 2)
        if num_constraints > 0:
            c = random.sample(CONSTRAINT_POOL, min(num_constraints, len(CONSTRAINT_POOL)))
            if parts:
                parts.append(". ".join(c))
            else:
                parts = c

        text = " ".join(parts).strip()
        text = _truncate_prompt(text)

        return Cell(
            task_id=task_id,
            model_key=model_key,
            system_prompt=text,
            decoding=DecodingConfig(
                temperature=random.choice(TEMP_OPTIONS),
                max_tokens=64,
                top_p=random.choice(TOP_P_OPTIONS),
                top_k=random.choice(TOP_K_OPTIONS),
                min_p=random.choice(MIN_P_OPTIONS),
                repeat_penalty=random.choice(REPEAT_PENALTY_OPTIONS),
            ),
            aggregation="single",
            name=f"fresh_{task_id}_{model_key}",
            generation=0,
        )

    def seed_generation_0(
        self,
        known_good_prompts: Optional[dict[str, list[str]]] = None,
        model_keys: Optional[list[str]] = None,
        cells_per_task: int = 3,
        workflow_probability: float = 0.25,
    ) -> list[Cell]:
        """Build generation 0 from known-good prompts and random variants.

        Args:
            known_good_prompts: dict mapping category name to list of prompt texts.
            model_keys: list of model keys to use.
            cells_per_task: number of random cells per task+model.
            workflow_probability: probability a random cell is a multi-step workflow.
        """
        if known_good_prompts is None:
            known_good_prompts = {
                "factual": ["Fact:", "Answer:", "Answer directly."],
                "math": ["Math:", "Solve step by step.", "Calculate:"],
                "sentiment": ["Sentiment:", "One word: positive, negative, neutral, or mixed."],
                "summarization": ["Summary:", "Summarize in 2 sentences."],
                "ner": ["Entities:", "Extract all named entities. Label as PERSON, ORG, LOC, DATE."],
            }
        if model_keys is None:
            model_keys = self.model_keys or ["qwen2.5-1.5b"]

        population: list[Cell] = []
        task_ids = {
            "factual": "T01", "math": "T02", "sentiment": "T03",
            "summarization": "T04", "ner": "T05",
        }

        for cat_name, prompts in known_good_prompts.items():
            tid = task_ids.get(cat_name, "T01")
            for model_key in model_keys:
                # Add known-good prompts as cells
                for i, prompt_text in enumerate(prompts):
                    population.append(Cell(
                        task_id=tid,
                        model_key=model_key,
                        system_prompt=prompt_text,
                        decoding=DecodingConfig(temperature=0.0, max_tokens=64,
                                                 top_p=1.0, top_k=40, min_p=0.0, repeat_penalty=1.0),
                        aggregation="single",
                        name=f"seed_{cat_name}_{model_key}_{i}",
                        generation=0,
                    ))
                # Add random cells for this task+model
                for _ in range(cells_per_task):
                    if random.random() < workflow_probability:
                        # Create a workflow cell from a random template
                        template_name = random.choice(
                            [k for k in WORKFLOW_TEMPLATES if k != "single_shot"]
                        )
                        template_steps = WORKFLOW_TEMPLATES[template_name]
                        # Deep-copy the template steps so each cell gets its own
                        import copy
                        steps_copy = copy.deepcopy(template_steps)
                        population.append(Cell(
                            task_id=tid,
                            model_key=model_key,
                            system_prompt="",  # workflow cell — prompt is in steps
                            decoding=DecodingConfig(temperature=0.0, max_tokens=64,
                                                     top_p=1.0, top_k=40, min_p=0.0, repeat_penalty=1.0),
                            steps=steps_copy,
                            name=f"seed_wf_{cat_name}_{model_key}_{template_name}_{_}",
                            generation=0,
                        ))
                    else:
                        population.append(self._random_cell(tid, model_key))

        return population

    # ── Internal operators ────────────────────────────────────────────────

    def _apply_operator(self, op: int, text: str, cell: Cell) -> str:
        """Apply a single mutation operator to the text."""
        if op == 0:
            # Rephrase
            for orig, replacements in THESAURUS.items():
                if orig in text:
                    return text.replace(orig, random.choice(replacements), 1)
            # Partial match fallback
            for orig in THESAURUS:
                for word in orig.split():
                    if word in text:
                        break

        elif op == 1:
            # Add constraint
            existing = [c for c in CONSTRAINT_POOL if c.lower() in text.lower()]
            available = [c for c in CONSTRAINT_POOL if c not in existing]
            if available:
                constraint = random.choice(available)
                if text and not text.endswith("."):
                    text += ". "
                text = text + " " + constraint

        elif op == 2:
            # Remove last sentence
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if len(sentences) > 1:
                text = " ".join(sentences[:-1])

        elif op == 3:
            # Swap verbosity
            for s in SHORT_STYLES:
                if s in text:
                    return text.replace(s, random.choice(MEDIUM_STYLES), 1)
            for s in MEDIUM_STYLES:
                if s in text:
                    return text.replace(s, random.choice(SHORT_STYLES), 1)

        elif op == 4:
            # Change prefix
            prefixes = _task_prefixes(cell)
            current_prefix = None
            for p in prefixes:
                if p and text.startswith(p):
                    current_prefix = p
                    break
            options = [p for p in prefixes if p != current_prefix]
            if options:
                new_prefix = random.choice(options)
                if current_prefix:
                    text = text.replace(current_prefix, new_prefix, 1)
                elif new_prefix:
                    text = new_prefix + " " + text

        elif op == 5:
            # Add word limit
            n = random.choice([5, 10, 15])
            for old_n in [5, 10, 15, 20]:
                if f"Keep under {old_n} words." in text:
                    text = text.replace(f"Keep under {old_n} words.", "").strip()
            if text and not text.endswith("."):
                text += "."
            text = text + f" Keep under {n} words."

        elif op == 6:
            # Add preamble guard
            if "No preamble" not in text and "No commentary" not in text:
                if text and not text.endswith("."):
                    text += "."
                text = text + " No preamble. No commentary."

        elif op == 7:
            # Change temperature
            options = [t for t in TEMP_OPTIONS if t != cell.decoding.temperature]
            if options:
                cell.decoding.temperature = random.choice(options)

        elif op == 8:
            # Swap model
            if self.model_keys:
                options = [m for m in self.model_keys if m != cell.model_key]
                if options:
                    cell.model_key = random.choice(options)

        elif op == 9:
            # Change max_tokens
            options = [32, 64, 96, 128, 192]
            current = cell.decoding.max_tokens
            cell.decoding.max_tokens = random.choice([o for o in options if o != current])

        elif op == 10:
            # Change top_p
            current = cell.decoding.top_p
            options = [o for o in TOP_P_OPTIONS if o != current]
            if options:
                cell.decoding.top_p = random.choice(options)

        elif op == 11:
            # Change top_k
            current = cell.decoding.top_k
            options = [o for o in TOP_K_OPTIONS if o != current]
            if options:
                cell.decoding.top_k = random.choice(options)

        elif op == 12:
            # Change min_p
            current = cell.decoding.min_p
            options = [o for o in MIN_P_OPTIONS if o != current]
            if options:
                cell.decoding.min_p = random.choice(options)

        elif op == 13:
            # Change repeat_penalty
            current = cell.decoding.repeat_penalty
            options = [o for o in REPEAT_PENALTY_OPTIONS if o != current]
            if options:
                cell.decoding.repeat_penalty = random.choice(options)

        elif op == 14:
            # All-params jitter — randomise all 4 sampling params at once
            cell.decoding.top_p = random.choice(TOP_P_OPTIONS)
            cell.decoding.top_k = random.choice(TOP_K_OPTIONS)
            cell.decoding.min_p = random.choice(MIN_P_OPTIONS)
            cell.decoding.repeat_penalty = random.choice(REPEAT_PENALTY_OPTIONS)

        elif op == 15:
            # Split into steps — convert single-shot to 2-step workflow
            if cell.steps is None:
                task_cat = cell.task_label
                max_tok = cell.decoding.max_tokens
                half_tokens = max(max_tok // 2, 32)

                answer_prompt = SPLIT_ANSWER_PROMPTS.get(
                    task_cat,
                    "Based on your analysis, output the final answer."
                )

                cell.steps = [
                    StepConfig(name="analyze", system_prompt=cell.system_prompt, input_from="_input"),
                    StepConfig(name="answer", system_prompt=answer_prompt, input_from="analyze"),
                ]
                cell.aggregation = "workflow"
                cell.decoding.max_tokens = half_tokens
                text = "Workflow cell — see steps"

        elif op == 16:
            # Add verification step
            verify_prompt = VERIFY_PROMPTS.get(
                cell.task_label,
                "Verify the output above. Correct any mistakes and output the final answer."
            )

            if cell.steps is None:
                # Convert single-shot to 2-step workflow with verification
                max_tok = cell.decoding.max_tokens
                half_tokens = max(max_tok // 2, 32)
                cell.steps = [
                    StepConfig(name="main", system_prompt=cell.system_prompt, input_from="_input"),
                    StepConfig(name="verify", system_prompt=verify_prompt, input_from="main"),
                ]
                cell.aggregation = "workflow"
                cell.decoding.max_tokens = half_tokens
                text = "Workflow cell — see steps"
            else:
                # Append verification step after the last step
                last_step = cell.steps[-1]
                cell.steps.append(
                    StepConfig(name="verify", system_prompt=verify_prompt, input_from=last_step.name)
                )

        elif op == 17:
            # Remove step — merge workflow back to single-shot
            if cell.steps:
                all_texts = [s.system_prompt for s in cell.steps if s.system_prompt]
                merged = " ".join(all_texts) if all_texts else cell.system_prompt
                merged = _truncate_prompt(re.sub(r"\s+", " ", merged).strip())
                cell.steps = None
                cell.aggregation = "single"
                text = merged

        elif op == 18:
            # Reorder steps — swap two adjacent steps
            if cell.steps and len(cell.steps) >= 3:
                idx = random.randint(0, len(cell.steps) - 2)
                cell.steps[idx], cell.steps[idx + 1] = cell.steps[idx + 1], cell.steps[idx]

        elif op == 19:
            # Swap model per step — assign a different model to one step
            if cell.steps and len(cell.steps) >= 2 and len(self.model_keys) >= 2:
                step_idx = random.randrange(len(cell.steps))
                step = cell.steps[step_idx]
                current = step.model_key or cell.model_key
                options = [m for m in self.model_keys if m != current]
                if options:
                    step.model_key = random.choice(options)

        return _truncate_prompt(re.sub(r"\s+", " ", text).strip())

    # ── Selection ─────────────────────────────────────────────────────────

    def _tournament_select(self, population: list[Cell], k: int = 3) -> Cell:
        """Select a cell via tournament (best accuracy wins)."""
        candidates = random.sample(population, min(k, len(population)))
        return max(candidates, key=lambda c: c.metadata.get("accuracy", 0.0))

    def _clone_elite(self, cell: Cell) -> Cell:
        """Clone a cell as an elite (preserve identity)."""
        return cell.clone(new_name=f"elite_{cell.name}", generation=cell.generation + 1)
