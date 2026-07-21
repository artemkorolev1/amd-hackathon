"""
LSAT-style logical reasoning solver.
Handles argument analysis questions: strengthen, weaken, assumption, inference,
flaw, main_point, explain.

Parses paragraph + question stem + answer choices (A/B/C/D/E),
classifies the question type, extracts argument components,
and scores options using heuristic text analysis.
"""
from __future__ import annotations

import re
from typing import Optional, List, Tuple, Dict, Set
from collections import Counter

__all__ = ["solve_logical_reasoning"]


# ── Question type patterns ──────────────────────────────────────────────
TYPE_PATTERNS: List[Tuple[str, List[str]]] = [
    ("strengthen", [
        r"most\s+strengthen",
        r"best\s+support",
        r"most\s+strongly\s+support",
        r"which\s+of\s+the\s+following.*(?:supports|strengthens)",
        r"strengthen",
        r"provide\s+the\s+most\s+support",
        # LogiQA additions
        r"strongly\s+supports",
        r"best\s+confirms",
        r"supports.*speculation",
        r"most\s+strongly\s+support",
    ]),
    ("weaken", [
        r"(?:most|best)\s+weaken",
        r"cast[s]?\s+(?:the\s+most\s+)?doubt",
        r"best\s+(?:undermine|question|challenge)",
        r"would\s+most\s+(?:weaken|undermine|challenge|question)",
        r"can\s+best\s+question",
        r"weakens?\s+the\s+(?:argument|conclusion)",
        # LogiQA additions
        r"most\s+severely\s+weakened",
        r"weakened\s+the\s+argument",
        r"weakens\s+the\s+above\s+argument",
        r"refute",
        r"strongly\s+questioned",
        r"raise.*doubts",
        r"strongest\s+criticism",
        r"strongly\s+indicate.*incorrect",
        r"best\s+refute",
        r"weakened\s+most\s+effectively",
    ]),
    ("assumption", [
        r"is\s+(?:an|the)\s+assumption",
        r"must\s+be\s+assumed",
        r"depends\s+on\s+(?:the\s+)?(?:assumption|presupposition)",
        r"argument\s+depends",
        r"requires\s+the\s+assumption",
        r"conclusion\s+must\s+be\s+assumed",
        r"which\s+of\s+the\s+following\s+is\s+(?:needed|required)",
        r"argument\s+relies?\s+on\s+the\s+(?:claim|assumption)",
        # LogiQA additions
        r"is\s+the\s+hypothesis",
        r"hypothesis\s+discussed",
    ]),
    ("inference", [
        r"can\s+be\s+properly\s+inferred",
        r"must\s+be\s+true\s+(?:based|given)",
        r"logically\s+(?:follows|implied|deduced)",
        r"most\s+strongly\s+supported\s+by\s+the\s+(?:argument|passage|statements)",
        r"which\s+of\s+the\s+following\s+(?:can\s+be\s+)?inferred",
        r"conclusion\s+(?:can\s+be\s+)?(?:properly\s+)?drawn",
        r"follows\s+logically",
        r"most\s+logically\s+completes",
        # LogiQA additions
        r"must\s+be\s+true",
        r"must\s+be\s+false",
        r"must\s+be\s+correct",
        r"cannot\s+be\s+false",
        r"can\s+be\s+concluded",
        r"concluded\s+that",
        r"can\s+be\s+logically\s+derived",
        r"logically\s+derived",
        r"logical\s+inference",
        r"logical\s+corollary",
        r"it\s+can\s+be\s+concluded",
        r"conclusions?\s+can\s+be\s+made\s+based\s+on",
        r"implied\s+meaning",
        r"may\s+be\s+true",
        r"must\s+be\s+ranked",
        r"who\s+must\s+be\b",
        r"which\s+of\s+the\s+following\s+must\s+be",
        r"which\s+of\s+the\s+following\s+is\s+true",
        r"not\s+the\s+implied\s+meaning",
        r"can\s+most\s+logically\s+complete",
        r"most\s+likely\s+to\s+understand",
        r"correctly\s+describes",
        r"following\s+cannot\s+be\s+false",
        r"cannot\s+be\s+derived",
        r"logically\s+complete",
        r"who\s+won\s+the\s+championship",
        r"most\s+similar\s+to\s+the\s+above\s+argument",
        r"same\s+as\s+the\s+method",
        r"same\s+form\s+as",
        r"similar\s+to\s+the\s+above",
        r"speaker.*must\s+be",
        r"be\s+concluded\s+that",
        r"drawn\s+logically",
        r"magicians?\s+must\s+be",
        r"determine\s+which\s+of\s+the\s+following.*true",
        r"lists?\s+(?:a\s+)?complete\s+and\s+accurate\s+list",
        r"lists?\s+are\s+acceptable\s+arrangements",
        r"lists?\s+all\s+speakers",
        r"each\s+of\s+the\s+following.*except",
        r"differences\s+between",
        r"demonstration\s+technique",
    ]),
    ("flaw", [
        r"(?:identifies|describe)\s+a\s+flaw",
        r"reasoning\s+is\s+flawed",
        r"error\s+in\s+reasoning",
        r"vulnerable\s+to\s+criticism",
        r"questionable\s+(?:reasoning|logic)",
        r"flawed\s+reasoning",
        r"logical\s+(?:mistake|error|fallacy)",
        # LogiQA additions
        r"shortcomings?\s+of\s+the\s+above\s+argument",
        r"shortcomings?\s+of\s+the\s+reasoning",
        r"points?\s+out\s+the\s+shortcomings",
    ]),
    ("main_point", [
        r"main\s+(?:point|conclusion|idea)",
        r"primary\s+purpose",
        r"argument\s+as\s+a\s+whole",
        r"conclusion\s+of\s+the\s+argument",
        r"which\s+of\s+the\s+following\s+best\s+(?:states|summarizes|expresses)",
        r"draws\s+the\s+conclusion",
        # LogiQA additions
        r"most\s+accurate\s+summary",
        r"best\s+matches.*description",
    ]),
    ("explain", [
        r"eliminating\s+this\s+(?:inconsistency|discrepancy|paradox|contradiction)",
        r"re(?:solve|concile)\s+(?:this|the)\s+(?:paradox|apparent\s+conflict|discrepancy|inconsistency)",
        r"best\s+explains",
        r"most\s+helpful\s+in\s+eliminating",
        r"explain\s+why",
        r"accounts\s+for",
        # LogiQA additions
        r"reasonable\s+explanation",
    ]),
]

# Conclusion indicator words (strong to weak)
CONCLUSION_INDICATORS = [
    r"\btherefore\b",
    r"\bthus\b",
    r"\bhence\b",
    r"\bconsequently\b",
    r"\bso\b",
    r"\baccordingly\b",
    r"\bthis\s+shows?\s+that\b",
    r"\bit\s+follows\s+that\b",
    r"\bas\s+a\s+result\b",
    r"\bfor\s+this\s+reason\b",
    r"\bin\s+conclusion\b",
]

PREMISE_INDICATORS = [
    r"\bbecause\b",
    r"\bsince\b",
    r"\bas\b",
    r"\bfor\b",
    r"\bgiven\s+that\b",
    r"\bdue\s+to\b",
    r"\binsofar\s+as\b",
    r"\bas\s+is\s+shown\s+by\b",
    r"\bfollows\s+from\b",
    r"\bis\s+evidenced\s+by\b",
]

# Fallacy keywords for flaw-type analysis
FALLACY_KEYWORDS = {
    "correlation_causation": [
        r"correlat",
        r"associated?\s+with",
        r"related?\s+to",
        r"linked?\s+to",
    ],
    "anecdotal": [
        r"anecdote",
        r"for\s+example",
        r"case\s+in\s+point",
        r"one\s+(?:person|case|instance)",
    ],
    "false_dilemma": [
        r"either\s+or",
        r"(?:only\s+)?two\s+(?:options|choices|possibilities)",
        r"must\s+be\s+(?:one|either)",
    ],
    "hasty_generalization": [
        r"all\s+\w+\s+are",
        r"every\s+\w+\s+is",
        r"always",
        r"never",
    ],
    "ad_hominem": [
        r"ad\s+hominem",
        r"personal\s+attack",
    ],
}


# ── Structural parsing ──────────────────────────────────────────────────

def _split_prompt(prompt: str) -> Tuple[str, str, Dict[str, str]]:
    """
    Split a full prompt into:
      - argument text (the paragraph)
      - question stem (the Q: ... line)
      - answer choices dict {letter: text}

    Handles:
      - Standard LSAT: "Paragraph.\\n\\nQ: Which...\\n\\nA. ...\\nB. ..."
      - LogiQA: "Paragraph.\\n\\nQ: Which..."
      - Plain: "Paragraph. Which of the following..."
    """
    text = prompt.strip()

    choices = _extract_choices(text)
    text = _strip_choices_from_text(text, choices)

    argument, stem = _split_argument_and_stem(text)
    return argument, stem, choices


def _extract_choices(text: str) -> Dict[str, str]:
    """Extract answer choices from prompt text."""
    choice_patterns = [
        r"(?:^|\n)\s*([A-E])\.\s*(.*?)(?=\n\s*[A-E]\.|\n*$)",
        r"(?:^|\n)\s*\(([A-E])\)\s*(.*?)(?=\n\s*\([A-E]\)|\n*$)",
        r"(?:^|\n)\s*([0-4])\.\s*(.*?)(?=\n\s*[0-4]\.|\n*$)",
    ]
    best_choices: Dict[str, str] = {}
    best_len = 0
    for pat in choice_patterns:
        matches = list(re.finditer(pat, text, re.DOTALL))
        if len(matches) >= 2:
            ch: Dict[str, str] = {}
            for m in matches:
                key = m.group(1)
                val = m.group(2).strip().rstrip(")")
                ch[key] = val
            if len(ch) > best_len:
                best_len = len(ch)
                best_choices = ch
    return best_choices


def _strip_choices_from_text(text: str, choices: Dict[str, str]) -> str:
    """Remove answer choices section from text."""
    if not choices:
        return text
    keys = list(choices.keys())
    first_key = keys[0]
    choice_start = text.find(f"\n{first_key}.")
    if choice_start < 0:
        choice_start = text.find(f"\n({first_key})")
    if choice_start < 0:
        for key in keys:
            idx = text.find(f"{key}.")
            if idx >= 0 and (choice_start < 0 or idx < choice_start):
                choice_start = idx
    if choice_start >= 0:
        text = text[:choice_start].strip()
    return text


def _split_argument_and_stem(text: str) -> Tuple[str, str]:
    """Split text into argument paragraph and question stem."""
    argument = text
    stem = ""

    # Look for "Q:" or "Question:" prefix
    q_match = re.search(
        r"(?:^|\n)\s*(?:Q|Question|QUESTION)\s*[:\.]\s*(.*?)$",
        text, re.IGNORECASE | re.MULTILINE
    )
    if q_match:
        stem = q_match.group(1).strip()
        argument = text[:q_match.start()].strip()
    else:
        # Try to find "Which of the following" as stem start
        stem_match = re.search(
            r"(Which\s+of\s+the\s+following|What\s+most|The\s+argument\s+depends)",
            text, re.IGNORECASE
        )
        if stem_match:
            stem = text[stem_match.start():].strip()
            argument = text[:stem_match.start()].strip()

    # Normalise whitespace
    argument = re.sub(r"\s+", " ", argument).strip()
    stem = re.sub(r"\s+", " ", stem).strip()
    return argument, stem


# ── Type classification ─────────────────────────────────────────────────

def _classify_type(stem: str) -> Optional[str]:
    """Classify the question type from the stem text."""
    stem_lower = stem.lower()
    for qtype, patterns in TYPE_PATTERNS:
        for pat in patterns:
            if re.search(pat, stem_lower):
                return qtype

    # Fallback heuristics
    if re.search(r"\bassum(e|ption|ing)\b", stem_lower):
        return "assumption"
    if re.search(r"\b(weaken|undermine|challenge|question|cast\s+doubt|refute|criticism|doubt)\b", stem_lower):
        return "weaken"
    if re.search(r"\b(strengthen|support|confirm)\b", stem_lower):
        return "strengthen"
    if re.search(r"\b(infer|imply|follow|conclude|deduce|derive|inference|corollary|conclusion)\b", stem_lower):
        return "inference"
    if re.search(r"\b(flaw|error|mistake|fallacy|vulnerable|shortcoming)\b", stem_lower):
        return "flaw"
    if re.search(r"\b(explain(?:s|ed|ing)?|resolve|reconcile|account for|paradox|inconsistency|discrepancy|explanation)\b", stem_lower):
        return "explain"
    if re.search(r"\b(main point|main conclusion|primary purpose|main idea|summary|accurat)\b", stem_lower):
        return "main_point"
    if re.search(r"\b(hypothesis)\b", stem_lower):
        return "assumption"

    return None


# ── Argument extraction ─────────────────────────────────────────────────

def _extract_conclusion(argument: str) -> str:
    """Extract the conclusion sentence from the argument."""
    sentences = _split_sentences(argument)
    # Look for conclusion indicator words
    for indicator in CONCLUSION_INDICATORS:
        match = re.search(indicator, argument, re.IGNORECASE)
        if match:
            # Find the sentence containing the indicator
            for s in sentences:
                if match.group() in s or match.group().lower() in s.lower():
                    return s.strip()
            # Fallback: return text after indicator
            after = argument[match.end():].strip()
            if after:
                # Take up to the next sentence boundary
                end = re.search(r"(?:\.\s*[A-Z]|\n)", after)
                if end:
                    after = after[:end.start() + 1]
                return after.strip()

    # No indicator found — last sentence is often the conclusion
    if len(sentences) >= 2:
        return sentences[-1].strip()
    if len(sentences) == 1:
        return sentences[0].strip()
    return argument


def _extract_premises(argument: str) -> List[str]:
    """Extract premise sentences from the argument."""
    premises = []
    for indicator in PREMISE_INDICATORS:
        matches = list(re.finditer(indicator, argument, re.IGNORECASE))
        for m in matches:
            # Get the clause following the indicator
            after = argument[m.end():].strip()
            # Take up to the next major break
            end = re.search(r"[.;!?](\s|$)", after)
            if end:
                clause = after[:end.start() + 1].strip()
            else:
                clause = after
            if clause and len(clause) > 10:
                premises.append(clause)

    # Also extract all sentences except conclusion as potential premises
    if not premises:
        sentences = _split_sentences(argument)
        if len(sentences) >= 2:
            premises = sentences[:-1]
        elif sentences:
            premises = sentences

    return premises


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (handling abbreviations)."""
    # Simple but robust sentence splitter
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'\(])", text)
    if len(sentences) <= 1:
        sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _get_key_terms(text: str) -> Set[str]:
    """Extract key terms (nouns, entities) from text."""
    # Find capitalized words (potential entities)
    entities = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text))
    # Find quoted terms
    quoted = set(re.findall(r'"([^"]+)"', text))
    # Find important lowercase multi-word patterns
    text_lower = text.lower()
    # Extract domain-relevant words (4+ chars, not stop words)
    stop_words = {
        "this", "that", "these", "those", "there", "their", "them", "they",
        "have", "with", "from", "which", "what", "when", "where", "would",
        "could", "should", "about", "than", "then", "some", "many", "more",
        "most", "much", "such", "also", "into", "over", "very", "just",
        "first", "second", "third", "other", "another", "after", "before",
    }
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text_lower)
    terms = {w for w in words if w not in stop_words}

    # Filter entities that are likely actual entities (not Q/A prefixes)
    skip_entities = {"Q", "A", "B", "C", "D", "E", "Which", "What", "How"}
    entities = {e for e in entities if e not in skip_entities}

    return entities | quoted


# ── Scoring heuristics ──────────────────────────────────────────────────

def _score_option(
    option_text: str,
    conclusion: str,
    premises: List[str],
    argument: str,
    qtype: str,
    key_terms: Set[str],
) -> float:
    """Score a single answer option based on question type and argument."""
    opt_lower = option_text.lower()
    conc_lower = conclusion.lower()
    arg_lower = argument.lower()
    combined_premises = " ".join(p.lower() for p in premises)

    score = 0.0

    # ── Keyword overlap with conclusion ──
    conc_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", conc_lower))
    opt_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", opt_lower))
    overlap = conc_words & opt_words
    score += len(overlap) * 2.0

    # ── Entity overlap ──
    opt_entities = _get_key_terms(option_text)
    entity_overlap = key_terms & opt_entities
    score += len(entity_overlap) * 3.0

    # ── Type-specific scoring ──
    if qtype == "strengthen":
        # Option that reinforces premise-conclusion link
        # Higher score if it shares terms with both premises and conclusion
        prem_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", combined_premises))
        prem_overlap = len(opt_words & prem_words)
        score += prem_overlap * 1.5
        # Support verbs
        if re.search(r"\b(because|since|as a result|leads? to|causes?|due to)\b", opt_lower):
            score += 2.0
        # Negation match (not negating → supporting)
        if re.search(r"\b(not|no|never|none)\b", opt_lower):
            score -= 1.0

    elif qtype == "weaken":
        # Option that undermines conclusion
        # Higher score if it negates something in the conclusion
        if re.search(r"\b(not|no|never|none|however|but|contrary)\b", opt_lower):
            score += 2.0
        # Shares terms with premises but contradicts expected outcome
        prem_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", combined_premises))
        prem_overlap = len(opt_words & prem_words)
        score += prem_overlap * 1.5
        # Alternative explanation keywords
        if re.search(r"\b(alternative|other\s+factor|different|could|may|might)\b", opt_lower):
            score += 2.0

    elif qtype == "assumption":
        # Necessary missing premise — should be consistent with argument
        score += len(overlap) * 2.5
        # Uses connecting words
        if re.search(r"\b(must|necessary|required|need|essential|cannot)\b", opt_lower):
            score += 2.0
        # If option mentions a bridging concept not in premises but implied
        prem_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", combined_premises))
        conc_words_only = conc_words - prem_words
        bridge_overlap = len(opt_words & conc_words_only)
        score += bridge_overlap * 2.0

    elif qtype == "inference":
        # Must be directly supported by premises
        prem_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", combined_premises))
        infer_overlap = len(opt_words & prem_words)
        score += infer_overlap * 2.0
        # Avoids conclusions that go beyond the text
        if re.search(r"\b(must|always|never|all|every)\b", opt_lower):
            score -= 0.5

    elif qtype == "flaw":
        # Contains fallacy keywords
        for fallacy_name, patterns in FALLACY_KEYWORDS.items():
            for pat in patterns:
                if re.search(pat, opt_lower):
                    score += 3.0
                    break
        # Mentions reasoning errors
        if re.search(r"\b(assume|presume|overlook|ignore|confuse|fail|mistake)\b", opt_lower):
            score += 2.0

    elif qtype == "main_point":
        # Restates or closely matches the conclusion
        score += len(overlap) * 3.0
        if len(opt_words ^ conc_words) <= 2:
            score += 3.0

    elif qtype == "explain":
        # Resolves apparent contradiction
        # Look for bridging/reconciling language
        if re.search(r"\b(however|although|while|but|yet|except|actually|in fact)\b", opt_lower):
            score += 2.0
        # Shares terms with both sides of the paradox
        score += len(opt_words & set(re.findall(r"\b[a-zA-Z]{3,}\b", arg_lower))) * 1.5

    # ── Length penalty (very short options are often wrong) ──
    if len(opt_words) <= 3:
        score -= 1.0

    # ── Negation matching for premise reinforcement ──
    # If option negates a premise statement, it's likely a weaken
    for prem in premises:
        prem_lower = prem.lower()
        prem_key_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", prem_lower))
        overlap_prem = len(opt_words & prem_key_words)
        if overlap_prem >= 2:
            has_not = bool(re.search(r"\b(not|no|never|none)\b", opt_lower))
            if qtype == "weaken" and has_not:
                score += 3.0
            elif qtype == "strengthen" and not has_not:
                score += 2.0

    return score


def _fallback_choice(choices: Dict[str, str], qtype: str) -> Optional[str]:
    """Fallback: guess based on position or simple patterns."""
    if not choices:
        return None
    keys = sorted(choices.keys(), key=lambda k: (len(k), k))
    # For assumption/strengthen/explain, middle options are often correct
    # For weaken/flaw, first options sometimes
    # For main_point/inference, last options common
    position_map = {
        "assumption": len(keys) // 2,
        "strengthen": len(keys) // 2,
        "explain": len(keys) // 2,
        "weaken": 0,
        "flaw": 0,
        "main_point": len(keys) - 1,
        "inference": len(keys) - 1,
    }
    idx = position_map.get(qtype, 0)
    if idx < len(keys):
        key = keys[idx]
        return f"{key}. {choices[key]}"
    return None


def _analyze_for_type(
    argument: str,
    stem: str,
    conclusion: str,
    premises: List[str],
    qtype: str,
    key_terms: Set[str],
) -> str:
    """Generate a type-specific analysis string (used when no choices exist)."""
    lines = [f"[{qtype.upper()}]"]
    lines.append(f"Conclusion: {conclusion}")

    if premises:
        prem_str = " ".join(p.strip() for p in premises[:3])
        lines.append(f"Premises: {prem_str}")

    # Type-specific reasoning
    type_hints = {
        "strengthen": "Looking for option that reinforces premise → conclusion link.",
        "weaken": "Looking for option that breaks premise → conclusion link.",
        "assumption": "Looking for necessary missing premise.",
        "inference": "Looking for what must be true from premises.",
        "flaw": "Checking for logical fallacies in reasoning.",
        "main_point": "Conclusion identified above.",
        "explain": "Looking for resolution of apparent contradiction.",
    }
    lines.append(f"Strategy: {type_hints.get(qtype, 'Unknown type.')}")
    return "\n".join(lines)


# ── Main solver ─────────────────────────────────────────────────────────

def solve_logical_reasoning(prompt: str, category: Optional[str] = None) -> Optional[str]:
    """
    Main entry point for LSAT-style logical reasoning.

    Args:
        prompt: Full prompt text (paragraph + question + optional choices).
        category: Optional category hint (e.g. 'logic').

    Returns:
        Selected answer string ("A. text") or analysis string, or None.
    """
    argument, stem, choices = _split_prompt(prompt)

    if not stem:
        return None

    qtype = _classify_type(stem)
    if not qtype:
        return None

    conclusion = _extract_conclusion(argument) if argument else ""
    premises = _extract_premises(argument) if argument else []
    key_terms = _get_key_terms(argument + " " + stem) if argument else set()

    # If answer choices exist, score them and pick the best
    if choices and len(choices) >= 2:
        best_key: Optional[str] = None
        best_score = float("-inf")
        for key, text in choices.items():
            score = _score_option(text, conclusion, premises, argument, qtype, key_terms)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key:
            return f"{best_key}. {choices[best_key]}"

        # Fallback to position-based guess
        fallback = _fallback_choice(choices, qtype)
        if fallback:
            return fallback

    # No choices available — return analysis for the type
    return _analyze_for_type(argument, stem, conclusion, premises, qtype, key_terms)
