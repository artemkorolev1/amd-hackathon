"""
secondary_nertweet.py — NER tweet/biomedical pattern detector.

Resolves NER↔factual confusions by detecting entity extraction instructions
that the primary NER scorer misses (biomedical, tweet-style, hashtag-heavy).

Pure regex/heuristic — zero model calls.
"""

import re

# ── Explicit NER instruction patterns ──
_NER_INSTRUCTION = re.compile(
    r'(?:^|\n)\s*(?:Extract|Identify|Find|List|Tag|Pull|Locate|'
    r'Retrieve|Annotate|Recognize|Detect|Classify|Label|Get all)'
    r'(?:\s+(?:all|the|any|each|every|named|unique|distinct))?'
    r'(?:\s+(?:of\s+the\s+)?(?:entities?|names?|people|persons?|'
    r'organizations?|companies?|cities?|countries?|locations?|'
    r'places?|dates?|diseases?|genes?|proteins?|drugs?|'
    r'symptoms?|conditions?|species?|chemicals?|compounds?|'
    r'pathogens?|viruses?|bacteria?|procedures?|'
    r'mention|mentions|references|references))',
    re.IGNORECASE | re.MULTILINE,
)

# ── Biomedical entity extraction (disease/gene/protein focused) ──
_BIOMEDICAL_NER = re.compile(
    r'(?:^|\n)\s*Extract\s+(?:all\s+)?(?:the\s+)?(?:disease|gene|protein|'
    r'chemical|compound|pathogen|virus|bacteria|organism|species|'
    r'mutation|variant|allele|biomarker|antibody|antigen|enzyme|receptor|'
    r'cell|tissue|organ|tumor|cancer|syndrome|disorder)',
    re.IGNORECASE | re.MULTILINE,
)

# ── Twitter/NER markers ──
_TWITTER_ENTITY = re.compile(
    r'\{@[^@]+@\}',  # {@Cleveland Browns@}, {@Nick Chubb@}
)
_TWITTER_HASHTAG_ENTITY = re.compile(
    r'#\s*[A-Z][A-Za-z]+',  # #Browns, #NFLPlayoffs
)

# ── NER source tags ──
_NER_SOURCE = re.compile(
    r'(?:from\s+(?:the\s+)?(?:following\s+)?(?:text|sentence|passage|corpus|'
    r'article|document|paragraph|tweet|post|comment|review|abstract|title))',
    re.IGNORECASE,
)

# ── Format indicators ──
_NER_OUTPUT_FORMAT = re.compile(
    r'(?:format|separated\s+by|output\s+as|return\s+as|list\s+them|'
    r'type\s*:\s*\w+|category\s*:\s*\w+|label\s*:\s*|tag\s*:\s*|'
    r'one\s+(?:per|on\s+each)\s+line)',
    re.IGNORECASE,
)

# ── Factual-subject words that override NER (educational narrative) ──
_FACTUAL_SUBJECT_WORDS = re.compile(
    r'\b(diagnosis|treatment|clinical|patient|patholog|etiology|prognosis|'
    r'which of the following|what is the most likely|'
    r'the function of|the role of|the purpose of)\b',
    re.IGNORECASE,
)

# ── Survey/sentiment words that override NER ──
_SURVEY_WORDS = re.compile(
    r'\b(agree|disagree|strongly|survey|questionnaire|rating|scale|'
    r'likely|unlikely|satisfied|satisfaction)\b',
    re.IGNORECASE,
)


def resolve_nertweet(category: str, prompt: str) -> str:
    """Detect NER prompts misclassified as factual or other categories.

    Fires BEFORE the factual_secondary so NER wins for entity extraction tasks.

    Args:
        category: Current category from primary classifier
        prompt: Original prompt text

    Returns:
        "ner" if NER instruction detected, original category otherwise
    """
    lower = prompt.lower()
    has_ner_instr = bool(_NER_INSTRUCTION.search(prompt))
    has_biomed = bool(_BIOMEDICAL_NER.search(prompt))
    has_twitter_entity = bool(_TWITTER_ENTITY.search(prompt))
    has_ner_source = bool(_NER_SOURCE.search(prompt))
    has_ner_format = bool(_NER_OUTPUT_FORMAT.search(prompt))

    # ── Strong NER signals → override to NER ──
    # Explicit "Extract entities:" or "Extract all disease names"
    if has_ner_instr or has_biomed:
        # But not if the subject is clearly educational/diagnostic MCQ
        # (e.g., "identify the correct diagnosis" is factual, not NER)
        if _FACTUAL_SUBJECT_WORDS.search(lower) and not has_twitter_entity:
            return category  # factual wins
        return "ner"

    # ── Twitter entity markers → NER ──
    if has_twitter_entity and category != "ner":
        # But not if it's a survey/sentiment
        if _SURVEY_WORDS.search(lower):
            return category
        return "ner"

    # ── NER source description + format instruction → NER ──
    if (
        has_ner_source
        and has_ner_format
        and category not in ("ner", "code_gen", "code_debug")
    ):
        return "ner"

    # ── Uncertain — return original ──
    return category
