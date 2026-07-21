"""
Layered Deterministic Decision Tree for Sentiment Classification.

Architecture: multiple lightweight deterministic layers stacked in order.
Each layer handles specific cases and passes through if uncertain.
First match wins — once a layer produces a classification, that's final.

Layers (in default order):
  1. SARCASM_PATTERN — sarcasm/backhanded/hedging regex → override to negative/neutral
  2. STRONG_SIGNAL — compound > pos_threshold or < neg_threshold → classify
  3. CONTRAST_SPLIT — contrastive clauses ("but", "however") → split and score independently
  4. NEGATION — negation-aware VADER scoring with word-level proximity
  5. DOMAIN_KEYWORDS — domain-specific lexicon matching (product reviews, movies)
  6. VADER_THRESHOLD — default VADER compound threshold fallback

Usage:
    tree = SentimentDecisionTree()
    label, source, confidence = tree.classify("This movie is fantastic!")
    path = tree.get_decision_path("This movie is fantastic!")
"""

import logging
import re
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# =============================================================================
# Layer 2: Sarcasm & Backhanded Patterns (from deterministic.py v1 + v2)
# =============================================================================

_RE_SARCASM_OH = re.compile(
    r'\b(?:Oh|Oh\s+(?:brilliant|great|wonderful|fantastic|amazing|perfect|nice|lovely|'
    r'just\s+(?:what|the)\s+\w+|really\?|is\s+that\s+so))\b',
    re.I
)

_RE_SARCASM_YEAH = re.compile(
    r'\b(?:yeah\s+right|sure\s+(?:thing|you\s+are|Jan)|'
    r'as\s+if|whatever\s+you\s+say|thanks\s+(?:for\s+)?nothing|'
    r'big\s+(?:deal|whoop)|whoop(?:ee|ie)\s+doo)\b',
    re.I
)

_RE_SARCASM_RHET = re.compile(
    r'(?:who\s+(?:doesn\'?t|wouldn\'?t)|is\s+that\s+supposed\s+to\s+be|'
    r'you\s+call\s+that|don\'?t\s+you\s+(?:just\s+)?love)',
    re.I
)

_RE_SARCASM_FAINT = re.compile(
    r'(?:'
    r'Efficiency\s+at\s+its\s+(?:finest|best)'
    r'|absolutely\s+stunning\s+(?:achievement|example|work|display)'
    r'|for\s+someone\s+with\s+your\s+(?:qualifications|background|experience|education|training|rank|position)'
    r'|I\s+(?:really\s+)?admire\s+your\s+(?:ability|capacity|dedication|commitment|patience|tolerance|courage)'
    r'|what\s+(?:a|an)\s+(?:wonderful|amazing|fantastic|beautiful|great|lovely)\s+\w+\s+(?:to|that|for|way)'
    r'|(?:you\'?re?|that\'?s?)\s+(?:so|very|really)\s+(?:brave|courageous|bold|generous|talented|'
    r'clever|smart|thoughtful|helpful|special|unique)\s+(?:to|that|for)'
    r')',
    re.I
)

_RE_BACKHANDED = re.compile(
    r'\b(?:'
    r'(?:you\'?re?\s+(?:so|very|really)\s+(?:brave|courageous|bold|generous|talented|'
    r'clever|smart|thoughtful|helpful|special|unique|something))\s+(?:to|that|for)'
    r'|(?:i\s+(?:really\s+)?admire|i\s+(?:must\s+)?(?:say|admit|confess))\s+'
    r'(?:your\s+(?:ability|capacity|dedication|commitment|patience|tolerance)\s+to\s+be)'
    r'|(?:what\s+(?:a|an)\s+(?:wonderful|amazing|fantastic|beautiful|great|lovely))'
    r'\s+\w+\s+(?:to|that|for)'
    r'|(?:efficiency|quality|service|design|customer\s*(?:service|support))'
    r'\s+(?:at\s+its\s+finest|at\s+its\s+best)'
    r'|(?:that\'?s?\s+(?:exactly|precisely|just)\s+what\s+(?:i|we|everyone)\s+\w+)'
    r')\b',
    re.I
)

_RE_GENERAL_BUT = re.compile(
    r'\b(?:'
    r'(?:i\s+wanted?\s+to\s+(?:love|like|enjoy)\s+(?:it|this|the)\w*\s*,?\s+but)'
    r'|(?:was\s+(?:excited|hoping)\s+(?:for|to)\s*,?\s+but)'
    r'|(?:had\s+high\s+(?:hopes|expectations)\s*,?\s+but)'
    r'|(?:started\s+(?:well|great|strong|promising)\s*,?\s+but)'
    r'|(?:looks?\s+(?:great|nice|good|beautiful|amazing|fantastic)\s*,?\s+but)'
    r'|(?:sounds?\s+(?:great|good|nice|promising)\s*,?\s+but)'
    r'|(?:promised?\s+(?:to|great|much|a\s+lot)\s*,?\s+but)'
    r'|(?:had\s+(?:so\s+)?much\s+potential\s*,?\s+but)'
    r')\b',
    re.I
)

_RE_HEDGING = re.compile(
    r'(?:'
    r'not\s+(?:entirely|quite|really|totally|fully|exactly|particularly|all\s+that)\s+'
    r'(?:terrible|bad|awful|horrible|good|great|wonderful|amazing|impressive|exciting|pleasant|unpleasant|shabby)'
    r'|I\s+(?:suppose|guess|imagine)\s+'
    r'|one\s+could\s+do\s+(?:worse|better)'
    r'|(?:that|it)\s+(?:went|worked\s+out)\s+(?:about\s+)?as\s+well\s+as\s+(?:expected|could\s+be\s+expected)'
    r'|interesting?\s+enough\s+to\s+keep\s+me\s+from'
    r'|(?:adequate|passable|tolerable|serviceable|decent\s+enough|good\s+enough)'
    r'|at\s+least\s+(?:it|he|she|they)\s+(?:was|were|did|had|has|have)\s+(?:not|never)'
    r'|at\s+least\s+(?:it|he|she|they|the)\s+\w+\s+(?:wasn|didn|hasn|hadn)'
    r'|flagged\s+.*?none\s+were\s+confirmed'
    r')',
    re.I
)

# =============================================================================
# Layer 3: Contrast clause detector
# =============================================================================

_RE_CONTRAST = re.compile(
    r'\b(?:but|however|although|though|yet|nevertheless|nonetheless|'
    r'on\s+the\s+(?:other\s+hand|contrary|flip\s+side)|that\s+said|'
    r'having\s+said\s+that|all\s+the\s+same|even\s+so|then\s+again)\b',
    re.I
)

# =============================================================================
# Layer 4: Negation detector
# =============================================================================

_RE_NEGATION = re.compile(
    r'\b(?:not|never|no|neither|nor|nothing|nowhere|none|nobody|'
    r'hardly|barely|scarcely|rarely|seldom|less|'
    r'don\'?t|doesn\'?t|didn\'?t|won\'?t|wouldn\'?t|shouldn\'?t|couldn\'?t|'
    r'can\'?t|isn\'?t|aren\'?t|ain\'?t|hasn\'?t|haven\'?t|hadn\'?t|'
    r'wasn\'?t|weren\'?t|without)\b',
    re.I
)

# Positive VADER words for negation proximity
_VADER_POS_WORDS = {
    'good', 'great', 'amazing', 'wonderful', 'fantastic', 'excellent', 'beautiful',
    'love', 'lovely', 'best', 'perfect', 'brilliant', 'awesome', 'impressive',
    'nice', 'happy', 'glad', 'joy', 'delight', 'pleased', 'terrific', 'superb',
    'outstanding', 'remarkable', 'magnificent', 'splendid', 'marvelous', 'fabulous',
    'delicious', 'pleasant', 'enjoyable', 'thrilled', 'exciting', 'fun', 'funny',
    'charming', 'elegant', 'graceful', 'warm', 'caring', 'thoughtful', 'helpful',
    'talented', 'skilled', 'intelligent', 'clever', 'smart',
    'succeed', 'succeeds', 'success', 'successful', 'sophisticated',
}

# Negative VADER words for negation proximity
_VADER_NEG_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'dreadful', 'poor', 'ugly', 'hate',
    'disgusting', 'disappointing', 'disappointed', 'frustrating', 'frustrated',
    'boring', 'dull', 'stupid', 'dumb', 'worst', 'worse', 'hideous',
    'painful', 'tragic', 'horrific', 'atrocious', 'abysmal', 'pathetic', 'laughable',
    'lousy', 'rotten', 'nasty', 'cruel', 'evil', 'vile', 'sick', 'wrong',
    'failure', 'fail', 'failed', 'fails', 'useless', 'worthless', 'pointless',
    'mediocre', 'underwhelming', 'overrated', 'messy', 'sloppy', 'shoddy',
    'crashes', 'crashed', 'crashing', 'bricked', 'scam', 'ripoff',
}

# =============================================================================
# Layer 5: Domain-specific keywords
# =============================================================================

_RE_DOMAIN_NEGATIVE = re.compile(
    r'\b(?:'
    r'sit\s+through|nothing\s+\'?s?\s+happening|well-worn|contrived|'
    r'cold\s+movie|dustbin\s+of\s+history|far\s+less\s+sophisticated|'
    r'off\s+his\s+game|off\s+her\s+game|off\s+their\s+game|'
    r'poorly\s+acted|badly\s+acted|badly\s+written|'
    r'waste\s+of|nothing\s+but\s+boilerplate|'
    r'cliché|clichés|hollow|shallow|empty|'
    r'fail\s+to|fails\s+to|failed\s+to|'
    r'no\s+(?:apparent|real|actual|genuine|true)\s+\w+'
    r'|the\s+horrors|shaggy\s+dog\s+story\b'
    r')\b',
    re.I
)

_RE_DOMAIN_POSITIVE = re.compile(
    r'\b(?:'
    r'enriched\s+by|imaginatively\s+mixed|'
    r'cross\s+swords.*best|'
    r'the\s+greatest|fresh|fresh\s+and|'
    r'masterpiece|masterful|brilliantly|'
    r'thought-provoking|must-see|'
    r'succeeds|succeeding|'
    r'a\s+wonderful|a\s+remarkable|a\s+masterpiece|'
    r'well-crafted|well-acted|well-written|well-made'
    r')\b',
    re.I
)

# =============================================================================
# Extended domain lexicon (tech/product review words)
# =============================================================================

_EXTRA_LEXICON = {
    # Tech/product negatives
    "crashes": -2.5, "crashed": -2.5, "crashing": -2.5,
    "freezes": -2.0, "frozen": -1.5, "freezing": -1.5,
    "glitch": -1.5, "glitchy": -2.0, "glitches": -1.5,
    "buggy": -2.0, "buggiest": -2.5, "laggy": -1.5,
    "bloated": -1.5, "overpriced": -2.0, "overhyped": -1.5,
    "underwhelming": -1.5, "mediocre": -1.0, "deleted": -1.5,
    "uninstalled": -1.5, "refund": -1.0, "refunded": -1.5,
    "overheats": -2.0, "overheating": -2.0, "malware": -3.0,
    "spyware": -3.0, "bloatware": -2.0, "hardware": -0.5,
    "bricked": -3.0, "bricking": -3.0,
    "slowly": -0.7, "barely": -0.5, "useless": -2.5,
    "worthless": -2.5, "pointless": -2.0, "terrible": -3.0,
    "horrible": -3.0, "dreadful": -3.0, "awful": -3.0,
    "scam": -3.0, "scammed": -3.0, "ripoff": -2.5,
    "dissatisfied": -2.0, "unhappy": -1.5, "disappointed": -1.5,
    "disappointment": -2.0, "frustrating": -2.0, "frustrated": -2.0,
    "infuriating": -3.0, "enraging": -3.0,
    "seamless": 1.5, "intuitive": 1.5, "lightning": 1.0,
    "lightweight": 1.0, "responsive": 1.5, "reliable": 1.5,
    "durable": 1.0, "polished": 1.5, "versatile": 1.0,
    "rushed": -2.0, "unsatisfying": -2.0, "undercooked": -2.0,
    "unfinished": -2.0, "unpolished": -2.0, "unstable": -2.0,
    "unreliable": -2.0, "unusable": -2.5, "unresponsive": -2.0,
    "unintuitive": -2.0, "unimpressive": -1.5, "underwhelmed": -1.5,
    "overrated": -2.0, "overcomplicated": -1.5, "overengineered": -1.5,
    "clunky": -2.0, "janky": -2.0, "messy": -1.0,
    "sloppy": -2.0, "lazy": -1.5, "careless": -2.0,
    "shoddy": -2.0, "cheap": -1.0, "flimsy": -2.0,
    "poorly": -0.7, "badly": -0.7, "terribly": -0.7,
    "dreadfully": -0.7, "horribly": -0.7, "atrociously": -2.0,
    "abysmal": -3.0, "appalling": -3.0, "pathetic": -3.0,
    "laughable": -2.0, "embarrassing": -2.0, "shameful": -2.5,
    "disgraceful": -3.0, "inexcusable": -3.0,
}

# =============================================================================
# Layer definitions
# =============================================================================

LAYER_NAMES = [
    "SARCASM_PATTERN",
    "STRONG_SIGNAL",
    "CONTRAST_SPLIT",
    "NEGATION",
    "DOMAIN_KEYWORDS",
    "VADER_THRESHOLD",
]

# Confidence levels
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"


class SentimentDecisionTree:
    """
    Layered deterministic sentiment classifier.

    Each layer can produce a decision (label, source_layer, confidence) or
    pass through (None). The first layer to produce a decision wins.

    Labels: "positive", "negative", "neutral", "mixed"
    """

    def __init__(
        self,
        # Layer 1: STRONG_SIGNAL thresholds
        pos_threshold: float = 0.5,
        neg_threshold: float = -0.3,
        # Layer 6: VADER_THRESHOLD thresholds
        vader_pos_thresh: float = 0.05,
        vader_neg_thresh: float = 0.0,
        # Advanced options
        sarcasm_enabled: bool = True,
        contrast_enabled: bool = True,
        negation_enabled: bool = True,
        domain_enabled: bool = True,
        # Layer ordering (list of layer name strings)
        layer_order: Optional[list] = None,
    ):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.vader_pos_thresh = vader_pos_thresh
        self.vader_neg_thresh = vader_neg_thresh
        self.sarcasm_enabled = sarcasm_enabled
        self.contrast_enabled = contrast_enabled
        self.negation_enabled = negation_enabled
        self.domain_enabled = domain_enabled

        # Allow custom layer ordering
        self.layer_order = layer_order or LAYER_NAMES.copy()

        # Lazy-init VADER analyzer
        self._analyzer = None

    def _get_analyzer(self):
        """Get VADER analyzer with extended lexicon."""
        if self._analyzer is None:
            self._analyzer = SentimentIntensityAnalyzer()
            self._analyzer.lexicon.update(_EXTRA_LEXICON)
        return self._analyzer

    def _get_vader_scores(self, text: str) -> dict:
        """Get standard VADER polarity scores."""
        analyzer = self._get_analyzer()
        return analyzer.polarity_scores(text)

    # ----- Layer 1: STRONG_SIGNAL -----

    def _layer_strong_signal(self, text: str) -> Optional[tuple]:
        """
        If compound is clearly positive (> pos_threshold) or clearly negative
        (< neg_threshold), classify directly with high confidence.
        """
        scores = self._get_vader_scores(text)
        compound = scores["compound"]

        if compound > self.pos_threshold:
            return ("positive", "STRONG_SIGNAL", CONFIDENCE_HIGH)
        elif compound < self.neg_threshold:
            return ("negative", "STRONG_SIGNAL", CONFIDENCE_HIGH)

        return None

    # ----- Layer 2: SARCASM_PATTERN -----

    def _layer_sarcasm(self, text: str) -> Optional[tuple]:
        """
        Check for sarcasm, backhanded compliment, hedging, and "X but Y" patterns.
        These override to negative or neutral with high confidence when matched.
        """
        # Hedging → neutral (unless strongly negative)
        if _RE_HEDGING.search(text):
            scores = self._get_vader_scores(text)
            if scores["compound"] <= -0.45:
                return ("negative", "SARCASM_PATTERN", CONFIDENCE_HIGH)
            return ("neutral", "SARCASM_PATTERN", CONFIDENCE_HIGH)

        # Faint praise / strong sarcasm → negative
        if _RE_SARCASM_FAINT.search(text):
            return ("negative", "SARCASM_PATTERN", CONFIDENCE_HIGH)

        # "Oh brilliant" sarcasm → negative (if not already strongly negative)
        if _RE_SARCASM_OH.search(text):
            scores = self._get_vader_scores(text)
            if scores["compound"] > -0.1:
                return ("negative", "SARCASM_PATTERN", CONFIDENCE_HIGH)

        # "Yeah right" → negative
        if _RE_SARCASM_YEAH.search(text):
            return ("negative", "SARCASM_PATTERN", CONFIDENCE_HIGH)

        # Rhetorical question sarcasm → negative (if compound > 0.0)
        if _RE_SARCASM_RHET.search(text):
            scores = self._get_vader_scores(text)
            if scores["compound"] > 0.0:
                return ("negative", "SARCASM_PATTERN", CONFIDENCE_HIGH)

        # Backhanded compliment → negative
        if _RE_BACKHANDED.search(text):
            return ("negative", "SARCASM_PATTERN", CONFIDENCE_HIGH)

        # "X but Y" pattern → negative (if not already negative)
        if _RE_GENERAL_BUT.search(text):
            scores = self._get_vader_scores(text)
            if scores["compound"] > -0.1:
                return ("negative", "SARCASM_PATTERN", CONFIDENCE_HIGH)

        return None

    # ----- Layer 3: CONTRAST_SPLIT -----

    def _layer_contrast(self, text: str) -> Optional[tuple]:
        """
        Split text on contrast clauses and score each part independently.
        Post-contrast clause gets 2x weight.
        """
        match = _RE_CONTRAST.search(text)
        if not match:
            return None

        split_pos = match.start()
        pre_text = text[:split_pos].strip()
        post_text = text[match.end():].strip()

        if not pre_text or not post_text:
            return None

        analyzer = self._get_analyzer()
        pre_scores = analyzer.polarity_scores(pre_text)
        post_scores = analyzer.polarity_scores(post_text)

        pre_compound = pre_scores["compound"]
        post_compound = post_scores["compound"]

        # Weighted scoring: post-contrast gets 2x weight
        weighted = (pre_compound + 2.0 * post_compound) / 3.0

        # Strong post-contrast signal wins
        if post_compound <= -0.3 and pre_compound >= 0.1:
            return ("negative", "CONTRAST_SPLIT", CONFIDENCE_HIGH)
        if post_compound >= 0.3 and pre_compound <= -0.1:
            return ("positive", "CONTRAST_SPLIT", CONFIDENCE_HIGH)

        # If weighted score is unambiguous
        if weighted >= 0.05:
            return ("positive", "CONTRAST_SPLIT", CONFIDENCE_MEDIUM)
        elif weighted <= -0.05:
            return ("negative", "CONTRAST_SPLIT", CONFIDENCE_MEDIUM)

        return None

    # ----- Layer 4: NEGATION -----

    def _has_negation_near_word(self, text: str, word_idx: int, window: int = 3) -> bool:
        """Check if a negation word appears within `window` tokens before position `word_idx`."""
        tokens = text.lower().split()
        if word_idx >= len(tokens):
            return False
        start = max(0, word_idx - window)
        for i in range(start, word_idx):
            if _RE_NEGATION.search(tokens[i]):
                return True
        return False

    def _layer_negation(self, text: str) -> Optional[tuple]:
        """
        Run VADER but apply negation-aware compound adjustment.
        When negation appears near a sentiment word, neutralize its effect.
        """
        scores = self._get_vader_scores(text)
        compound = scores["compound"]
        lower = text.lower()
        tokens = lower.split()

        negated_pos = 0
        total_pos = 0
        negated_neg = 0
        total_neg = 0

        for i, tok in enumerate(tokens):
            tok_clean = tok.strip(".,!?;:'\"()[]{}")
            if tok_clean in _VADER_POS_WORDS:
                total_pos += 1
                if self._has_negation_near_word(lower, i):
                    negated_pos += 1
            elif tok_clean in _VADER_NEG_WORDS:
                total_neg += 1
                if self._has_negation_near_word(lower, i):
                    negated_neg += 1

        adjusted = compound

        # Negated positive words: "not good" → reduce compound
        if negated_pos > 0 and total_pos > 0 and negated_pos / total_pos >= 0.3:
            adjusted = compound - 0.20 * negated_pos

        # Negated negative words: "not terrible" → increase compound
        if negated_neg > 0 and total_neg > 0 and negated_neg / total_neg >= 0.3:
            adjusted = compound + 0.20 * negated_neg

        adjusted = max(-1.0, min(1.0, adjusted))

        # Only make a decision here if negation meaningfully changed the score
        if adjusted != compound:
            if adjusted >= self.vader_pos_thresh:
                return ("positive", "NEGATION", CONFIDENCE_MEDIUM)
            elif adjusted <= self.vader_neg_thresh:
                return ("negative", "NEGATION", CONFIDENCE_MEDIUM)
            else:
                return ("neutral", "NEGATION", CONFIDENCE_LOW)

        return None

    # ----- Layer 5: DOMAIN_KEYWORDS -----

    def _layer_domain_keywords(self, text: str) -> Optional[tuple]:
        """
        Match domain-specific keywords for movie/product review sentiment.
        Only triggers when VADER compound is near zero (no strong signal).
        """
        scores = self._get_vader_scores(text)
        compound = scores["compound"]

        # Only use domain fallback when VADER is uncertain
        if abs(compound) > 0.05:
            return None

        lower = text.lower().strip()
        if not lower:
            return ("neutral", "DOMAIN_KEYWORDS", CONFIDENCE_LOW)

        neg_match = _RE_DOMAIN_NEGATIVE.search(lower)
        pos_match = _RE_DOMAIN_POSITIVE.search(lower)

        if neg_match and pos_match:
            return ("neutral", "DOMAIN_KEYWORDS", CONFIDENCE_LOW)
        if neg_match:
            return ("negative", "DOMAIN_KEYWORDS", CONFIDENCE_MEDIUM)
        if pos_match:
            return ("positive", "DOMAIN_KEYWORDS", CONFIDENCE_MEDIUM)

        return None

    # ----- Layer 6: VADER_THRESHOLD -----

    def _layer_vader_threshold(self, text: str) -> Optional[tuple]:
        """
        Default VADER compound threshold fallback.
        Low confidence — this is the last resort.
        """
        scores = self._get_vader_scores(text)
        compound = scores["compound"]
        pos = scores["pos"]
        neg = scores["neg"]

        # Mixed detection (both strong pos and neg)
        if pos > 0.35 and neg > 0.35:
            return ("neutral", "VADER_THRESHOLD", CONFIDENCE_LOW)

        if compound >= self.vader_pos_thresh:
            return ("positive", "VADER_THRESHOLD", CONFIDENCE_LOW)
        elif compound <= self.vader_neg_thresh:
            return ("negative", "VADER_THRESHOLD", CONFIDENCE_LOW)
        else:
            return ("neutral", "VADER_THRESHOLD", CONFIDENCE_LOW)

    # ----- Public API -----

    def classify(self, text: str) -> tuple:
        """
        Run the decision tree and return (label, source_layer, confidence).

        Returns:
            tuple: (label: str, source_layer: str, confidence: str)
            label is one of "positive", "negative", "neutral", "mixed"
        """
        # Map layer names to methods
        layer_methods = {
            "STRONG_SIGNAL": self._layer_strong_signal,
            "SARCASM_PATTERN": self._layer_sarcasm,
            "CONTRAST_SPLIT": self._layer_contrast,
            "NEGATION": self._layer_negation,
            "DOMAIN_KEYWORDS": self._layer_domain_keywords,
            "VADER_THRESHOLD": self._layer_vader_threshold,
        }

        for layer_name in self.layer_order:
            method = layer_methods.get(layer_name)
            if method is None:
                continue

            # Skip disabled layers
            if layer_name == "SARCASM_PATTERN" and not self.sarcasm_enabled:
                continue
            if layer_name == "CONTRAST_SPLIT" and not self.contrast_enabled:
                continue
            if layer_name == "NEGATION" and not self.negation_enabled:
                continue
            if layer_name == "DOMAIN_KEYWORDS" and not self.domain_enabled:
                continue

            result = method(text)
            if result is not None:
                return result

        # Should never reach here (VADER_THRESHOLD always returns)
        return ("neutral", "VADER_THRESHOLD", CONFIDENCE_LOW)

    def get_decision_path(self, text: str) -> list:
        """
        Returns the full decision path for debugging/inspection.

        Each entry: {
            "layer": str,
            "decision": str or None,
            "reason": str,
            "compound": float or None,
        }
        """
        path = []
        analyzer = self._get_analyzer()
        scores = analyzer.polarity_scores(text)
        base_compound = scores["compound"]

        layer_methods = {
            "STRONG_SIGNAL": self._layer_strong_signal,
            "SARCASM_PATTERN": self._layer_sarcasm,
            "CONTRAST_SPLIT": self._layer_contrast,
            "NEGATION": self._layer_negation,
            "DOMAIN_KEYWORDS": self._layer_domain_keywords,
            "VADER_THRESHOLD": self._layer_vader_threshold,
        }

        for layer_name in self.layer_order:
            method = layer_methods.get(layer_name)
            if method is None:
                continue

            entry = {
                "layer": layer_name,
                "compound": base_compound,
                "decision": None,
                "reason": "",
            }

            # Try the method
            result = method(text)
            if result is not None:
                label, source, confidence = result
                entry["decision"] = label
                entry["confidence"] = confidence
                entry["reason"] = f"{source} decided: {label} (confidence={confidence})"
            else:
                entry["decision"] = None
                entry["reason"] = f"{layer_name}: no decision (passed through)"

            path.append(entry)

            # If a decision was made, stop
            if result is not None:
                break

        return path

    def classify_with_path(self, text: str) -> dict:
        """
        Classify and return full details including decision path.
        """
        result = self.classify(text)
        path = self.get_decision_path(text)

        return {
            "label": result[0],
            "source_layer": result[1],
            "confidence": result[2],
            "decision_path": path,
        }

    def get_config(self) -> dict:
        """Return current configuration."""
        return {
            "pos_threshold": self.pos_threshold,
            "neg_threshold": self.neg_threshold,
            "vader_pos_thresh": self.vader_pos_thresh,
            "vader_neg_thresh": self.vader_neg_thresh,
            "sarcasm_enabled": self.sarcasm_enabled,
            "contrast_enabled": self.contrast_enabled,
            "negation_enabled": self.negation_enabled,
            "domain_enabled": self.domain_enabled,
            "layer_order": self.layer_order,
        }


# =============================================================================
# Factory with default config (matches v1 performance baseline)
# =============================================================================

def create_default_tree() -> SentimentDecisionTree:
    """Create a default tree with best-known threshold configuration."""
    return SentimentDecisionTree(
        pos_threshold=0.5,
        neg_threshold=-0.3,
        vader_pos_thresh=0.05,
        vader_neg_thresh=0.0,
        sarcasm_enabled=True,
        contrast_enabled=True,
        negation_enabled=True,
        domain_enabled=True,
    )


def create_v1_baseline_tree() -> SentimentDecisionTree:
    """
    Create a tree that mimics VADER v1 behavior (only STRONG_SIGNAL
    + SARCASM_PATTERN + VADER_THRESHOLD) for comparison.
    """
    return SentimentDecisionTree(
        pos_threshold=0.05,
        neg_threshold=0.0,
        vader_pos_thresh=0.05,
        vader_neg_thresh=0.0,
        sarcasm_enabled=True,
        contrast_enabled=False,
        negation_enabled=False,
        domain_enabled=False,
        layer_order=["SARCASM_PATTERN", "VADER_THRESHOLD"],
    )
