"""
Bitmorphic Complexity Score classifier - 7-signal weighted complexity scorer.

Produces a continuous score (0-1) for routing decisions.
Used as a fallback classifier in the deterministic router.
"""
import re


class BitmorphicClassifier:
    """Weighted signal-based complexity scoring for task routing."""

    COMPLEX_KEYWORDS = [
        "reason",
        "reasoning",
        "explain",
        "analyze",
        "analyse",
        "evaluate",
        "compare",
        "contrast",
        "critique",
        "prove",
        "proof",
        "derive",
        "deduce",
        "infer",
        "justify",
        "argue",
        "hypothesis",
        "calculate",
        "compute",
        "solve",
        "equation",
        "integral",
        "derivative",
        "probability",
        "statistics",
        "algebra",
        "geometry",
        "theorem",
        "mathematical",
        "formula",
        "irrational",
        "implement",
        "debug",
        "refactor",
        "algorithm",
        "function",
        "class",
        "optimize",
        "complexity",
        "backpropagation",
        "neural",
        "api",
        "endpoint",
        "authentication",
        "schema",
        "architect",
        "design",
        "system",
        "step",
        "first",
        "then",
        "finally",
        "multi-step",
        "strategy",
        "plan",
        "outline",
        "essay",
        "story",
        "compose",
        "draft",
        "comprehensive",
        "detailed",
        "in-depth",
        "thorough",
        "elaborate",
        "generalize",
        "generalise",
        "extend",
        "apply",
        # Domain-specific keywords for better task differentiation
        "speed", "distance", "kmph", "velocity", "acceleration",
        "percentage", "percent", "ratio", "proportion",
        "fraction", "decimal",
        "extract", "entity", "disease", "gene",
        "sentiment", "positive", "negative", "opinion",
    ]

    SIMPLE_KEYWORDS = [
        "translate",
        "summarize",
        "summarise",
        "define",
        "what is",
        "who is",
        "when was",
        "where is",
        "yes or no",
        "true or false",
        "short answer",
        "brief",
        "one word",
        "hello",
        "hi",
        "thanks",
        "thank you",
        "greet",
        "capital",
        "name",
        "how many",
        "how old",
        "how far",
        "how long",
        "what color",
        "what colour",
        "what year",
        "what day",
        "meaning of",
        "definition of",
        "synonym",
        "antonym",
        "spell",
        "abbreviation",
        "acronym",
        "convert",
        "temperature",
        "currency",
        "largest",
        "smallest",
        "tallest",
        "fastest",
        "president",
        "founder",
        "inventor",
        "author",
        "continent",
        "country",
        "city",
        "planet",
        "simple",
        "basic",
        "quick",
        "easy",
    ]

    STRUCTURED_PATTERN = re.compile(r"\b(json|table|csv|markdown|yaml|xml)\b", re.I)
    MULTI_PART_PATTERN = re.compile(
        r"\b(and also|additionally|furthermore|moreover|firstly|"
        r"secondly|finally|as well as|in addition)\b",
        re.I,
    )

    WEIGHTS = {
        "length": 0.10,
        "complex_keywords": 0.30,
        "simple_keywords": 0.15,
        "structured_output": 0.10,
        "multi_part": 0.10,
        "question_depth": 0.05,
        "sentence_count": 0.20,
    }

    def _length_signal(self, word_count: int) -> float:
        if word_count <= 10:
            return 0.1
        if word_count <= 30:
            return 0.3
        if word_count <= 80:
            return 0.5
        if word_count <= 150:
            return 0.7
        return 0.9

    def _keyword_signal(self, lower: str, keywords: list) -> float:
        hits = sum(1 for kw in keywords if kw in lower)
        return min(hits / 2.0, 1.0)

    def _question_depth_signal(self, prompt: str) -> float:
        q = prompt.count("?")
        if q <= 1:
            return 0.1
        if q <= 3:
            return 0.5
        return 0.9

    def _sentence_count_signal(self, prompt: str) -> float:
        count = max(1, len(re.split(r'[.!?]+', prompt)) - 1)
        if count <= 1:
            return 0.05
        if count <= 2:
            return 0.2
        if count <= 4:
            return 0.5
        if count <= 6:
            return 0.75
        return 1.0

    def classify(self, prompt: str) -> dict:
        """
        Returns dict with score, difficulty, route, and all signal values.
        """
        lower = prompt.lower()
        words = prompt.split()
        word_count = len(words)

        signals = {
            "length": self._length_signal(word_count),
            "complex_keywords": self._keyword_signal(lower, self.COMPLEX_KEYWORDS),
            "simple_keywords": self._keyword_signal(lower, self.SIMPLE_KEYWORDS),
            "structured_output": self._keyword_signal(
                lower, ["json", "table", "csv", "markdown", "yaml", "xml"]
            ),
            "multi_part": min(
                len(self.MULTI_PART_PATTERN.findall(lower)) / 3.0, 1.0
            ),
            "question_depth": self._question_depth_signal(prompt),
            "sentence_count": self._sentence_count_signal(prompt),
        }

        # Aggregate with weights
        score = 0.0
        for name, value in signals.items():
            weight = self.WEIGHTS[name]
            if name == "simple_keywords":
                # INVERTED: simple keywords push score DOWN
                score += weight * (1.0 - value)
            else:
                score += weight * value

        score = max(0.0, min(1.0, score))

        # Difficulty buckets
        if score < 0.35:
            difficulty = "SIMPLE"
        elif score < 0.65:
            difficulty = "MODERATE"
        else:
            difficulty = "COMPLEX"

        route = "remote" if score >= 0.6 else "local"

        return {
            "classifier": "bitmorphic",
            "score": round(score, 4),
            "difficulty": difficulty,
            "route": route,
            "signals": signals,
            "word_count": word_count,
        }

    def __call__(self, prompt: str) -> dict:
        return self.classify(prompt)
