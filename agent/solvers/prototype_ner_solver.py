"""
Prototype: Improved NER solver for tweetner7-style tasks.

Key insight: tweetner7 uses {@entity@} markers that the existing solver ignores.
Simply extracting these markers gives ~68% recall on the training set.
Combining with context-based type classification + unmarked entity extraction
should reach 50-60%+.

Category fix: eval uses "ner", deterministic.py checks "named_entity_recognition".
"""
import re
from typing import Optional

# Entity type classification based on context clues
_TYPE_INDICATORS = {
    "person": {
        "before": {"person", "singer", "actor", "player", "rapper", "his", "her",
                    "called", "named", "said", "asked", "told", "like", "follow",
                    "mention", "reply", "via", "with", "against", "vs", "and"},
        "after": {"said", "replied", "asked", "told", "says", "plays", "plays as",
                  "portrays", "plays", "sing", "sang", "performs"},
        "title_words": {"mr", "mrs", "ms", "dr", "prof", "sir", "lord", "queen",
                        "king", "prince", "princess", "saint", "president",
                        "minister", "senator", "governor", "mayor", "captain",
                        "general", "colonel", "sergeant", "chief"},
    },
    "group": {
        "before": {"group", "band", "crew", "team", "members", "choir", "the",
                    "kpop", "pop", "rock", "music", "dance"},
        "suffix_words": {"members", "band", "group", "crew", "squad", "team", "choir",
                         "orchestra", "ensemble"},
    },
    "corporation": {
        "before": {"corporation", "inc", "ltd", "corp", "company", "police",
                    "airline", "mtv", "cnn", "bbc", "nbc", "fox", "abc", "cbs",
                    "company", "brand", "organization", "department", "bureau"},
        "after": {"inc", "ltd", "corp", "corporation", "company", "llc"},
    },
    "location": {
        "before": {"location", "city", "country", "place", "at", "in", "near",
                    "street", "road", "avenue", "boulevard", "drive", "lane",
                    "area", "region", "state", "province", "island", "river",
                    "mountain", "park", "beach", "building"},
        "after": {"city", "town", "village", "county", "state", "province",
                  "island", "river", "mountain", "lake", "ocean", "sea",
                  "street", "road", "avenue", "square", "park"},
    },
    "event": {
        "before": {"event", "game", "match", "festival", "concert", "show",
                    "tournament", "cup", "final", "race", "election", "protest",
                    "rally", "parade", "conference", "meeting", "party",
                    "competition", "championship", "series", "season", "episode"},
        "hashtag": True,  # Events are often hashtags (#Ufc259, #FridayLivestream)
    },
    "product": {
        "before": {"product", "app", "software", "device", "phone", "computer",
                    "platform", "tool", "game", "console", "service", "website",
                    "site", "channel", "brand"},
    },
    "creative_work": {
        "before": {"song", "album", "movie", "film", "book", "show", "series",
                    "episode", "track", "single", "play", "musical", "opera",
                    "poem", "novel", "story", "article", "essay", "documentary"},
    },
}


def _classify_entity_type(entity_text: str, before_context: str, 
                           after_context: str, is_hashtag: bool) -> str:
    """Classify entity type based on surrounding context."""
    ctx_before = before_context.lower()
    ctx_after = after_context.lower()
    entity_lower = entity_text.lower()
    
    # Check for hashtag entities (events)
    if is_hashtag:
        return "event"
    
    # Check for person indicators in context
    for indicator in _TYPE_INDICATORS["person"]["before"]:
        if indicator in ctx_before:
            return "person"
    for indicator in _TYPE_INDICATORS["person"]["after"]:
        if indicator in ctx_after:
            return "person"
    
    # Check for corporate/org indicators
    for indicator in _TYPE_INDICATORS["corporation"]["before"]:
        if indicator in ctx_before:
            return "corporation"
    for indicator in _TYPE_INDICATORS["corporation"]["after"]:
        if indicator in ctx_after:
            return "corporation"
    
    # Check for location indicators
    for indicator in _TYPE_INDICATORS["location"]["before"]:
        if indicator in ctx_before:
            return "location"
    
    # Check for event indicators
    for indicator in _TYPE_INDICATORS["event"]["before"]:
        if indicator in ctx_before:
            return "event"
    
    # Check for creative_work indicators
    for indicator in _TYPE_INDICATORS["creative_work"]["before"]:
        if indicator in ctx_before:
            return "creative_work"
    
    # Check for group indicators
    for indicator in _TYPE_INDICATORS["group"]["before"]:
        if indicator in ctx_before:
            return "group"
    
    # Check for product indicators
    for indicator in _TYPE_INDICATORS["product"]["before"]:
        if indicator in ctx_before:
            return "product"
    
    # Default: if entity has title-like prefix, it's a person
    first_word = entity_text.split()[0].lower() if entity_text.split() else ""
    if first_word in _TYPE_INDICATORS["person"]["title_words"]:
        return "person"
    
    # If entity starts with @ (Twitter handle style), it's a person
    if entity_text.startswith("@"):
        return "person"
    
    # Default: person for capitalized names
    if entity_text[0].isupper():
        return "person"
    
    return "group"  # conservative default


def _extract_annotated_entities(text: str) -> list[dict]:
    """
    Extract entities marked with {@...@} in tweetner7 format.
    Returns list of {type, text, start, end} dicts.
    """
    entities = []
    
    # Match {@entity@} markers
    pattern = re.compile(r'\{@([^@]+)@\}')
    
    for m in pattern.finditer(text):
        entity_text = m.group(1).strip()
        start = m.start()
        end = m.end()
        
        # Context: 80 chars before and after
        before_ctx = text[max(0, start - 80):start]
        after_ctx = text[end:min(len(text), end + 80)]
        
        # Check if this is a hashtag entity (for events)
        is_hashtag = False
        hash_before = text[max(0, start - 20):start]
        if '#' in hash_before:
            is_hashtag = True
        
        entity_type = _classify_entity_type(entity_text, before_ctx, after_ctx, is_hashtag)
        
        entities.append({
            'type': entity_type,
            'text': entity_text,
            'context_before': before_ctx.strip(),
        })
    
    return entities


def _extract_unmarked_entities(text: str) -> list[dict]:
    """
    Extract entities NOT marked with {@...@}.
    Handles: capitalized names, hashtags, @mentions.
    """
    entities = []
    seen_spans = set()  # Avoid duplicating {@...@} spans
    
    # Track which character positions are inside {@...@} markers
    inside_marker = set()
    for m in re.finditer(r'\{@[^@]+@\}', text):
        for i in range(m.start(), m.end()):
            inside_marker.add(i)
    
    # Extract unmarked capitalized names
    # Pattern: single capitalized word that's not sentence-start
    for m in re.finditer(r'(?<!\.\s)(?<!\A)(?<!\{@)\b([A-Z][a-z]+)\b(?!@\})', text):
        pos = m.start()
        if pos in inside_marker:
            continue
        word = m.group(1)
        # Filter out common non-entity capitalized words (sentence starts, days, months)
        if word in ("The", "This", "That", "These", "Those", "It", "He", "She",
                     "They", "We", "You", "I", "A", "An", "And", "But", "Or",
                     "For", "Nor", "Yet", "So", "If", "Then", "Than", "Also",
                     "Very", "Just", "Here", "There", "When", "Where", "Why",
                     "How", "What", "Which", "Who", "Whom", "Whose",
                     "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                     "Saturday", "Sunday", "January", "February", "March",
                     "April", "May", "June", "July", "August", "September",
                     "October", "November", "December"):
            continue
        # Skip words that are the start of a sentence
        before = text[max(0, pos - 3):pos]
        if before in ("\n", ". ", "! ", "? "):
            continue
        
        word_end = pos + len(word)
        before_ctx = text[max(0, pos - 40):pos]
        after_ctx = text[word_end:min(len(text), word_end + 40)]
        entity_type = _classify_entity_type(word, before_ctx, after_ctx, False)
        
        # Only accept if context strongly suggests it's an entity
        if entity_type in ("person", "location", "corporation"):
            if word not in seen_spans:
                seen_spans.add(word)
                entities.append({
                    'type': entity_type,
                    'text': word,
                    'context_before': before_ctx.strip(),
                })
    
    # Extract hashtag entities (#Ufc259, #FridayLivestream)
    for m in re.finditer(r'#(\w+)', text):
        tag = m.group(1)
        if tag not in seen_spans:
            seen_spans.add(tag)
            # Skip common non-entity hashtags
            if tag.lower() in ("nowplaying", "music", "trending", "nflplayoffs",
                               "brownstwitter", "v", "btc", "cancelapboardexams2021",
                               "ufc257", "ufc259", "theonlyhope", "sheikhjarrah"):
                # These ARE entities, keep them
                pass
            entities.append({
                'type': 'event',
                'text': tag,
            })
    
    return entities


def _format_ner_output(annotated: list[dict], unmarked: list[dict]) -> str:
    """Format entities as expected by the eval: type: text on each line."""
    lines = []
    seen = set()
    
    for e in annotated + unmarked:
        key = (e['type'], e['text'].lower())
        if key in seen:
            continue
        seen.add(key)
        
        # Output format: "type: text"
        lines.append(f"{e['type']}: {e['text']}")
    
    return "\n".join(lines)


def solve_ner_prototype(task: str, category: str) -> Optional[str]:
    """
    Improved NER solver that extracts {@...@} markers and unmarked entities.
    
    Returns formatted entity list or None if no entities found.
    """
    # FIX: Also handle "ner" category (used by eval)
    if category not in ("ner", "named_entity_recognition"):
        return None
    
    text = task.strip()
    
    # Remove the "Extract entities:" prefix if present
    text = re.sub(r'^(?:Extract entities:)\s*', '', text)
    
    # Phase 1: Extract {@...@} annotated entities
    annotated = _extract_annotated_entities(text)
    
    # Phase 2: Extract unmarked entities
    unmarked = _extract_unmarked_entities(text)
    
    combined = annotated + unmarked
    
    if not combined:
        return None
    
    return _format_ner_output(annotated, unmarked)


if __name__ == "__main__":
    # Test against training data
    import json
    
    data = json.load(open("/home/artem/dev/amd-hackathon/data/eval/training-v3.json"))
    ner_questions = [q for q in data if q['category'] == 'ner']
    
    correct = 0
    total = 0
    
    for q in ner_questions:
        result = solve_ner_prototype(q['prompt'], q['category'])
        expected = q['expected_answer']
        
        # Normalize for comparison
        result_norm = result.strip() if result else ""
        expected_norm = expected.strip()
        
        # Check if expected entities are present in result
        expected_entities = set()
        for line in expected_norm.split('\n'):
            line = line.strip()
            if ':' in line:
                expected_entities.add(line)
        
        result_entities = set()
        for line in result_norm.split('\n'):
            line = line.strip()
            if ':' in line:
                result_entities.add(line)
        
        intersection = expected_entities & result_entities
        precision = len(intersection) / len(result_entities) if result_entities else 0
        recall = len(intersection) / len(expected_entities) if expected_entities else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        total += 1
        if result_norm == expected_norm:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} | {q['task_id']}")
        print(f"  Expected: {expected_norm[:80]}")
        print(f"  Got:      {result_norm[:80]}")
        print(f"  P={precision:.2f} R={recall:.2f} F1={f1:.2f}")
        print()
    
    print(f"Exact match: {correct}/{total} = {100*correct/total:.0f}%")
