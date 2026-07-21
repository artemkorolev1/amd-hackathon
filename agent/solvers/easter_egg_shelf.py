"""
Easter egg tools — not critical but fun to have.
Registered in ToolRegistry with category="fun".
"""
import json
import csv
import io
import re
from datetime import date, datetime
from typing import Optional
from collections import Counter


# ── EGG 1: CSV Formatter ──
def format_csv(text: str) -> str:
    """Pretty-print CSV data with aligned columns.

    Args:
        text: Raw CSV text with comma-separated values

    Returns:
        Formatted table with aligned columns
    """
    try:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return "(empty)"

        # Compute column widths
        col_widths = []
        for row in rows:
            while len(col_widths) < len(row):
                col_widths.append(0)
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell.strip()))

        # Build formatted output
        output = []
        for row_idx, row in enumerate(rows):
            formatted = []
            for i, cell in enumerate(row):
                width = col_widths[i] if i < len(col_widths) else 0
                formatted.append(cell.strip().ljust(width))
            output.append(" | ".join(formatted))

            # Separator after header
            if row_idx == 0:
                output.append("-+-".join("-" * w for w in col_widths))

        return "\n".join(output)
    except Exception:
        return f"(could not parse CSV: {text[:100]}...)"


# ── EGG 2: Text Statistics ──
def text_stats(text: str) -> dict:
    """Return quirky text statistics: reading time, grade level, most common words.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary with stats
    """
    if not text.strip():
        return {"error": "empty text"}

    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    char_count = len(text)
    sentence_count = max(len(sentences), 1)
    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
    avg_sentence_len = word_count / sentence_count

    # Reading time (avg 200 wpm)
    reading_time_min = word_count / 200
    reading_time_sec = reading_time_min * 60

    # Flesch-Kincaid grade level
    # 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    # Rough syllable count: count vowel groups
    syllable_count = 0
    for word in words:
        # Count vowel groups
        vowel_groups = len(re.findall(r'[aeiouy]+', word.lower()))
        syllable_count += max(vowel_groups, 1)

    grade_level = 0.39 * avg_sentence_len + 11.8 * (syllable_count / max(word_count, 1)) - 15.59
    grade_level = max(0, min(grade_level, 20))

    # Word frequency
    word_freq = Counter(w.lower().strip(".,!?\"';:()[]") for w in words)
    most_common = dict(word_freq.most_common(10))

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_len, 2),
        "avg_sentence_length": round(avg_sentence_len, 1),
        "reading_time": f"{int(reading_time_min)}m {int(reading_time_sec % 60)}s",
        "reading_time_seconds": round(reading_time_sec, 1),
        "flesch_kincaid_grade": round(grade_level, 1),
        "most_common_words": most_common,
        "unique_words": len(word_freq),
        "fun_fact": _fun_text_fact(word_count, word_freq),
    }


def _fun_text_fact(word_count: int, word_freq: Counter) -> str:
    """Generate a fun fact about the text."""
    if word_count == 0:
        return "This text is empty — perfect for a blank stare."

    # Check for long words
    long_words = [w for w in word_freq if len(w) > 12]
    if long_words:
        biggest = max(long_words, key=len)
        return f"Contains the monstrous word '{biggest}' ({len(biggest)} chars!)"

    # Most common non-stop word
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                  'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
                  'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                  'did', 'will', 'would', 'could', 'should', 'may', 'might',
                  'shall', 'can', 'not', 'no', 'it', 'its', 'this', 'that',
                  'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they',
                  'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
                  'their', 'our', 'what', 'which', 'who', 'whom'}
    content_words = {w: c for w, c in word_freq.items() if w not in stop_words}
    if content_words:
        top_word = max(content_words, key=content_words.get)
        return f"Your favorite word is '{top_word}' ({content_words[top_word]}x)"

    return f"Exactly {word_count} words — neat!"


# ── EGG 3: Reverse Text ──
def reverse_text(text: str) -> str:
    """Reverse the input text (just for fun).

    Args:
        text: Any text

    Returns:
        Reversed text
    """
    return text[::-1]


# ── EGG 4: Word Cloud ──
def top_words(text: str, n: int = 10) -> str:
    """Return the N most common words with frequencies.

    Args:
        text: Input text
        n: Number of top words to show (default: 10)

    Returns:
        Formatted frequency table
    """
    if not text.strip():
        return "(empty text)"

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return "(no words found)"

    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                  'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
                  'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                  'did', 'will', 'would', 'could', 'should', 'may', 'might',
                  'shall', 'can', 'not', 'no', 'it', 'its', 'this', 'that',
                  'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they',
                  'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
                  'their', 'our', 'what', 'which', 'who', 'whom'}

    content_words = [w for w in words if w not in stop_words and len(w) > 1]
    if not content_words:
        content_words = words

    counter = Counter(content_words)
    top = counter.most_common(n)

    # Format as a simple bar chart
    max_count = max(c for _, c in top) if top else 1
    max_word_len = max(len(w) for w, _ in top) if top else 1
    bar_max = 30

    lines = [f"📊 Top {len(top)} words:"]
    for word, count in top:
        bar_len = int((count / max_count) * bar_max)
        bar = "█" * bar_len
        pct = (count / len(content_words)) * 100
        lines.append(f"  {word.rjust(max_word_len)} ({count:3d}x) {bar} {pct:.1f}%")

    lines.append(f"\n   Total content words: {len(content_words)}")
    lines.append(f"   Total words (incl. stop words): {len(words)}")

    return "\n".join(lines)


# ── EGG 5: Leetspeak Translator ──
def to_leetspeak(text: str) -> str:
    """Convert text to leetspeak (e -> 3, a -> 4, etc.)

    Args:
        text: Normal English text

    Returns:
        Leetspeak version
    """
    leet_map = {
        'a': '4', 'e': '3', 'i': '1', 'o': '0', 'u': '(_)',
        'b': '8', 'g': '9', 'l': '1', 's': '5', 't': '7',
        'z': '2', 'y': '`/',
    }

    result = []
    for char in text:
        lower = char.lower()
        if lower in leet_map:
            # Capitalize if original was uppercase
            if char.isupper():
                result.append(leet_map[lower].upper())
            else:
                # 50% chance to use an alternate form for fun
                result.append(leet_map[lower])
        else:
            result.append(char)

    return "".join(result)


# ── EGG 6: Palindrome Check ──
def is_palindrome(text: str) -> bool:
    """Check if text (cleaned) is a palindrome.

    Args:
        text: Text to check

    Returns:
        True if palindrome, False otherwise
    """
    # Strip non-alphanumeric and lowercase
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text).lower()
    return cleaned == cleaned[::-1]


# ── EGG 7: Countdown until next April Fools ──
def days_until_april_fools() -> int:
    """Days until next April Fools' Day (fun counter).

    Returns:
        Integer number of days
    """
    today = date.today()
    year = today.year

    # This year's April Fools
    april_fools = date(year, 4, 1)

    if today > april_fools:
        # Next year
        april_fools = date(year + 1, 4, 1)

    delta = (april_fools - today).days
    return delta


# ── EGG 8: Weather hot-take ──
def weather_hot_take(temp_c: float) -> str:
    """Given a temperature in Celsius, return a hot take.

    Args:
        temp_c: Temperature in degrees Celsius

    Returns:
        A humorous take on the temperature
    """
    if temp_c < -20:
        return (
            f"🥶 {temp_c}°C? That's not weather, that's a cryogenic challenge. "
            "Even penguins are wearing tiny scarves."
        )
    elif temp_c < -10:
        return (
            f"🥶 {temp_c}°C — Cold enough to make a snowman reconsider his life choices."
        )
    elif temp_c < 0:
        return (
            f"❄️ {temp_c}°C — Freezing! Your breath is now visible and judgmental."
        )
    elif temp_c < 10:
        return (
            f"🧥 {temp_c}°C — Chilly. The perfect excuse to stay under a blanket "
            "and pretend you're a burrito."
        )
    elif temp_c < 20:
        return (
            f"🌤️ {temp_c}°C — 'Room temperature' for some, 'sweater weather' for others. "
            "No one is happy."
        )
    elif temp_c < 25:
        return (
            f"🌞 {temp_c}°C — Goldilocks would approve. Not too hot, not too cold. "
            "Flawless weather."
        )
    elif temp_c < 30:
        return (
            f"😎 {temp_c}°C — Getting toasty! Time to break out the lemonade "
            "and questionable summer fashion."
        )
    elif temp_c < 35:
        return (
            f"🥵 {temp_c}°C — Hot! The pavement is cooking, and so are your plans "
            "to do anything productive."
        )
    elif temp_c < 40:
        return (
            f"🔥 {temp_c}°C — 'Hot enough for ya?' asked every person you've ever met. "
            "The answer is no."
        )
    else:
        return (
            f"☀️🔥 {temp_c}°C — Surface of the sun vibes. "
            "Eggs fry on sidewalks. Existential crises on patios. "
            "Stay hydrated!"
        )


# ── EGG 9: Emoji Translator ──
def to_emoji(text: str) -> str:
    """Convert common words/phrases to emoji.

    Args:
        text: Text containing emoji-able words

    Returns:
        Text with emoji substitutions
    """
    emoji_map = {
        "happy": "😊", "sad": "😢", "love": "❤️", "heart": "💖",
        "fire": "🔥", "cool": "😎", "awesome": "🔥", "great": "👍",
        "good": "👌", "bad": "👎", "yes": "✅", "no": "❌",
        "pizza": "🍕", "taco": "🌮", "burger": "🍔", "pasta": "🍝",
        "coffee": "☕", "tea": "🍵", "beer": "🍺", "wine": "🍷",
        "dog": "🐕", "cat": "🐱", "bird": "🐦", "fish": "🐟",
        "sun": "☀️", "moon": "🌙", "star": "⭐", "rain": "🌧️",
        "snow": "❄️", "cloud": "☁️", "thunder": "⛈️",
        "rocket": "🚀", "computer": "💻", "phone": "📱",
        "music": "🎵", "party": "🎉", "cake": "🎂",
        "money": "💰", "time": "⏰", "book": "📚",
        "robot": "🤖", "ghost": "👻", "alien": "👽",
        "sleep": "😴", "think": "🤔", "wink": "😉",
    }

    result = text
    for word, emoji in emoji_map.items():
        # Case-insensitive replacement (word boundaries)
        result = re.sub(rf'\b{re.escape(word)}\b', emoji, result, flags=re.IGNORECASE)

    return result


# ── EGG 10: Coin Flip ──
def flip_coin() -> str:
    """Flip a virtual coin.

    Returns:
        'Heads' or 'Tails'
    """
    import random
    return random.choice(["Heads", "Tails"])
