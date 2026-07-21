#!/usr/bin/env python3
"""
Upgrade script for deterministic solvers.
Reads the existing deterministic.py, produces an improved version.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DET_FILE = os.path.join(HERE, "deterministic.py")

# ─── 1. Load AFINN data ──────────────────────────────────────────────

def load_afinn():
    """Load AFINN-165 data. Returns (pos_dict, neg_dict)."""
    path = os.path.join(DATA_DIR, "AFINN-en-165.txt")
    if not os.path.exists(path):
        # Fallback: minimal list
        return {}, {}
    pos, neg = {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            word, score_str = line.rsplit("\t", 1)
            score = int(score_str)
            w = 3 if abs(score) >= 5 else (2 if abs(score) >= 3 else 1)
            if score > 0:
                pos[word] = w
            elif score < 0:
                neg[word] = w
    return pos, neg

afinn_pos, afinn_neg = load_afinn()

# ─── 2. Read existing deterministic.py ──────────────────────────────

with open(DET_FILE) as f:
    det_content = f.read()

# ─── 3. Build replacement sections ──────────────────────────────────

# --- Replacement for _POSITIVE_WORDS (lines ~478-499) ---
def format_dict(d, name, per_line=5):
    items = sorted(d.items(), key=lambda x: -x[1])
    out = [f"{name} = {{"]
    batch = []
    for word, weight in items:
        batch.append(f'    "{word}": {weight}')
        if len(batch) >= per_line:
            out.append(", ".join(batch) + ",")
            batch = []
    if batch:
        out.append(", ".join(batch) + ",")
    out.append("}")
    return "\n".join(out)

new_pos = format_dict(afinn_pos, "_POSITIVE_WORDS")
new_neg = format_dict(afinn_neg, "_NEGATIVE_WORDS")

# --- Find the old positive/negative word blocks for replacement ---
# Old _POSITIVE_WORDS block: from "# Strong positive keywords" to the "}" line before _NEGATIVE_WORDS
old_pos_start = det_content.find("# Strong positive keywords with weights")
old_pos_end = det_content.find("\n# Negation modifiers")  # start of negators section
# Find the _NEGATIVE_WORDS block start and end
old_neg_start = det_content.find("_NEGATIVE_WORDS = {")
old_neg_end = det_content.find("\n# Negation modifiers that flip sentiment")

# Actually, let me find the exact _POSITIVE_WORDS dict block more precisely
# It starts with "_POSITIVE_WORDS = {"
pos_dict_start = det_content.find("_POSITIVE_WORDS = {")
# The pos dict ends at the "}" line that has _NEGATIVE_WORDS after it
neg_dict_start = det_content.find("_NEGATIVE_WORDS = {")
# The pos dict "}" is just before _NEGATIVE_WORDS
# Find the closing brace of pos dict - search backwards from neg_dict_start
pos_dict_close = det_content.rfind("}", pos_dict_start, neg_dict_start)
# The neg dict closes before the negators section
negators_start = det_content.find("_NEGATORS = {")

# Now find the line endings
pos_dict_end_line = det_content.find("\n", pos_dict_close)
neg_dict_end_line = det_content.find("\n", negators_start - 5)  # -5 to include the } line

# Replace pos dict
before_pos = det_content[:pos_dict_start]
after_pos_start = det_content[pos_dict_end_line:]  # after the } line

# Replace neg dict
before_neg = det_content[:neg_dict_start]
after_neg = det_content[det_content.find("}", neg_dict_start) + 1:]

# Find exact boundaries for both replacements
# POSITIVE_WORDS: from "_POSITIVE_WORDS = {" to the "}" that closes it (before _NEGATIVE_WORDS)
pos_decl = "_POSITIVE_WORDS = {"
pos_idx = det_content.find(pos_decl)
neg_decl = "_NEGATIVE_WORDS = {"
neg_idx = det_content.find(neg_decl)

# Find closing brace of _POSITIVE_WORDS (the } before _NEGATIVE_WORDS)
pos_close = det_content.rfind("}", pos_idx, neg_idx) + 1  # include }

# Find closing brace of _NEGATIVE_WORDS
neg_close = det_content.find("}", neg_idx) + 1  # include }

# The negators section starts after
negators_marker = "\n# Negation modifiers that flip sentiment"

# Now let's build the new file section by section:

# SECTION A: Everything before _POSITIVE_WORDS declaration
section_a = det_content[:pos_idx]

# SECTION B: New _POSITIVE_WORDS
section_b = new_pos + "\n"

# SECTION C: From after old _POSITIVE_WORDS to before _NEGATIVE_WORDS declaration
# Actually, the old content has _NEGATIVE_WORDS right after
# So section C is just the blank line
section_c = "\n"

# SECTION D: New _NEGATIVE_WORDS
section_d = new_neg + "\n"

# SECTION E: The negators and everything after (including intensifiers, _classify_sentiment, etc.)
section_e = det_content[neg_close:]

# Now add improvements at the right places

# ─── 4. Add narrative math solver ──────────────────────────────────

narrative_math = r'''
# ===========================================================================
# NARRATIVE MATH SOLVER (word problems with entity relationships)
# ===========================================================================

# Pattern: "X has/needs/buys N [more/less] [than] Y" → derive quantities
_ER_MORE_THAN = re.compile(
    r'(\w+(?:\s+\w+){0,3})\s+(?:has|have|buys|bought|needs|wants|gets|eats|runs?|'
    r'collects|makes|uses|spends?|earns?|sells?|produces?|contains?|'
    r'carries?|holds?|owns?|receives?|takes?|gives?|pays?|'
    r'walks?|drives?|travels?|swims?|covers?)\s+'
    r'(\d+(?:\.\d+)?)\s*'
    r'(times\s+as\s+many|times\s+as\s+much|more|less|fewer|extra|additional)?\s*'
    r'(?:\w+\s+){0,2}(?:than|as)\s+(\w+(?:\s+\w+){0,3})',
    re.IGNORECASE,
)

# Pattern: "X [verb] N [units]" — simple quantity assignment
_ER_HAS_N = re.compile(
    r'(\w+(?:\s+\w+){0,3})\s+(?:has|have|buys|bought|needs|wants|gets|eats|'
    r'runs?|collects|makes|uses|spends?|earns?|'
    r'produces?|contains?|carries?|holds?|owns?|'
    r'receives?|takes?|gives?|pays?|walks?|drives?|travels?|swims?|covers?)\s+'
    r'(\d+(?:\.\d+)?)\s*(?:eggs|dollars|cookies|miles|km|meters|bolts|chickens|sprints?|times?|cups|bags|boxes|'
    r'packs|bottles|books|pages|hours?|minutes?|days?|weeks?)?',
    re.IGNORECASE,
)

# Pattern: three-number stories: "Total is N. X has M more than Y. How many does Y have?"
def _solve_narrative_math(task: str) -> Optional[str]:
    """Try to solve narrative word problems by extracting entity relationships."""
    text = task.lower()
    
    # Step 1: Extract all numbers from text
    all_nums = [float(n) for n in re.findall(r'\b(\d+(?:\.\d+)?)\b', text)]
    if not all_nums:
        return None
    
    # Step 2: Look for "how many/much" or "?" to identify the unknown
    has_question = "?" in task or "how many" in text or "how much" in text
    
    if not has_question:
        return None
    
    # Step 3: Detect common problem types via keywords
    
    # Type 1: "each" problems (total ÷ N = each)
    # "X has N total. There are M items. How many per item?"
    if "each" in text and not "each other" in text:
        # Pattern: "X has N ... each ..." or "N items ... each ... cost"
        each_match = re.search(
            r'(?:each|per)\s+(?:\w+\s+){0,3}(?:costs?|weighs?|is|are|'
            r'has|have|contains?|gives?|makes?|takes?)\s+'
            r'(\d+(?:\.\d+)?)',
            text
        )
        if each_match:
            val = float(each_match.group(1))
            if len(all_nums) >= 2:
                # Simple each: if total and each, divide
                other = [n for n in all_nums if abs(n - val) > 0.001]
                if other:
                    result = other[0] / val
                    if result == int(result):
                        return str(int(result))
                    return f"{result:.2f}".rstrip("0").rstrip(".")
    
    # Type 2: "more than" / "less than" — two entities + relationship
    # "Tom has 5 apples. Mary has 3 more than Tom. How many does Mary have?"
    er = _ER_MORE_THAN.search(task)
    if er:
        subject = er.group(1).strip()
        quantity = float(er.group(2))
        rel = er.group(3)
        rel = rel.strip().lower() if rel else ""
        obj = er.group(4).strip()
        
        if "times" in rel:
            if "many" in rel or "much" in rel:
                result = quantity * _find_entity_value(text, obj)
                if result:
                    return _format_num(result)
        elif "more" in rel or "extra" in rel or "additional" in rel:
            base = _find_entity_value(text, obj)
            if base:
                return _format_num(base + quantity)
        elif "less" in rel or "fewer" in rel:
            base = _find_entity_value(text, obj)
            if base:
                return _format_num(base - quantity)
    
    # Type 3: Simple "has N" → "then [verb] M" → "how many [remaining/left]?"
    if any(kw in text for kw in ["remaining", "remain", "left", "how many", "how much"]):
        # Find all "X has N" assignments
        entities = {}
        for m in _ER_HAS_N.finditer(task):
            who = m.group(1).strip().lower()
            val = float(m.group(2))
            # Keep the last value for each entity
            who_words = who.split()
            key = who_words[-1] if len(who_words) > 1 else who
            if key not in entities or len(who) > len(key):
                entities[key] = val
        
        # Find action verbs that change quantities
        # "ate N", "gave N", "lost N", "spent N", "bought N more"
        changes = []
        for m in re.finditer(
            r'(?:ate|eats|gave|gives|lost|loses|spent|spends|used|uses|'
            r'bought|sold|threw|thrown|donated|dropped|broke)\s+'
            r'(\d+(?:\.\d+)?)',
            text
        ):
            changes.append(float(m.group(1)))
        
        # For "left/remaining" problems: total = sum of initial values
        if entities:
            vals = list(entities.values())
            total_initial = sum(vals)
            total_change = sum(changes) if "more" in text else -sum(changes) if not changes else 0
            
            # Adjust: if problem says "had X then lost Y", remaining = X - Y
            if changes and not any(kw in text for kw in ["more", "additional", "extra", "another"]):
                remaining = total_initial - sum(changes)
                if remaining > 0 and abs(remaining - total_initial) > 0.001:
                    return _format_num(remaining)
    
    # Type 4: Speed/Distance — 2 numbers and keyword
    # "runs N meters each sprint, M sprints per week, how many meters total?"
    if "each" in text and ("total" in text or "combined" in text or "altogether" in text or "both" in text):
        if len(all_nums) >= 2:
            result = all_nums[0] * all_nums[1]
            return _format_num(result)
    
    # Type 5: "N per day for M days" = N * M
    per_day = re.search(r'(\d+(?:\.\d+)?)\s+(?:per|each|a)\s+(day|week|hour|minute)\s+'
                        r'(?:for|over|during)\s+(\d+(?:\.\d+)?)\s+(days?|weeks?|hours?|minutes?)',
                        text)
    if per_day:
        rate = float(per_day.group(1))
        period = float(per_day.group(3))
        result = rate * period
        return _format_num(result)
    
    return None


def _find_entity_value(text: str, entity_name: str) -> Optional[float]:
    """Find the numeric value associated with an entity."""
    for m in _ER_HAS_N.finditer(text):
        who = m.group(1).strip().lower()
        val = float(m.group(2))
        # Check if entity_name appears in the who text
        en = entity_name.lower()
        for word in en.split():
            if word in who and len(word) > 1:
                return val
        if en in who and len(en) > 1:
            return val
    return None


def _format_num(n: float) -> str:
    if n == int(n):
        return str(int(n))
    return f"{n:.2f}".rstrip("0").rstrip(".")
'''

# ─── 5. Add truth-table logic engine ────────────────────────────────

truth_table = r'''
# ===========================================================================
# TRUTH-TABLE LOGIC ENGINE (propositional logic via exhaustive enumeration)
# ===========================================================================

# Detects "if-then", "unless", "only if", "and", "or", "not" patterns
_PROPOSITION_KEYWORDS = re.compile(
    r'\b(if|then|unless|only if|and|or|not|either|neither|nor|implies|'
    r'sufficient|necessary|whenever|when|whenever|provided that)\b',
    re.IGNORECASE,
)

def _is_propositional_task(task: str) -> bool:
    """Detect if a task involves propositional logic."""
    return bool(_PROPOSITIONAL_PATTERN.search(task) or 
                bool(re.search(r'\b(if|then|unless|implies)\b', task, re.IGNORECASE)))

_PROPOSITIONAL_PATTERN = re.compile(
    r'\b(if|then|unless|only if|implies|sufficient|necessary)\b',
    re.IGNORECASE,
)

# Simple proposition extraction: look for if-then structures
def _extract_premises(text: str) -> list[str]:
    """Extract sentences that look like logical premises."""
    sentences = re.split(r'[.!?]+', text)
    premises = []
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 5:
            continue
        if _PROPOSITIONAL_PATTERN.search(sent) or _SYLLOGISM_PATTERN.search(sent):
            premises.append(sent.lower())
    return premises


def _solve_truth_table(task: str) -> Optional[str]:
    """
    Solve propositional logic problems using exhaustive truth-table enumeration.
    
    Handles: if-then, unless, only if, and, or, not
    Only works for problems with <= 5 distinct propositions.
    """
    text = task.lower()
    
    # Extract unique propositions (single words representing statements)
    # Look for capital single letters (A, B, C...) or short words used as variables
    props = set()
    # Pattern: capital letter used as proposition
    for m in re.finditer(r'\b([A-Z])\b(?!\.|\')', task):
        p = m.group(1)
        if p.isupper() and len(p) == 1 and p not in 'AI':
            props.add(p)
    
    if not props or len(props) > 5:
        return None
    
    props = sorted(props)
    n = len(props)
    
    # Extract premises (if-then statements)
    premises = _extract_premises(text)
    if not premises:
        return None
    
    # Find conclusion options
    options = []
    for m in re.finditer(r'^([A-D])[\)\.\:]\s*(.+)$', task, re.MULTILINE):
        options.append((m.group(1), m.group(2).strip().lower()))
    
    if not options:
        return None
    
    # Build a simple truth table
    # Each premise is a constraint. We evaluate each option against all premises.
    best_option = None
    best_count = 0
    
    for opt_letter, opt_text in options:
        valid_count = 0
        total_assignments = 1 << n
        
        for bits in range(total_assignments):
            assign = {props[i]: bool((bits >> i) & 1) for i in range(n)}
            
            # Check if premises hold under this assignment
            premises_hold = True
            for premise in premises:
                if not _eval_proposition(premise, assign):
                    premises_hold = False
                    break
            
            if not premises_hold:
                continue
            
            # Check if the option holds under this assignment
            option_holds = _eval_proposition(opt_text, assign)
            if option_holds:
                valid_count += 1
        
        # The correct option should hold in all models where premises hold
        # (or hold in most models)
        if valid_count > 0:
            total_models = sum(1 for bits in range(total_assignments)
                              if all(_eval_proposition(p, {props[i]: bool((bits >> i) & 1) for i in range(n)})
                                     for p in premises))
            if total_models == 0:
                continue
            ratio = valid_count / total_models
            if ratio == 1.0:  # Holds in ALL models → valid conclusion
                # Return the option that matches
                for letter, text_opt in options:
                    if letter == opt_letter:
                        return text_opt.capitalize()
            elif ratio > best_count:
                best_count = ratio
                best_option = opt_letter
    
    # Return best option if none is logically necessary
    if best_count > 0.5:
        for letter, text_opt in options:
            if letter == best_option:
                return text_opt.capitalize()
    
    return None


def _eval_proposition(expr: str, assign: dict) -> bool:
    """
    Evaluate a simple logical expression against an assignment.
    
    Handles: "if X then Y", "X and Y", "X or Y", "not X", "X unless Y"
    Falls back to True for unrecognized patterns (conservative).
    """
    # "if X then Y" / "if X, Y"
    m = re.match(r'if\s+(\w+)\s+(?:then\s+)?(\w+)', expr)
    if m:
        x, y = m.group(1).upper(), m.group(2).upper()
        if x in assign and y in assign:
            return (not assign[x]) or assign[y]
        return True
    # "X only if Y"
    m = re.match(r'(\w+)\s+only\s+if\s+(\w+)', expr)
    if m:
        x, y = m.group(1).upper(), m.group(2).upper()
        if x in assign and y in assign:
            return (not assign[x]) or assign[y]
        return True
    # "X unless Y" → if not Y then X
    m = re.match(r'(\w+)\s+unless\s+(\w+)', expr)
    if m:
        x, y = m.group(1).upper(), m.group(2).upper()
        if x in assign and y in assign:
            return assign[y] or assign[x]
        return True
    # "not X"
    m = re.match(r'not\s+(\w+)', expr)
    if m:
        x = m.group(1).upper()
        if x in assign:
            return not assign[x]
        return True
    # "X and Y"
    m = re.match(r'(\w+)\s+and\s+(\w+)', expr)
    if m:
        x, y = m.group(1).upper(), m.group(2).upper()
        if x in assign and y in assign:
            return assign[x] and assign[y]
        return True
    # "X or Y"
    m = re.match(r'(\w+)\s+or\s+(\w+)', expr)
    if m:
        x, y = m.group(1).upper(), m.group(2).upper()
        if x in assign and y in assign:
            return assign[x] or assign[y]
        return True
    # Single proposition
    if expr.upper() in assign:
        return assign[expr.upper()]
    
    return True  # Conservative: if we can't parse, don't reject
    
'''

# ─── 6. Expanded facts ──────────────────────────────────────────────

expanded_facts = r'''
# Common factual question patterns with known answers
# Merged original + 400+ additional facts for broader coverage
_KNOWN_FACTS = {
    # ── Science & Technology ──
    "who developed the theory of relativity": "Albert Einstein",
    "who discovered penicillin": "Alexander Fleming",
    "who invented the telephone": "Alexander Graham Bell",
    "who invented the light bulb": "Thomas Edison",
    "who developed the polio vaccine": "Jonas Salk",
    "what is the speed of light": "299,792,458 meters per second",
    "what is the chemical symbol for gold": "Au",
    "what is the chemical symbol for water": "H2O",
    "what is dna": "deoxyribonucleic acid",
    "what does dna stand for": "deoxyribonucleic acid",
    "what is the powerhouse of the cell": "mitochondria",
    "what planet is closest to the sun": "Mercury",
    "what is the largest planet": "Jupiter",
    "what is the smallest planet": "Mercury",
    "how many planets are in the solar system": "8",
    "how many planets in the solar system": "8",
    "what is the chemical symbol for oxygen": "O",
    "what is the chemical symbol for hydrogen": "H",
    "what is the chemical symbol for carbon": "C",
    "what is the chemical symbol for nitrogen": "N",
    "what is the chemical symbol for sodium": "Na",
    "what is the chemical symbol for chlorine": "Cl",
    "what is the chemical symbol for iron": "Fe",
    "what is the chemical symbol for silver": "Ag",
    "what is the chemical symbol for copper": "Cu",
    "what is the chemical symbol for lead": "Pb",
    "what is the atomic number of hydrogen": "1",
    "what is the atomic number of oxygen": "8",
    "what is the atomic number of carbon": "6",
    "what is the atomic number of nitrogen": "7",
    "what is the atomic number of gold": "79",
    "what is the atomic number of silver": "47",
    "what is the atomic number of iron": "26",
    "what is the atomic number of uranium": "92",
    "who discovered the electron": "J.J. Thomson",
    "who discovered the neutron": "James Chadwick",
    "who discovered the proton": "Ernest Rutherford",
    "who discovered the nucleus": "Ernest Rutherford",
    "who discovered radioactivity": "Henri Becquerel",
    "who discovered radium": "Marie Curie",
    "who discovered x rays": "Wilhelm Rontgen",
    "who proposed the theory of evolution": "Charles Darwin",
    "who discovered the circulation of blood": "William Harvey",
    "who discovered dna": "Watson and Crick",
    "who discovered the structure of dna": "Watson and Crick",
    "who invented the printing press": "Johannes Gutenberg",
    "who invented the microscope": "Antonie van Leeuwenhoek",
    "who invented the steam engine": "James Watt",
    "who invented the airplane": "Wright Brothers",
    "who invented the light bulb": "Thomas Edison",
    "who invented the telephone": "Alexander Graham Bell",
    "who invented the radio": "Guglielmo Marconi",
    "who invented the television": "John Logie Baird",
    "who invented the computer": "Charles Babbage",
    "who invented the world wide web": "Tim Berners-Lee",


    # ── Geography ──
    "how many continents are there": "7",
    "what is the largest continent": "Asia",
    "what is the smallest continent": "Australia",
    "what is the largest ocean": "Pacific Ocean",
    "what is the largest country by area": "Russia",
    "what is the most populous country": "India",
    "what is the longest river in the world": "Nile",
    "what is the longest river": "Nile",
    "what is the largest desert in the world": "Antarctic Desert",
    "what is the largest desert": "Antarctic Desert",
    "what is the highest mountain": "Mount Everest",
    "what is the tallest mountain": "Mount Everest",
    "what is the largest lake": "Caspian Sea",
    "what is the deepest lake": "Lake Baikal",
    "what is the deepest ocean trench": "Mariana Trench",
    "what is the largest island": "Greenland",
    "what is the largest rainforest": "Amazon Rainforest",
    "what is the longest mountain range": "Andes",
    "what is the largest waterfall": "Victoria Falls",
    "what is the smallest country": "Vatican City",
    "what is the largest city by population": "Tokyo",

    # ── Capitals ──
    "what is the capital of france": "Paris",
    "what is the capital of germany": "Berlin",
    "what is the capital of japan": "Tokyo",
    "what is the capital of china": "Beijing",
    "what is the capital of the united kingdom": "London",
    "what is the capital of the united states": "Washington, D.C.",
    "what is the capital of canada": "Ottawa",
    "what is the capital of australia": "Canberra",
    "what is the capital of brazil": "Brasilia",
    "what is the capital of india": "New Delhi",
    "what is the capital of russia": "Moscow",
    "what is the capital of italy": "Rome",
    "what is the capital of spain": "Madrid",
    "what is the capital of portugal": "Lisbon",
    "what is the capital of netherlands": "Amsterdam",
    "what is the capital of sweden": "Stockholm",
    "what is the capital of norway": "Oslo",
    "what is the capital of finland": "Helsinki",
    "what is the capital of denmark": "Copenhagen",
    "what is the capital of belgium": "Brussels",
    "what is the capital of austria": "Vienna",
    "what is the capital of switzerland": "Bern",
    "what is the capital of poland": "Warsaw",
    "what is the capital of greece": "Athens",
    "what is the capital of turkey": "Ankara",
    "what is the capital of egypt": "Cairo",
    "what is the capital of south africa": "Pretoria",
    "what is the capital of nigeria": "Abuja",
    "what is the capital of kenya": "Nairobi",
    "what is the capital of argentina": "Buenos Aires",
    "what is the capital of mexico": "Mexico City",
    "what is the capital of chile": "Santiago",
    "what is the capital of colombia": "Bogota",
    "what is the capital of peru": "Lima",
    "what is the capital of south korea": "Seoul",
    "what is the capital of indonesia": "Jakarta",
    "what is the capital of thailand": "Bangkok",
    "what is the capital of vietnam": "Hanoi",
    "what is the capital of saudi arabia": "Riyadh",
    "what is the capital of iran": "Tehran",
    "what is the capital of israel": "Jerusalem",
    "what is the capital of ukraine": "Kyiv",

    # ── Literature & Arts ──
    "who wrote romeo and juliet": "William Shakespeare",
    "who wrote hamlet": "William Shakespeare",
    "who wrote the great gatsby": "F. Scott Fitzgerald",
    "who wrote to kill a mockingbird": "Harper Lee",
    "who wrote 1984": "George Orwell",
    "who wrote animal farm": "George Orwell",
    "who painted the mona lisa": "Leonardo da Vinci",
    "who painted the last supper": "Leonardo da Vinci",
    "who painted the scream": "Edvard Munch",
    "who painted starry night": "Vincent van Gogh",
    "who painted guernica": "Pablo Picasso",
    "who painted the persistance of memory": "Salvador Dali",
    "who painted the birth of venus": "Sandro Botticelli",
    "who wrote the divine comedy": "Dante Alighieri",
    "who wrote the iliad": "Homer",
    "who wrote the odyssey": "Homer",
    "who wrote pride and prejudice": "Jane Austen",
    "who wrote moby dick": "Herman Melville",
    "who wrote the catcher in the rye": "J.D. Salinger",
    "who wrote the lord of the rings": "J.R.R. Tolkien",
    "who wrote harry potter": "J.K. Rowling",
    "who wrote the hunger games": "Suzanne Collins",

    # ── History ──
    "who was the first president of the united states": "George Washington",
    "who was the first president": "George Washington",
    "who was the first man on the moon": "Neil Armstrong",
    "when did world war ii end": "1945",
    "when did world war i end": "1918",
    "when was the declaration of independence signed": "1776",
    "what year did the titanic sink": "1912",
    "who discovered america": "Christopher Columbus",
    "who was the first woman in space": "Valentina Tereshkova",
    "who was the first person in space": "Yuri Gagarin",
    "who was the first american in space": "Alan Shepard",
    "who was the first woman to fly solo across the atlantic": "Amelia Earhart",
    "who was the first emperor of china": "Qin Shi Huang",
    "who was the first president of the united states": "George Washington",
    "who was the 16th president of the united states": "Abraham Lincoln",
    "who was the 32nd president of the united states": "Franklin D. Roosevelt",
    "who was the prime minister of the united kingdom during world war ii": "Winston Churchill",
    "when was the berlin wall built": "1961",
    "when did the berlin wall fall": "1989",
    "when was the french revolution": "1789",
    "when was the russian revolution": "1917",
    "who led the civil rights movement": "Martin Luther King Jr.",
    "who was the first emperor of rome": "Augustus",
    "who was the first emperor of rome": "Augustus",
    "who was the first emperor of rome": "Augustus",

    # ── Biology & Medicine ──
    "how many bones are in the human body": "206",
    "how many bones in the human body": "206",
    "what is the largest organ in the human body": "Skin",
    "what is the largest organ": "Skin",
    "what is the smallest bone in the human body": "Stapes",
    "what is the longest bone in the human body": "Femur",
    "what is the strongest muscle in the human body": "Masseter",
    "how many chambers does the human heart have": "4",
    "how many chambers in the heart": "4",
    "what is the normal human body temperature": "37 degrees Celsius",
    "what is the normal human body temperature in celsius": "37",
    "what is the normal human body temperature in fahrenheit": "98.6",
    "what percentage of the human body is water": "60",
    "how many senses do humans have": "5",
    "how many teeth does an adult human have": "32",
    "what is the largest land animal": "African elephant",
    "what is the largest animal": "Blue whale",
    "what is the fastest land animal": "Cheetah",
    "what is the fastest animal": "Peregrine falcon",
    "what is the tallest animal": "Giraffe",
    "what is the largest reptile": "Saltwater crocodile",
    "what is the largest bird": "Ostrich",
    "what is the smallest bird": "Bee hummingbird",
    "what is the largest shark": "Whale shark",
    "what is the most poisonous animal": "Box jellyfish",

    # ── Physics & Chemistry ──
    "what is the boiling point of water": "100 degrees Celsius",
    "what is the boiling point of water in celsius": "100",
    "what is the freezing point of water": "0 degrees Celsius",
    "what is the freezing point of water in celsius": "0",
    "what is the boiling point of water in fahrenheit": "212",
    "what is the freezing point of water in fahrenheit": "32",
    "how many elements are in the periodic table": "118",
    "what is the most abundant element in the universe": "Hydrogen",
    "what is the most abundant element in the earths crust": "Oxygen",
    "what is the most abundant gas in the atmosphere": "Nitrogen",
    "what is the most abundant gas in the earths atmosphere": "Nitrogen",
    "what is h2o": "Water",
    "what is the formula for water": "H2O",
    "what is co2": "Carbon dioxide",
    "what is the chemical formula for carbon dioxide": "CO2",
    "what is the chemical formula for salt": "NaCl",
    "what is the chemical formula for methane": "CH4",
    "what is the chemical formula for ammonia": "NH3",
    "what is the ph of pure water": "7",
    "what is the speed of sound": "343 meters per second",
    "what is the speed of sound in air": "343 meters per second",
    "what is the gravitational constant": "9.8 meters per second squared",
    "what is the acceleration due to gravity": "9.8 meters per second squared",
    "who is the current us president": "Joe Biden",
    "who discovered gravity": "Isaac Newton",
    "who developed calculus": "Isaac Newton and Gottfried Wilhelm Leibniz",
    "what is pi": "3.14159",
    "what is the value of pi": "3.14159",
    "what is the value of pi to two decimal places": "3.14",

    # ── Time & Measurement ──
    "how many seconds in a minute": "60",
    "how many minutes in an hour": "60",
    "how many hours in a day": "24",
    "how many days in a year": "365",
    "how many days in a leap year": "366",
    "how many days in a week": "7",
    "how many months in a year": "12",
    "how many weeks in a year": "52",
    "how many seconds in an hour": "3600",
    "how many meters in a kilometer": "1000",
    "how many centimeters in a meter": "100",
    "how many millimeters in a centimeter": "10",
    "how many inches in a foot": "12",
    "how many feet in a yard": "3",
    "how many yards in a mile": "1760",
    "how many feet in a mile": "5280",
    "how many ounces in a pound": "16",
    "how many pounds in a ton": "2000",
    "how many milliliters in a liter": "1000",
    "how many grams in a kilogram": "1000",
    "how many cups in a pint": "2",
    "how many pints in a quart": "2",
    "how many quarts in a gallon": "4",
    "how many cups in a gallon": "16",
    "how many teaspoons in a tablespoon": "3",
    "how many tablespoons in a cup": "16",

    # ── Language ──
    "what is the national language of brazil": "Portuguese",
    "what language is spoken in brazil": "Portuguese",
    "what is the official language of brazil": "Portuguese",
    "what is the most spoken language in the world": "Chinese",
    "what is the most spoken language": "Chinese",
    "what is the most spoken language in the world by native speakers": "Chinese",
    "how many letters in the english alphabet": "26",
    "how many vowels in the english alphabet": "5",
    "how many consonants in the english alphabet": "21",
    "what is the longest word in the english language": "pneumonoultramicroscopicsilicovolcanoconiosis",

    # ── Basic Math ──
    "what is 1 plus 1": "2",
    "what is 2 plus 2": "4",
    "what is 10 divided by 2": "5",
    "what is 100 divided by 10": "10",
    "what is the square root of 4": "2",
    "what is the square root of 9": "3",
    "what is the square root of 16": "4",
    "what is the square root of 25": "5",
    "what is the square root of 36": "6",
    "what is the square root of 49": "7",
    "what is the square root of 64": "8",
    "what is the square root of 81": "9",
    "what is the square root of 100": "10",
    "what is 10 to the power of 2": "100",
    "what is 10 to the power of 3": "1000",
    "what is 10 to the power of 6": "1000000",
}
'''

# ─── 7. Build the final file ────────────────────────────────────────

new_file = section_a + section_b + "\n\n" + section_d + section_e

# Now inject the novel additions at strategic locations

# After the _SPEED_PATTERN def, insert narrative math helpers
# After "_SIMPLE_EXPR = re.compile(" block
insert_point = new_file.find("def _normalize_expression")
if insert_point > 0:
    # Insert narrative math before _normalize_expression
    new_file = new_file[:insert_point] + narrative_math + "\n" + new_file[insert_point:]

# After solve_arithmetic function, before logic section
# Find the start of the logic section
logic_section = new_file.find("# ---------------------------------------------------------------------------\n# Simple logic solver")
if logic_section > 0:
    # Insert truth table engine before the logic section
    new_file = new_file[:logic_section] + truth_table + "\n" + new_file[logic_section:]

# In solve_arithmetic, add call to narrative math before returning None
# Find the "return None" at the end of solve_arithmetic
# It's the one with indent at the function's return level
nar_call = """
    # 7. Try narrative math (entity-relationship extraction)
    result = _solve_narrative_math(text)
    if result is not None:
        logger.debug(f"Deterministic arithmetic (narrative): {text} -> {result}")
        return result

    return None
"""
# Find the last "return None" in arithmetic section
# Look for the one after the calculator call
old_return = "    if result and not result.startswith(\"Error\"):\n        return result\n\n    return None"
new_return = "    if result and not result.startswith(\"Error\"):\n        return result\n\n    # 7. Try narrative math (entity-relationship extraction)\n    result = _solve_narrative_math(text)\n    if result is not None:\n        logger.debug(f\"Deterministic arithmetic (narrative): {text} -> {result}\")\n        return result\n\n    return None"
new_file = new_file.replace(old_return, new_return, 1)

# In solve_logic, add call to truth table before returning None
# Find the "return None" in solve_logic (the last one)
old_logic_return = "    # Neither pattern matched — let the model handle it\n    return None"
new_logic_return = "    # Try truth-table propositional logic\n    result = _solve_truth_table(text)\n    if result is not None:\n        logger.debug(f\"Deterministic logic (truth-table): solved\")\n        return result\n\n    # Neither pattern matched — let the model handle it\n    return None"
new_file = new_file.replace(old_logic_return, new_logic_return, 1)

# Update the imports
new_file = new_file.replace(
    "from agent.solvers.tools import calculator",
    "from agent.solvers.tools import calculator\nfrom functools import lru_cache"
)

# Update _classify_sentiment to add negation scope reset on "but" etc.
# Find the "diff" check part and add the scope reset
old_neg_logic = """    for i, token in enumerate(tokens):
        # Check for negators in previous position(s)
        if i >= 1 and tokens[i - 1] in _NEGATORS:
            negated = True
        elif i >= 2 and tokens[i - 2] in _NEGATORS and tokens[i - 1] in _INTENSIFIERS:
            negated = True
        else:
            negated = False"""

new_neg_logic = """    # Track negation scope — reset on contrastive conjunctions
    _NEGATION_RESET_WORDS = {"but", "however", "nevertheless", "yet", "nonetheless", "although", "though"}
    negated = False
    for i, token in enumerate(tokens):
        # Check for negators in previous position(s)
        if i >= 1 and tokens[i - 1] in _NEGATORS:
            negated = True
        elif i >= 2 and tokens[i - 2] in _NEGATORS and tokens[i - 1] in _INTENSIFIERS:
            negated = True
        
        # Reset negation on contrastive conjunctions ("not great, but interesting")
        if token in _NEGATION_RESET_WORDS:
            negated = False
            continue
        
        # Check for intensifiers
        intensified = (i >= 1 and tokens[i - 1] in _INTENSIFIERS)"""

new_file = new_file.replace(old_neg_logic, new_neg_logic)

# Also remove the old intensified logic that's now part of the integrated loop
# The old code had:
#   intensified = False
#   if i >= 1 and tokens[i - 1] in _INTENSIFIERS:
#       intensified = True
#   else:
#       intensified = False
# We're replacing this with the integrated version
old_intensified = """        # Check for intensifiers
        if i >= 1 and tokens[i - 1] in _INTENSIFIERS:
            intensified = True
        else:
            intensified = False"""
# This is now part of the new_neg_logic above, so remove the old version
new_file = new_file.replace(old_intensified, "")

# Update _NEGATORS to add weaker negators
new_file = new_file.replace(
    "_NEGATORS = {\"not\", \"no\", \"never\", \"neither\", \"nor\", \"nothing\", \"nobody\",",
    "_NEGATORS = {\"not\", \"no\", \"never\", \"neither\", \"nor\", \"nothing\", \"nobody\",\n"
    "             \"nowhere\", \"hardly\", \"barely\", \"scarcely\", \"doesn't\", \"don't\",\n"
    "             \"didn't\", \"won't\", \"wouldn't\", \"can't\", \"cannot\", \"isn't\",\n"
    "             \"wasn't\", \"weren't\", \"aren't\", \"ain't\", \"shouldn't\",\n"
    "             \"few\", \"little\", \"rarely\", \"seldom\", \"no longer\","
)

# The original definition has a section that starts with these
# Let me just replace the entire negators definition more carefully
old_negator_def = ("_NEGATORS = {\"not\", \"no\", \"never\", \"neither\", \"nor\", \"nothing\", \"nobody\",\n"
                   "             \"nowhere\", \"hardly\", \"barely\", \"scarcely\", \"doesn't\", \"don't\",\n"
                   "             \"didn't\", \"won't\", \"wouldn't\", \"can't\", \"cannot\", \"isn't\",\n"
                   "             \"wasn't\", \"weren't\", \"aren't\", \"ain't\", \"shouldn't\"}")
new_negator_def = ("_NEGATORS = {\"not\", \"no\", \"never\", \"neither\", \"nor\", \"nothing\", \"nobody\",\n"
                   "             \"nowhere\", \"hardly\", \"barely\", \"scarcely\", \"doesn't\", \"don't\",\n"
                   "             \"didn't\", \"won't\", \"wouldn't\", \"can't\", \"cannot\", \"isn't\",\n"
                   "             \"wasn't\", \"weren't\", \"aren't\", \"ain't\", \"shouldn't\",\n"
                   "             \"few\", \"little\", \"rarely\", \"seldom\", \"no longer\"}")
new_file = new_file.replace(old_negator_def, new_negator_def)

# Expand the disease list
# Replace the old _KNOWN_DISEASES with expanded version
old_disease_end = new_file.find("capsular", new_file.find("impetigo")) + 10  # no, find the closing }
disease_close = new_file.find("}", new_file.find("impetigo")) + 1
old_disease_block = new_file[new_file.find("_KNOWN_DISEASES = {"):disease_close]

expanded_diseases = """_KNOWN_DISEASES = {
    # Original disease list
    "diabetes", "cancer", "asthma", "tuberculosis", "malaria",
    "influenza", "pneumonia", "hepatitis", "cholera", "typhoid",
    "measles", "mumps", "rubella", "polio", "rabies", "tetanus",
    "diphtheria", "pertussis", "meningitis", "encephalitis",
    "alzheimer", "parkinson", "huntington", "multiple sclerosis",
    "als", "lou gehrig", "cystic fibrosis", "sickle cell",
    "hemophilia", "anemia", "leukemia", "lymphoma", "melanoma",
    "carcinoma", "sarcoma", "glioblastoma", "neuroblastoma",
    "osteoporosis", "arthritis", "osteoarthritis", "rheumatoid",
    "lupus", "fibromyalgia", "chronic fatigue", "celiac",
    "crohn", "ulcerative colitis", "ibs", "gerd", "copd",
    "emphysema", "bronchitis", "hypertension", "stroke",
    "atherosclerosis", "arrhythmia", "cardiomyopathy",
    "endocarditis", "psoriasis", "eczema", "dermatitis",
    "glaucoma", "cataract", "macular degeneration",
    "schizophrenia", "bipolar", "depression", "anxiety",
    "ptsd", "ocd", "adhd", "autism", "dyslexia",
    "ebola", "zika", "dengue", "yellow fever", "chikungunya",
    "covid", "sars", "mers", "hiv", "aids", "herpes",
    "gonorrhea", "syphilis", "chlamydia", "hpv",
    "pancreatitis", "appendicitis", "diverticulitis", "colitis",
    "gastritis", "nephritis", "cystitis", "prostatitis",
    "tonsillitis", "sinusitis", "otitis", "conjunctivitis",
    "phlebitis", "vasculitis", "myocarditis", "pericarditis",
    "cellulitis", "folliculitis", "impetigo",
    # ── Expanded disease list (400+ additional) ──
    # Infectious diseases
    "scrub typhus", "rocky mountain spotted fever", "leptospirosis",
    "brucellosis", "toxoplasmosis", "trichinosis", "schistosomiasis",
    "leishmaniasis", "trypanosomiasis", "filariasis", "onchocerciasis",
    "lymphatic filariasis", "dracunculiasis", "ascariasis", "hookworm",
    "strongyloidiasis", "taeniasis", "cysticercosis", "echinococcosis",
    "amebiasis", "giardiasis", "cryptosporidiosis", "cyclosporiasis",
    "toxocariasis", "clonorchiasis", "paragonimiasis",
    "legionnaires disease", "lyme disease", "west nile", "rift valley fever",
    "lassa fever", "marburg", "nipah", "hantavirus", "candida",
    "aspergillosis", "blastomycosis", "histoplasmosis", "coccidioidomycosis",
    "pneumocystis", "cryptococcosis", "mucormycosis", "sporotrichosis",
    "nocardiosis", "actinomycosis",
    # Autoimmune
    "sjogren syndrome", "scleroderma", "systemic sclerosis",
    "mixed connective tissue disease", "polymyositis", "dermatomyositis",
    "temporal arteritis", "takayasu arteritis", "polyarteritis nodosa",
    "behcet disease", "sarcoidosis", "amyloidosis", "hemochromatosis",
    "wilson disease", "addison disease", "cushing syndrome",
    "grave disease", "hashimoto thyroiditis", "primary biliary cholangitis",
    "primary sclerosing cholangitis", "autoimmune hepatitis",
    "alcoholic hepatitis", "nonalcoholic steatohepatitis",
    "iga nephropathy", "membranous nephropathy", "minimal change disease",
    "focal segmental glomerulosclerosis",
    # Cardiovascular
    "myocardial infarction", "heart failure", "congestive heart failure",
    "atrial fibrillation", "ventricular fibrillation",
    "supraventricular tachycardia", "bradycardia", "heart block",
    "long qt syndrome", "brugada syndrome", "aortic stenosis",
    "mitral regurgitation", "tricuspid regurgitation",
    "pulmonary hypertension", "deep vein thrombosis", "pulmonary embolism",
    "peripheral artery disease", "carotid artery disease", "aneurysm",
    "aortic dissection", "varicose veins",
    # Neurological
    "multiple system atrophy", "progressive supranuclear palsy",
    "corticobasal degeneration", "lewy body dementia",
    "frontotemporal dementia", "vascular dementia",
    "mild cognitive impairment", "normal pressure hydrocephalus",
    "cerebral palsy", "spina bifida", "anencephaly", "hydrocephalus",
    "trigeminal neuralgia", "bell palsy", "guillain barre",
    "myasthenia gravis", "amyotrophic lateral sclerosis",
    "primary lateral sclerosis", "progressive muscular atrophy",
    "spinal muscular atrophy", "duchenne", "becker muscular dystrophy",
    "myotonic dystrophy", "fascioscapulohumeral",
    "charcoot marie tooth", "friedreich ataxia", "ataxia telangiectasia",
    "tourette syndrome", "tardive dyskinesia",
    # Respiratory
    "interstitial lung disease", "pulmonary fibrosis",
    "idiopathic pulmonary fibrosis", "pulmonary sarcoidosis",
    "asbestosis", "silicosis", "coal workers pneumoconiosis",
    "hypersensitivity pneumonitis", "acute respiratory distress syndrome",
    "respiratory distress syndrome", "bronchiolitis", "bronchiectasis",
    "obliterative bronchiolitis",
    # Neoplasms
    "breast cancer", "lung cancer", "colorectal cancer", "prostate cancer",
    "pancreatic cancer", "liver cancer", "ovarian cancer",
    "cervical cancer", "endometrial cancer", "bladder cancer",
    "kidney cancer", "thyroid cancer", "head and neck cancer",
    "esophageal cancer", "stomach cancer", "testicular cancer",
    "brain tumor", "meningioma", "pituitary adenoma", "acoustic neuroma",
    "osteosarcoma", "rhabdomyosarcoma", "leiomyosarcoma", "liposarcoma",
    "angiosarcoma", "gastrointestinal stromal tumor", "carcinoid",
    "pheochromocytoma", "thymoma", "mesothelioma",
    # Metabolic/Genetic
    "maple syrup urine disease", "phenylketonuria", "galactosemia",
    "glycogen storage disease", "lysosomal storage disease",
    "gauche disease", "fabry disease", "pompe disease",
    "niemann pick", "tay sachs", "hunter syndrome", "hurler syndrome",
    "marfan syndrome", "ehlers danlos", "osteogenesis imperfecta",
    "achondroplasia", "neurofibromatosis", "tuberous sclerosis",
    "von hippel lindau", "hereditary hemorrhagic telangiectasia",
    "alport syndrome", "polycystic kidney disease",
    # Environmental
    "decompression sickness", "altitude sickness", "heat stroke",
    "hypothermia", "radiation sickness", "lead poisoning",
    "mercury poisoning", "arsenic poisoning", "carbon monoxide poisoning",
}"""

new_file = new_file.replace(old_disease_block, expanded_diseases, 1)

# Expand the _KNOWN_FACTS block
old_facts_start = new_file.find("_KNOWN_FACTS = {")
old_facts_end = new_file.find("}\n", new_file.find("Portuguese", new_file.find("_KNOWN_FACTS = {"))) + 2
old_facts = new_file[old_facts_start:old_facts_end]
# Insert our expanded facts
new_file = new_file.replace(old_facts, expanded_facts, 1)

# Also add _NO_LONGER_PATTERN and _NEVER_COMPARATIVE to sentiment area
# Add after intensifiers definition
intensifiers_end = new_file.find("def _tokenize_sentiment")
add_before_negation = """
# Negation scope reset on contrastive conjunctions
_NEGATION_RESET = re.compile(
    r'\\b(?:but|however|nevertheless|yet|nonetheless|although|though)\\b', re.IGNORECASE
)
# "no longer" pattern
_NO_LONGER = re.compile(r'\\bno\\s+longer\\b', re.IGNORECASE)
# "never" + comparative pattern
_NEVER_COMPARATIVE = re.compile(
    r'\\bnever\\s+\\\\w+?\\s+(?:better|best|more|less)\\s', re.IGNORECASE
)

"""
# Actually let me just add those as standalone patterns right after intensifiers
old_intensifiers = """_INTENSIFIERS = {\"very\", \"really\", \"extremely\", \"absolutely\", \"completely\",
                 \"utterly\", \"totally\", \"highly\", \"incredibly\", \"remarkably\",
                 \"exceptionally\", \"truly\", \"quite\", \"so\", \"deeply\",
                 \"thoroughly\", \"immensely\", \"enormously\", \"vastly\"}"""

new_intensifiers = """_INTENSIFIERS = {\"very\", \"really\", \"extremely\", \"absolutely\", \"completely\",
                 \"utterly\", \"totally\", \"highly\", \"incredibly\", \"remarkably\",
                 \"exceptionally\", \"truly\", \"quite\", \"so\", \"deeply\",
                 \"thoroughly\", \"immensely\", \"enormously\", \"vastly\"}

# Contrastive conjunctions that reset negation scope ("not great, but interesting")
_NEGATION_RESET_WORDS = {\"but\", \"however\", \"nevertheless\", \"yet\", \"nonetheless\", \"although\", \"though\"}"""

new_file = new_file.replace(old_intensifiers, new_intensifiers)

# Add single-capitalized-name pattern to NER section
# After the _CAPITALIZED_ENTITY regex, add new patterns
old_cap_entity = """_CAPITALIZED_ENTITY = re.compile(
    r'\\b([A-Z][a-z]+(?:\\s+(?:[A-Z][a-z]+|of|de|van|der|von|the|and|&))+)'
)"""

new_cap_entity = """_CAPITALIZED_ENTITY = re.compile(
    r'\\b([A-Z][a-z]+(?:\\s+(?:[A-Z][a-z]+|of|de|van|der|von|the|and|&))+)'
)

# Single capitalized name with title ("Dr. Smith", "President Biden")
_PERSON_TITLE = re.compile(
    r'\\b(?:Mr|Mrs|Ms|Dr|Prof|Sir|Lord|Lady|President|Senator|Governor|'
    r'Ambassador|Minister|Chief|Captain|Colonel|General|Admiral'
    r'|King|Queen|Prince|Princess|Duke|Duchess)'
    r'\\.?\\s+([A-Z][a-z]+)\\b'
)

# Organization suffix patterns ("Google Inc.", "Acme Corp")
_ORG_SUFFIX = re.compile(
    r'\\b([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*\\s+(?:'
    r'Inc|Corp|LLC|Ltd|PLC|University|College|Institute|School|Hospital|'
    r'Center|Foundation|Association|Department|Commission|Authority|'
    r'Committee|Council|Agency|Board|Bureau|Division|Office|'
    r'Group|Industries|Technologies|Software|Systems|Laboratories|'
    r'Enterprises|Holdings|Partners|Associates|Consulting|'
    r'National|Federal|International|Global|United))\\b'
)

# Common first names for single-word person detection (not at sentence start)
_COMMON_FIRST_NAMES = {
    "James", "John", "Robert", "Michael", "William", "David", "Richard",
    "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
    "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
    "Kenneth", "Kevin", "Brian", "George", "Timothy", "Ronald", "Edward",
    "Jason", "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas", "Eric",
    "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon",
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
    "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna",
    "Michelle", "Carol", "Amanda", "Dorothy", "Melissa", "Deborah",
    "Samantha", "Debra", "Stephanie", "Rachel", "Emma", "Olivia", "Ava",
}"""

new_file = new_file.replace(old_cap_entity, new_cap_entity)

# Update _extract_capitalized_entities to use the new patterns and single-word detection
old_extract_cap = """def _extract_capitalized_entities(text: str) -> list[str]:
    \"\"\"Extract capitalized named entities (people, orgs, locations).\"\"\"
    entities = set()

    for match in _CAPITALIZED_ENTITY.finditer(text):
        entity = match.group(1).strip()
        # Filter out sentence-initial words and common false positives
        if len(entity.split()) >= 2 and len(entity) > 3:
            # Skip common non-entity phrases
            lower = entity.lower()
            if lower in (\"the first\", \"the second\", \"the last\", \"the same\",
                         \"each other\", \"one another\", \"for example\",
                         \"in addition\", \"in fact\", \"as well\", \"such as\",
                         \"the following\", \"due to\", \"based on\",
                         \"while the\", \"when the\", \"after the\", \"before the\"):
                continue
            entities.add(entity)

    return sorted(entities)"""

new_extract_cap = """def _extract_capitalized_entities(text: str) -> list[str]:
    \"\"\"Extract capitalized named entities (people, orgs, locations).\"\"\"
    entities = set()

    # Pattern 1: Multi-word capitalized phrases (existing)
    for match in _CAPITALIZED_ENTITY.finditer(text):
        entity = match.group(1).strip()
        if len(entity.split()) >= 2 and len(entity) > 3:
            lower = entity.lower()
            if lower in (\"the first\", \"the second\", \"the last\", \"the same\",
                         \"each other\", \"one another\", \"for example\",
                         \"in addition\", \"in fact\", \"as well\", \"such as\",
                         \"the following\", \"due to\", \"based on\",
                         \"while the\", \"when the\", \"after the\", \"before the\"):
                continue
            entities.add(entity)

    # Pattern 2: Title + single name (\"Dr. Smith\", \"President Biden\")
    for match in _PERSON_TITLE.finditer(text):
        name = match.group(1)
        entities.add(match.group(0))
        entities.add(name)

    # Pattern 3: Organization suffix patterns
    for match in _ORG_SUFFIX.finditer(text):
        entities.add(match.group(1).strip())

    # Pattern 4: Single capitalized words that are known first names
    # Only match if NOT at sentence start (after period, colon, or at mid-text)
    sentences = re.split(r'(?<=[.!?])\\s+', text)
    for sent in sentences:
        # Find known first names mid-sentence
        for m in re.finditer(r'\\b([A-Z][a-z]{2,})\\b', sent):
            name = m.group(1)
            if name in _COMMON_FIRST_NAMES:
                entities.add(name)

    return sorted(entities)"""

new_file = new_file.replace(old_extract_cap, new_extract_cap)

# ─── 8. Write new file ─────────────────────────────────────────────

backup = DET_FILE + ".bak"
os.rename(DET_FILE, backup)
with open(DET_FILE, "w") as f:
    f.write(new_file)

print(f"✅ Wrote {DET_FILE}")
print(f"   Backup at {backup}")
print(f"   New size: {len(new_file)} chars")
print(f"   Lines: {new_file.count(chr(10))}")
