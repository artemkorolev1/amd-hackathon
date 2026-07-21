"""
Deterministic logic puzzle solver using python-constraint.
Handles: syllogisms, seating arrangements, scheduling, constraint puzzles.
"""
from constraint import Problem, AllDifferentConstraint
import re
from typing import Optional, List


def solve_logic_puzzle(prompt: str) -> Optional[str]:
    """
    Attempt to solve a logic puzzle using constraint propagation.

    Detects puzzle types:
    - Seating/scheduling: "X sits to the left of Y", "X is before Y"
    - Attribute matching: "X drives a red car", "Y likes cats"
    - Ordering: "X is taller than Y", "X is older than Z"
    - Classification: "X is a doctor", "Y is a lawyer"

    Returns solution string or None if can't solve.
    """
    try:
        puzzle = prompt.strip()
        if not puzzle:
            return None

        # --- Try to detect and solve known puzzle types ---

        # 1. Seating arrangement detection
        seating_patterns = [
            r"sits?\s+(to\s+the\s+)?(left|right|next\s+to|beside|adjacent)",
            r"is\s+sitting\s+(between|next\s+to)",
            r"seat\w*\s+(order|arrangement)",
            r"arrange\w*\s+(the\s+)?(people|persons|students|friends|colleagues)",
        ]
        is_seating = any(re.search(pat, puzzle, re.IGNORECASE) for pat in seating_patterns)

        # 2. Ordering / ranking detection
        ordering_patterns = [
            r"(taller|shorter|older|younger|faster|slower|higher|lower|more|less)\s+than",
            r"ranked?\s+(higher|lower|above|below)",
            r"(first|second|third|last)\s+(place|rank|position)",
        ]
        is_ordering = any(re.search(pat, puzzle, re.IGNORECASE) for pat in ordering_patterns)

        # 3. Attribute / classification matching
        attribute_patterns = [
            r"(drives|likes|owns|has|wears|plays|studies|teaches)\s+a\w*\s+\w+",
            r"is\s+(a|an|the)\s+\w+",
            r"works\s+as\s+a\w*",
            r"lives\s+in\s+\w+",
        ]
        is_attribute = any(re.search(pat, puzzle, re.IGNORECASE) for pat in attribute_patterns)

        # 4. Scheduling / time ordering
        scheduling_patterns = [
            r"(before|after|at\s+\d|from|until)\s+\w+",
            r"(scheduled|planned|occurs|happens)\s+(at|on|before|after)",
            r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)",
            r"(hour|minute|o'clock|am|pm)",
        ]
        is_scheduling = any(re.search(pat, puzzle, re.IGNORECASE) for pat in scheduling_patterns)

        puzzle_type = "unknown"
        if is_seating:
            puzzle_type = "seating"
        elif is_scheduling:
            puzzle_type = "scheduling"
        elif is_ordering:
            puzzle_type = "ordering"
        elif is_attribute:
            puzzle_type = "attribute_matching"

        # --- Build constraint model ---

        # Extract names (capitalized words)
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        all_words = re.findall(name_pattern, puzzle)

        # Filter to plausible names
        skip_words = {
            "The", "This", "That", "These", "Those", "There", "Here",
            "One", "Two", "Three", "Four", "Five", "First", "Second",
            "Third", "Last", "Next", "Each", "Every", "Some", "Many",
            "Both", "Neither", "Either", "All", "What", "Which", "When",
            "Where", "How", "Why", "Who", "Whom", "Whose", "If", "Then",
            "Else", "Also", "Only", "Just", "Very", "Too", "So", "But",
            "And", "Or", "Nor", "For", "With", "Out", "Up", "Down",
            "Off", "Over", "Under", "Again", "Further", "Once", "Here",
            "There", "When", "Where", "Why", "How", "Not", "Yes", "No",
            "Please", "Help", "Solve", "Find", "Given", "Using", "Use",
            "Assume", "Suppose", "Let", "Consider", "Take", "Make",
            "True", "False", "None", "Maybe",
        }
        names = list(dict.fromkeys(
            w for w in all_words if w not in skip_words and len(w) > 1
        ))

        if len(names) < 2:
            return None

        # --- Attribute matching ---
        if is_attribute and names:
            common_attrs = [
                "doctor", "lawyer", "teacher", "engineer", "artist", "writer",
                "nurse", "chef", "pilot", "driver", "programmer", "designer",
                "red", "blue", "green", "yellow", "white", "black", "brown",
                "cats", "dogs", "birds", "fish", "horses", "rabbits",
                "pizza", "pasta", "salad", "sushi", "burger", "tacos",
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday",
            ]
            found_attrs = [a for a in common_attrs if a.lower() in puzzle.lower()]

            if len(names) >= 2 and len(found_attrs) >= 2:  # noqa: PLR2004
                problem = Problem()
                for name in names:
                    problem.addVariable(name, found_attrs)
                problem.addConstraint(AllDifferentConstraint(), names)

                # Parse direct constraints: X is Y / X drives Y / X likes Y
                _add_attribute_constraints(problem, puzzle, names, found_attrs)

                solutions = problem.getSolutions()
                if solutions:
                    sol = solutions[0]
                    lines = [f"🧩 Solved ({puzzle_type}):"]
                    for name in names:
                        lines.append(f"  {name} = {sol[name]}")
                    return "\n".join(lines)

        # --- Ordering/ranking ---
        if is_ordering:
            problem = Problem()
            n = len(names)
            for name in names:
                problem.addVariable(name, list(range(1, n + 1)))
            problem.addConstraint(AllDifferentConstraint(), names)

            _add_order_constraints(problem, puzzle, names)

            solutions = problem.getSolutions()
            if solutions:
                sol = solutions[0]
                lines = [f"🧩 Solved ({puzzle_type}):"]
                sorted_names = sorted(names, key=lambda n: sol[n])
                for pos, name in enumerate(sorted_names, 1):
                    lines.append(f"  #{pos}: {name}")
                return "\n".join(lines)

        # --- Seating arrangement ---
        if is_seating:
            problem = Problem()
            n = len(names)
            for name in names:
                problem.addVariable(name, list(range(n)))
            problem.addConstraint(AllDifferentConstraint(), names)

            _add_seating_constraints(problem, puzzle, names)

            solutions = problem.getSolutions()
            if solutions:
                sol = solutions[0]
                lines = [f"🧩 Solved ({puzzle_type}):"]
                sorted_names = sorted(names, key=lambda n: sol[n])
                for pos, name in enumerate(sorted_names):
                    lines.append(f"  Seat {pos}: {name}")
                return "\n".join(lines)

        return None

    except Exception:
        return None


def _add_attribute_constraints(
    problem: Problem,
    puzzle: str,
    names: List[str],
    attrs: List[str],
) -> None:
    """Add attribute matching constraints like 'X is Y', 'X drives Y'."""
    for name in names:
        for attr in attrs:
            patterns = [
                rf"{re.escape(name)}\s+is\s+(a|an|the)?\s*{re.escape(attr)}\b",
                rf"{re.escape(name)}\s+(drives|likes|owns|has|wears|plays|studies|teaches)\s+a\w*\s*{re.escape(attr)}\b",
                rf"{re.escape(name)}\s+drives\s+{re.escape(attr)}",
                rf"{re.escape(name)}\s+likes\s+{re.escape(attr)}",
                rf"{re.escape(name)}\s+owns\s+{re.escape(attr)}",
            ]
            for pat in patterns:
                if re.search(pat, puzzle, re.IGNORECASE):
                    # Constrain name to the specific attr
                    problem.addConstraint(
                        lambda v, a=attr: v == a,
                        (name,)
                    )
                    break

        # "X is not Y" constraints
        for attr in attrs:
            pat = rf"{re.escape(name)}\s+is\s+not\s+(a|an|the)?\s*{re.escape(attr)}\b"
            if re.search(pat, puzzle, re.IGNORECASE):
                problem.addConstraint(
                    lambda v, a=attr: v != a,
                    (name,)
                )


def _add_order_constraints(
    problem: Problem,
    puzzle: str,
    names: List[str],
) -> None:
    """Add ordering constraints like 'X is taller than Y', 'X is before Y'."""
    for name_a in names:
        for name_b in names:
            if name_a == name_b:
                continue

            # X is {comparator} than Y
            for comparator in ["taller", "shorter", "older", "younger", "faster",
                               "slower", "higher", "lower", "more", "less",
                               "greater", "smaller", "bigger"]:
                pat = rf"{re.escape(name_a)}\s+is\s+{re.escape(comparator)}\s+than\s+{re.escape(name_b)}"
                if re.search(pat, puzzle, re.IGNORECASE):
                    if comparator in ("shorter", "younger", "slower", "lower", "less", "smaller"):
                        problem.addConstraint(
                            lambda a, b: a < b,
                            (name_a, name_b)
                        )
                    else:
                        problem.addConstraint(
                            lambda a, b: a > b,
                            (name_a, name_b)
                        )
                    break

            # X is before Y
            pat = rf"{re.escape(name_a)}\s+is\s+before\s+{re.escape(name_b)}"
            if re.search(pat, puzzle, re.IGNORECASE):
                problem.addConstraint(
                    lambda a, b: a < b,
                    (name_a, name_b)
                )

            # X is after Y
            pat = rf"{re.escape(name_a)}\s+is\s+after\s+{re.escape(name_b)}"
            if re.search(pat, puzzle, re.IGNORECASE):
                problem.addConstraint(
                    lambda a, b: a > b,
                    (name_a, name_b)
                )

            # X is older/younger (without 'than') — implicit comparison
            for adj in ["older", "younger"]:
                pat = rf"{re.escape(name_a)}\s+is\s+{re.escape(adj)}\s+than\s+{re.escape(name_b)}"
                if re.search(pat, puzzle, re.IGNORECASE):
                    if adj == "younger":
                        problem.addConstraint(lambda a, b: a < b, (name_a, name_b))
                    else:
                        problem.addConstraint(lambda a, b: a > b, (name_a, name_b))
                    break


def _add_seating_constraints(
    problem: Problem,
    puzzle: str,
    names: List[str],
) -> None:
    """Add seating arrangement constraints."""
    for name_a in names:
        for name_b in names:
            if name_a == name_b:
                continue

            # adjacency helper
            def _adjacent(a, b):
                return abs(a - b) == 1

            # X sits next to Y
            pat = rf"{re.escape(name_a)}\s+sits?\s+(next\s+to|beside|adjacent\s+to)\s+{re.escape(name_b)}"
            if re.search(pat, puzzle, re.IGNORECASE):
                problem.addConstraint(_adjacent, (name_a, name_b))

            # X sits to the left of Y
            pat = rf"{re.escape(name_a)}\s+sits?\s+to\s+the\s+left\s+of\s+{re.escape(name_b)}"
            if re.search(pat, puzzle, re.IGNORECASE):
                problem.addConstraint(
                    lambda a, b: a < b,
                    (name_a, name_b)
                )

            # X sits to the right of Y
            pat = rf"{re.escape(name_a)}\s+sits?\s+to\s+the\s+right\s+of\s+{re.escape(name_b)}"
            if re.search(pat, puzzle, re.IGNORECASE):
                problem.addConstraint(
                    lambda a, b: a > b,
                    (name_a, name_b)
                )

            # X sits between Y and Z
            match = re.search(
                rf"{re.escape(name_a)}\s+sits?\s+between\s+{re.escape(name_b)}\s+and\s+(\w+)",
                puzzle, re.IGNORECASE
            )
            if match:
                name_c = match.group(1)
                if name_c in names:
                    problem.addConstraint(
                        lambda a, b, c: (a - b) * (a - c) < 0,
                        (name_a, name_b, name_c)
                    )


def solve_syllogism(premises: List[str]) -> Optional[str]:
    """
    Solve categorical syllogisms (All A are B, Some B are C, etc.)
    Returns conclusion or None.
    """
    try:
        if len(premises) < 2:  # noqa: PLR2004
            return None

        all_relations = []
        some_relations = []
        no_relations = []

        for premise in premises:
            p = premise.strip().lower()
            m = re.match(r"all\s+(.+?)\s+are\s+(.+)", p)
            if m:
                all_relations.append((m.group(1).strip(), m.group(2).strip()))
                continue
            m = re.match(r"some\s+(.+?)\s+are\s+(.+)", p)
            if m:
                some_relations.append((m.group(1).strip(), m.group(2).strip()))
                continue
            m = re.match(r"no\s+(.+?)\s+(?:are|is)\s+(.+)", p)
            if m:
                no_relations.append((m.group(1).strip(), m.group(2).strip()))
                continue

        all_dict = {}
        for a, b in all_relations:
            all_dict[a] = all_dict.get(a, []) + [b]

        conclusions = []

        # Transitivity: All A are B, All B are C → All A are C
        for a in all_dict:
            for b in all_dict[a]:
                if b in all_dict:
                    for c in all_dict[b]:
                        if a != c:
                            conclusions.append(f"All {a} are {c}")

        # Some A are B, All B are C → Some A are C
        for a, b in some_relations:
            if b in all_dict:
                for c in all_dict[b]:
                    if a != c:
                        conclusions.append(f"Some {a} are {c}")

        # No A are B, All B are C → No A are C
        for a, b in no_relations:
            if b in all_dict:
                for c in all_dict[b]:
                    if a != c:
                        conclusions.append(f"No {a} are {c}")

        # Also show direct some-relations
        for a, b in some_relations:
            conclusions.append(f"Some {a} are {b}")

        if conclusions:
            unique = list(dict.fromkeys(conclusions))
            return "\n".join(unique)

        return None

    except Exception:
        return None
