#!/usr/bin/env python3
"""Shared fuzzy_match for eval workers. Imported, not string-interpolated."""

import re

def fuzzy_match(answer: str, expected: str) -> bool:
    a, e = answer.strip().lower(), expected.strip().lower()
    if not a or not e:
        return False
    if a == e:
        return True
    if len(e) <= 20 and e in a:
        return True
    if len(a) <= 20 and a in e:
        return True
    na = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    ne = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", e)]
    if na and ne:
        an, en = na[-1], ne[-1]
        if en != 0 and abs((an - en) / en) <= 0.01:
            return True
        if an == en:
            return True
    ta = set(t for t in re.split(r"[^a-zA-Z0-9.]+", a) if t)
    te = set(t for t in re.split(r"[^a-zA-Z0-9.]+", e) if t)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8:
        return True
    return False
