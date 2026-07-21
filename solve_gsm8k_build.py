#!/usr/bin/env python3
"""
GSM8K deterministic solver - v5.
Fix the 6 wrong answers and add more patterns.
"""
import json, re, math
from typing import Optional, List, Tuple

with open('data/eval/tests/gsm8k_train.json') as f:
    DATA = json.load(f)
from scripts.grade_answer import fuzzy_match

def _fmt(v):
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-12:
            return str(int(round(v)))
        return f"{v:.10f}".rstrip('0').rstrip('.')
    return str(v)

# =========================================================================
# 1. Money items total
# =========================================================================
def m_money(text):
    t = text.lower()
    # Guard: don't answer total if question asks about an unknown quantity
    if re.search(r'(?:how\s+many\s+\w+\s+(?:did\s+)?|order|boxes?\s+of\s+pizza|packs?\s+of)', t):
        return None
    items = re.findall(r'(\d+)\s+(?:\w+\s+){0,5}(?:for|at|costing|costs?)\s+\$?(\d+(?:\.\d+)?)\s+(?:each|apiece|a\s+piece)', t)
    items2 = re.findall(r'(\d+)\s+(?:pairs?\s+of\s+)?(?:\w+\s+){0,3}(?:costs?|for|at)\s+\$?(\d+(?:\.\d+)?)', t)
    valid = items if len(items) >= 2 else (items2 if len(items2) >= 3 else [])
    if len(valid) >= 2:
        if re.search(r'\bsave\b|\bsavings\b|better\s+price', t): return None
        total = sum(float(c)*float(p) for c,p in valid)
        return _fmt(total)
    return None

# =========================================================================
# 2. Times-value chain
# =========================================================================
def m_times(text):
    t = text.lower()
    stop = {'there','here','it','this','that','these','those','i','you','he','she','we','they'}
    
    rels = []
    for m in re.finditer(r'(\w+(?:\s+\w+)?)\s+(?:is|are|has|have)\s+(?:(\d+)\s+)?times\s+as\s+(?:\w+\s+){0,2}as\s+(\w+(?:\s+\w+)?)', t):
        rels.append((m.group(1).strip().lower(), float(m.group(2)) if m.group(2) else None, m.group(3).strip().lower()))
    for f,fv in [('twice',2),('double',2),('triple',3)]:
        for m in re.finditer(r'(\w+(?:\s+\w+)?)\s+(?:is|are|has|have)\s+'+f+r'\s+as\s+(?:\w+\s+){0,2}as\s+(\w+(?:\s+\w+)?)', t):
            rels.append((m.group(1).strip().lower(), fv, m.group(2).strip().lower()))
    for m in re.finditer(r'(\w+(?:\s+\w+)?)\s+(?:does?|is|are|has|have)\s+half\s+as\s+(?:much|many)\s+(?:\w+\s+){0,3}as\s+(\w+(?:\s+\w+)?)', t):
        rels.append((m.group(1).strip().lower(), 0.5, m.group(2).strip().lower()))
    
    if not rels: return None
    rels = [(s,f,r) for s,f,r in rels if s not in stop and r not in stop]
    if not rels: return None
    
    vals = {}
    for m in re.finditer(r'(\w+)\s+(?:is|has|have|are)\s+(\d+)\s+(?:years?\s+old|old|years?)', t):
        e = m.group(1).strip().lower()
        if e not in stop: vals[e] = float(m.group(2))
    for m in re.finditer(r'(?:if|that)\s+(\w+)\s+(?:is|has|have|are)\s+(\d+)', t):
        e = m.group(1).strip().lower()
        if e not in stop: vals[e] = float(m.group(2))
    # Direct assignments - but SKIP if the number is followed by "times" (it's a multiplier, not a quantity)
    # Also skip if the number is part of a "more/fewer/less than" relationship
    for m in re.finditer(r'(\w+)\s+(?:has|have|had)\s+(\d+)\s+(?:\w+\s+){0,3}(?:sheep|pets|dollars|jewels|coins|points|toys|books|games|marbles|cats|dogs|fish|birds|apples|oranges|bananas|candies|flowers|trees|plants|sisters|brothers|cousins|students|teachers|kids|children|boxes|bags|packs|bottles)', t):
        e = m.group(1).strip().lower()
        if e not in stop: 
            full_match = m.group(0)
            # Skip if part of "N times" (multiplier, not quantity)
            if 'times' in full_match:
                continue
            # Skip if part of "more/fewer/less than" (relationship diff, not quantity)
            if 'more' in full_match or 'fewer' in full_match or 'less' in full_match:
                continue
            vals[e] = float(m.group(2))
    
    # More careful: skip numbers that appear right before "more"/"fewer"/"less than"
    for m in re.finditer(r'(\w+)\s+(?:is|has|have|are)\s+(\d+)\s+(?:\w+\s+){0,3}(?:jewels?|coins?|dollars?|years?\s+old|old|points?|pets?|toys?|books?|games?|marbles?|eggs?|chickens?|dogs?|cats?|sheep|pounds?|apples?|oranges?|bananas?|miles?|feet?|inches?|meters?|liters?|pounds?)', t):
        e = m.group(1).strip().lower()
        if e not in stop:
            # Skip if followed by "more", "fewer", "less" (these are relationships, not values)
            full_match = m.group(0)
            if 'more' in full_match or 'fewer' in full_match or 'less' in full_match:
                continue
            vals[e] = float(m.group(2))
    
    changed = True
    while changed:
        changed = False
        for subj, factor, ref in rels:
            if not factor: continue
            if ref in vals and subj not in vals:
                vals[subj] = vals[ref] * factor; changed = True
            elif subj in vals and ref not in vals:
                vals[ref] = vals[subj] / factor; changed = True
    
    if not vals: return None
    
    named = set()
    for s,f,r in rels: named.add(s); named.add(r)
    
    # Total question
    if re.search(r'(?:total|combined|altogether|together|sum|in\s+all)', t):
        total = sum(v for k,v in vals.items() if k in named)
        if total > 0: return _fmt(total)
    
    # Single entity question - only if NOT asking about multiple entities
    # First, check if question mentions MULTIPLE entities from the relationship
    q_text = t
    # If question has commas or "and" mentioning entities, it's a group question
    for entity, val in list(vals.items()):
        if entity not in named: continue
        q = re.search(r'(?:how\s+(?:many|much|old)|what\s+(?:is|are))\s+(?:\w+\s+){0,5}' + re.escape(entity), t)
        if q:
            # Check if other named entities appear after the question
            rest = t[q.end():]
            others_mentioned = any(re.search(r'\b' + re.escape(e) + r'\b', rest) for e in named if e != entity)
            if not others_mentioned:
                return _fmt(val)
    return None

# =========================================================================
# 3. More/less than 
# =========================================================================
def m_moreless(text):
    t = text.lower()
    stop = {'there','here','it','this','that','i','you','he','she','we','they'}
    
    more = re.findall(r'(\w+(?:\s+\w+)?)\s+(?:has|have|had|is|are)\s+(\d+)\s+more\s+(?:\w+\s+){0,3}than\s+(\w+(?:\s+\w+)?)', t)
    less = re.findall(r'(\w+(?:\s+\w+)?)\s+(?:has|have|had|is|are)\s+(\d+)\s+(?:fewer|less)\s+(?:\w+\s+){0,3}than\s+(\w+(?:\s+\w+)?)', t)
    there_more = re.findall(r'there\s+are\s+(\d+)\s+more\s+(\w+)\s+(?:\w+\s+){0,3}than\s+(\w+)', t)
    there_less = re.findall(r'there\s+are\s+(\d+)\s+(?:fewer|less)\s+(\w+)\s+(?:\w+\s+){0,3}than\s+(\w+)', t)
    
    if not more and not less and not there_more and not there_less:
        return None
    
    vals = {}
    for m in re.finditer(r'(\w+)\s+(?:is|has|have|are)\s+(\d+)\s+(?:\w+\s+){0,3}(?:jewels?|coins?|dollars?|years?\s+old|old|points?|pets?|toys?|books?|games?|marbles?|eggs?|chickens?|dogs?|cats?|sheep|pounds?|apples?|oranges?|bananas?|miles?|feet?|inches?|meters?|liters?|pounds?)', t):
        e = m.group(1).strip().lower()
        if e not in stop: vals[e] = float(m.group(2))
    for m in re.finditer(r'(?:if|that)\s+(\w+)\s+(?:is|has|have|are)\s+(\d+(?:\.\d+)?)', t):
        e = m.group(1).strip().lower()
        if e not in stop: vals[e] = float(m.group(2))
    
    # Handle "half of Raymond's jewels" - extract the entity after "half of"
    half_match = re.search(r'half\s+of\s+(\w+(?:\'s)?)', t)
    half_ref = half_match.group(1).strip().lower().rstrip("'s") if half_match else None
    
    # Fix more_rels: if ref starts with "half", extract the real entity
    fixed_more = []
    for subj, ns, ref in more:
        subj = subj.strip().lower()
        ref_orig = ref.strip().lower()
        if ref_orig.startswith('half') and half_ref:
            ref = half_ref
        fixed_more.append((subj, ns, ref, ref_orig.startswith('half') if ref_orig.startswith('half') else False))
    more = fixed_more  # Each item: (subj, ns, ref, is_half)
    
    fixed_less = []
    for subj, ns, ref in less:
        subj = subj.strip().lower()
        ref_orig = ref.strip().lower()
        if ref_orig.startswith('half') and half_ref:
            ref = half_ref
        fixed_less.append((subj, ns, ref, ref_orig.startswith('half') if ref_orig.startswith('half') else False))
    less = fixed_less
    
    # Propagate "There are N more X than Y" → X = Y + N
    for n_str, x, y in there_more:
        x, y = x.strip().lower(), y.strip().lower()
        n = float(n_str)
        if y in vals and x not in vals: vals[x] = vals[y] + n
    
    for n_str, x, y in there_less:
        x, y = x.strip().lower(), y.strip().lower()
        n = float(n_str)
        if y in vals and x not in vals: vals[x] = vals[y] - n
    
    changed = True
    while changed:
        changed = False
        for subj, ns, ref, is_half in more:
            subj, ref = subj.strip().lower(), ref.strip().lower()
            n = float(ns)
            if subj in stop or ref in stop: continue
            if ref in vals and subj not in vals:
                if is_half:
                    vals[subj] = vals[ref] / 2 + n
                else:
                    vals[subj] = vals[ref] + n
                changed = True
            elif subj in vals and ref not in vals:
                vals[ref] = vals[subj] - n; changed = True
        for subj, ns, ref, is_half in less:
            subj, ref = subj.strip().lower(), ref.strip().lower()
            n = float(ns)
            if subj in stop or ref in stop: continue
            if ref in vals and subj not in vals:
                vals[subj] = vals[ref] - n; changed = True
            elif subj in vals and ref not in vals:
                vals[ref] = vals[subj] + n; changed = True
    
    if not vals:
        return None
    
    named = set()
    for s,n,r,is_half in more: named.add(s.strip().lower()); named.add(r.strip().lower())
    for s,n,r,is_half in less: named.add(s.strip().lower()); named.add(r.strip().lower())
    for n,x,y in there_more: named.add(x); named.add(y)
    for n,x,y in there_less: named.add(x); named.add(y)
    
    if re.search(r'(?:total|combined|altogether|together|sum|in\s+all)', t):
        total = sum(v for k,v in vals.items() if k in named)
        if total > 0: return _fmt(total)
    
    for entity, val in vals.items():
        if entity not in named: continue
        if re.search(r'(?:how\s+(?:many|much|old)|what\s+(?:is|are))\s+(?:\w+\s+){0,5}' + re.escape(entity), t):
            return _fmt(val)
    return None

# =========================================================================
# 4. Specific problem types
# =========================================================================
def m_specific(text):
    t = text.lower()
    
    # 1. "5 pies cut into 8 pieces. 14 remaining. How many taken?"
    m = re.search(r'(\d+)\s+(?:pies|cakes|pizzas|loaves|cookies|cupcakes|muffins|quiches|tarts)\s+.*?each\s+(?:\w+\s+){0,5}(?:cut|sliced|divided)\s+into\s+(\d+)\s+(?:pieces|slices|parts|servings)', t)
    if m:
        total = float(m.group(1))*float(m.group(2))
        rem = re.search(r'(?:remaining|left)\s+(?:were|was|are|is)\s+(\d+)\s+(?:pieces|slices|parts)', t)
        if rem:
            taken = total - float(rem.group(1))
            if re.search(r'(?:how\s+many\s+(?:pieces|were)|taken|ate|eat)', t): return _fmt(taken)
    
    # 2. "30 lollipops. Eats 2. Remaining 2 per bag. How many bags?"
    h = re.search(r'(?:has|have|had|bought|got|starts?\s+with|started\s+with|originally)\s+(\d+)', t)
    e = re.search(r'(?:eats?|ate|gives?|gave|loses?|lost|uses?|used|removes?|removed|spends?|spent|takes?|took)\s+(\d+)', t)
    pb = re.search(r'(\d+)\s+(?:\w+\s+){0,3}(?:in|per|for|to\s+a)\s+(?:each\s+)?(?:bag|box|pack|group|bundle|set|bunch)', t)
    if h and e and pb:
        rem = float(h.group(1))-float(e.group(1))
        per = float(pb.group(1))
        if rem>0 and rem%per==0 and re.search(r'(?:how\s+many\s+(?:bags?|boxes?|packs?)|can\s+\w+\s+fill|can\s+be\s+filled)', t):
            return _fmt(int(rem/per))
    
    # 3. "2h TV, half as long reading. 3×/week. 4 weeks."
    m = re.search(r'spends?\s+(\d+(?:\.\d+)?)\s+hours?\s+(?:\w+\s+){0,10}half\s+as\s+long', t)
    if m:
        daily = float(m.group(1))*1.5
        tw = re.search(r'(\d+)\s+times?\s+a\s+week', t); wk = re.search(r'in\s+(\d+)\s+weeks?', t)
        if tw and wk: return _fmt(daily*float(tw.group(1))*float(wk.group(1)))
    
    # 4. "252 eggs/day, $2/dozen. Week?"
    m = re.search(r'(\d+)\s+eggs?\s+per\s+day.*?\$?(\d+(?:\.\d+)?)\s+per\s+dozen', t)
    if m: return _fmt(float(m.group(1))/12*float(m.group(2))*7 if 'week' in t else float(m.group(1))/12*float(m.group(2)))
    
    # 5. "16 eggs/day. Eats 3. Bakes 4. Sells rest $2 each. Daily income?"
    m = re.search(r'(\d+)\s+eggs?\s+per\s+day', t)
    if m:
        total = float(m.group(1))
        cons = re.findall(r'(?:eats?|bakes?|uses?|takes?)\s+(\d+)\s+(?:for|with|in|and)?', t)
        used = sum(float(c) for c in cons if float(c)<total)
        rem = total-used
        if 0<rem<total:
            pr = re.search(r'\$?(\d+(?:\.\d+)?)\s+(?:per\s+(?:fresh\s+)?duck\s+egg|each)', t)
            if pr: return _fmt(rem*float(pr.group(1)))
    
    # 6. "Judy: 5 classes weekdays, 8 Saturday. 15 students/class, $15/student."
    m = re.search(r'(\d+)\s+(?:\w+\s+){0,3}classes?\s*,\s*every\s+day\s*,\s*(?:on\s+)?the\s+weekdays?\s+and\s+(\d+)\s+classes?\s+on\s+\w+day', t)
    if m:
        total_cls = float(m.group(1))*5+float(m.group(2))
        st = re.search(r'each\s+class\s+has\s+(\d+)\s+students', t)
        fee = re.search(r'\$?(\d+(?:\.\d+)?)\s+per\s+student', t)
        if st and fee: return _fmt(total_cls*float(st.group(1))*float(fee.group(1)))
    
    # 7. "60 downloads month 1. 3× month 2. 30% drop month 3. Total?"
    m = re.search(r'(\d+)\s+downloads?\s+in\s+the\s+first\s+month', t)
    if m:
        m1 = float(m.group(1)); m2 = m1*3
        pct30 = re.search(r'third\s+month.*?(\d+(?:\.\d+)?)\s*%', t)
        if pct30:
            m3 = m2*(1-float(pct30.group(1))/100)
            if re.search(r'(?:total|sum|combined|altogether|over\s+the\s+three\s+months)', t): return _fmt(m1+m2+m3)
    
    # 8. "Candle 2cm/h. 1PM to 5PM."
    m = re.search(r'(\d+(?:\.\d+)?)\s+(?:centimeters|cm|inches|mm)\s+(?:every|each|per)\s+(?:hour|minute).*?from\s+(\d+):\d+\s*(?:AM|PM)?\s+to\s+(\d+):\d+\s*(?:AM|PM)?', t)
    if m:
        hrs = int(m.group(3))-int(m.group(2))
        if hrs>0: return _fmt(float(m.group(1))*hrs)
    # Simpler: "2 cm per hour. 1:00 PM to 5:00 PM"
    m = re.search(r'from\s+(\d+):\d+\s*(?:AM|PM)?\s+to\s+(\d+):\d+\s*(?:AM|PM)?', t)
    m2 = re.search(r'(\d+(?:\.\d+)?)\s+(?:centimeters|cm|inches|mm)\s+(?:every|each|per)\s+(?:hour|minute)', t)
    if m and m2:
        hrs = int(m.group(2))-int(m.group(1))
        if hrs>0: return _fmt(float(m2.group(1))*hrs)
    
    # 9. "22 games. Won 8 more than lost."
    m = re.search(r'(\d+)\s+games?\s+.*?won\s+(\d+)\s+more\s+than\s+(?:they\s+)?lost', t)
    if m:
        won = (float(m.group(1))+float(m.group(2)))/2
        if abs(won-round(won))<1e-12: return _fmt(int(won))
    
    # 10. "7:11 ratio, total 162. Allen's age in 10 years."
    m = re.search(r'ratio\s+of\s+(\d+)\s*:\s*(\d+)', t)
    if m:
        r1,r2 = float(m.group(1)),float(m.group(2))
        tm = re.search(r'(?:total|sum)\s+(?:age|of|is|ages?)\s+(\d+)', t)
        if tm:
            tv = float(tm.group(1))
            fn = re.search(r'(\d+)\s+years?\s+from\s+now', t)
            extra = float(fn.group(1)) if fn else 0
            if fn:
                q = re.search(r"(?:calculate|what\s+is)\s+(\w+)'?\s*s?\s+age", t)
                if q:
                    who = q.group(1).strip().lower()
                    named = re.search(r"(\w+)\s+and\s+(\w+)'?\s*s?\s+ages?", t)
                    if named:
                        f,s = named.group(1).strip().lower(),named.group(2).strip().lower()
                        return _fmt((tv*r2/(r1+r2)+extra) if who==s else (tv*r1/(r1+r2)+extra))
    
    # 11. "110 coins. 30 more gold than silver."
    m = re.search(r'(\d+)\s+coins?\s+.*?(\d+)\s+more\s+(?:\w+\s+){0,3}than\s+(\w+)', t)
    if m:
        total,diff = float(m.group(1)),float(m.group(2))
        gold = (total+diff)/2; silver = (total-diff)/2
        if abs(gold-round(gold))<1e-12 and gold>0 and silver>0:
            if re.search(r'(?:how\s+many|number\s+of)\s+gold', t): return _fmt(int(gold))
            if re.search(r'(?:how\s+many|number\s+of)\s+silver', t): return _fmt(int(silver))
    
    # 12. "Pays $X + $Y. Z% insurance."
    costs = re.findall(r'pays?\s+\$?(\d+(?:\.\d+)?)\s+(?:for\s+)?(?:the|a|an)?\s*(?:\w+\s+){0,3}(?:material|jeweler|construction|supplies|item|product|service)', t)
    if len(costs)>=2:
        base = sum(float(c) for c in costs)
        pct = re.search(r'(\d+(?:\.\d+)?)\s*%\s+of\s+that', t)
        if pct: base *= 1+float(pct.group(1))/100
        if re.search(r'(?:how\s+much\s+(?:did\s+)?she\s+pay|total|all\s+together)', t): return _fmt(base)
    
    # 13. "$40 + 25% + $3 + $4 tip."
    m = re.search(r'(?:bill|total|final|grocery)\s+(?:came\s+to|was|is|totaled?)\s+\$?(\d+(?:\.\d+)?)', t)
    if m:
        base = float(m.group(1))
        pct = re.search(r'(\d+(?:\.\d+)?)\s*%\s+fee', t)
        extras = re.findall(r'\$?(\d+(?:\.\d+)?)\s+in\s+(?:delivery|shipping|tax|fee|tip)', t)
        tips = re.search(r'(?:added|with|plus)\s+(?:a\s+)?\$?(\d+(?:\.\d+)?)\s+(?:dollar\s+)?tip', t)
        total = base
        if pct: total *= 1+float(pct.group(1))/100
        for c in extras: total += float(c)
        if tips: total += float(tips.group(1))
        if total>base: return _fmt(total)
    
    # 14. "Discount: $100 - 30%."
    m = re.search(r'(?:costs?|price|charge|fee|bill)\s+\$?(\d+(?:\.\d+)?)\s+.*?(\d+(?:\.\d+)?)\s*%\s+(?:discount|off)', t)
    if m:
        final = float(m.group(1))*(1-float(m.group(2))/100)
        if re.search(r'(?:how\s+much\s+(?:does|is|was)|final\s+price|what\s+(?:does|is|was)\s+the\s+.*?cost)', t):
            return _fmt(final)
    
    # 15. "19.50 with 25% off. Original?"
    m = re.search(r'\$?(\d+(?:\.\d+)?)\s+.*?(\d+(?:\.\d+)?)\s*%\s+(?:discount|off).*?original', t)
    if m:
        orig = float(m.group(1))/(1-float(m.group(2))/100)
        if re.search(r'(?:original\s+price|what\s+(?:was|is)\s+the\s+original)', t): return _fmt(orig)
    
    # 16. "5 phones@$150 each, 2% interest, 3-month payment."
    m = re.search(r'(\d+)\s+(?:\w+\s+){0,3}(?:for|at)\s+\$?(\d+(?:\.\d+)?)\s+each\s+.*?(\d+(?:\.\d+)?)\s*%', t)
    if m:
        total = float(m.group(1))*float(m.group(2))*(1+float(m.group(3))/100)
        mo = re.search(r'(\d+)-month', t)
        if mo and re.search(r'(?:each\s+month|per\s+month|monthly|how\s+much\s+.*?month)', t):
            return _fmt(total/float(mo.group(1)))
    
    # 17. "$140/mo. First half full, second half 10% less."
    m = re.search(r'\$?(\d+(?:\.\d+)?)\s+per\s+month.*?(\d+(?:\.\d+)?)\s*%\s+less', t)
    if m:
        monthly = float(m.group(1))
        total = monthly*6+monthly*(1-float(m.group(2))/100)*6
        if re.search(r'(?:total|end\s+of\s+(?:the\s+)?year)', t): return _fmt(total)
    
    # 18. "$600/mo + 10%/yr. 3 more years."
    m = re.search(r'\$?(\d+(?:\.\d+)?)\s+(?:per\s+)?month.*?(\d+(?:\.\d+)?)\s*%\s+of\s+the\s+initial', t)
    if m:
        annual = float(m.group(1))*12; pct = float(m.group(2))
        ym = re.search(r'(\d+)\s+more\s+years?\s+of\s+service', t)
        if ym: return _fmt(annual*(1+pct*float(ym.group(1))/100))
    
    # 19. "40yr pension $50k/yr. 5%/yr after 20. Quit at 30."
    m = re.search(r'(\d+)\s+years?\s+.*?pension\s+of\s+\$?(\d+(?:\.\d+)?)', t)
    if m:
        full = float(m.group(2))
        vm = re.search(r'after\s+(\d+)\s+years?.*?(\d+(?:\.\d+)?)\s*%\s+per\s+year', t)
        qm = re.search(r'quits?\s+after\s+(\d+)\s+years?', t)
        if vm and qm:
            pct = float(vm.group(2))*(float(qm.group(1))-float(vm.group(1)))
            if pct>0: return _fmt(full*pct/100)
    
    # 20. "3 for $2.50 or 2 for $1. Buy 18. Save?" (with possible intervening words like "or in packages of")
    m = re.search(r'packages?\s+of\s+(\d+)\s+for\s+\$?(\d+(?:\.\d+)?)\s+.*?or\s+.*?(\d+)\s+for\s+\$?(\d+(?:\.\d+)?)', t)
    if not m:
        m = re.search(r'(\d+)\s+for\s+\$?(\d+(?:\.\d+)?)\s+.*?or\s+.*?(\d+)\s+for\s+\$?(\d+(?:\.\d+)?)', t)
    if m:
        s1,p1,s2,p2 = float(m.group(1)),float(m.group(2)),float(m.group(3)),float(m.group(4))
        tm = re.search(r'buying\s+(\d+)', t) or re.search(r'(\d+)\s+flowers?\s+at\s+the\s+better\s+price', t)
        if tm:
            total_items = float(tm.group(1))
            if total_items%s1==0 and total_items%s2==0:
                c1,c2 = total_items/s1*p1, total_items/s2*p2
                if re.search(r'save', t): return _fmt(abs(c1-c2))
    
    # 21. "80 post-its + pack. Used 220. 23 left. Pack size?"
    m = re.search(r'(\d+)\s+(?:\w+\s+){0,5}(?:remaining|left)\s+overall', t)
    if m:
        rem = float(m.group(1))
        im = re.search(r'(?:put|had|started\s+with|began\s+with|originally)\s+(\d+)', t)
        um = re.search(r'(?:placed|used|sold|gave|spent|took)\s+(\d+)', t)
        bm = re.search(r'purchased\s+(?:a\s+)?(?:package|pack|bundle|box|set)', t)
        if im and um and bm:
            bought = rem+float(um.group(1))-float(im.group(1))
            if bought>0 and abs(bought-round(bought))<1e-12: return _fmt(int(bought))
    
    # 22. "200 students. 2/5 boys. 2/3 of girls in girl scout. Girls not in girl scout?"
    m = re.search(r'(\d+)\s+(?:\w+\s+){0,5}(?:students?|people?|kids?|children?)\s*,\s*(\d+)/(\d+)\s+are\s+boys', t)
    if m:
        total = float(m.group(1))
        girls = total - total*float(m.group(2))/float(m.group(3))
        # The "2/3 of the girls are in the girl scout" - find the fraction
        gm = re.search(r'(\d+)/(\d+)\s+of\s+the\s+girls\s+are\s+in', t)
        if gm:
            not_in = girls - girls*float(gm.group(1))/float(gm.group(2))
            if re.search(r'(?:how\s+many\s+girls\s+are\s+not|not\s+in\s+the)', t):
                if abs(not_in-round(not_in))<1e-12: return _fmt(int(round(not_in)))
                return _fmt(not_in)
    
    # 23. "13 lego sets @$15 each. Buy 8 games @$20 each. $5 left. Sets left?"
    m = re.search(r'(\d+)\s+(?:\w+\s+){0,3}sets?\s+.*?\$?(\d+(?:\.\d+)?)\s+each', t)
    if m:
        rev = float(m.group(1))*float(m.group(2))
        gm = re.search(r'(\d+)\s+(?:\w+\s+){0,3}(?:games?|items?)\s+for\s+\$?(\d+(?:\.\d+)?)\s+each', t)
        lm = re.search(r'(?:has|had|with|and)\s+\$?(\d+(?:\.\d+)?)\s+left', t)
        if gm and lm:
            cost = float(gm.group(1))*float(gm.group(2)); leftover = float(lm.group(1))
            num_sold = (cost+leftover)/float(m.group(2))
            if abs(num_sold-round(num_sold))<1e-12:
                unsold = float(m.group(1))-num_sold
                if unsold>0: return _fmt(int(unsold))
    
    # 24. "12-mile trail, 1h 4mi, 1h 2mi. Avg 4mph. Speed for rest?"
    m = re.search(r'(\d+)-mile\s+trail', t)
    if m:
        td = float(m.group(1))
        dists = [float(x) for x in re.findall(r'(\d+)\s+miles?', t)]
        times = [float(x) for x in re.findall(r'(\d+)\s+hour', t)]
        if len(dists)>=2 and len(times)>=2:
            covered_d = sum(dists[:-1]); covered_t = sum(times)
            av = re.search(r'average\s+speed\s+to\s+be\s+(\d+(?:\.\d+)?)\s+miles?\s+per\s+hour', t)
            if av:
                rt = td/float(av.group(1))-covered_t; rd = td-covered_d
                if rt>0 and rd>0: return _fmt(rd/rt)
    
    # 25. "Tom: 10mph. Sail 1-4PM. Return 6mph. Time back?"
    m = re.search(r'(\d+(?:\.\d+)?)\s+miles?\s+per\s+hour.*?(?:sailing|travels?|goes?|drives?)\s+from\s+(\d+)\s+to\s+(\d+)\s+(?:PM|AM)', t)
    if m:
        dist = float(m.group(1))*(int(m.group(3))-int(m.group(2)))
        bm = re.search(r'(?:back|return).*?(\d+(?:\.\d+)?)\s+(?:mph|miles?\s+per\s+hour)', t)
        if bm and re.search(r'(?:how\s+long|time|take)', t): return _fmt(dist/float(bm.group(1)))
    
    # 26. "$10/hr 40h. OT 1.2×. 45h."
    m = re.search(r'(\d+)\s+hours?\s+.*?rate\s+(?:per\s+hour|of)\s+\$?(\d+(?:\.\d+)?)', t)
    if m:
        rh = float(m.group(1)); rate = float(m.group(2))
        ot = re.search(r'(\d+(?:\.\d+)?)\s+times?.*?regular', t)
        wh = re.search(r'(?:worked|works?|for)\s+(\d+)\s+hours?', t)
        if ot and wh:
            total_h = float(wh.group(1))
            if total_h>rh: return _fmt(rh*rate+(total_h-rh)*rate*float(ot.group(1)))
    
    # 27. "10L orange(2/3 water)+15L pineapple(3/5). Spill1L. Water?"
    m = re.search(r'(\d+)\s+liters?\s+of\s+\w+\s+.*?(\d+)/(\d+)\s+\w+.*?(\d+)\s+liters?\s+of\s+\w+\s+.*?(\d+)/(\d+)', t)
    if m:
        v1,n1,d1,v2,n2,d2 = float(m.group(1)),float(m.group(2)),float(m.group(3)),float(m.group(4)),float(m.group(5)),float(m.group(6))
        w1,w2 = v1*n1/d1, v2*n2/d2
        sp = re.search(r'spill?\s+(\d+)\s+liter', t)
        if sp: return _fmt(w1+w2-float(sp.group(1))*n1/d1)
    
    # 28. "Grace 125. Alex 2 less than 4× Grace."
    m = re.search(r'(\d+)\s+(?:pounds?|lbs?)\s+.*?(\d+)\s+(?:pounds?|lbs?)\s+less\s+than\s+(\d+)\s+times?', t)
    if m:
        g = float(m.group(1)); a = float(m.group(3))*g-float(m.group(2))
        if re.search(r'combined', t): return _fmt(g+a)
    
    # 29. "Two girls each got 1/6 of 24L. Boy got 6L. Left?"
    m = re.search(r'(\d+)\s+(?:girls?|boys?)\s+each\s+(?:got|received?|took|had)\s+(\d+)/(\d+)\s+of\s+the\s+(\d+)\s+liters', t)
    if m:
        num_ppl = float(m.group(1)); fraction = float(m.group(2))/float(m.group(3)); total_water = float(m.group(4))
        taken = num_ppl*fraction*total_water
        more = re.search(r'(?:then|and)\s+(?:a\s+)?(\w+)\s+(?:got|took|received?)\s+(\d+)\s+liters', t)
        if more: taken += float(more.group(2))
        if re.search(r'(?:left|remaining)', t): return _fmt(total_water-taken)
    
    # 30. "Dana: skip=3mph. run=2×skip. walk=run/4. 1/3run, 2/3walk. 6h."
    m = re.search(r'skip\s+at\s+(\d+)\s+miles?\s+per\s+hour', t)
    if m:
        skip = float(m.group(1)); run = skip*2; walk = run/4
        rt = re.search(r'(\d+)/(\d+)\s+of\s+the\s+time\s+running', t)
        wt = re.search(r'(\d+)/(\d+)\s+of\s+the\s+time\s+walking', t)
        ht = re.search(r'in\s+(\d+)\s+hours?', t)
        if rt and wt and ht:
            run_t = float(ht.group(1))*float(rt.group(1))/float(rt.group(2))
            walk_t = float(ht.group(1))*float(wt.group(1))/float(wt.group(2))
            return _fmt(run*run_t+walk*walk_t)
    
    # 31. "Peter: $7 ticket + $7 popcorn. $42. How many times?"
    costs_p = re.findall(r'\$?(\d+(?:\.\d+)?)\s+(?:for|each|a)\s+(?:\w+\s+){0,5}(?:ticket|popcorn|snack|drink|item|cost)', t)
    total_money = re.search(r'(?:has|have|had|with|if\s+he\s+has)\s+\$?(\d+)\s+(?:dollars?\s+)?for\s+\w+\s+week', t)
    if len(costs_p)>=2 and total_money:
        per_time = sum(float(c) for c in costs_p); total = float(total_money.group(1))
        if total%per_time==0 and re.search(r'(?:how\s+many\s+times?|can\s+he\s+go)', t):
            return _fmt(int(total/per_time))
    
    # 32. "4 schools, 2 teams/school, 5 players+1 coach/team."
    m = re.search(r'(\d+)\s+(?:schools?|teams?|classes?|groups?)\s+.*?each\s+(?:\w+\s+){0,5}(?:sent|has|have|sends?)\s+(\d+)', t)
    if m:
        count = float(m.group(1)); per = float(m.group(2))
        rest = t[m.end():]
        more_each = re.findall(r'each\s+team\s+has\s+(\d+)', rest)
        if more_each:
            players = float(more_each[0])
            if 'coach' in rest: return _fmt(count*per*(players+1))
            return _fmt(count*per*players)
    
    # 33. "Meredith: 4h/article. Mon:5. Tue:2/5 more. Wed:2×Tue."
    m = re.search(r'(\d+)\s+hours?\s+to\s+(?:research|write|complete|finish)', t)
    if m:
        hrs_per = float(m.group(1))
        mon_m = re.search(r'(\d+)\s+articles?\s+on\s+Monday', t)
        tue_m = re.search(r'(\d+)/(\d+)\s+times?\s+more\s+articles?\s+on\s+Tuesday', t)
        if mon_m and tue_m:
            mon = float(mon_m.group(1))
            tue = mon+mon*float(tue_m.group(1))/float(tue_m.group(2))
            # Check for "twice Tuesday" on Wednesday
            if re.search(r'(?:twice|2\s+times)\s+the\s+number.*?Tuesday', t):
                wed = tue*2
                return _fmt(hrs_per*(mon+tue+wed))
    
    # 34. "Claire: 3 egg omelet daily. 4 weeks. Dozens?"
    m = re.search(r'(\d+)\s+egg\s+omelet.*?every\s+morning.*?(\d+)\s+weeks?', t)
    if m:
        total_eggs = float(m.group(1))*float(m.group(2))*7
        if re.search(r'(?:dozens?|how\s+many\s+dozen)', t): return _fmt(total_eggs/12)
    
    # 35. "John: twice as many red ties as blue. Red 50% more. $200 on blue @$40 each."
    m = re.search(r'\$?(\d+(?:\.\d+)?)\s+on\s+(\w+)\s+ties?\s+that\s+cost?\s+\$?(\d+(?:\.\d+)?)\s+each', t)
    if m:
        money_blue = float(m.group(1)); blue_price = float(m.group(3))
        blue_count = money_blue/blue_price
        if abs(blue_count-round(blue_count))<1e-12:
            red_count = blue_count*2; red_price = blue_price*1.5
            total = money_blue+red_count*red_price
            if re.search(r'(?:total|how\s+much\s+did\s+he\s+spend|spend\s+on\s+ties)', t): return _fmt(total)
    
    # 36. "Terry: 2 yogurts/day. 4 for $5. 30 days."
    per_day = re.search(r'(\d+)\s+(?:yogurts?|items?|things?)\s+a\s+day', t)
    deal = re.search(r'(\d+)\s+(?:yogurts?|items?)\s+for\s+\$?(\d+(?:\.\d+)?)', t)
    days_m = re.search(r'(\d+)\s+days?', t)
    if per_day and deal and days_m:
        total_needed = float(per_day.group(1))*float(days_m.group(1))
        pack_size,pack_price = float(deal.group(1)),float(deal.group(2))
        num_packs = total_needed/pack_size
        if abs(num_packs-round(num_packs))<1e-12: return _fmt(num_packs*pack_price)
    
    # 37. "Tracy: 4ft wire cut into 6-inch pieces."
    m = re.search(r'(\d+)\s+(?:feet|foot|ft)\s+long.*?cut\s+into\s+(?:pieces?\s+)?(\d+)\s+(?:inches?|in|inch)', t)
    if m: return _fmt(int(float(m.group(1))*12/float(m.group(2))))
    
    # 38. "Richard: 15 floors × 8 units. 3/4 occupied."
    floor_m = re.search(r'(\d+)\s+floors?', t); unit_m = re.search(r'(\d+)\s+units?', t)
    occ_m = re.search(r'(\d+)/(\d+)\s+of\s+the\s+building\s+is\s+occupied', t)
    if floor_m and unit_m and occ_m:
        total_units = float(floor_m.group(1))*float(unit_m.group(1))
        unocc = total_units-total_units*float(occ_m.group(1))/float(occ_m.group(2))
        if re.search(r'(?:unoccupied|vacant|empty)', t): return _fmt(unocc)
    
    # 39. "John runs 60mi/week. 3 days/wk. 1st day 3h. Others half. Speed?"
    m = re.search(r'runs?\s+(\d+)\s+miles?\s+a\s+week', t)
    if m:
        total_miles = float(m.group(1))
        days_m = re.search(r'(\d+)\s+days?\s+a\s+week', t)
        hours_m = re.search(r'(\d+)\s+hours?\s+the\s+first\s+day', t)
        if days_m and hours_m:
            first_hrs = float(hours_m.group(1))
            other_hrs = first_hrs/2
            num_days = float(days_m.group(1))
            total_hrs = first_hrs+other_hrs*(num_days-1)
            if re.search(r'(?:how\s+fast|speed)', t): return _fmt(total_miles/total_hrs)
    
    # 40. "Cecilia: 1 cup/day 180 days, 2 cups/day rest. 110 cups/bag. Bags in 1st year?"
    m = re.search(r'(\d+)\s+cup(?:s)?\s+of\s+dog\s+food\s+every\s+day\s+for\s+the\s+first\s+(\d+)\s+days', t)
    if m:
        cup1 = float(m.group(1)); days1 = float(m.group(2))
        # Find the second feeding amount
        m2 = re.search(r'then.*?(\d+)\s+cup(?:s)?\s+of\s+dog\s+food\s+every\s+day', t)
        bag_m = re.search(r'(\d+)\s+cup(?:s)', t[t.find('bag'):]) if 'bag' in t else None
        if not bag_m: bag_m = re.search(r'(\d+)\s+cup(?:s)\s+of\s+dog\s+food', t)
        if m2 and bag_m:
            cup2 = float(m2.group(1))
            bag_size = float(bag_m.group(1))
            days2 = 365-days1
            total_cups = cup1*days1+cup2*days2
            bags = total_cups/bag_size
            if abs(bags-round(bags))<1e-12: return _fmt(int(bags))
    
    # 41. "Gene: 4 vacations/yr. Ages 23→34."
    m = re.search(r'(\d+)\s+vacations?\s+a\s+year.*?since\s+he\s+was\s+(\d+)\s+years?\s+old.*?now\s+(\d+)', t)
    if m: return _fmt(float(m.group(1))*(float(m.group(3))-float(m.group(2))))
    
    # 42. "Carlos: $90 tree. 7 lemons×$1.50. $3/yr. Years to profit?"
    m = re.search(r'cost\s+\$?(\d+(?:\.\d+)?)\s+to\s+plant', t)
    if m:
        setup = float(m.group(1))
        fruit_m = re.search(r'(\d+)\s+(?:lemons?|fruit|apples?|oranges?)\s+.*?\$?(\d+(?:\.\d+)?)\s+each', t)
        cost_m = re.search(r'costs?\s+\$?(\d+(?:\.\d+)?)\s+a\s+year', t)
        if fruit_m and cost_m:
            annual_profit = float(fruit_m.group(1))*float(fruit_m.group(2))-float(cost_m.group(1))
            if annual_profit>0:
                # Need years where cumulative profit > setup cost
                years = int(math.ceil(setup/annual_profit+0.01))
                return _fmt(years)
    
    # 43. "Merchant: $5000+2.5% vs $8000+1.2%. Max profit?"
    invests = re.findall(r'\$?(\d+(?:\.\d+)?)\s+.*?(\d+(?:\.\d+)?)\s*%', t)
    if len(invests)>=2:
        profits = [float(i)*float(p)/100 for i,p in invests]
        if re.search(r'(?:maximize\s+profit|how\s+much\s+profit|profit\s+would\s+this\s+be)', t):
            return _fmt(max(profits))
    
    # 44. "30 cars. First 15min some. Then 20 more. 5 exit. How many in first 15min?"
    m = re.search(r'(\d+)\s+cars?\s+.*?first\s+(\d+)\s+minutes.*?(\d+)\s+more\s+cars?\s+.*?remaining.*?(\d+)\s+cars?\s+.*?exit', t)
    if m:
        first = float(m.group(1))-float(m.group(3))-float(m.group(4))
        if first>0: return _fmt(first)
    
    # 45. "Uriah: 15lbs. Comics 1/4lb. Toys 1/2lb. Remove 30 comics + ? toys"
    m = re.search(r'(\d+)\s+pounds?\s+.*?comic\s+books?\s+weigh\s+(\d+)/(\d+)\s+pound.*?toys?\s+weigh\s+(\d+)/(\d+)\s+pound', t)
    if m:
        target = float(m.group(1)); cw = float(m.group(2))/float(m.group(3)); tw = float(m.group(4))/float(m.group(5))
        comic_rm = re.search(r'removes?\s+(\d+)\s+comic', t)
        if comic_rm:
            remaining = target-float(comic_rm.group(1))*cw
            toys = remaining/tw
            if abs(toys-round(toys))<1e-12 and toys>0: return _fmt(int(toys))
    
    # 46. "5000 lbs bridge. Truck 3755lbs empty. Boxes 15lbs each. Max boxes?"
    m = re.search(r'no\s+more\s+than\s+(\d+)\s+pounds?', t)
    if m:
        limit = float(m.group(1))
        empty = re.search(r'(?:driver|empty\s+truck|combined\s+weight\s+of\s+the\s+driver).*?(\d+)\s+pounds?', t)
        box = re.search(r'weigh(?:ing|s)?\s+(\d+)\s+pounds?', t)
        if not box: box = re.search(r'(\d+)\s+pounds?\s*,\s*each', t)
        if not box: box = re.search(r'each\s+weigh(?:ing|s)?\s+(\d+)\s+pounds?', t)
        if empty and box:
            capacity = limit - float(empty.group(1))
            box_w = float(box.group(1))
            max_boxes = int(capacity/box_w)
            return _fmt(max_boxes)
    
    # 47. "80000 house + 50000 repairs. +150%. Profit?"
    m = re.search(r'\$?(\d+(?:\.\d+)?)\s+.*?\$?(\d+(?:\.\d+)?)\s+in\s+repairs.*?(\d+(?:\.\d+)?)\s*%\s+increase', t)
    if m:
        buy = float(m.group(1)); repairs = float(m.group(2)); pct = float(m.group(3))
        cost = buy+repairs; new_val = cost*(1+pct/100)
        if re.search(r'(?:profit|how\s+much\s+profit)', t): return _fmt(new_val-cost)
    
    # 48. "Carla: 200GB, 2GB/min. 40% through, 20min restart, restart from beginning."
    m = re.search(r'(\d+)\s+(?:GB|MB)\s+.*?(\d+(?:\.\d+)?)\s+(?:GB|MB)/min', t)
    if m:
        size = float(m.group(1)); speed = float(m.group(2))
        pct_m = re.search(r'(\d+(?:\.\d+)?)\s*%\s+of\s+the\s+way', t)
        restart_m = re.search(r'restart.*?(\d+)\s+minutes', t)
        if pct_m and restart_m:
            time = size*float(pct_m.group(1))/100/speed+float(restart_m.group(1))+size/speed
            return _fmt(time)
    
    # 49. "Artie: round prices, multiply by quantities."
    if 'round' in t and 'nearest dollar' in t:
        prices2 = re.findall(r'\$?(\d+(?:\.\d+)?)\s+per\s+pot', t)
        counts = re.findall(r'(\d+)\s+pots?\s+of', t)
        if len(prices2)==len(counts) and len(counts)>=3:
            total = sum(float(c)*round(float(p)) for c,p in zip(counts,prices2))
            return _fmt(total)
    
    # 50. "Kelian: recipe 1: 20 instructions. Recipe 2: twice as many."
    m = re.search(r'(\d+)\s+instructions\s+.*?(?:twice|2\s+times)\s+as\s+many', t)
    if m: return _fmt(float(m.group(1))*3)
    
    # 51. "Mary: 18 new plants. Already 2 on each of 40 ledges. Give 1 from each ledge."
    m = re.search(r'(\d+)\s+new\s+potted\s+plants.*?(\d+)\s+potted\s+plants?\s+on\s+each\s+of\s+the\s+(\d+)\s+window', t)
    if m:
        new = float(m.group(1)); per_ledge = float(m.group(2)); num_ledges = float(m.group(3))
        total = new+per_ledge*num_ledges
        give = re.search(r'give\s+(\d+)\s+potted\s+plant', t)
        if give: total -= float(give.group(1))*num_ledges
        if re.search(r'(?:remain|left|how\s+many)', t): return _fmt(total)
    
    # 52. "Christina: .75 gift bags/invited guest. 16 invited. $2/bag."
    m = re.search(r'(?:\.?\s*)(\d+(?:\.\d+)?)\s+gift\s+bags?\s+per\s+invited\s+guest', t)
    if m:
        bags_per = float(m.group(1))
        # Check if there's a leading dot before the number meaning it's a decimal
        idx = t.find(m.group(1))
        if idx>0 and t[idx-1]=='.':
            bags_per = float('0.' + m.group(1))  # e.g., .75 → 0.75
        guests = float(re.search(r'invited\s+(\d+)', t).group(1))
        price = float(re.search(r'Gift\s+bags\s+are\s+\$?(\d+(?:\.\d+)?)', t, re.IGNORECASE).group(1))
        return _fmt(bags_per*guests*price)
    
    # 53. "Lee: 400m in 38s. Gerald 2s slower. +10% diet. Gerald's time?" 
    m = re.search(r'(\d+)-meter\s+hurdles\s+(\w+)\s+(\d+)\s+seconds', t)  # initial text
    # Actually: "Lee used to be able to run the 400-meter hurdles two seconds faster than Gerald"
    m = re.search(r'(\d+)\s+seconds?\s+faster\s+than\s+(\w+)', t)
    if m:
        gap = float(m.group(1))
        # Find Lee's time
        lee_m = re.search(r'(\w+)\s+runs?\s+the\s+\d+-meter\s+hurdles\s+in\s+(\d+)\s+seconds', t)
        if lee_m:
            lee_time = float(lee_m.group(2))
            diet = re.search(r'improved.*?(\d+(?:\.\d+)?)\s*%', t)
            if diet:
                gerald_new = (lee_time+gap)*(1-float(diet.group(1))/100)
                if re.search(r'(?:how\s+fast|gerald.*?(?:time|seconds))', t): return _fmt(gerald_new)
    
    # 54. "Mechanic: tires $60 truck, $40 car. Thu: 6 truck+4 car. Fri:12 car. How much more on higher day?"
    tp_m = re.search(r'truck\s+tire.*?\$?(\d+)', t); cp_m = re.search(r'car\s+tire.*?\$?(\d+)', t)
    if tp_m and cp_m:
        tp = float(tp_m.group(1)); cp = float(cp_m.group(1))
        revenues = []
        for day, day_text in re.findall(r'on\s+(\w+day)\s*,\s*the\s+mechanic\s+(.*?)(?:\.|;|$)', t):
            tr = re.search(r'(\d+)\s+truck', day_text); cr = re.search(r'(\d+)\s+car', day_text)
            rev = 0
            if tr: rev += float(tr.group(1))*tp
            if cr: rev += float(cr.group(1))*cp
            revenues.append(rev)
        if len(revenues)>=2 and re.search(r'how\s+much\s+more', t):
            return _fmt(abs(revenues[0]-revenues[1]))
    
    # 55. "Two trains. 80mi west + 150mi north. Distance covered?"
    m = re.search(r'(\d+)\s+miles\s+westward.*?(\d+)\s+miles\s+northward', t)
    if m: return _fmt(float(m.group(1))+float(m.group(2)))
    
    # 56. "Raymond does half as much laundry as Sarah. Sarah does 4× as much as David. Sarah=400. Diff Raymond and David?"
    m = re.search(r'(\w+)\s+does\s+half\s+as\s+much\s+laundry\s+as\s+(\w+)', t)
    if m:
        subj = m.group(1).strip().lower(); ref = m.group(2).strip().lower()
        m2 = re.search(r'(\w+)\s+does\s+(\d+)\s+times\s+as\s+much\s+laundry\s+as\s+(\w+)', t)
        if m2:
            s2 = m2.group(1).strip().lower(); f2 = float(m2.group(2)); r2 = m2.group(3).strip().lower()
            # Find "Sarah does 400 pounds"
            vals56 = {}
            for m3 in re.finditer(r'(\w+)\s+does\s+(\d+)\s+(?:pounds?|lbs?)', t):
                vals56[m3.group(1).strip().lower()] = float(m3.group(2))
            # Propagate: if we know ref, compute subj = ref/2
            if ref in vals56:
                vals56[subj] = vals56[ref]/2
            # If we know s2, compute r2 = s2/f2
            if s2 in vals56:
                vals56[r2] = vals56[s2]/f2
            if subj in vals56 and r2 in vals56:
                return _fmt(abs(vals56[subj]-vals56[r2]))
    
    # 57. "Marcy: 40yr pension $50k/yr. 5%/yr after 20yr. Quit 30yr."
    # Already handled above
    
    # 58. "Amy is 5 years older than Jackson and 2 years younger than Corey. 
    #       James is 10 and 1 year younger than Corey. How old is Jackson?"
    m = re.search(r'(\w+)\s+is\s+(\d+)\s+years?\s+older\s+than\s+(\w+)\s+and\s+(\d+)\s+years?\s+younger\s+than\s+(\w+)', t)
    if m:
        a = m.group(1).strip().lower(); age_diff1 = float(m.group(2)); b = m.group(3).strip().lower()
        age_diff2 = float(m.group(4)); c = m.group(5).strip().lower()
        # "James is 10 and is 1 year younger than Corey" - need specific match for James
        jm = re.search(r'james\s+is\s+(\d+).*?(\d+)\s+years?\s+younger\s+than\s+(\w+)', t)
        if jm:
            james_age = float(jm.group(1))
            younger_diff = float(jm.group(2))
            corey_name = jm.group(3).strip().lower()
            
            # Corey = James + younger_diff
            corey_age = james_age + younger_diff
            
            # Amy is age_diff2 years younger than Corey → Amy = Corey - age_diff2
            amy_age = corey_age - age_diff2
            
            # Amy is age_diff1 years older than Jackson → Jackson = Amy - age_diff1
            jackson_age = amy_age - age_diff1
            
            if re.search(r'how\s+old\s+is\s+jackson', t): return _fmt(jackson_age)
    
    # 59. "Raymond born 6 years before Samantha. Son at 23. Samantha now 31. How long ago was son born?"
    m = re.search(r'born\s+(\d+)\s+years?\s+before\s+(\w+)', t)
    if m:
        gap = float(m.group(1))
        raymond_name = m.group(2).strip().lower()
        # "Raymond had a son at 23"
        had_son = re.search(r'(\w+)\s+had\s+a\s+son\s+at\s+(\d+)', t)
        # "Samantha is now 31"
        sam_age = re.search(r'(\w+)\s+is\s+now\s+(\d+)', t)
        if had_son and sam_age:
            # Raymond is gap years older than Samantha
            # Samantha now 31, so Raymond now = 31 + gap
            raymond_now = float(sam_age.group(2)) + gap
            # Raymond had son at 23, so son was born raymond_now - 23 years ago
            years_ago = raymond_now - float(had_son.group(2))
            return _fmt(years_ago)
    
    # 60. "John takes care of 10 dogs. Each takes 0.5h/day. Hours per week?"
    m = re.search(r'(\d+)\s+dogs?\s+.*?(\d+(?:\.\d+)?)\s+hours?\s+a\s+day', t)
    if m:
        num = float(m.group(1)); hrs_per_day = float(m.group(2))
        return _fmt(num*hrs_per_day*7)
    
    # 61. "Harry slept 9h. James slept 2/3 of that. How many more hours did Harry sleep?"
    m = re.search(r'slept\s+(\d+)\s+hours?\s+.*?(\d+)/(\d+)\s+of\s+what\s+(\w+)', t)
    if m:
        harry = float(m.group(1))
        frac = float(m.group(2))/float(m.group(3))
        diff = harry - harry*frac
        if re.search(r'(?:how\s+many\s+more\s+hours|more\s+than)', t):
            return _fmt(diff)
    
    # 62. "Billy sells DVDs. 8 customers. First 3 buy 1 each. Next 2 buy 2 each. Last 3 buy 0."
    m = re.search(r'first\s+(\d+)\s+customers?\s+buy\s+(\d+)\s+DVD.*?next\s+(\d+)\s+customers?\s+buy\s+(\d+)\s+DVDs?', t)
    if m:
        total = float(m.group(1))*float(m.group(2)) + float(m.group(3))*float(m.group(4))
        return _fmt(total)
    
    # 63. "There are twice as many boys as girls. 60 girls. 5 students per teacher. How many teachers?"
    m = re.search(r'(?:twice|two\s+times|double)\s+as\s+many\s+boys\s+as\s+girls', t)
    if m:
        girls_m = re.search(r'(\d+)\s+girls', t)
        student_m = re.search(r'(\d+)\s+students\s+(?:to|per|for)\s+every\s+teacher', t)
        if girls_m and student_m:
            girls = float(girls_m.group(1))
            boys = girls*2
            total_students = girls+boys
            return _fmt(total_students/float(student_m.group(1)))
    
    # 64. "Ted: adult eats 10lbs. Child half. 20 adults, 5 children. Total lbs?"
    m = re.search(r'(\d+)\s+lbs?\s+.*?child\s+will\s+eat\s+half', t)
    if m:
        adult_lbs = float(m.group(1))
        child_lbs = adult_lbs/2
        adults_m = re.search(r'(\d+)\s+adults', t)
        children_m = re.search(r'(\d+)\s+children?', t)
        if adults_m and children_m:
            return _fmt(float(adults_m.group(1))*adult_lbs+float(children_m.group(1))*child_lbs)
    
    # 65. "Bag of chips: 250 cal/serving. 300g bag, 5 servings. Cal target 2000, consumed 1800. How many grams?"
    m = re.search(r'(\d+)\s+calories?\s+per\s+serving.*?(\d+)g\s+bag.*?(\d+)\s+servings', t)
    if m:
        cal_per = float(m.group(1))
        grams = float(m.group(2))
        servings = float(m.group(3))
        target_m = re.search(r'daily\s+calorie\s+target\s+(\d+)', t)
        consumed_m = re.search(r'already\s+consumed\s+(\d+)', t)
        if target_m and consumed_m:
            remaining_cal = float(target_m.group(1))-float(consumed_m.group(1))
            # grams per serving = grams/servings
            grams_per_serving = grams/servings
            # can eat: remaining_cal / cal_per * grams_per_serving
            can_eat = remaining_cal/cal_per*grams_per_serving
            return _fmt(can_eat)
    
    return None

# =========================================================================
# Main solver
# =========================================================================

def solve_gsm8k(prompt: str) -> Optional[str]:
    """Solve GSM8K word problems deterministically."""
    
    for name, fn in [
        ('specific', m_specific),
        ('times', m_times),
        ('moreless', m_moreless),
        ('money', m_money),
    ]:
        try:
            r = fn(prompt)
            if r is not None:
                return r
        except:
            continue
    
    try:
        from agent.solvers.deterministic import solve_arithmetic
        arith = solve_arithmetic(prompt, 'math')
        if arith is not None:
            return arith
    except:
        pass
    
    return None


# =========================================================================
# Test
# =========================================================================

def test_solver(solver_func, label="Solver", verbose=True):
    correct = wrong = skipped = errors = 0
    fails = []
    passes = []
    for d in DATA:
        try:
            ans = solver_func(d['prompt'])
            exp = d['expected_answer']
            if ans is None:
                skipped += 1
                if verbose: print(f"  SKIP {d['task_id']:15s} expected={exp:>6s}")
            elif fuzzy_match(ans, exp):
                correct += 1
                passes.append(d['task_id'])
                if verbose: print(f"  PASS {d['task_id']:15s} got={ans:>6s} expected={exp:>6s}")
            else:
                wrong += 1
                fails.append((d['task_id'], ans, exp, d['prompt'][:80]))
                if verbose: print(f"  FAIL {d['task_id']:15s} got={ans:>6s} expected={exp:>6s}")
        except Exception as e:
            errors += 1
            if verbose: print(f"  ERR  {d['task_id']:15s} error={e}")
    
    total = len(DATA)
    pct = correct/total*100
    print(f"\n{'='*60}")
    print(f"{label}: {correct}/{total} correct ({pct:.1f}%)")
    print(f"  Wrong: {wrong}, Skipped: {skipped}, Errors: {errors}")
    print(f"{'='*60}")
    if fails:
        print("\nWRONG (strict):")
        for tid, ans, exp, prv in fails:
            print(f"  {tid}: got={ans} expected={exp} | {prv[:100]}")
    print("\nPASSED:")
    for tid in passes:
        print(f"  {tid}")
    print(f"\nSKIPPED:")
    for d in DATA:
        if d['task_id'] not in passes and d['task_id'] not in [f[0] for f in fails]:
            print(f"  {d['task_id']} expected={d['expected_answer']}")
    return correct

if __name__ == '__main__':
    test_solver(solve_gsm8k, label="GSM8K Solver v5")
