#!/usr/bin/env python3
"""
Deep root cause analysis — FINAL.
Builds structured failure taxonomy for all 678 unique sentiment questions.
"""

import json, re
from pathlib import Path
BASE = Path("/home/artem/dev/amd-hackathon/data/eval")

def fuzzy_match(a, e):
    a, e = a.strip().lower(), e.strip().lower()
    if not a or not e: return False
    if a == e: return True
    if len(e) <= 20 and e in a: return True
    if len(a) <= 20 and a in e: return True
    na = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a)]
    ne = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", e)]
    if na and ne:
        an, en = na[-1], ne[-1]
        if en != 0 and abs((an-en)/en) <= 0.01: return True
        if an == en: return True
    ta = set(t for t in re.split(r"[^a-zA-Z0-9.]+", a) if t)
    te = set(t for t in re.split(r"[^a-zA-Z0-9.]+", e) if t)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8: return True
    return False

def load_json(path):
    with open(path) as f: return json.load(f)

def extract_questions(data, fl):
    r = []
    if isinstance(data, dict) and "questions" in data:
        for q in data["questions"]:
            if "sentiment" in str(q.get("category","")).lower(): r.append(fq(q,fl))
    elif isinstance(data, list):
        for q in data:
            if "sentiment" in str(q.get("category","")).lower(): r.append(fq(q,fl))
    return r

def fq(q, fl):
    e = q.get("expected_answer","")
    if not e and "gold" in q and isinstance(q["gold"], dict): e = q["gold"].get("answer","")
    return {"task_id":q.get("task_id",f"x-{hash(q['prompt'])%100000}"),
            "prompt":q["prompt"],"expected_answer":e,
            "source":q.get("source",fl),"file":fl}

# ---- Lexicons ----
POS = {"great","good","excellent","wonderful","fantastic","amazing","brilliant",
       "beautiful","love","loved","lovely","perfect","best","superb","outstanding",
       "remarkable","impressive","enjoyable","delightful","charming","funny",
       "enjoy","enjoyed","succeed","success","successfully","friendly","delicious",
       "talented","riveting","mesmerizing","astounding","refreshing","enriched",
       "appreciate","knowledgeable","helpful","happy","glad","thank","thanks",
       "pleased","satisfied","recommend","innovative","powerful","smooth","seamless",
       "gorgeous","magnificent","stunning","breathtaking","masterpiece",
       "entertaining","captivating","engaging","compelling","thrilling",
       "hilarious","insightful","clever","classic","masterful","refreshing","witty",
       "skillful","sublime","exquisite","flawless","delight","pleasure","joy",
       "incredible","terrific","awesome","superior","phenomenal","extraordinary",
       "genius","triumph","gem","effective","nicely","solid","decent","fine","nice",
       "efficient","efficiency","finest","improved","intuitive","fast","quick","easy",
       "worth","worthwhile","favorite","sweet","warm","touching","moving","poignant",
       "fascinating","exhilarating","passionate","vibrant","luminous","radiant",
       "exuberant","vivid","grace","graceful","enchantment","enchanting",
       "thoughtful","sincere","honest","genuine","authentic","sensitive","nuanced",
       "layered","complex","exciting","uplifting","inspired","expressive","electric",
       "energetic","lively","colorful","captures","embodies","transcendent",
       "triumphant","rousing","bracing","hopeful","affecting","resonant",
       "immersive","spellbinding","hauntingly","unexpectedly","surprisingly",
       "likable","likeable","engrossing","unpretentious","down-to-earth",
       "earthy","crisp","sharp","incisive","perceptive","illuminating",
       "rewarding","satisfying","fulfilling","wholesome","heartwarming"}

NEG = {"terrible","awful","horrible","atrocious","appalling","dreadful","hideous",
       "repulsive","disgusting","abysmal","godawful","unwatchable","insufferable",
       "intolerable","abomination","travesty","disaster","catastrophe","debacle",
       "bad","poor","worse","worst","lousy","mediocre","pathetic","boring","dull",
       "tedious","tiresome","monotonous","pointless","useless","lame","weak",
       "shallow","flat","bland","predictable","formulaic","stale","unoriginal",
       "uninspired","unconvincing","unfunny","frustrating","infuriating","annoying",
       "irritating","disappointing","disappointment","disappointed","letdown",
       "offensive","insulting","embarrassing","laughable","ridiculous","absurd",
       "nonsense","stupid","dumb","confusing","mess","muddled","overrated",
       "overhyped","pretentious","hate","hated","sucks","suck","stinks","stink",
       "waste","wasted","crashes","crash","buggy","broken","malfunction",
       "suffers","suffer","lack","lacking","failed","failure","fail","missing",
       "missed","lost","loss","painful","painfully","agonizing","slog","chore",
       "wooden","stiff","forced","contrived","miscast","underwhelming",
       "depressing","miserable","bleak","grim","dark","violence","violent",
       "unpleasant","unappealing","unsatisfying","unfortunately","alas",
       "shock","shocking","upset","upsetting","disturbing","troubling",
       "uncomfortable","jarring","numbing","numb","harsh","brutal",
       "grating","cringeworthy","awkward","embarrassing","hollow","empty",
       "meaningless","uninteresting","forgettable","disjointed","incoherent",
       "improbable","unbelievable","unconvincing","unmemorable","unremarkable",
       "unexceptional","pedestrian","commonplace","derivative","generic",
       "cliched","clichéd","soulless","heartless","cold","mechanical",
       "amateurish","amateur","half-baked","unfinished","sloppy","careless",
       "disjointed","uneven","erratic","inconsistent","messy","chaotic",
       "dismal","hopeless","fruitless","futile","pointless"}

def get_text(prompt):
    t = prompt
    for p in [
        r'^Classify the sentiment[^:]*:\s*\n*\n*["\']?(.*?)["\']?\s*$',
        r'^Classify as [^:]*:\s*\n*\n*["\']?(.*?)["\']?\s*$',
        r'^Classify the sentiment\s+of\s+(?:this\s+)?\w*\s*review\s+as\s+[^:]+:\s*["\']?(.*?)["\']?\s*$',
        r'^Classify the sentiment\s+of\s+the\s+following\s+text[^:]*:\s*\n*["\']?(.*?)["\']?\s*$',
        r'^Classify the sentiment\s+of\s+this\s+review:\s*["\']?(.*?)["\']?\s*$',
        r'^Classify the sentiment:\s*["\']?(.*?)["\']?\s*$',
        r'^Classify the sentiment\s+of\s+this\s+(?:customer|restaurant|flight|product|service)\s+review\s+as\s+[^:]+:\s*(.*?)$',
        r'^Classify the sentiment\s+of\s+this\s+feedback\s+as\s+[^:]+:\s*(.*?)$',
    ]:
        m = re.search(p, prompt, re.DOTALL)
        if m:
            t = m.group(1).strip().strip('"').strip("'")
            break
    return t

def analyze(text):
    tl = text.lower()
    words = re.findall(r"[a-zA-Z0-9']+", tl)
    ws = set(w for w in words)
    pc = sum(1 for k in POS if k in ws)
    nc = sum(1 for k in NEG if k in ws)
    wc, cc = len(words), len(text)
    
    sarc = any(re.search(p,tl) for p in [
        r'\boh\s+(brilliant|great|wonderful|fantastic|love)\b',
        r"\bjust\s+what\s+i\s+(needed|wanted)\b",
        r"\bhighlight\s+of\s+my\s+day\b", r"\befficiency\s+at\s+its\s+finest\b",
        r"\btruly\s+['\"]?(improved|intuitive|amazing)['\"]?\b",
        r"\banother\s+(brilliant|great|fantastic)\b",
        r"\byou're\s+so\s+(brave|smart|clever|talented)\b",
        r"\bi\s+(really\s+)?admire\b.{0,40}\bwrong\b",
        r"\bfor\s+someone\s+with\s+your\s+qualifications\b",
        r"\byou('ve| have) done remarkably well\b",
        r"\ba\s+new\s+low\b",
    ])
    hedge = any(re.search(p,tl) for p in [
        r"\bnot\s+(entirely|completely|totally|wholly)\s+(terrible|bad|awful)\b",
        r"\bcould\s+be\s+worse\b", r"\bcould\s+do\s+worse\b",
        r"\bnot\s+too\s+bad\b", r"\bi\s+(suppose|guess)\s+it'?s?\s+not\b",
    ])
    faint = any(re.search(p,tl) for p in [
        r"\b(at\s+least|i\s+guess|i\s+suppose)\s+(it'?s?|it\s+is)\s+(ok|okay|fine|acceptable)\b",
        r"\bis\s+not\s+(the\s+worst|that\s+bad|a\s+total)\b",
        r"\b(they|the\s+band|he|she)\s+(really\s+)?seem(ed)?\s+like\s+(they\s+were\s+)?having\s+fun\b",
    ])
    has_but = bool(re.search(r'\bbut\b', tl))
    has_contrast = has_but or bool(re.search(r'\b(however|although|though|even though)\b', tl))
    scare = bool(re.search(r"['\"](intuitive|improved|amazing|great|best|love|brilliant|fantastic|wonderful|perfect)['\"]", tl))
    factual = any(m in tl for m in ["flagged","transactions","according to","analysts",
        "quarterly earnings","forecast","reported","announced","algorithm",
        "the new policy","shares traded","market share","earnings report"])
    long = cc > 300 or wc > 50
    
    # Keyword-level simulation
    net = pc - nc
    if long and has_contrast and pc > 0 and nc > 0:
        model_output = "mixed"
    elif sarc and nc > 0:
        # Sarcasm with positive words but negative context → keyword model picks wrong
        model_output = "positive" if pc > 0 else "negative"
    elif hedge:
        model_output = "neutral"
    elif faint:
        model_output = "positive"
    elif long and pc > 0 and nc > 0:
        model_output = "mixed"
    elif has_but and pc > 0 and nc > 0:
        model_output = "positive" if pc > nc else "negative" if nc > pc else "mixed"
    elif pc > nc + 1: model_output = "positive"
    elif nc > pc + 1: model_output = "negative"
    elif pc > 0 and nc > 0 and pc == nc and wc < 20: model_output = "neutral"
    else: model_output = "positive" if pc > nc else "negative" if nc > pc else "neutral"
    
    return {"pc":pc,"nc":nc,"sarc":sarc,"hedge":hedge,"faint":faint,"but":has_but,
            "contrast":has_contrast,"scare":scare,"factual":factual,"long":long,
            "wc":wc,"cc":cc,"model_output":model_output}

def classify(prompt, expected, ft):
    el = expected.lower().strip()
    
    # TIER 1: Explicit markers in expected answer
    for m in ["sarcastic","sarcasm","ironic","irony","dismissive","condescending",
              "backhanded","passive-aggressive","deadpan","mockery"]:
        if m in el: return "SARCASM"
    if "hedging" in el or "understatement" in el: return "HEDGING"
    if "mixed" in el and "mixed" not in expected[:6].lower(): return "MIXED_SIGNALS"
    if "both positive and negative" in el or "both sides" in el: return "MIXED_SIGNALS"
    if "faint praise" in el or ("predominantly negative" in el and "faint" in el): return "FAINT_PRAISE"
    if "neutral factual" in el or "factual report" in el: return "NEUTRAL_NEGATIVE"
    if "charged keywords" in el: return "NEUTRAL_NEGATIVE"
    if "contradict" in el: return "CONTRADICTION"
    
    # Rubric detection
    if len(expected) > 80 and any(w in el for w in ["acceptable","does not pass","must cover","acknowledge"]):
        return "EXPLANATION_BIAS"
    if len(expected) > 120 and not any(w in el for w in ["positive","negative","neutral","mixed"]):
        return "EXPLANATION_BIAS"
    if not any(w in el for w in ["positive","negative","neutral","mixed"]):
        return "FORMAT_MISMATCH" if len(expected) > 50 else "FORMAT_MISMATCH"
    
    gt = ""
    for w in ["positive","negative","neutral","mixed"]:
        if w in el: gt = w; break
    
    # TIER 2: Structural
    if ft["sarc"] or ft["scare"]: return "SARCASM"
    if ft["hedge"]: return "HEDGING"
    if ft["faint"]: return "FAINT_PRAISE"
    if ft["factual"] and ft["nc"] > 0: return "NEUTRAL_NEGATIVE"
    if ft["long"] and ft["pc"] > 0 and ft["nc"] > 0: return "MIXED_SIGNALS"
    if ft["but"] and ft["pc"] > 0 and ft["nc"] > 0: return "MIXED_SIGNALS"
    
    # TIER 3: Keyword misalignment
    if gt == "negative" and ft["pc"] > ft["nc"]: return "KEYWORD_MISMATCH"
    if gt == "positive" and ft["nc"] > ft["pc"]: return "KEYWORD_MISMATCH"
    
    # TIER 4: Subtle language
    if ft["pc"] == 0 and ft["nc"] == 0 and gt: return "SUBTLE_LANGUAGE"
    if ft["pc"] <= 1 and ft["nc"] <= 1 and gt and ft["wc"] < 30: return "SUBTLE_LANGUAGE"
    
    return "NONE"

def extract_gt(expected):
    el = expected.lower().strip()
    if el in ("positive","negative","neutral","mixed"): return el
    for p in ["sentiment:","label:","answer:"]:
        if p in el:
            a = el.split(p,1)[1].strip()
            for w in ["positive","negative","neutral","mixed"]:
                if w in a: return w
    ws = re.findall(r'(?:^|\W)(positive|negative|neutral|mixed)(?:\W|$)', el)
    if ws: return ws[0]
    for w in ["positive","negative","neutral","mixed"]:
        if w in el: return w
    return ""

def main():
    print("="*70)
    print("SENTIMENT FAILURE ANALYSIS — STRUCTURED TAXONOMY")
    print("="*70)
    print()
    
    srcs = {"training-v1":BASE/"training-v1.json","training-v2":BASE/"training-v2.json",
            "training-v3":BASE/"training-v3.json","validation-v1":BASE/"validation-v1.json",
            "validation-v2":BASE/"validation-v2.json","validation-v3":BASE/"validation-v3.json",
            "sentiment_combined_25":BASE/"sentiment_combined_25.json",
            "sst2_100":BASE/"tests/sst2_100.json",
            "complexity_eval_40":BASE/"tests/complexity_eval_40.json",
            "eval_60_medium_hard":BASE/"primary/eval_60_medium_hard.json",
            "eval_hard_218":BASE/"primary/eval_hard_218.json",
            "eval_60_docx_style":BASE/"primary/eval_60_docx_style.json",
            "build-A-40":BASE/"generated/build-A-40.json",
            "build-B-40":BASE/"generated/build-B-40.json",
            "sentiment_comprehensive_hard":BASE/"generated/sentiment_comprehensive_hard.json",
            "gen1":BASE/"generated/eval_from_datasets_20260712_172357.json",
            "gen2":BASE/"generated/eval_from_datasets_20260712_172426.json",
            "gen3":BASE/"generated/eval_from_datasets_20260712_172443.json"}
    
    all_q = []
    for lb, p in srcs.items():
        if p.exists():
            d = load_json(p)
            all_q.extend(extract_questions(d, lb))
    
    seen = set()
    uniq = []
    for q in all_q:
        if q["prompt"].strip() not in seen:
            seen.add(q["prompt"].strip())
            uniq.append(q)
    
    total = len(uniq)
    print(f"Loaded {len(all_q)} raw → {total} unique\n")
    
    # Analyze + test each label against expected
    tc = {}
    pq = []
    sb = {}
    failures = []
    
    for q in uniq:
        p, e = q["prompt"], q["expected_answer"]
        text = get_text(p)
        ft = analyze(text)
        
        # Test ALL 4 labels against expected via fuzzy_match
        labels = ["positive","negative","neutral","mixed"]
        fm_results = {l: fuzzy_match(l, e) for l in labels}
        model_out = ft["model_output"]
        would_pass = fm_results.get(model_out, False)
        
        # Alternative: any valid label passes
        any_pass = any(fm_results.values())
        
        ct = classify(p, e, ft)
        gt = extract_gt(e)
        
        # Track type stats
        if ct not in tc:
            tc[ct] = {"c":0,"ex":[],"lbls":set(),"srcs":{},"correct":0,"wrong":0}
        tc[ct]["c"] += 1
        tc[ct]["lbls"].add(gt)
        tc[ct]["correct"] += 1 if would_pass else 0
        tc[ct]["wrong"] += 1 if not would_pass else 0
        if len(tc[ct]["ex"]) < 5:
            tc[ct]["ex"].append(p[:150])
        s = q.get("source", q.get("file","?"))
        tc[ct]["srcs"][s] = tc[ct]["srcs"].get(s, 0) + 1
        
        fl = q.get("file","?")
        if fl not in sb: sb[fl] = {"t":0,"bt":{}}
        sb[fl]["t"] += 1
        sb[fl]["bt"][ct] = sb[fl]["bt"].get(ct, 0) + 1
        
        # Reason text
        reason_map = {
            "SARCASM": "Positive-sounding words (brilliant, great, highlight, finest) used in clearly negative/ironic context; keyword-biased model latches onto surface positivity",
            "HEDGING": "Understatement/hedging ('not entirely terrible', 'could be worse') resists binary classification; model misled by keywords within hedged constructions",
            "MIXED_SIGNALS": "Text contains both positive and negative elements with contrastive structure (but/however/although); model picks one polarity based on keyword density rather than recognizing mixed nature",
            "FAINT_PRAISE": "Faint/muted praise ('not that bad', 'at least it's ok') used as concession within essentially negative content; model overweights faint positive signal",
            "NEUTRAL_NEGATIVE": "Factual reporting of objectively negative events without evaluative language; model misled by charged keywords in neutral register",
            "EXPLANATION_BIAS": "Expected answer is a grading rubric rather than a single sentiment label; format mismatch with one-word output prompt",
            "CONTRADICTION": "Text contradicts itself between positive and negative poles",
            "FORMAT_MISMATCH": "Expected answer contains no standard sentiment label",
            "KEYWORD_MISMATCH": "Text has sentiment keywords but they point to wrong polarity; model misled by salient but misleading lexical items",
            "SUBTLE_LANGUAGE": "Sentiment conveyed through nuanced vocabulary or contextual inference rather than explicit sentiment keywords; standard lexicons insufficient",
            "NONE": "Standard question classifiable via keyword-based approach",
        }
        
        pq.append({"task_id":q["task_id"],"prompt":p[:200]+"...",
                   "expected":e[:150]+"...",
                   "would_classify_as":model_out if not would_pass else model_out,
                   "failure_type":ct if ct != "NONE" else "NONE",
                   "reason":reason_map.get(ct, "")})
        
        if not would_pass:
            failures.append(q)
    
    # Summary
    fail_count = sum(tc[t]["wrong"] for t in tc if t != "NONE")
    print(f"{'='*70}")
    print(f"TAXONOMY")
    print(f"{'='*70}")
    print(f"Estimated overall accuracy (keyword model): "
          f"{(sum(tc[t]['correct'] for t in tc))/total*100:.1f}%")
    print(f"Questions where one of 4 labels CAN match: "
          f"{sum(1 for q in uniq if any(fuzzy_match(l, q['expected_answer']) for l in ['positive','negative','neutral','mixed']))}/{total}")
    print()
    
    for ct, s in sorted(tc.items(), key=lambda x:-x[1]["c"]):
        pct = s["c"]/total*100
        acc = s["correct"]/(s["correct"]+s["wrong"])*100 if (s["correct"]+s["wrong"]) > 0 else 0
        bar = "█"*int(pct/2)+"░"*(50-int(pct/2))
        print(f"  {ct:25s} {s['c']:4d} ({pct:5.1f}%) {bar}")
        print(f"    Labels: {s['lbls']}  |  Simulated acc: {acc:.0f}%")
        print(f"    Top: {dict(sorted(s['srcs'].items(),key=lambda x:-x[1])[:3])}")
    
    # Print detailed per-type info
    print(f"\n{'='*70}")
    print("PER-CATEGORY ANALYSIS")
    print(f"{'='*70}")
    for ct, s in sorted(tc.items(), key=lambda x:-x[1]["c"]):
        if ct == "NONE": continue
        print(f"\n--- {ct} ({s['c']} questions, {s['correct']/(s['correct']+s['wrong'])*100:.0f}% simulated acc) ---")
        if s["ex"]:
            print(f'  Example: "{s["ex"][0][:100]}..."')
    
    print(f"\n{'='*70}")
    print("SOURCE BREAKDOWN")
    print(f"{'='*70}")
    for fl, s in sorted(sb.items(), key=lambda x:-x[1]["t"]):
        bt = ", ".join(f"{t}:{c}" for t,c in sorted(s["bt"].items(),key=lambda x:-x[1]) if c >= 3)
        print(f"  {fl}: {s['t']} — {bt if bt else 'scattered'}")
    
    # Build final JSON
    p_map = {
        "SARCASM": "Positive keywords used in clearly negative/ironic context; keyword model misses pragmatic negation",
        "HEDGING": "Hedged/understated language that avoids clear sentiment polarity",
        "MIXED_SIGNALS": "Contrastive structure with both positive and negative elements",
        "FAINT_PRAISE": "Faint/minimal praise as concession within negative context",
        "NEUTRAL_NEGATIVE": "Factual reporting of negative events without evaluative language",
        "EXPLANATION_BIAS": "Expected answer is grading rubric, not single label",
        "CONTRADICTION": "Text explicitly contradicts itself",
        "FORMAT_MISMATCH": "Expected answer lacks standard sentiment label",
        "KEYWORD_MISMATCH": "Keywords point to wrong polarity",
        "SUBTLE_LANGUAGE": "Nuanced vocabulary not in standard sentiment lexicons",
        "NONE": "Standard keyword-classifiable question",
    }
    f_map = {
        "SARCASM": "Detect irony: scare-quote checks, contrast between positive lexis and negative context",
        "HEDGING": "Recognize hedging constructions; classify heavily qualified statements as neutral",
        "MIXED_SIGNALS": "Detect contrastive structure; classify mixed when both polarities present",
        "FAINT_PRAISE": "Distinguish faint praise from genuine enthusiasm; check if positive is undermined",
        "NEUTRAL_NEGATIVE": "Detect factual register; charged keywords in reporting do not indicate sentiment",
        "EXPLANATION_BIAS": "Format expected answers as canonical labels; use LLM judge for rubric grading",
        "CONTRADICTION": "Classify as mixed when text has explicit self-contradiction",
        "FORMAT_MISMATCH": "Standardize expected answers to include canonical sentiment label",
        "KEYWORD_MISMATCH": "Add negation handling, context-weighted keywords, longer-range dependencies",
        "SUBTLE_LANGUAGE": "Expand sentiment lexicon; use embedding-based classifiers; fine-tune on SST-2",
        "NONE": "No fix needed",
    }
    
    out = {"meta":{"total_questions":total,
                   "failure_count":sum(tc[t]["wrong"] for t in tc if t != "NONE"),
                   "accuracy_estimated":round(sum(tc[t]["correct"] for t in tc)/total, 3)},
           "failure_types":{},
           "per_question":pq}
    
    for ct, s in sorted(tc.items(), key=lambda x:-x[1]["c"]):
        acc = round(s["correct"]/(s["correct"]+s["wrong"]), 3) if (s["correct"]+s["wrong"])>0 else 0
        out["failure_types"][ct] = {
            "count":s["c"],"accuracy_on_these":acc,
            "examples":[e[:250] for e in s["ex"][:3]],
            "pattern":p_map.get(ct,""),"fix_hint":f_map.get(ct,""),
        }
    
    outpath = BASE / "sentiment_failure_analysis.json"
    with open(outpath, "w") as f: json.dump(out, f, indent=2, ensure_ascii=False)
    
    print(f"\nWritten to {outpath}")
    print(f"  {len(pq)} per-question entries")
    print(f"  {len(out['failure_types'])} failure type categories")
    
    # Final conclusions
    print(f"\n{'='*70}")
    print("ROOT CAUSE CONCLUSIONS")
    print(f"{'='*70}")
    hard = {k:v for k,v in tc.items() if k != "NONE" and v["c"] > 0}
    total_hard = sum(v["c"] for v in hard.values())
    for ct, s in sorted(hard.items(), key=lambda x:-x[1]["c"]):
        print(f"  {ct:<20s}: {s['c']:4d} q ({s['c']/total*100:5.1f}%) — sim acc {s['correct']/(s['correct']+s['wrong'])*100:.0f}%")
    print(f"\n  Total: {total_hard} hard ({total_hard/total*100:.1f}%) + {tc.get('NONE',{}).get('c',0)} easy")
    print(f"\n  Primary root causes of 25-54% accuracy on 300-set:")
    print(f"  1. SUBTLE_LANGUAGE ({tc.get('SUBTLE_LANGUAGE',{}).get('c',0)}q, {tc.get('SUBTLE_LANGUAGE',{}).get('c',0)/total*100:.0f}%):")
    print(f"     SST-2 derived texts use nuanced vocabulary beyond standard lexicons.")
    print(f"  2. MIXED_SIGNALS ({tc.get('MIXED_SIGNALS',{}).get('c',0)}q, {tc.get('MIXED_SIGNALS',{}).get('c',0)/total*100:.0f}%):")
    print(f"     Long mixed IMDB reviews with contrastive positive/negative structure.")
    print(f"  3. KEYWORD_MISMATCH ({tc.get('KEYWORD_MISMATCH',{}).get('c',0)}q, {tc.get('KEYWORD_MISMATCH',{}).get('c',0)/total*100:.0f}%):")
    print(f"     Keywords point to the wrong polarity label.")
    print(f"  4-6. SARCASM/HEDGING/FAINT_PRAISE/NEUTRAL_NEGATIVE ({sum(tc.get(t,{}).get('c',0) for t in ['SARCASM','HEDGING','FAINT_PRAISE','NEUTRAL_NEGATIVE'])}q):")
    print(f"     Pragmatic phenomena that simple keyword models cannot handle.")

if __name__ == "__main__":
    main()
