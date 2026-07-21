#!/usr/bin/env python3
"""
Test script for summarization approaches on 1B models.
Run: python3 test_summarization_approaches.py

Tests 8 prompt strategies on 5 xsum questions across 3 models.
Can also test Fireworks API if FIREWORKS_API_KEY is set.
"""

import json
import re
import sys
import time
import gc
import os

MODEL_PATHS = {
    "qwen2.5-1.5b": "/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-coder": "/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "gemma-3-1b": "/home/artem/models/gemma-3-1b-it-Q4_K_M.gguf",
}

def fuzzy_match(answer: str, expected: str) -> bool:
    a, e = answer.strip().lower(), expected.strip().lower()
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


# ── Test questions ──
TEST_QUESTIONS = [
    {
        "id": "xsum-260eec64733b",
        "prompt": "Summarize: The 26-year-old has made just one appearance since joining Posh in November, scoring in the 3-2 win over former club Barnsley.\nGraham Westley's side currently have injuries to fellow defenders Callum Elder, Gabriel Zakuani and Kgosi Ntlhe.\nPosh, currently sixth in the table, are at home against struggling Chesterfield on Boxing Day.",
        "expected": "Peterborough United defender Miles Addison has signed a new one-month contract with the League One side."
    },
    {
        "id": "xsum-60f2ba73e141",
        "prompt": "Summarize: The men, aged 26 and 24, were in a house in Melrose Street when three men armed with a knife, hammer and batons forced their way into the property just before midnight on Saturday.\nAfter assaulting the men, the gang left with a sum of cash and personal items. They also smashed a number of windows.\nThe men in the house received medical treatment for their injuries.\nPolice have appealed for anyone with information to contact them.",
        "expected": "Two men have been assaulted by an armed gang in south Belfast."
    },
    {
        "id": "xsum-7756c474bf25",
        "prompt": "Summarize: Police said officers from the North West counter terrorism unit searched an address on Peakdale Avenue, Crumpsall, Manchester on Friday.\nThe suspect, 26, was arrested the following day on suspicion of offences under the Terrorism Act.\nPolice would not comment on the nature of the alleged offence but said it believed it caused \"no threat\" to the community.",
        "expected": "A man is being held in Manchester on suspicion of terrorism offences."
    },
    {
        "id": "xsum-975d27e940d2",
        "prompt": "Summarize: Two men, aged 23 and 24, had a noxious substance thrown over them at 19:00 BST on Tuesday on Roman Road, Bethnal Green, east London.\nRahad Hussain, 23, has been charged with wounding with intent to do grievous bodily harm and possession of an offensive weapon, namely acid.\nHe was remanded in custody when he appeared at Thames Magistrates' Court.\nMr Hussain, of no fixed address,  gave no indication of a plea.\nHe is due to appear at Snaresbrook Crown Court on 29 August.",
        "expected": "A man has appeared in court over an acid attack that left two people with \"life-changing\" injuries."
    },
    {
        "id": "xsum-ef3d0b8ec1b2",
        "prompt": "Summarize: Thousands of animals, many of them endangered, are part of the count which is required by law as part of the zoo's licence.\nImportant details about each and every individual are noted down so that the zoo can help worldwide breeding programmes.\nNewsround's Martin headed to the zoo, which houses over 400 different species, to find out how it's done.",
        "expected": "Keepers at Chester Zoo are making sure every creature, from the biggest elephant to the smallest beetle, is present and correct as part of their annual animal count."
    }
]


# ── Prompt strategies ──
PROMPT_STRATEGIES = {
    "v1-headline": {
        "desc": "'Write a news headline' — Qwen",
        "model": "qwen2.5-1.5b",
        "sys": "Write a short news headline for this article. One sentence only, capturing the core event. Use exact names, numbers. No preamble.",
        "max_tok": 100,
    },
    "v2-prefill": {
        "desc": "Prefill HEADLINE: — Qwen",
        "model": "qwen2.5-1.5b",
        "sys": "Output a single-sentence news headline capturing the main event. Use exact names and numbers. No explanation.",
        "max_tok": 80,
        "prefill": "HEADLINE: ",
    },
    "v3-title": {
        "desc": "'What is the title?' — Qwen",
        "model": "qwen2.5-1.5b",
        "sys": "What is the title of this news article? Answer with the title only, in 10-15 words. No explanation.",
        "max_tok": 64,
    },
    "v4-fewshot": {
        "desc": "Few-shot examples — Qwen",
        "model": "qwen2.5-1.5b",
        "sys": "You generate BBC-style news headlines.\n\nExample 1:\nArticle: Rescue crews saved a family of five from a beach in Somerset after their car got stuck in the mud.\nHeadline: A family of five, including three young children, had to be rescued from a Somerset beach after their car got stuck in the mud on Saturday evening.\n\nExample 2:\nArticle: A man attacked two people with acid in London. He has been charged and appeared in court.\nHeadline: A man has appeared in court over an acid attack that left two people with \"life-changing\" injuries.\n\nNow generate the headline for the article below. Output ONLY the headline, one sentence, 8-20 words.",
        "max_tok": 100,
    },
    "v5-prefill-gemma": {
        "desc": "Prefill HEADLINE: — Gemma",
        "model": "gemma-3-1b",
        "sys": "Write a news headline. One sentence. Max 15 words. Use exact names and numbers.",
        "max_tok": 80,
        "prefill": "HEADLINE: ",
    },
    "v6-qwhat": {
        "desc": "What happened? — Qwen",
        "model": "qwen2.5-1.5b",
        "sys": "Read the article. Answer: what happened? One short sentence with names and numbers.",
        "max_tok": 80,
    },
    "v7-extractive": {
        "desc": "Extract WHO WHAT WHERE — Qwen",
        "model": "qwen2.5-1.5b",
        "sys": "Extract: WHO (main person/group), WHAT (key event), WHERE (location). Then write a one-sentence headline combining them. No explanation.",
        "max_tok": 120,
    },
    "v8-prefill-coder": {
        "desc": "Prefill HEADLINE: — Coder",
        "model": "qwen2.5-coder",
        "sys": "Write a news headline summarizing this article. One sentence only. Use exact names, numbers, places.",
        "max_tok": 80,
        "prefill": "HEADLINE: ",
    },
}

def test_fireworks():
    """Test Fireworks API approach if key is available."""
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key:
        print("\n ⚠ No FIREWORKS_API_KEY set — skipping Fireworks test")
        return None
    
    sys.path.insert(0, "/home/artem/dev/amd-hackathon")
    from agent.solvers.fireworks import FireworksSolver
    
    fw = FireworksSolver(api_key=api_key)
    prompt = (
        "You write BBC-style news headlines. "
        "Given a news article, output a single headline sentence (8-20 words) "
        "that captures the core event. Use exact names, numbers, and places. "
        "Just output the bare headline text."
    )
    
    print(f"\n{'='*60}")
    print("FIREWORKS (kimi-k2p7-code)")
    print(f"{'='*60}")
    
    correct = 0
    for q in TEST_QUESTIONS:
        try:
            article_text = q["prompt"]
            if article_text.startswith("Summarize:"):
                article_text = "Article: " + article_text[len("Summarize:"):].strip()
            
            answer = fw.solve(
                model="accounts/fireworks/models/kimi-k2p7-code",
                system_prompt=prompt,
                user_prompt=article_text,
                max_tokens=80,
                temperature=0.0,
                task_type="summarization",
                timeout=29,
            )
            answer = answer.strip().strip('"').strip("'")
            ok = fuzzy_match(answer, q["expected"])
            if ok: correct += 1
            print(f"  {'✓' if ok else '✗'} {answer[:100]}")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
    
    print(f"  → Accuracy: {correct}/{len(TEST_QUESTIONS)} = {correct/len(TEST_QUESTIONS)*100:.1f}%")
    return correct / len(TEST_QUESTIONS) * 100


def main():
    from llama_cpp import Llama
    import torch
    
    print("=" * 80)
    print("SUMMARIZATION APPROACH TEST — 8 strategies on 3 models")
    print("=" * 80)
    
    # Load models
    models = {}
    for name, path in MODEL_PATHS.items():
        print(f"Loading {name}...", file=sys.stderr)
        t0 = time.time()
        models[name] = Llama(model_path=path, n_gpu_layers=-1, n_ctx=2048, verbose=False)
        print(f"  Done in {time.time()-t0:.1f}s", file=sys.stderr)
    
    results = {}
    
    for strategy_name, cfg in PROMPT_STRATEGIES.items():
        desc = cfg["desc"]
        model_name = cfg["model"]
        sys_prompt = cfg["sys"]
        max_tok = cfg["max_tok"]
        prefill = cfg.get("prefill", "")
        
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy_name} — {desc}")
        print(f"{'='*60}")
        
        llm = models[model_name]
        correct = 0
        details = []
        
        for q in TEST_QUESTIONS:
            qid = q["id"]
            expected = q["expected"]
            prompt_text = q["prompt"]
            
            messages = [{"role": "system", "content": sys_prompt}]
            if prefill:
                messages.append({"role": "user", "content": prompt_text})
                messages.append({"role": "assistant", "content": prefill})
            else:
                messages.append({"role": "user", "content": prompt_text})
            
            t0 = time.time()
            r = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tok,
                temperature=0.0,
                stop=["\n\n"]
            )
            elapsed = time.time() - t0
            
            raw = r["choices"][0]["message"]["content"].strip()
            answer = raw
            if prefill and not answer.startswith(prefill.strip()):
                answer = prefill + answer
            
            ok = fuzzy_match(answer, expected)
            if ok:
                correct += 1
                marker = "✓ ✓ ✓"
            else:
                marker = "✗"
            
            details.append({
                "qid": qid,
                "expected": expected,
                "got": answer[:150],
                "correct": ok,
                "time_s": round(elapsed, 2),
            })
            
            print(f"  {marker} {answer[:100]}")
            if ok:
                print(f"     expected: {expected[:80]}")
        
        acc = correct / len(TEST_QUESTIONS) * 100
        results[strategy_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": len(TEST_QUESTIONS),
            "details": details,
        }
        print(f"  → {correct}/{len(TEST_QUESTIONS)} = {acc:.1f}%")
    
    # Unload models
    for name in list(models.keys()):
        del models[name]
    gc.collect()
    torch.cuda.empty_cache()
    
    # Test Fireworks
    fw_acc = test_fireworks()
    
    # Summary
    print(f"\n\n{'='*70}")
    print("ALL RESULTS")
    print(f"{'='*70}")
    all_accs = sorted(
        [(sname, sres["accuracy"], PROMPT_STRATEGIES[sname]["desc"], PROMPT_STRATEGIES[sname]["model"])
         for sname, sres in results.items()],
        key=lambda x: x[1], reverse=True
    )
    for sname, acc, desc, model in all_accs:
        print(f"  {acc:5.1f}% — {sname:25s} | {model:20s} | {desc}")
    
    if fw_acc is not None:
        print(f"  {fw_acc:5.1f}% — fireworks-kimi            | kimi-k2p7-code          | Fireworks API headline")
    
    # Save
    out_path = "/home/artem/dev/amd-hackathon/data/eval/summarization_final_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
