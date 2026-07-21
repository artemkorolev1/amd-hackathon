#!/usr/bin/env python3
"""NER eval subprocess worker: loads ONE model, runs ALL questions with ONE prompt strategy, outputs JSON to stdout."""
import json, sys, time, re, gc

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
        if en != 0 and abs((an - en) / en) <= 0.01: return True
        if an == en: return True
    ta = set(t for t in re.split(r"[^a-zA-Z0-9.]+", a) if t)
    te = set(t for t in re.split(r"[^a-zA-Z0-9.]+", e) if t)
    if len(te) > 0 and len(ta & te) / len(te) >= 0.8: return True
    return False

def compute_entity_f1(result_str: str, expected_str: str) -> dict:
    """Compute entity-level precision, recall, F1."""
    def norm(s):
        return s.replace('\u2019', "'").replace('\u2018', "'").replace('\u201c', '"').replace('\u201d', '"')
    rn, en = norm(result_str), norm(expected_str)
    exp_lines = set()
    for line in en.split('\n'):
        line = line.strip()
        if ':' in line:
            exp_lines.add(line)
    res_lines = set()
    for line in rn.split('\n'):
        line = line.strip()
        if ':' in line:
            res_lines.add(line)
    intersection = exp_lines & res_lines
    precision = len(intersection) / len(res_lines) if res_lines else 0.0
    recall = len(intersection) / len(exp_lines) if exp_lines else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact = rn == en
    return {
        "exact_match": exact,
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "correct_lines": len(intersection),
        "expected_lines": len(exp_lines),
        "result_lines": len(res_lines),
        "intersection": list(intersection),
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--prompt-name", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--user-prefix", default="")
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--workflow", action="store_true", help="Use NER_2STEP_WORKFLOW")
    args = parser.parse_args()

    # Load eval data
    with open(args.eval_file) as f:
        questions = json.load(f)

    # Load model
    from llama_cpp import Llama
    print(f"[worker] Loading {args.model_name} ...", file=sys.stderr)
    t0 = time.time()
    llm = Llama(model_path=args.model_path, n_gpu_layers=-1, n_ctx=2048, verbose=False)
    print(f"[worker] Loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    results = []
    total_ok_exact = 0
    total_ok_fuzzy = 0
    total_f1 = 0.0
    total_time = 0.0

    for i, q in enumerate(questions):
        prompt = q["prompt"]
        expected = q["expected_answer"]
        task_id = q["task_id"]

        t_start = time.time()

        if args.workflow:
            # NER_2STEP_WORKFLOW: extract then verify
            import sys as _sys
            _sys.path.insert(0, "/home/artem/dev/amd-hackathon")
            from agent.workflow import WorkflowEngine, ToolRegistry, NER_2STEP_WORKFLOW
            from agent.cell import Cell, StepConfig, DecodingConfig

            def llm_infer(messages, max_tok, temp):
                r = llm.create_chat_completion(messages=messages, max_tokens=max_tok, temperature=temp)
                return r["choices"][0]["message"]["content"].strip()

            engine = WorkflowEngine(llm_infer, ToolRegistry())
            cell = Cell(
                model_key=args.model_name,
                system_prompt="",
                steps=NER_2STEP_WORKFLOW,
                decoding=DecodingConfig(max_tokens=args.max_tokens, temperature=0.0)
            )
            wf_result = engine.run(cell, prompt)
            answer = wf_result["final_answer"]
        elif args.system_prompt:
            # System prompt + user message
            msgs = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": prompt},
            ]
            r = llm.create_chat_completion(messages=msgs, max_tokens=args.max_tokens, temperature=0.0)
            answer = r["choices"][0]["message"]["content"].strip()
        elif args.user_prefix:
            # User message prefixed
            msgs = [{"role": "user", "content": args.user_prefix + "\n\n" + prompt}]
            r = llm.create_chat_completion(messages=msgs, max_tokens=args.max_tokens, temperature=0.0)
            answer = r["choices"][0]["message"]["content"].strip()
        else:
            # Just the prompt as user message
            msgs = [{"role": "user", "content": prompt}]
            r = llm.create_chat_completion(messages=msgs, max_tokens=args.max_tokens, temperature=0.0)
            answer = r["choices"][0]["message"]["content"].strip()

        elapsed = time.time() - t_start
        total_time += elapsed

        entity_metrics = compute_entity_f1(answer, expected)
        fuzzy_ok = fuzzy_match(answer, expected)

        total_ok_exact += 1 if entity_metrics["exact_match"] else 0
        total_ok_fuzzy += 1 if fuzzy_ok else 0
        total_f1 += entity_metrics["f1"]

        results.append({
            "task_id": task_id,
            "expected": expected,
            "got": answer,
            "elapsed_s": round(elapsed, 3),
            "exact_match": entity_metrics["exact_match"],
            "fuzzy_match": fuzzy_ok,
            "f1": entity_metrics["f1"],
            "precision": entity_metrics["precision"],
            "recall": entity_metrics["recall"],
            "correct_entities": entity_metrics["correct_lines"],
            "expected_entities": entity_metrics["expected_lines"],
            "result_entities": entity_metrics["result_lines"],
        })

        marker = "✓" if entity_metrics["exact_match"] else ("~" if fuzzy_ok else "✗")
        print(f"  [{i+1:2d}/{len(questions)}] {marker} F1={entity_metrics['f1']:.2f}  ({elapsed:.1f}s)  {answer[:50]}", file=sys.stderr)

    n = len(questions)
    output = {
        "model": args.model_name,
        "prompt_name": args.prompt_name,
        "system_prompt": args.system_prompt,
        "user_prefix": args.user_prefix,
        "workflow": args.workflow,
        "questions": n,
        "exact_match_accuracy": round(total_ok_exact / n * 100, 1) if n else 0,
        "fuzzy_match_accuracy": round(total_ok_fuzzy / n * 100, 1) if n else 0,
        "avg_f1": round(total_f1 / n, 4) if n else 0.0,
        "total_time_s": round(total_time, 1),
        "results": results,
    }

    print(json.dumps(output), file=sys.stdout)

if __name__ == "__main__":
    main()
