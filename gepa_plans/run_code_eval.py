#!/usr/bin/env python3
"""
Code Gen + Code Debug GEPA evaluation for AMD ACT II hackathon.

Tests 4 prompt strategies per sub-category on two models.
Uses llama-cpp-python in subprocess workers.
Grades with fuzzy_match from gepa_plans/eval_common.py.
For code_gen, also tries executing output via python3 -c.
"""
import json, sys, os, subprocess, tempfile, re, time, math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from eval_common import fuzzy_match

# ── Paths ──
DATA_DIR = Path('/home/artem/dev/amd-hackathon/data/eval')
GEPA_DIR = Path('/home/artem/dev/amd-hackathon/gepa_plans')
WORKER_SCRIPT = GEPA_DIR / 'code_eval_worker.py'
RESULTS_PATH = GEPA_DIR / 'code_eval_results.json'
PYTHON = sys.executable

MODELS = {
    'qwen2.5-coder-1.5b-instruct': '/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf',
    'qwen2.5-1.5b-instruct': '/home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf',
}

CODE_DEBUG_PROMPTS = [
    'Fix the bug:',
    'Debug:',
    'Fix the following code:',
    'Fix:',
]

CODE_GEN_PROMPTS = [
    'Write the requested function:',
    'Implement:',
    'Generate code:',
    'Code:',
]


def load_data():
    with open(DATA_DIR / 'training-v3.json') as f:
        train = json.load(f)
    with open(DATA_DIR / 'validation-v3.json') as f:
        val = json.load(f)
    combined = train + val
    code_debug = [item for item in combined if item['category'] == 'code_debug']
    code_gen = [item for item in combined if item['category'] == 'code_gen']
    print(f"  code_debug: {len(code_debug)} entries, code_gen: {len(code_gen)} entries")
    return code_debug, code_gen


def extract_task_from_debug(prompt):
    """Extract (buggy_code, task_desc) from a code_debug prompt."""
    parts = prompt.split('\n\nTask:')
    if len(parts) >= 2:
        buggy_code = parts[0].replace('Fix this Python function:\n', '', 1).replace('Fix this Python function:', '', 1).strip()
        task_part = 'Task:' + parts[1]
        return buggy_code, task_part
    return None, None


def make_code_debug_prompt(entry, prefix):
    """Create a code_debug prompt with a different prefix."""
    original = entry['prompt']
    buggy_code, task_part = extract_task_from_debug(original)
    if buggy_code is None:
        return original
    return f"{prefix}\n{buggy_code}\n\n{task_part}"


def make_code_gen_prompt(entry, prefix):
    """Create a code_gen prompt with a different prefix."""
    original = entry['prompt']
    # Original format: "Write a Python function: <description>"
    for old_prefix in ['Write a Python function: ', 'Write a Python function:']:
        if original.startswith(old_prefix):
            description = original[len(old_prefix):].strip()
            break
    else:
        description = original
    return f"{prefix} {description}"


def extract_code_from_output(output):
    """Extract Python code from model output (handle ```python blocks, etc)."""
    # Try to extract from code blocks
    code_block_pattern = r'```(?:python)?\n(.*?)```'
    matches = re.findall(code_block_pattern, output, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try to extract function definition
    func_pattern = r'(def \w+\(.*?\):.*?)(?:\n\S|$)'
    matches = re.findall(func_pattern, output, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Fall back to the whole output (up to reasonable length)
    return output.strip()


def try_execute_python(code, timeout=5):
    """Try to execute Python code and return (success, output)."""
    try:
        result = subprocess.run(
            [PYTHON, '-c', code],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip()[:200]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def run_batch(model_path, tasks):
    """Run a batch of prompts through a model subprocess worker.
    
    Args:
        model_path: Path to GGUF model
        tasks: List of dicts with 'prompt' and 'expected'
    
    Returns:
        List of dicts with 'generated'
    """
    if not tasks:
        return []

    # Write tasks to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(tasks, f)
        tasks_path = f.name

    result = None
    try:
        result = subprocess.run(
            [PYTHON, str(WORKER_SCRIPT), model_path, tasks_path],
            capture_output=True, text=True, timeout=600,
            env={**os.environ},
        )
        if result.returncode != 0:
            print(f"    Worker error (rc={result.returncode}): {result.stderr[:300]}")
            return [{'generated': ''} for _ in tasks]

        stdout = result.stdout.strip()
        if not stdout:
            print(f"    Worker produced no output. stderr: {result.stderr[:200]}")
            return [{'generated': ''} for _ in tasks]

        return json.loads(stdout)
    except json.JSONDecodeError as e:
        preview = result.stdout[:300] if result is not None else "no output"
        print(f"    JSON decode error: {e}")
        print(f"    stdout preview: {preview}")
        return [{'generated': ''} for _ in tasks]
    except subprocess.TimeoutExpired:
        print(f"    Worker timed out after 600s")
        return [{'generated': ''} for _ in tasks]
    except Exception as e:
        print(f"    Worker exception: {e}")
        return [{'generated': ''} for _ in tasks]
    finally:
        os.unlink(tasks_path)


def evaluate_category(model_name, model_path, entries, category, prompt_prefixes):
    """Evaluate all prompt strategies for one category on one model.
    
    Returns list of result dicts.
    """
    all_results = []

    for prefix in prompt_prefixes:
        label = f"{prefix}"
        print(f"    Strategy: '{prefix}'")

        # Build tasks for this strategy
        tasks = []
        for entry in entries:
            if category == 'code_debug':
                prompt = make_code_debug_prompt(entry, prefix)
            else:
                prompt = make_code_gen_prompt(entry, prefix)

            tasks.append({
                'prompt': prompt,
                'expected': entry['expected_answer'],
                'task_id': entry.get('task_id', ''),
                'source': entry.get('source', ''),
            })

        # Run batch
        outputs = run_batch(model_path, tasks)

        if len(outputs) != len(tasks):
            print(f"      WARNING: got {len(outputs)} results for {len(tasks)} tasks")
            # Pad with empty results
            while len(outputs) < len(tasks):
                outputs.append({'generated': ''})

        # Grade
        correct = 0
        for i, (task, out) in enumerate(zip(tasks, outputs)):
            generated = out.get('generated', '')
            expected = task['expected']

            # Grading with fuzzy_match
            score = fuzzy_match(generated, expected)

            # For code_gen, also try executing
            exec_success = None
            if category == 'code_gen':
                code = extract_code_from_output(generated)
                exec_success, exec_output = try_execute_python(code)
                # If execution succeeded but fuzzy_match failed, still count execution
                # But we rely on fuzzy_match for the primary score

            task_result = {
                'model': model_name,
                'category': category,
                'prompt_strategy': label,
                'task_id': task['task_id'],
                'source': task['source'],
                'expected': expected,
                'generated': generated,
                'score': score,
                'exec_success': exec_success if category == 'code_gen' else None,
            }
            all_results.append(task_result)
            if score:
                correct += 1

        acc = correct / len(entries) * 100 if entries else 0
        print(f"      Accuracy: {correct}/{len(entries)} = {acc:.1f}%")

    return all_results


def print_summary(all_results):
    """Print per-(category, model, prompt) accuracy summary."""
    print("\n" + "=" * 80)
    print("SUMMARY: Code Gen + Code Debug GEPA Evaluation")
    print("=" * 80)

    # Group results
    groups = defaultdict(lambda: {'correct': 0, 'total': 0, 'exec_ok': 0})

    for r in all_results:
        key = (r['model'], r['category'], r['prompt_strategy'])
        groups[key]['total'] += 1
        if r['score']:
            groups[key]['correct'] += 1
        if r['exec_success'] is True:
            groups[key]['exec_ok'] += 1

    # Print table
    print(f"\n{'Model':<30} {'Category':<15} {'Prompt Strategy':<30} {'Acc':<8} {'Exec%':<8}")
    print("-" * 95)

    model_order = list(MODELS.keys())
    categories = ['code_debug', 'code_gen']

    for model in model_order:
        for cat in categories:
            for ps_name in (CODE_DEBUG_PROMPTS if cat == 'code_debug' else CODE_GEN_PROMPTS):
                # The prompt_strategy stored includes the prefix text
                key = (model, cat, ps_name)
                if key in groups:
                    g = groups[key]
                    acc = g['correct'] / g['total'] * 100
                    epct = g['exec_ok'] / g['total'] * 100 if g['total'] > 0 else 0
                    epct_str = f"{epct:.0f}%" if cat == 'code_gen' else "N/A"
                    print(f"{model:<30} {cat:<15} {ps_name:<30} {acc:>5.1f}%   {epct_str:<8}")

    # Overall averages
    print("\n" + "-" * 95)
    for model in model_order:
        for cat in categories:
            key_prefix = (model, cat)
            total_correct = sum(g['correct'] for k, g in groups.items() if k[:2] == key_prefix)
            total = sum(g['total'] for k, g in groups.items() if k[:2] == key_prefix)
            if total > 0:
                print(f"{model:<30} {cat:<15} {'AVERAGE':<30} {total_correct/total*100:>5.1f}%")
    
    # Best prompt per (model, category)
    print("\n" + "-" * 95)
    print("Best prompt per (model, category):")
    for model in model_order:
        for cat in categories:
            best_acc = -1
            best_ps = ""
            best_g = None
            for key, g in groups.items():
                if key[0] == model and key[1] == cat:
                    acc = g['correct'] / g['total'] * 100
                    if acc > best_acc:
                        best_acc = acc
                        best_ps = key[2]
                        best_g = g
            if best_g:
                print(f"  {model:<30} {cat:<15} best: '{best_ps:<30}' → {best_g['correct']}/{best_g['total']} = {best_acc:.1f}%")

    # Cross-model comparison
    print("\n" + "-" * 95)
    print("Code specialist vs general model comparison:")
    for cat in categories:
        coder_key = ('qwen2.5-coder-1.5b-instruct', cat)
        gen_key = ('qwen2.5-1.5b-instruct', cat)
        coder_correct = sum(g['correct'] for k, g in groups.items() if k[:2] == coder_key)
        coder_total = sum(g['total'] for k, g in groups.items() if k[:2] == coder_key)
        gen_correct = sum(g['correct'] for k, g in groups.items() if k[:2] == gen_key)
        gen_total = sum(g['total'] for k, g in groups.items() if k[:2] == gen_key)
        if coder_total > 0 and gen_total > 0:
            print(f"  {cat}: coder={coder_correct}/{coder_total}={coder_correct/coder_total*100:.1f}% vs gen={gen_correct}/{gen_total}={gen_correct/gen_total*100:.1f}%")


def main():
    print("=" * 80)
    print("Code Gen + Code Debug GEPA Evaluation")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    code_debug, code_gen = load_data()

    all_results = []

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Path: {model_path}")
        print(f"{'='*60}")

        # Code Debug
        print(f"\n  --- Code Debug ---")
        start = time.time()
        results_debug = evaluate_category(
            model_name, model_path, code_debug, 'code_debug', CODE_DEBUG_PROMPTS
        )
        elapsed = time.time() - start
        print(f"  Code Debug completed in {elapsed:.1f}s")
        all_results.extend(results_debug)

        # Code Gen
        print(f"\n  --- Code Gen ---")
        start = time.time()
        results_gen = evaluate_category(
            model_name, model_path, code_gen, 'code_gen', CODE_GEN_PROMPTS
        )
        elapsed = time.time() - start
        print(f"  Code Gen completed in {elapsed:.1f}s")
        all_results.extend(results_gen)

    # Save results
    print(f"\nSaving {len(all_results)} results to {RESULTS_PATH}...")
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print_summary(all_results)

    print(f"\nResults saved to {RESULTS_PATH}")
    print("Done!")


if __name__ == '__main__':
    main()
