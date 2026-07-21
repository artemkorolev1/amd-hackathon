#!/usr/bin/env python3
"""v12e runner — produces one answer per line on stdout for evaluate.py to grade.
Usage: python3 run_v12e.py > /tmp/answers.txt
Then:   cd /home/artem/dev/amd-hackathon && python3 evaluate.py --no-run < /tmp/answers.txt"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["COMPLEXITY_MODEL_DIR"] = "/home/artem/dev/amd-hackathon-shared/classifiers/best_complexity_model"
os.environ["FIREWORKS_API_KEY"] = os.environ.get("FIREWORKS_API_KEY", "")

from agent.complexity_filter import score as c_score
from agent.category_filter import classify
from agent.solvers import local_model
from agent.dynamic_prompts import build_system_prompt, get_max_tokens, NER_ONE_SHOT_EXAMPLE, SENTIMENT_EXAMPLES, MATH_EXAMPLES
from agent.solvers.deterministic import solve_arithmetic, solve_logic, solve_sentiment, solve_ner, solve_factual_qa, solve_code_debugging
from agent.pre_filter import stage0

EVAL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/artem/dev/amd-hackathon-shared/eval_all_300.json"

with open(EVAL_PATH) as f:
    questions = json.load(f)["questions"]

DET_CAT_MAP = {"math":"math_arithmetic","logic":"logical_reasoning","sentiment":"sentiment","ner":"named_entity_recognition","factual":"factual_knowledge","code_debug":"code_debugging","summarization":"summarization","general":"other_complex"}
DET_SOLVERS = [solve_arithmetic, solve_logic, solve_sentiment, solve_ner, solve_factual_qa, solve_code_debugging]
LORA_CATS = {"logic","ner","summarization","sentiment","factual","code_debug","code_gen","math"}

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger("eval")

for q in questions:
    prompt = q["prompt"]
    # Stage 0
    s0 = stage0(prompt)
    if s0.action == "bypass" and s0.direct_answer:
        print(s0.direct_answer)
        continue
    # Stage 2 + complexity
    cat, conf, _ = classify(prompt)
    cx = c_score(prompt, cat)
    # Deterministic
    det_ans = None
    for sfn in DET_SOLVERS:
        try:
            a = sfn(prompt, DET_CAT_MAP.get(cat, "general"))
            if a:
                det_ans = a
                break
        except Exception:
            logger.debug("Deterministic solver %s skipped for question %s", sfn.__name__, q.get("task_id", "?"))
    if det_ans:
        print(det_ans)
        continue
    # LoRA
    custom_instr = None
    if os.environ.get("FEW_SHOT", "1") == "1":
        if cat == "ner":
            custom_instr = NER_ONE_SHOT_EXAMPLE
        elif cat == "sentiment":
            custom_instr = SENTIMENT_EXAMPLES
        elif cat == "math":
            custom_instr = MATH_EXAMPLES
    sys_p = build_system_prompt(cat, cx, custom_instructions=custom_instr)
    msgs = [{"role":"system","content":sys_p},{"role":"user","content":prompt}]
    ans = local_model.chat_completion(msgs, category=cat if cat in LORA_CATS else "",
                                      max_tokens=get_max_tokens(cat, cx))
    # Flatten to single line for one-answer-per-line output contract
    flat = (ans or "").replace("\n", "\\n").replace("\r", "\\r")
    print(flat)
