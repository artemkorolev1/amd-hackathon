<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Like suppose I have like a math specialist and I'm not sure if he can do these types of like splitting into plan build idea Identify plan proceed does it work for smaller agents as well? Or yeah, how does it work? Would you be able to define uh for my use case right I have these eight eight types of tasks would you be able to define what I can what can be used there.

Short answer: yes, your math specialist (and other small specialists) can do “plan → split → solve pieces → refine → compose” if you bake that structure into the prompts and orchestration. This works with small reasoning models like Qwen‑1.5B; you just have to keep each step tight and tool‑assisted.[^1][^2][^3]

Below I’ll assume your main task types are roughly:

- Factual / RAG
- Math reasoning
- Sentiment
- Constrained summarization
- NER
- Coding
- Debugging
- “Logic” / general structured reasoning

I’ll give you a **per‑task pattern** for how to use planning, splitting, tools, and refinement.

***

## 1. Does multi-step planning work for small math specialists?

Research on **plan‑and‑solve** and **Self‑Refine** shows that multi‑step prompting is about *prompt structure*, not model size: the pattern is “first understand and plan, then execute subtasks, then refine,” and it improves reasoning on math for a range of models.[^4][^5][^6][^7][^8][^9]

For math:

- Plan‑and‑Solve prompting explicitly instructs:
*“Let’s first understand the problem and devise a plan to solve it. Then, let’s carry out the plan and solve step by step.”*[^5][^7][^9]
- Self‑Refine prompting is a simple loop: generate initial solution → ask the same model for feedback → refine using that feedback, repeating until criteria are met.[^6][^10][^8][^4]
- Small Qwen‑family reasoning models (1.5–1.8B) have been shown to handle competitive math and coding when tuned for reasoning, especially when given structured chain‑of‑thought prompts.[^2][^3][^1]

So your math specialist absolutely *can* do this, as long as you:

- Make **Run 1** a *planning prompt* (“analyze the problem and list substeps/tools”).
- Use **Run 2–3** for executing substeps (parse text → build equations → call a math tool like SymPy).
- Use **Run 4** as a self‑critique/refinement pass.
- Use **Run 5** to compose the final answer + explanation.

The orchestration enforces the structure; the small model fills in the content.

***

## 2. Patterns per task type in your system

### Factual / RAG (T01)

Use a short **2‑ or 3‑step loop**:

1. **Plan**: identify what facts are needed and which RAG indexes / tools to call.
2. **Retrieve + draft**: call RAG tools, synthesize an answer.
3. **Check + refine**: run deterministic checks (source presence, date consistency) and a simple Self‑Refine pass for clarity or missing details.[^8][^11][^12][^6]

You don’t need deep multi‑branch reasoning here; just plan, retrieve, and refine.

***

### Math reasoning (T02)

Use full **Plan‑and‑Solve + Self‑Refine**, because math benefits the most:

1. **Run 1 – Plan**:
    - Extract variables and numbers.
    - Decide substeps (set up equations, choose method, consider edge cases).[^7][^9][^5]
2. **Run 2 – Structured setup**:
    - Build equations or structured representation; optionally call a parsing tool.
3. **Run 3 – Solve with tools**:
    - Call SymPy or a math solver tool to compute exact results; keep the LLM focused on explaining and selecting the right operation.[^13][^14]
4. **Run 4 – Self‑critique**:
    - Ask the math specialist to check for calculation mistakes, missing conditions, and alternative solutions (Self‑Refine pass).[^10][^4][^6][^8]
5. **Run 5 – Compose**:
    - Combine the best reasoning path and numerical solution into a clean final answer + justification.

This pattern is exactly the “plan → subtasks → refine” structure that PS / PS+ use, adapted to your tools.[^9][^5][^7]

***

### Sentiment classification (T03)

You probably don’t need five heavy runs; small models and classical classifiers do well here:

1. **Run 1 – Plan + read**:
    - Decide if the text is short/long and whether mixed sentiment is likely.
2. **Run 2 – Label**:
    - LLM assigns a sentiment label and rationale.
3. **Run 3 – Validate with classical model**:
    - A small sentiment classifier confirms or challenges the label.
4. **Run 4 (optional) – Self‑Refine**:
    - If disagreement or low confidence, ask the LLM to reconsider using the classifier’s output.[^6][^8]

No need for branching; keep it shallow and deterministic.

***

### Constrained summarization (T04)

Summarization benefits from plan‑first + constraint checks:

1. **Run 1 – Plan**:
    - Identify main points, required length, format (bullets vs sentences).
2. **Run 2 – Draft summary**:
    - Produce a first summary focusing on coverage.
3. **Run 3 – Constraint check**:
    - Run deterministic tools to verify sentence count, word limit, and JSON/structure if needed.[^15][^16][^17][^18]
4. **Run 4 – Self‑Refine**:
    - Ask the model to improve clarity or coverage while strictly respecting constraints.
5. **Run 5 – Final formatting**:
    - Ensure final output exactly matches your T04 schema (e.g., `N` bullets, capped length).

Here the tools are length/format checkers and maybe ROUGE/BERTScore for quality if you want more sophistication.[^19]

***

### Named-Entity Recognition (T05)

Use a short decomposition + cross‑check:

1. **Run 1 – Span proposal**:
    - LLM marks candidate spans and labels (PERSON/ORG/LOC/DATE).
2. **Run 2 – Classical NER check**:
    - spaCy or transformers NER validates or corrects spans.[^20][^21][^22]
3. **Run 3 – Self‑Refine**:
    - LLM revises its output to align with the classical tags, especially mixed or ambiguous cases.[^8][^6]

You don’t need full five phases unless the text is very long or ambiguous.

***

### Coding

For your coding specialist:

1. **Run 1 – Problem analysis and plan**:
    - List functions/modules to write and test cases needed.
2. **Run 2 – Initial implementation**:
    - Generate code according to the plan.
3. **Run 3 – Tool‑based execution**:
    - Call a `PythonInterpreterTool` (smolagents) or similar to run tests and capture errors.[^23][^24][^25]
4. **Run 4 – Self‑Refine / debug loop**:
    - Self‑Refine style: use trace + error messages to improve code iteratively.[^25][^10][^6]
5. **Run 5 – Final code + explanation**:
    - Return cleaned code and a short explanation of how it works.

***

### Debugging (code or reasoning)

Your debugging specialist is almost pure Self‑Refine:

1. **Run 1 – Read and hypothesize issues**.
2. **Run 2 – Instrument / test** (using tools like Python execution, assertion checks).
3. **Run 3 – Propose fixes**.
4. **Run 4 – Re‑run tests and re‑evaluate**.
5. **Run 5 – Summarize root cause + fix**.[^10][^25][^6][^8]

This can be reused for non‑code logic debugging too: test assumptions, compare against ground truth, revise reasoning.

***

### “Logic” / general structured reasoning

For general logical tasks (argument analysis, multi-step reasoning):

- Use a **lighter Plan‑and‑Solve**:
    - Run 1: identify premises, conclusion, and subquestions.
    - Runs 2–3: check consistency and evaluate subquestions or counterexamples.
    - Run 4: Self‑Refine the reasoning for clarity and error removal.
    - Run 5: Compose final, structured explanation (e.g., numbered steps).

***

## How to wire this into your GEPA system

For each task type, you can define a **cell-level micro-workflow**:

- A **planning prompt** template (Run 1).
- A list of **substeps**, each with its own prompt and tools.
- A **refinement step** using Self‑Refine‑style prompts.
- A **final composition step**.

Your small specialists (Qwen2.5‑Math‑1.5B, Qwen‑Coder‑1.5B, SmolLM2, etc.) can all follow this structure; the key is that your orchestrator enforces the sequence and passes intermediate artifacts between runs, instead of relying on the model to remember everything in one call.[^26][^2]

If you want, next step I can sketch one concrete “prompt skeleton” for the math specialist and another for summarization, in the exact multi‑run shape you’re imagining.
<span style="display:none">[^27][^28][^29][^30]</span>

<div align="center">⁂</div>

[^1]: https://www.reddit.com/r/LocalLLaMA/comments/1ou1emx/we_put_a_lot_of_work_into_a_15b_reasoning_model/

[^2]: https://thesalt.substack.com/p/finding-the-optimal-reasoning-budget

[^3]: https://dev.co/ai/llms/deepseek-r1-distill-qwen-1-5b

[^4]: https://openreview.net/pdf?id=S37hOerQLB

[^5]: https://arxiv.org/abs/2305.04091

[^6]: https://learnprompting.org/docs/advanced/self_criticism/self_refine

[^7]: https://aclanthology.org/2023.acl-long.147/

[^8]: https://arxiv.org/abs/2303.17651

[^9]: https://fnl.es/Science/Papers/Prompt+Engineering/Plan-and-Solve+Prompting

[^10]: https://mirascope.com/docs/v1/guides/prompt-engineering/chaining-based/self-refine

[^11]: https://www.prakashbhandari.com.np/posts/building-rag-pipeline-for-summarization-and-q-and-a-with-llamaindex-and-openai/

[^12]: https://docs.mistral.ai/resources/cookbooks/third_party-llamaindex-routerqueryengine

[^13]: https://arxiv.org/html/2407.17544v1

[^14]: https://github.com/sdiehl/sympy-mcp

[^15]: https://langchain-doc.readthedocs.io/en/latest/use_cases/summarization.html

[^16]: https://milvus.io/ai-quick-reference/how-can-i-use-llamaindex-for-document-summarization

[^17]: https://developers.llamaindex.ai/python/examples/index_structs/doc_summary/docsummary/

[^18]: https://www.promptfoo.dev/docs/configuration/expected-outputs/deterministic/

[^19]: https://blog.mozilla.ai/on-model-selection-for-text-summarization/

[^20]: https://spacy.io/usage/large-language-models

[^21]: https://explosion.ai/_/project/spacy

[^22]: https://aclanthology.org/2026.nlp4dh-1.35/

[^23]: https://github.com/huggingface/agents-course/blob/main/units/en/unit2/smolagents/tools.mdx

[^24]: https://github.com/huggingface/smolagents

[^25]: https://arxiv.org/html/2502.02928v2

[^26]: https://news.mit.edu/2025/enabling-small-language-models-solve-complex-reasoning-tasks-1212

[^27]: https://arxiv.org/html/2512.16649v1

[^28]: https://promptengineering.org/plan-and-solve-plus-ps-a-prompting-framework-for-enhanced-llm-reasoning/

[^29]: https://qwenlm.github.io/blog/qwen1.5/

[^30]: https://www.reddit.com/r/machinelearningnews/comments/12f2iby/this_ai_paper_introduces_selfrefine_a_framework/

