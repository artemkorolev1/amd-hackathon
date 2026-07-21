<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Deep Research: Deterministic Routing, Classical ML Classifiers, and Tools Around Small LLMs

I’m building a multi‑model LLM system inside a constrained container:
Hardware: 2 CPUs (4 GB RAM each, ~8 GB total), no GPU
Models: small open‑weight LLMs between 0.5B and 1.7B parameters, quantized to GGUF Q4_K_M
Tasks: factual Q\&A, math word problems, sentiment classification, constrained summarization, named‑entity recognition, and possibly code generation/debugging
I want a highly deterministic, efficient architecture with:
Deterministic routing (classical ML, not LLM-based)
Deterministic / algorithmic tools for math, code, NER, summarization, and factual lookup
Small LLMs used only where they really add value
Please perform deep research and help me design this system.

1. Best practices for deterministic routing in multi‑model LLM systems
Summarize current best practices for multi‑model routing and model cascades, focusing on pre‑judgment routing (classify the request before any generation) rather than LLM‑as‑router.arxiv+4
Compare:
Simple rule‑based routing (regex, keyword features)
Classical ML classifiers (logistic regression, SVM, random forests)
Small transformers or text classifiers (<100M parameters) used as routers

Discuss latency and resource trade‑offs for each approach in a low‑resource container (≤200 ms per routing decision, ≤200 MB RAM).
Provide concrete examples of routing rules or feature sets that distinguish:
factual questions vs math problems vs summarization vs sentiment vs NER vs code

2. Deterministic / classical classifiers for task routing and validation
I want non‑LLM classifiers for both:
Task routing (decide which LLM(s) and which tools should run)
Output validation (check or correct LLM outputs)
Please:
Identify suitable classical or small ML models for task classification:
e.g., TF‑IDF + logistic regression, fastText, small BERT‑style models, etc.
Must run quickly on CPU, and be trainable on my own labeled data.

Identify small sentiment classifiers (VADER, TextBlob, fastText, tiny transformers) that can:
Detect mixed sentiment (positive and negative in same text)
Produce deterministic outputs given fixed thresholds

Identify small NER models (spaCy, Flair, stanza, tiny BERT) suitable for:
Deterministically extracting PERSON / ORG / LOC / DATE labels
Running as validators for LLM‑produced entities

Provide benchmark numbers (accuracy/F1, inference latency, RAM usage) where available.
3. Deterministic tools for math, code, summarization, and factual lookup
I want to augment small LLMs with external tools that are as deterministic as possible:
Math / numeric reasoning:
Symbolic math libraries (e.g., SymPy, mpmath) that can solve word problems once the expression is extracted.
Equation or expression parsers + validation routines.
Any open‑source systems that combine LLMs with symbolic math to reduce errors.

Code generation, debugging, and analysis:
Static analyzers, linters, formatters (e.g., ruff, flake8, black, mypy, pyright) that can validate and correct generated code.
Tools or frameworks for “coding agents” that let small LLMs call deterministic tools (run tests, check compilation) and use the results.
Any sub‑1.7B code‑focused models that pair well with such tools.

Summarization:
Extractive summarization algorithms (TextRank, LexRank, position‑based summaries) that can produce deterministic, format‑constrained outputs:
exactly N sentences
or exactly N bullet points, each under a given word limit

Compare these to small LLM summarizers in terms of quality and robustness.

Factual knowledge:
Lightweight retrieval systems (e.g., BM25 + local index, SQLite/JSON knowledge bases) that can:
answer factual queries deterministically from a corpus
or provide retrieved passages for the LLM to condition on

Discuss how to use small LLMs primarily as explainers over retrieved facts rather than primary knowledge sources.

For each tool category, recommend specific libraries, give typical resource usage, and show how to call them from Python in a small container.
4. End‑to‑end deterministic routing and validation design
Using the above pieces, help design an end‑to‑end architecture where:
A classical router (or small ML classifier) decides:
Which LLM model(s) to call (within my 3‑model sub‑1.7B set)
Which tools to call (math solver, NER model, summarizer, etc.)

LLMs mostly:
Interpret the problem
Generate intermediate representations (equations, structured JSON, summaries)

Tools:
Perform final computation (math, code checks, factual lookup, extractive summarization)

A deterministic judge / validator layer:
Checks LLM outputs against tool results (e.g., numeric equality, entity label agreement, format constraints)
Applies fixed rules to choose final answers or to fall back to tool‑only results.

Please:
Propose at least one concrete system design (diagram + component list) optimized for:
Small LLMs (≤1.7B params)
8 GB RAM, 2 CPUs
GGUF Q4_K_M models via llama.cpp/Ollama

Provide pseudocode or code sketches for:
Routing logic
Integration of tools with LLM outputs
Deterministic validation rules.

5. Model selection under ≤1.7B constraint
Under the hard constraint that LLMs must be ≤1.7B parameters:
Identify the best available small models (general, math, code, summarization, NER) in that range.
For each:
Name, parameter count, GGUF availability
Benchmarks on tasks relevant to my pipeline (MMLU, GSM8K, IFEval, CoNLL NER, etc.)
License and suitability for local container deployment.

Recommend:
A 3‑model combination that covers my tasks well
How each model should be paired with deterministic tools as described above.

6. Practical best practices and pitfalls
Finally, summarize:
Best practices for:
deterministic routing (pre‑judgment, rule‑based vs ML classifiers)
keeping latency low while using multiple tools
logging and monitoring disagreements between LLMs and tools

Common pitfalls:
over‑complicating routing logic
relying on unstable small LLMs where a simple tool would suffice
resource leaks or performance problems in a constrained CPU‑only container

Please include:
References to papers, blog posts, and GitHub repos on multi‑model routing, PromptEval / multi‑prompt evaluation, and hybrid LLM+tool systems.arxiv+4
Tables comparing candidate models and tools.
Concrete suggestions I can implement directly in Python.

You can get a highly deterministic, low‑latency system by using a classical router plus a small set of specialized LLMs and deterministic tools (symbolic math, static analysis, BM25 retrieval, extractive summarization), with a final validator that always prefers tool‑verified results over raw generations.[^1][^2]

Below I’ll walk through routing practices, concrete tools, and one end‑to‑end design tailored to 2 CPUs, ~8 GB RAM, and ≤1.7B‑parameter GGUF models.

***

## 1. Deterministic routing best practices

Modern multi‑LLM routing work stresses: start with simple, deterministic routing; only escalate to complex routers (RL, LLM‑as‑router) when necessary. Pre‑generation “pre‑judgment” routing is especially recommended when cost and latency matter, as it lets you avoid calling any LLM for trivial tasks.[^3][^4][^5][^6]

### Rule‑based vs ML vs small transformers

Routing strategies used in recent surveys and case studies:

- **Rule‑based routing**
    - Uses handcrafted rules on query metadata (length, language) and lexical features (keywords, regexes).
Surveys on LLM prompt routing highlight rule‑based thresholds and domain tags as the baseline routing method because it’s transparent and needs no training.[^4][^5]
    - Pros: trivial to implement, essentially zero compute cost; decisions are deterministic by construction.[^5]
    - Cons: brittle on edge cases, doesn’t generalize to new task types without manual updates.[^4]
- **Classical ML classifiers (TF‑IDF + LR/SVM, fastText, etc.)**
    - Represent text with bag‑of‑words or TF‑IDF and train a small classifier for task labels or route IDs.[^1]
TF‑IDF + logistic regression is tiny and essentially a sparse dot‑product, giving very small models (few MB) and very fast inference.[^1]
    - fastText (C++ with Python bindings) can classify half a million sentences among 312k classes in under a minute on a standard multi‑core CPU, with per‑query latencies around 8–11 ms in production tests.[^7]
    - Pros: good accuracy for short text classification; deterministic once you fix the model and thresholds; very low CPU and RAM usage.[^7][^1]
    - Cons: limited contextual understanding (no word order), may blur task boundaries when labels are subtle.[^1]
- **Small transformers (<100M) as routers**
    - Surveys on prompt routing note that supervised transformers (e.g. RoBERTa‑style) can be fine‑tuned to map queries to model IDs or task classes, achieving higher accuracy than bag‑of‑words on subtle distinctions.[^5][^4]
    - Small sentence‑transformer style encoders (e.g., MiniLM‑like) can be quantized to INT8 with <1% accuracy loss, reducing model size by ~4× while keeping good routing performance.[^1]
    - Pros: better semantics; can detect fuzzy categories like “math‑ish but also question‑answering”.[^5]
    - Cons: heavier than TF‑IDF/fastText; a 50–80M router will take tens of milliseconds per query on CPU and tens of MB of RAM, which is still acceptable but high compared to classical models.[^1]

Given your constraints (≤200 ms routing, ≤200 MB RAM), rule‑based + a classical classifier is usually enough; small transformers can be reserved for borderline cases that your classical classifier flags as “uncertain”.[^4][^1]

### Concrete routing features for your tasks

You can design a deterministic feature set combining simple regex/keywords with bag‑of‑words features:

- **Factual Q\&A vs math problems**
    - Math indicators: presence of many digits, `+ - * / % =`, phrases like “how many”, “total”, “difference”, “ratio”, “probability”, “per”, “each”, “average”.
Symbolic math benchmarks (e.g. GSM8K) explicitly use multi‑step arithmetic expressions with such lexical cues.[^8][^9]
    - Factual indicators: “who/what/when/where/which”, “define”, “explain”, named entities, but few arithmetic tokens.
- **Summarization**
    - Indicators: verbs like “summarize”, “condense”, “tl;dr”, “brief overview”, “in N sentences”, “bullet points”, plus long input length (e.g. >500 characters).[^10][^11]
- **Sentiment**
    - Indicators: emotional adjectives (“great”, “awful”), polarity words (“love”, “hate”), and presence of first‑person opinions (“I think”, “I feel”).
Rule‑based lexicon sentiment tools like VADER are built around such features.[^12][^13]
- **NER**
    - Indicators: many capitalized tokens, dates, locations, organizations; presence of patterns like email addresses, titles (“Mr.”, “Dr.”, “Inc.”, “LLC”).
NER libraries like spaCy and Flair train on such entity patterns.[^14][^15]
- **Code**
    - Indicators: code fences ````, high frequency of symbols like `(){}[]<>`, keywords like `def`, `class`, `for`, `if`, `return`, language‑specific keywords (“import”, “function”, “console.log”).
Code‑focused models like Qwen2.5‑Coder are explicitly trained on source code with these tokens.[]

A practical pattern is:

1. Simple regex/keyword routing for obvious cases (e.g., “summarize … in 3 bullets” → summarization pipeline).
2. fastText or TF‑IDF+LR as a multi‑class task classifier for ambiguous cases, trained on your own labeled examples for `FACT_QA`, `MATH`, `SUMM`, `SENTIMENT`, `NER`, `CODE`.[][]

This keeps routing latency well under your 200 ms budget while using <100 MB RAM total for router + task‑specific tools.[][][]

***

## 2. Classical / small models for routing and validation

### Task classification (router)

Suitable non‑LLM classifiers:

- **TF‑IDF + Logistic Regression / Linear SVM**
    - Implementation: scikit‑learn’s `TfidfVectorizer` + `LogisticRegression` or `LinearSVC`.[]
    - Pros: tiny models (weights matrix + vocabulary mapping), good accuracy for short texts; easy to retrain on new labels.[]
    - Resource: models in the low MB range; inference is a sparse dot‑product and softmax, typically a few ms on CPU per query.[]
- **fastText supervised classifier**
    - Implementation: `fasttext.train_supervised` on labeled data; binds into Python via the official library.[]
    - fastText can train on over 1B words in <10 minutes on a standard multi‑core CPU and classify 500k sentences among 312k classes in under a minute, with average per‑query latency ~8 ms and max ~11 ms in one deployment report.[]
    - Models can be compressed to a few hundred kilobytes for mobile and embedded devices while maintaining similar accuracy.[]
    - This makes fastText an excellent router for your container.
- **Small sentence transformers / SetFit‑style routers**
    - A practical guide on small on‑device classifiers notes that sentence transformers (MiniLM‑like) can be quantized to INT8 with ≲1% accuracy loss and used with SetFit to get strong performance with few labeled examples.[]
    - Use case: second‑stage router that only runs when fastText or LR says “uncertain”.

Given your latency and RAM budget, I’d use:

- Primary router: fastText (or TF‑IDF+LR) for task classification.[][]
- Optional secondary: a tiny transformer (e.g., MiniLM‑size) for borderline cases.


### Sentiment classifiers

You want mixed sentiment detection and deterministic outputs:

- **VADER**
    - Lexicon and rule‑based sentiment analyzer optimized for social media and short text.[]
    - Studies report accuracy around 58–60% on ternary classification (pos/neu/neg) in some datasets, outperforming TextBlob and Flair baselines in one comparative analysis.[][]
    - Because it’s purely rule‑based, it’s deterministic given fixed thresholds on compound scores.[][]
- **TextBlob**
    - Pattern‑based with polarity scores; simpler but often less accurate than VADER on nuanced language.[][]
    - Deterministic but more brittle on sarcasm and finance‑style text.[]
- **fastText sentiment models**
    - You can train a fastText classifier on your own sentiment labels (e.g. `POS`, `NEG`, `MIXED`, `NEUTRAL`). fastText’s speed and compression make it suitable for real‑time dashboards and mixed sentiment detection.[][]
- **Hybrid VADER + transformer**
    - A recent framework combines VADER (fast, lexicon) with DistilBERT (contextual) to balance speed and accuracy for real‑time sentiment, using VADER for coarse routing and DistilBERT when deeper context is needed.[]
    - You can mimic this with fastText instead of DistilBERT to stay fully CPU‑friendly.

Deterministic mixed sentiment detection pattern:

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    pos = scores["pos"]; neg = scores["neg"]
    if pos > 0.3 and neg > 0.3:
        return "MIXED"
    elif compound >= 0.05:
        return "POS"
    elif compound <= -0.05:
        return "NEG"
    else:
        return "NEUTRAL"
```

This is fully deterministic given the thresholds.[][]

### NER models for validation

You want deterministic PERSON / ORG / LOC / DATE extraction:

- **spaCy**
    - `en_core_web_sm` is a small English model with NER, POS, and parser, optimized for speed on CPU.[][]
    - spaCy docs emphasize it as a good general starting point: small footprint, fast pipeline in Cython, and balanced accuracy vs speed.[][]
    - Performance notes: enabling NER and parser, you can process roughly `100,000 * available_RAM_in_GB` characters at a time before hitting memory limits; larger inputs require chunking.[]
    - For your container, `en_core_web_sm` is a reasonable NER validator.

Example:

```python
import spacy
nlp = spacy.load("en_core_web_sm")  # small, CPU-optimized NER[^54]

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
```

- **Flair**
    - Flair NER offers richer accuracy but is significantly slower: one issue report notes that spaCy NER processed >50 GB of text in 2 days while Flair only managed ~500 MB over the same period, illustrating a large speed gap.[]
    - Flair provides “fast” NER variants (e.g., `ner-fast`, `ner-ontonotes-fast`) that trade some accuracy for speed.[][]
    - For validation, you could use Flair “fast” for smaller snippets if you need a second NER opinion, but spaCy alone is often enough for deterministic validation.
- **Stanza / tiny BERT variants**
    - Stanza and transformers‑based NER models exist, but they tend to be heavier than spaCy’s CNN‑style models and may not fit your tight latency budget as well.[]

For validation, run spaCy NER on:

- The **input** (to extract entities used in the question).
- The **LLM output** (to extract entities claimed in the answer).

Then compare sets; if the LLM introduces extra entities or mislabels them, you can flag or correct based on spaCy’s labels.[][]

***

## 3. Deterministic tools for math, code, summarization, and factual lookup

### Math / numeric reasoning

You can build a neuro‑symbolic math pipeline:

- **Symbolic backends: SymPy + mpmath**
    - Neuro‑symbolic frameworks for math reasoning explicitly combine LLMs with symbolic math solvers to guarantee correctness, using the LLM to translate between natural language and formal expressions and the solver to compute exact answers.[][]
    - GSM‑Symbolic introduces benchmarks based on symbolic templates to stress such pipelines.[][]
- **Pattern**

1. LLM interprets the word problem and outputs a structured representation: a sequence of operations or a `{"equation": "...", "answer": ...}` JSON.[][]
2. You parse and validate the expression (e.g. only allow `+ - * / **` and numeric literals).
3. SymPy/mpmath compute the result deterministically.

Example:

```python
import sympy as sp

def solve_expression(expr_str: str):
    # Very conservative parsing
    expr = sp.sympify(expr_str, dict())
    result = sp.N(expr)
    return result
```

You can also validate that the numeric answer in the LLM’s explanation equals the SymPy result within a tolerance.

### Code generation, debugging, analysis

Deterministic tools:

- **Ruff (linting)**
    - Ruff is an extremely fast Python linter written in Rust, re‑implementing rules from Flake8, isort, pyupgrade, etc. and running in a single AST pass.[]
    - It’s often >10× faster than traditional Python linters on large codebases.[]
- **Black (formatting)**
    - Black is a deterministic, “uncompromising” Python formatter that ignores existing formatting to avoid non‑determinism; the same input always yields the same output given its configuration.[][]
- **Static type checkers (mypy, pyright)**
    - mypy is a popular static type checker that enforces type hints without running code, helping catch type mismatches ahead of time.[][]
    - pyright offers similar static analysis and powers the VS Code Pylance extension.[]

Pattern:

1. LLM proposes code.
2. You run: black → ruff → mypy/pyright.
3. If any tool fails, send back diagnostics or trigger a repair pass.

Example Python integration:

```python
import subprocess
from pathlib import Path

def format_and_check(path: Path):
    subprocess.run(["black", str(path)], check=True)          # deterministic formatting[^30]
    subprocess.run(["ruff", "check", str(path)], check=True)  # fast linting[^29]
    subprocess.run(["mypy", str(path)], check=False)          # type checking[^31]
```

- **Coding agents with small LLMs**
    - Frameworks like SuperAGI and similar agent stacks show patterns where LLMs call tools (compilers, tests) and use feedback to refine code, though many examples assume larger models.[]
    - For small models, Qwen2.5‑Coder‑1.5B‑Instruct‑GGUF is explicitly designed for self‑hosted code generation, reasoning, and fixing via llama.cpp or Ollama, with Apache‑2.0 license.[][]
    - Combine Qwen2.5‑Coder with black+ruff+mypy to create a deterministic coding agent.


### Extractive summarization (deterministic)

Extractive algorithms are deterministic given fixed parameters:

- **TextRank / LexRank / Luhn / LSA**
    - Tutorials from IBM and others explain extractive methods like Luhn (frequency‑based), LexRank (graph‑based centrality), and LSA (latent semantic analysis) for summarizing text.[][]
    - Sumy is a lightweight Python library that implements Luhn, LexRank, Edmundson, LSA, KL‑Sum, and other algorithms, with built‑in tokenization and stemming.[][][]

Example with Sumy (LexRank):

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def lexrank_n_sentences(text: str, n: int):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    sentences = summarizer(parser.document, n)  # exactly n sentences[^13][^33]
    return [str(s) for s in sentences]
```

For “exactly N bullet points under M words”, post‑process the selected sentences: truncate to M words, prepend bullet markers.

Quality comparison:

- Classical extractive summarizers are robust and deterministic for news, technical docs, and long passages; they preserve key sentences but can be less abstractive or coherent than LLM‑based summarizers.[][]
- For constrained formats (N sentences, N bullets, strict word limits), extractive methods often outperform small LLMs, which can violate format or hallucinate details.[][]


### Factual knowledge: deterministic retrieval

You can use lexical BM25 retrieval and light knowledge bases:

- **BM25S / Rank‑BM25**
    - Rank‑BM25 is a simple Python BM25 implementation using NumPy; BM25S is a newer library that uses SciPy sparse matrices and precomputes term‑document scores to achieve up to 500× speedups over Rank‑BM25 while remaining pure Python.[][][]
    - BM25S indexes a corpus in memory and provides fast top‑k retrieval with deterministic scoring given fixed parameters (k1, b).[][]

Example BM25S:

```python
import bm25s

corpus = ["...", "..."]  # your docs
retriever = bm25s.BM25(corpus=corpus)
retriever.index(bm25s.tokenize(corpus))      # deterministic index[^22][^32]

def retrieve_top_k(query: str, k: int = 5):
    q_tokens = bm25s.tokenize([query])
    results, scores = retriever.retrieve(q_tokens, k=k)
    return results.tolist(), scores.tolist()
```

- **Pyserini / Lucene**
    - Pyserini wraps Anserini/Lucene to provide Lucene‑grade BM25 retrieval from Python, with prebuilt indexes and configurable analyzers.[][]
    - Example: `LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')` followed by `search("query")`.[]
- **SQLite / JSON KBs**
    - For small structured facts (e.g. product tables, internal docs), you can store them in SQLite or JSON and query deterministically by key or simple SQL.

Using LLMs as explainers:

- Surveys on LLM routing and hybrid systems recommend using retrieval as the primary knowledge source and LLMs mainly to **explain or compose** answers from retrieved facts.[][][]
- Pipeline:

1. BM25S/Pyserini retrieves top‑k passages deterministically.[][]
2. LLM is prompted with “Using only the following passages, explain …”, and its output is later validated against those passages (e.g., check that claimed facts appear in retrieved text).

***

## 4. End‑to‑end deterministic routing and validation design

### High‑level architecture

This aligns with routing papers (Lessons Learned from LLM Routing, UniRoute, Router‑R1) but replaces LLM‑as‑router with classical components.[][][][][]

**Components:**

1. **Input Pre‑processor**
    - Detects language, length, presence of code fences, etc.
2. **Rule‑based Pre‑Router**
    - Hard routes obvious commands:
        - “summarize … in N sentences/bullets” → Summarization pipeline.
        - “run this code”, “fix this Python function” → Code pipeline.
        - Very short opinion text → Sentiment pipeline.
3. **Classical Task Classifier**
    - fastText or TF‑IDF+LR multi‑class classifier: `FACT_QA`, `MATH`, `SUMM`, `SENTIMENT`, `NER`, `CODE`.[][]
4. **Tool \& Model Selector**
    - Based on task class:
        - `MATH`: math LLM + SymPy.
        - `CODE`: code LLM + black/ruff/mypy.
        - `FACT_QA`: general LLM + BM25S/Pyserini.
        - `SUMM`: extractive summarization; LLM only for style if needed.
        - `SENTIMENT`: VADER/fastText; LLM optional.
        - `NER`: spaCy; LLM optional.
5. **Small LLM(s) (≤1.7B, GGUF)**
    - Used mainly for:
        - Interpreting problems.
        - Generating structured representations (equations, JSON, bullet skeletons).
        - Explanations over retrieved facts.
6. **Deterministic Tool Layer**
    - SymPy/mpmath, BM25S/Pyserini, spaCy, VADER, Sumy, black/ruff/mypy, etc.
7. **Validator / Judge**
    - Compares LLM outputs with tool results:
        - Numeric equality, entity overlaps, format constraints.
    - Applies fixed rules: “prefer tool result when disagreement”, “reject outputs that violate schema”.
8. **Logging \& Monitoring**
    - Logs routing decisions, tool vs LLM disagreements, fallback usage.

### Textual diagram

Think of it as:

```text
User input
   ↓
Pre-router (rules)
   ↓
Task classifier (fastText / TF-IDF+LR)
   ↓
Task-specific pipeline
   ├─ MATH:    LLM → equation JSON → SymPy → validator
   ├─ CODE:    LLM → code → black/ruff/mypy → validator
   ├─ FACT_QA: BM25S → LLM explanation → validator
   ├─ SUMM:    Sumy (LexRank) → LLM polishing (optional) → validator
   ├─ SENT:    VADER/fastText → LLM explanation (optional)
   └─ NER:     spaCy → LLM explanation (optional)
```


### Routing logic pseudocode

```python
def route_request(request: str) -> dict:
    # 1. Rule-based pre-routing
    if "summarize" in request.lower():
        if "bullet" in request.lower():
            return {"task": "SUMM", "mode": "BULLETS"}
        return {"task": "SUMM", "mode": "SENTENCES"}

    if "```" in request or "def " in request or "class " in request:
        return {"task": "CODE"}

    # 2. Classical classifier (fastText or TF-IDF+LR)
    task = task_classifier.predict(request)  # e.g., "MATH", "FACT_QA", etc.[web:21][web:11]
    return {"task": task}
```


### Tool + LLM integration sketches

**Math pipeline:**

```python
def math_pipeline(question: str, llm):
    # LLM generates structured representation
    prompt = (
        "Read the math word problem and output JSON with 'equation' and 'answer'. "
        "Use only + - * / and numbers.\n\nProblem:\n" + question
    )
    llm_json = llm.generate_json(prompt)  # small math-capable model[web:12][web:27]

    equation_str = llm_json["equation"]
    llm_answer = float(llm_json["answer"])

    sympy_answer = solve_expression(equation_str)  # deterministic[web:20]

    if abs(sympy_answer - llm_answer) < 1e-6:
        final_answer = llm_json  # trusted
    else:
        # Prefer SymPy result, keep LLM explanation but correct number
        llm_json["answer"] = float(sympy_answer)
        final_answer = llm_json

    return final_answer
```

**Code pipeline:**

```python
def code_pipeline(prompt: str, llm):
    code = llm.generate_code(prompt)  # Qwen2.5-Coder-1.5B[web:43]
    path = Path("/tmp/generated.py")
    path.write_text(code)

    format_and_check(path)  # black + ruff + mypy[web:29][web:30][web:31]
    formatted = path.read_text()

    return formatted
```

**Summarization pipeline:**

```python
def summarization_pipeline(text: str, mode: str, n: int, llm=None):
    if mode == "SENTENCES":
        sentences = lexrank_n_sentences(text, n)  # Sumy LexRank[web:13][web:33]
        return " ".join(sentences)
    else:
        sentences = lexrank_n_sentences(text, n)
        bullets = []
        for s in sentences:
            words = s.split()
            bullets.append("- " + " ".join(words[:30]))  # limit length
        return "\n".join(bullets)
```

**Factual Q\&A pipeline:**

```python
def fact_qa_pipeline(question: str, llm, retriever):
    docs, scores = retrieve_top_k(question, k=5)  # BM25S[web:22][web:32]
    context = "\n\n".join(docs)

    prompt = (
        "Answer the question using ONLY the facts in the context. "
        "If information is missing, say 'I don't know'.\n\n"
        "Context:\n" + context + "\n\nQuestion:\n" + question
    )
    answer = llm.generate(prompt)  # small general LLM[web:34][web:52]

    # Validator: ensure key facts appear in context
    # e.g., check that named entities in answer are in context
    entities_context = set(e for d in docs for e, _ in extract_entities(d))  # spaCy[web:25][web:54]
    entities_answer = set(e for e, _ in extract_entities(answer))

    if not entities_answer.issubset(entities_context):
        # fallback: return top passage or conservative answer
        return docs[^0]

    return answer
```


### Deterministic validation rules

You can codify validators as pure functions with no randomness:

- **Math:** numeric equality or tolerance check between LLM answer and solver result; reject if mismatch.
- **Code:** linting and type‑checking must succeed; otherwise mark answer as “invalid” and either ask the LLM to fix or return compilation errors.
- **Summarization:** validate:
    - Exactly N sentences / N bullets.
    - Each bullet under M words.
    - No hallucinated facts (optional: NER overlap with source).
- **Factual:** entity overlap, absence of entities not in retrieved context, optional keyword overlap.
- **Sentiment:** VADER scores must align with label; for mixed sentiment, both positive and negative proportions above threshold.

All of these are deterministic given fixed thresholds and tool versions.[^13][^2][^16][^17]

***

## 5. Model selection under ≤1.7B constraint

Below are candidate small LLMs (≤1.7B) that fit your container and can be quantized to GGUF Q4_K_M for llama.cpp/Ollama.[^18][^19][^20][^21]

### Candidate models

| Role | Model | Params | GGUF / llama.cpp notes | Benchmarks / strengths | License |
| :-- | :-- | :-- | :-- | :-- | :-- |
| General QA \& summarization | **LFM2.5‑1.2B‑Instruct** | ~1.2B | Liquid’s LFM2.5 family ships GGUF checkpoints compatible with llama.cpp and MLX; designed for edge deployment.[^21] | Blog reports strong instruction following; GSM8K and IFEval scores are high for larger variants (2.6B: 82.41 GSM8K, 79.56 IFEval), suggesting good reasoning for the family; 1.2B variant is tuned for on‑device instruction tasks.[^22][^21] | Likely permissive; Liquid positions LFM2.5 as edge‑friendly with broad deployment options including GGUF.[^21] |
| Math reasoning | **FM2.5‑1.2B‑Thinking** | 1.2B | Reasoning‑optimized; community posts describe it as running fully on‑device under 1 GB RAM and achieving ~85.6 on GSM8K benchmark.[^23] | Very strong GSM8K performance for its size; excellent fit as math specialist behind SymPy pipeline.[^23] | Not clearly specified in snippet; you’d need to confirm on official repo, but intended for on‑device use.[^23] |
| Code generation | **Qwen2.5‑Coder‑1.5B‑Instruct‑GGUF** | 1.5B | Pre‑quantized GGUF version of Qwen2.5‑Coder; supports multiple quantization levels including Q4_K; optimized for llama.cpp/Ollama, with context up to 32k tokens.[^24][^25] | Trained on 5.5T tokens of source code and text‑code data; intended for code generation, reasoning, and fixing.[^24] | Apache‑2.0; suitable for commercial, local deployment.[^24] |
| General chat / backup | **Qwen2.5‑1.5B‑Instruct‑GGUF** | 1.5B | GGUF quantization by bartowski with Q4_K variants in 0.78–3.09 GB file size; designed for CPU/edge inference via llama.cpp.[^19] | Qwen2.5 edge models (0.5B, 1.5B, 3B) maintain strong performance across general benchmarks; 1.5B is a good generalist.[^26][^19] | Apache‑2.0.[^19] |
| Tiny fallback / router tasks | **Qwen2.5‑0.5B‑Instruct (GGUF)** | 0.5B | Community deployments show Qwen2.5‑0.5B GGUF running on low‑spec devices via llama.cpp, recommended as a small edge model.[^20] | Good enough for light chat and explanations; can serve as a low‑cost explainer when you don’t need full 1.5B capability.[^20] | Apache‑2.0 via Qwen.[^26][^20] |
| General small baseline | **TinyLlama‑1.1B** | 1.1B | Llama‑2‑style architecture for constrained hardware; not GGUF in snippet but widely converted to GGUF elsewhere.[^18][^27] | Open LLM Leaderboard: average score ~36.42 with MMLU ~26.04, GSM8K ~1.44, so weak on math reasoning but usable as general text generator.[^27] | Apache‑2.0.[^18][^27] |

For NER and sentiment, you’ll rely on classical tools (spaCy/VADER/fastText) rather than LLMs, avoiding model size constraints altogether.[^15][^13][^28][^7]

### Recommended 3‑model combination

Given your tasks and constraints:

1. **Math specialist** – **FM2.5‑1.2B‑Thinking** (or smallest available math‑tuned LFM2.5 variant)
    - Use for math word problems: generate equations and explanations, then validate with SymPy.[^23][^29]
2. **Code specialist** – **Qwen2.5‑Coder‑1.5B‑Instruct‑GGUF**
    - Use for code generation, debugging, and explanations around static analysis results.[^24][^25]
3. **Generalist / explainer** – **Qwen2.5‑1.5B‑Instruct‑GGUF** *or* **LFM2.5‑1.2B‑Instruct**
    - Use for factual Q\&A explanations over retrieved passages, structured summarization prompts, and glue tasks (e.g., interpreting instructions).[^19][^21]

In deployment you don’t need all three loaded concurrently:

- Load math + general for math‑heavy tasks; unload code model.
- Load code + general for coding tasks.
- Load general only for pure summarization/Q\&A.

GGUF Q4_K_M quantizations for 1.5B and 1.2B models typically sit well under your total RAM; Qwen2.5‑1.5B Q4_K files are ~1–2 GB, and Liquid emphasizes on‑device GGUF support for LFM2.5.[^21][^19]

Pairings with deterministic tools:

- FM2.5‑1.2B‑Thinking + SymPy/mpmath + GSM‑Symbolic‑style neuro‑symbolic pipeline.[^29][^8][^23]
- Qwen2.5‑Coder‑1.5B + black/ruff/mypy for Python, plus language‑specific linters (e.g., ESLint for JS).
- Qwen2.5‑1.5B/LFM2.5‑1.2B + BM25S/Pyserini + spaCy for factual Q\&A and entity‑aware validation.[^30][^2][^31][^15]

***

## 6. Practical best practices and pitfalls

### Deterministic routing

Best practices distilled from routing surveys and case studies:[^6][^3][^4][^5]

- **Start simple:**
    - Use rule‑based pre‑routing for obvious task classes (e.g., explicit “summarize in N sentences”, “here is some Python code”).[^4][^5]
    - Use fastText or TF‑IDF+LR as your main router; only consider small transformers if you see systematic misclassification.[^7][^1]
- **Keep routers stateless and deterministic:**
    - No randomness, no temperature, no sampling; just fixed thresholds and trained weights.[^5][^4]
    - Persist router models as versioned artifacts; changes go through offline evaluation first.[^32][^4]
- **Separate routing from generation:**
    - Avoid LLM‑as‑router in your container; surveys show classifier‑based routing is more predictable and cheaper than LLM‑based routers and RL policies, especially for clear task boundaries.[^33][^4][^5]


### Latency with multiple tools

- **Pipeline optimization:**
    - Do routing in ≤50 ms with fastText or TF‑IDF+LR on CPU.[^7][^1]
    - Index retrieval corpora offline with BM25S; query‑time operations are simple sparse matrix operations, keeping retrieval latency low.[^2][^31]
    - Use spaCy with batch processing (`nlp.pipe`) and disable unused components when validating, as recommended in performance checklists.[^34][^15]
- **Parallelization:**
    - When necessary, run independent tools (e.g., retrieval and NER) in parallel threads; but be mindful of your 2‑CPU limit.


### Logging and monitoring disagreements

- **PromptEval‑style evaluation:**
    - PromptEval provides a framework to estimate performance across many prompts and models using shared evaluation harnesses, helpful for offline analysis of routing and tool vs LLM disagreements.[^32]
    - You can log:
        - Router decisions and downstream model/tool calls.
        - LLM answers and tool‑verified results.
        - A “disagreement score” (e.g., numeric mismatch, entity mismatch).
- **Use disagreements to improve tools:**
    - When tools consistently override LLM outputs (e.g., SymPy often corrects math), consider tightening validator thresholds or adjusting LLM prompting to focus more on explanation than answer selection.[^9][^29]


### Common pitfalls

From multi‑LLM routing literature and practical talks:[^3][^6][^4][^5]

- **Over‑complicating routing logic:**
    - Complex RL‑based routers (Router‑R1, MoMA) are powerful but require training data, cost modeling, and LLM‑as‑judge; in a small container this adds latency and unpredictability.[^35][^36][^37]
    - Stick to rule‑based + classical classifier routing.
- **Using small LLMs where tools suffice:**
    - Small LLMs are weaker on math and exact logic than symbolic tools; GSM8K benchmarks show that generic 1.1B models can score as low as ~1.44.[^27]
    - Use LLMs to explain and structure, not to compute whenever a deterministic solver exists.[^29][^9]
- **Resource leaks and performance issues:**
    - Loading multiple large GGUF models simultaneously can exhaust RAM; unload unused models or run only one heavy model at a time.[^19][^21]
    - For spaCy, processing extremely large texts with NER and parser enabled can hit memory limits; split text or disable components as recommended.[^34][^15]
    - Ensure tools like BM25S and Pyserini are initialized once and reused, not re‑created per request, to avoid repeated indexing costs.[^30][^2]

***

## Summary of concrete Python‑implementable suggestions

- **Routing:**
    - Implement rule‑based pre‑router with regex and keyword triggers for `SUMM`, `CODE`, and simple `SENTIMENT`.
    - Train a fastText or TF‑IDF+LR classifier on your labeled data for `FACT_QA`, `MATH`, `SUMM`, `SENTIMENT`, `NER`, `CODE` task classes.[^7][^1]
- **Math:**
    - Use FM2.5‑1.2B‑Thinking (or similar math‑tuned ≤1.7B model) to produce equations and reasoning; validate answers with SymPy/mpmath.[^23][^29]
- **Code:**
    - Use Qwen2.5‑Coder‑1.5B‑Instruct‑GGUF via llama.cpp as your code generator; immediately run black, ruff, and mypy/pyright on outputs.[^16][^17][^38][^24]
- **Summarization:**
    - Use Sumy’s LexRank/Luhn to produce deterministic N‑sentence or N‑bullet summaries; optionally run a small LLM for style polishing but keep extractive content as ground truth.[^39][^40][^11]
- **Factual Q\&A:**
    - Index your corpus with BM25S or Pyserini and route factual questions to retrieval first; have a general LLM explain the retrieved facts and validate entity overlap with spaCy NER.[^31][^15][^2][^30]
- **Sentiment:**
    - Use VADER or a fastText sentiment classifier to assign deterministic labels, including MIXED sentiment based on positive/negative score thresholds.[^12][^13][^7]
- **NER:**
    - Use spaCy `en_core_web_sm` as NER validator to enforce entity consistency between question, retrieved context, and LLM answer.[^15][^28][^34]
- **Evaluation:**
    - Use PromptEval‑style harnesses for offline evaluation of routing decisions and prompt variants, particularly to tune router thresholds and model selection.[^32][^4]

If you’d like, I can next help you turn this into a concrete repo layout (modules for router, pipelines, validators) and detailed llama.cpp/Ollama configurations for the chosen models.
<span style="display:none">[^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^55][^56]</span>

<div align="center">⁂</div>

[^1]: https://rickwinfrey.com/writings/small-models-for-text-classification

[^2]: https://arxiv.org/html/2407.03618v1

[^3]: https://arxiv.org/html/2502.00409v2

[^4]: https://aclanthology.org/2024.insights-1.15.pdf

[^5]: https://www.emergentmind.com/topics/llm-based-prompt-routing

[^6]: https://www.youtube.com/watch?v=2ld7cR9y0Po

[^7]: https://www.opensourceforu.com/2017/11/fast-text-incredibly-fast-text-classification/

[^8]: https://arxiv.org/html/2410.05229v1

[^9]: https://www.emergentmind.com/topics/gsm8k-dataset

[^10]: https://www.ibm.com/think/tutorials/text-summarization-python

[^11]: https://machinelearninggeek.com/text-summarization-using-python/

[^12]: https://kth.diva-portal.org/smash/get/diva2:1890072/FULLTEXT02.pdf

[^13]: https://arxiv.org/html/2504.15448v2

[^14]: https://danlp-alexandra.readthedocs.io/en/latest/docs/tasks/ner.html

[^15]: https://deepnote.com/blog/ultimate-guide-to-the-spacy-library-in-python

[^16]: https://lwn.net/Articles/930487/

[^17]: https://news.ycombinator.com/item?id=17151813

[^18]: https://github.com/jzhang38/TinyLlama

[^19]: https://dev.co/ai/llms/bartowski-qwen2-5-1-5b-instruct-gguf

[^20]: https://forum.dfinity.org/t/llama-cpp-on-the-internet-computer/33471

[^21]: https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai

[^22]: https://www.liquid.ai/blog/introducing-lfm2-2-6b-redefining-efficiency-in-language-models

[^23]: https://www.facebook.com/groups/DeepNetGroup/posts/2710154086044123/

[^24]: https://dev.co/ai/llms/qwen2-5-coder-1-5b-instruct-gguf

[^25]: https://www.reddit.com/r/Qwen_AI/comments/1uskyw1/superlite_cyber_coder_qwen25_15b_4bit_gguf_for/

[^26]: https://qwenlm.github.io/blog/qwen2.5-llm/

[^27]: https://dev.co/ai/llms/tinyllama-1-1b-intermediate-step-1431k-3t

[^28]: https://spacy.io/usage

[^29]: https://arxiv.org/html/2412.04857v1

[^30]: https://pypi.org/project/pyserini/

[^31]: https://www.emergentmind.com/topics/bm25s

[^32]: https://github.com/felipemaiapolo/prompteval

[^33]: https://openreview.net/pdf?id=ka82fvJ5f1

[^34]: https://stackoverflow.com/questions/74181750/a-checklist-for-spacy-optimization

[^35]: https://arxiv.org/html/2509.07571v1

[^36]: https://arxiv.org/html/2603.04444v4

[^37]: https://ulab-uiuc.github.io/Router-R1/

[^38]: https://realpython.com/ref/glossary/static-type-checker/

[^39]: https://www.geeksforgeeks.org/nlp/mastering-text-summarization-with-sumy-a-python-library-overview/

[^40]: https://www.linkedin.com/posts/analytics-vidhya_automated-text-summarization-with-sumy-library-activity-7385593054826717184-bub8

[^41]: https://ceur-ws.org/Vol-3181/paper59.pdf

[^42]: https://raw.githubusercontent.com/mudler/LocalAI/refs/heads/master/gallery/index.yaml

[^43]: https://www.youtube.com/watch?v=x4KsNDHT_XM

[^44]: https://www.reddit.com/r/AI_Agents/comments/1hir48s/best_ai_agent_framework_low_code_or_no_code/

[^45]: https://github.com/flairNLP/flair/issues/1996

[^46]: https://huggingface.co/blog/xhluca/bm25s

[^47]: https://github.com/QwenLM/Qwen2.5-Math

[^48]: https://stackoverflow.com/questions/50487495/what-is-difference-between-en-core-web-sm-en-core-web-mdand-en-core-web-lg-mod

[^49]: https://talkpython.fm/episodes/show/400/ruff-the-fast-rust-based-python-linter

[^50]: https://stackoverflow.com/questions/59183863/in-python-how-to-tweak-black-formatter-if-possible

[^51]: https://learn.scientific-python.org/development/guides/mypy/

[^52]: https://localai.io/models/

[^53]: https://www.josedavidbaena.com/blog/tiny-language-models/tiny-llm-architecture-comparison

[^54]: https://jds-online.org/journal/JDS/article/1441/file/pdf

[^55]: https://github.com/explosion/spacy-models

[^56]: https://pubmed.ncbi.nlm.nih.gov/41570547/

