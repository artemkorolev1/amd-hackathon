# Summarization Classification Failure Analysis

**Date:** Analysis run
**Total summarization-labeled items found:** 651
**Correctly classified as summarization:** 487 (74.8%)
**Misclassified:** 164 (25.2%)

**Eval files scanned:** 35

## Summary of Misclassifications

| Predicted As | Count | % of Misclassified |
|-------------|-------|-------------------|
| logic | 70 | 42.7% |
| math | 49 | 29.9% |
| code_gen | 40 | 24.4% |
| code_debug | 3 | 1.8% |
| ner | 1 | 0.6% |
| factual | 1 | 0.6% |

## Confusion (Summarization Row Only)

| True\Pred | code_debug | code_gen | factual | logic | math | ner | sentiment | summarization |
|---|---|---|---|---|---|---|---|---|
| summarization | 3 | 40 | 1 | 70 | 49 | 1 | 0 | 487 |

## Top 3 Misclassification Patterns

### 1. Summarization → logic (70 cases)

**Example 1:** task_id=`cnndm-0e18c7febaac`
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nKim Kardashian has revealed that she is going to great lengths to make sure that her luxurious designer wardrobe stays in mint condition because she plans on han
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| logic | 4.0 ← **WINS** |
| summarization | 3.5 |
| factual | 2.5 |
| code_debug | 0.0 |
| code_gen | 0.0 |
| math | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

**Example 2:** task_id=`cnndm-15df7e9a228e`
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nFerguson, Missouri (CNN)Change has come to Ferguson. After months of turmoil and upheaval, months of frustration and anger, the beleaguered city has a new govern
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| logic | 4.5 ← **WINS** |
| summarization | 3.5 |
| factual | 1.5 |
| code_debug | 0.0 |
| code_gen | 0.0 |
| math | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

**Example 3:** task_id=`cnndm-32e8dd065117`
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nSian Harkin, 54, gave £30,000 to a builder using the school chequebook claiming the work was to be carried out at Llwyncelyn Primary School in Porth, when the wo
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| logic | 4.5 ← **WINS** |
| summarization | 3.5 |
| code_debug | 0.0 |
| code_gen | 0.0 |
| factual | 0.0 |
| math | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

**Example 4:** task_id=`cnndm-4608fd83269e`
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nI see Usain Bolt on the circuit and he’s always good fun. He likes to make jokes of me! I remember in Moscow, at the last World Championships, when I was competi
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| logic | 4.0 ← **WINS** |
| summarization | 3.5 |
| code_debug | 0.0 |
| code_gen | 0.0 |
| factual | 0.0 |
| math | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

**Example 5:** task_id=`cnndm-5826dcbc787c`
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nA lonely man has taken his search for love - or lust - online, posting an advertisement complete with his desires, or 'requirements'. The outback Casanova, who l
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| logic | 4.0 ← **WINS** |
| summarization | 3.5 |
| code_debug | 0.0 |
| code_gen | 0.0 |
| factual | 0.0 |
| math | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

... and 65 more similar misclassifications.

### 2. Summarization → math (49 cases)

**Example 1:** task_id=`q_90`
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
On December 5, 2022, scientists at the National Ignition Facility (NIF) at Lawrence Livermore National Laboratory achieved a historic milestone: a fusion reaction that produced more energy than it con
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| math | 4.0 ← **WINS** |
| summarization | 3.5 |
| logic | 1.5 |
| code_debug | 0.0 |
| code_gen | 0.0 |
| factual | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

**Example 2:** task_id=`q_94`
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
The International Energy Agency's World Energy Outlook 2024 presents a comprehensive analysis of the global transition to renewable energy. In 2023, global renewable energy capacity additions reached 
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| math | 4.0 ← **WINS** |
| summarization | 3.0 |
| code_debug | 0.0 |
| code_gen | 0.0 |
| factual | 0.0 |
| logic | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

**Example 3:** task_id=`q_97`
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
A new study published in Nature Communications Earth and Environment has analyzed 40 years of satellite data to quantify Arctic sea ice loss. The study found that September sea ice extent (the annual 
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| math | 4.0 ← **WINS** |
| summarization | 4.0 |
| logic | 1.5 |
| code_debug | 0.0 |
| code_gen | 0.0 |
| factual | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

**Example 4:** task_id=`q_190`
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
On December 5, 2022, scientists at the National Ignition Facility (NIF) at Lawrence Livermore National Laboratory achieved a historic milestone: a fusion reaction that produced more energy than it con
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| math | 4.0 ← **WINS** |
| summarization | 3.5 |
| logic | 1.5 |
| code_debug | 0.0 |
| code_gen | 0.0 |
| factual | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

**Example 5:** task_id=`q_194`
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
The International Energy Agency's World Energy Outlook 2024 presents a comprehensive analysis of the global transition to renewable energy. In 2023, global renewable energy capacity additions reached 
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| math | 4.0 ← **WINS** |
| summarization | 3.0 |
| code_debug | 0.0 |
| code_gen | 0.0 |
| factual | 0.0 |
| logic | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

... and 44 more similar misclassifications.

### 3. Summarization → code_gen (40 cases)

**Example 1:** task_id=`q_91`
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
LEGAL BRIEF - PLAINTIFFS' ARGUMENT (Artists vs. Stability AI, US District Court for the Northern District of California):\n\nThe plaintiffs, representing a class of visual artists, allege that Stability
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| code_gen | 8.0 ← **WINS** |
| summarization | 7.5 |
| code_debug | 0.0 |
| factual | 0.0 |
| logic | 0.0 |
| math | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

**Example 2:** task_id=`q_191`
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
LEGAL BRIEF - PLAINTIFFS' ARGUMENT (Artists vs. Stability AI, US District Court for the Northern District of California):\n\nThe plaintiffs, representing a class of visual artists, allege that Stability
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| code_gen | 8.0 ← **WINS** |
| summarization | 7.5 |
| code_debug | 0.0 |
| factual | 0.0 |
| logic | 0.0 |
| math | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

**Example 3:** task_id=`eval_hard_218.json_q91`
**Source:** `data/eval/primary/eval_hard_218.json`
**Prompt (first 200 chars):**
```
LEGAL BRIEF - PLAINTIFFS' ARGUMENT (Artists vs. Stability AI, US District Court for the Northern District of California):\n\nThe plaintiffs, representing a class of visual artists, allege that Stability
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| code_gen | 8.0 ← **WINS** |
| summarization | 7.5 |
| code_debug | 0.0 |
| factual | 0.0 |
| logic | 0.0 |
| math | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

**Example 4:** task_id=`eval_hard_218.json_q191`
**Source:** `data/eval/primary/eval_hard_218.json`
**Prompt (first 200 chars):**
```
LEGAL BRIEF - PLAINTIFFS' ARGUMENT (Artists vs. Stability AI, US District Court for the Northern District of California):\n\nThe plaintiffs, representing a class of visual artists, allege that Stability
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| code_gen | 8.0 ← **WINS** |
| summarization | 7.5 |
| code_debug | 0.0 |
| factual | 0.0 |
| logic | 0.0 |
| math | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

**Example 5:** task_id=`eval_60_medium_hard.json_q56`
**Source:** `data/eval/primary/eval_60_medium_hard.json`
**Prompt (first 200 chars):**
```
SOURCE 1 (NY Federal Reserve Research Paper, 2025): 'Buy Now, Pay Later (BNPL) users are 3.2x more likely to carry revolving credit card debt. The average BNPL user has a credit score 47 points below 
```

**Raw scores from all 8 scorers:**

| Scorer | Score |
|--------|-------|
| code_gen | 6.0 ← **WINS** |
| summarization | 6.0 |
| math | 2.0 |
| logic | 1.5 |
| code_debug | 0.0 |
| factual | 0.0 |
| ner | 0.0 |
| sentiment | 0.0 |

---

... and 35 more similar misclassifications.

## Complete List of All Misclassified Items

### `q_90` → predicted **math**
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
On December 5, 2022, scientists at the National Ignition Facility (NIF) at Lawrence Livermore National Laboratory achieved a historic milestone: a fusion reaction that produced more energy than it con
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.5
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `q_91` → predicted **code_gen**
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
LEGAL BRIEF - PLAINTIFFS' ARGUMENT (Artists vs. Stability AI, US District Court for the Northern District of California):\n\nThe plaintiffs, representing a class of visual artists, allege that Stability
```

**All scorer scores:**

- code_gen: 8.0 ← WINS
- summarization: 7.5
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `q_94` → predicted **math**
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
The International Energy Agency's World Energy Outlook 2024 presents a comprehensive analysis of the global transition to renewable energy. In 2023, global renewable energy capacity additions reached 
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `q_97` → predicted **math**
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
A new study published in Nature Communications Earth and Environment has analyzed 40 years of satellite data to quantify Arctic sea ice loss. The study found that September sea ice extent (the annual 
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 4.0
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `q_190` → predicted **math**
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
On December 5, 2022, scientists at the National Ignition Facility (NIF) at Lawrence Livermore National Laboratory achieved a historic milestone: a fusion reaction that produced more energy than it con
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.5
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `q_191` → predicted **code_gen**
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
LEGAL BRIEF - PLAINTIFFS' ARGUMENT (Artists vs. Stability AI, US District Court for the Northern District of California):\n\nThe plaintiffs, representing a class of visual artists, allege that Stability
```

**All scorer scores:**

- code_gen: 8.0 ← WINS
- summarization: 7.5
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `q_194` → predicted **math**
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
The International Energy Agency's World Energy Outlook 2024 presents a comprehensive analysis of the global transition to renewable energy. In 2023, global renewable energy capacity additions reached 
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `q_197` → predicted **math**
**Source:** `input/cx_300.json`
**Prompt (first 200 chars):**
```
A new study published in Nature Communications Earth and Environment has analyzed 40 years of satellite data to quantify Arctic sea ice loss. The study found that September sea ice extent (the annual 
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 4.0
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_hard_218.json_q90` → predicted **math**
**Source:** `data/eval/primary/eval_hard_218.json`
**Prompt (first 200 chars):**
```
On December 5, 2022, scientists at the National Ignition Facility (NIF) at Lawrence Livermore National Laboratory achieved a historic milestone: a fusion reaction that produced more energy than it con
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.5
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_hard_218.json_q91` → predicted **code_gen**
**Source:** `data/eval/primary/eval_hard_218.json`
**Prompt (first 200 chars):**
```
LEGAL BRIEF - PLAINTIFFS' ARGUMENT (Artists vs. Stability AI, US District Court for the Northern District of California):\n\nThe plaintiffs, representing a class of visual artists, allege that Stability
```

**All scorer scores:**

- code_gen: 8.0 ← WINS
- summarization: 7.5
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_hard_218.json_q94` → predicted **math**
**Source:** `data/eval/primary/eval_hard_218.json`
**Prompt (first 200 chars):**
```
The International Energy Agency's World Energy Outlook 2024 presents a comprehensive analysis of the global transition to renewable energy. In 2023, global renewable energy capacity additions reached 
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_hard_218.json_q97` → predicted **math**
**Source:** `data/eval/primary/eval_hard_218.json`
**Prompt (first 200 chars):**
```
A new study published in Nature Communications Earth and Environment has analyzed 40 years of satellite data to quantify Arctic sea ice loss. The study found that September sea ice extent (the annual 
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 4.0
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_hard_218.json_q190` → predicted **math**
**Source:** `data/eval/primary/eval_hard_218.json`
**Prompt (first 200 chars):**
```
On December 5, 2022, scientists at the National Ignition Facility (NIF) at Lawrence Livermore National Laboratory achieved a historic milestone: a fusion reaction that produced more energy than it con
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.5
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_hard_218.json_q191` → predicted **code_gen**
**Source:** `data/eval/primary/eval_hard_218.json`
**Prompt (first 200 chars):**
```
LEGAL BRIEF - PLAINTIFFS' ARGUMENT (Artists vs. Stability AI, US District Court for the Northern District of California):\n\nThe plaintiffs, representing a class of visual artists, allege that Stability
```

**All scorer scores:**

- code_gen: 8.0 ← WINS
- summarization: 7.5
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_hard_218.json_q194` → predicted **math**
**Source:** `data/eval/primary/eval_hard_218.json`
**Prompt (first 200 chars):**
```
The International Energy Agency's World Energy Outlook 2024 presents a comprehensive analysis of the global transition to renewable energy. In 2023, global renewable energy capacity additions reached 
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_hard_218.json_q197` → predicted **math**
**Source:** `data/eval/primary/eval_hard_218.json`
**Prompt (first 200 chars):**
```
A new study published in Nature Communications Earth and Environment has analyzed 40 years of satellite data to quantify Arctic sea ice loss. The study found that September sea ice extent (the annual 
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 4.0
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_60_medium_hard.json_q56` → predicted **code_gen**
**Source:** `data/eval/primary/eval_60_medium_hard.json`
**Prompt (first 200 chars):**
```
SOURCE 1 (NY Federal Reserve Research Paper, 2025): 'Buy Now, Pay Later (BNPL) users are 3.2x more likely to carry revolving credit card debt. The average BNPL user has a credit score 47 points below 
```

**All scorer scores:**

- code_gen: 6.0 ← WINS
- summarization: 6.0
- math: 2.0
- logic: 1.5
- code_debug: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_60_medium_hard.json_q57` → predicted **code_gen**
**Source:** `data/eval/primary/eval_60_medium_hard.json`
**Prompt (first 200 chars):**
```
SOURCE: A Wikipedia article on the history of programming languages:\n\n'Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability with significant whitespace. Py
```

**All scorer scores:**

- code_gen: 4.5 ← WINS
- math: 4.0
- summarization: 4.0
- factual: 3.5
- code_debug: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_clean_val.json_q2810` → predicted **math**
**Source:** `data/eval/primary/eval_clean_val.json`
**Prompt (first 200 chars):**
```
The International Energy Agency's World Energy Outlook 2024 presents a comprehensive analysis of the global transition to renewable energy. In 2023, global renewable energy capacity additions reached 
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_clean_val.json_q10431` → predicted **math**
**Source:** `data/eval/primary/eval_clean_val.json`
**Prompt (first 200 chars):**
```
On December 5, 2022, scientists at the National Ignition Facility (NIF) at Lawrence Livermore National Laboratory achieved a historic milestone: a fusion reaction that produced more energy than it con
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.5
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-0b86e96e2f9f` → predicted **code_gen**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nA valet parking attendant managed to plough a 200mph supercar through the front of a shop after mistaking the brake for the throttle in a Ferrari 599 GTO worth £
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- factual: 2.5
- code_debug: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-0e18c7febaac` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nKim Kardashian has revealed that she is going to great lengths to make sure that her luxurious designer wardrobe stays in mint condition because she plans on han
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-0e8e76f08eff` → predicted **code_gen**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nMore than a third of GPs are considering retirement in the next five years, a survey shows. Another one in ten is thinking about moving abroad to countries inclu
```

**All scorer scores:**

- code_gen: 3.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-15df7e9a228e` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nFerguson, Missouri (CNN)Change has come to Ferguson. After months of turmoil and upheaval, months of frustration and anger, the beleaguered city has a new govern
```

**All scorer scores:**

- logic: 4.5 ← WINS
- summarization: 3.5
- factual: 1.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-191cee6e240b` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nThe National Rifle Association gathered on Saturday to condemn Barack Obama and Hillary Clinton as 'elitists' who will 'dismantle our freedoms and reshape Americ
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-32e8dd065117` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nSian Harkin, 54, gave £30,000 to a builder using the school chequebook claiming the work was to be carried out at Llwyncelyn Primary School in Porth, when the wo
```

**All scorer scores:**

- logic: 4.5 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-4608fd83269e` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nI see Usain Bolt on the circuit and he’s always good fun. He likes to make jokes of me! I remember in Moscow, at the last World Championships, when I was competi
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-5826dcbc787c` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nA lonely man has taken his search for love - or lust - online, posting an advertisement complete with his desires, or 'requirements'. The outback Casanova, who l
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-5988a797ddda` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nThree teenage girls were rescued after their car careened over a 100-foot cliff in Arizona and landed upside down on top of the 16-year-old driver who was thrown
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-609e57bcda34` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nThe saying 'two is company but three's a crowd' clearly didn't apply to James Corden as he met two Beckham sporting stars. The comedian tweeted a picture on Thur
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-6b65004a4b8d` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nIf you are an aspiring manager in English football and you are black, you get used to one thing: when an opportunity comes your way, it’s going to be the manager
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-7763f8c99c2b` → predicted **code_gen**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\n(CNN)Robert Boardwine's path to fatherhood was unconventional, but Virginia's appeals court said Tuesday he is legally entitled to be a part of his son's life. B
```

**All scorer scores:**

- code_gen: 3.0 ← WINS
- logic: 3.0
- summarization: 3.0
- factual: 1.5
- code_debug: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-7c1c4ec3e1f9` → predicted **code_gen**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nSome teachers draw pictures on the whiteboard to explain new concepts to students. But one chemistry lecturer draws on her own skin, due to an unusual medical co
```

**All scorer scores:**

- code_gen: 3.0 ← WINS
- summarization: 3.0
- factual: 1.5
- code_debug: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-96ca82f022d8` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nAdele Sarno (above) has been living in her two-bedroom apartment for more than 50 years . An Italian-American grandmother is facing eviction from her $820-a-mont
```

**All scorer scores:**

- logic: 6.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-9edd33058ca5` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nThe story of Rapunzel, the girl trapped in a tower who is rescued by a prince climbing up her long locks, has captivated little would-be princesses the world ove
```

**All scorer scores:**

- logic: 6.0 ← WINS
- factual: 4.0
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-b5e037e0aeec` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nA Chicago single mother has revealed what it's really like to live on McDonald's wages, where she makes only $10.50 an hour and can only afford to sleep in a mol
```

**All scorer scores:**

- math: 6.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-cf4162cd0efb` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nWhether you sprout the odd grey hair, spend hours eradicating them or let your silver locks flow free, the latest hair trend is good news for women everywhere. G
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-d48e20422198` → predicted **code_gen**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nEd Balls was today accused of 'letting the cat out of the bag' on tax rises after leaving the door open to trapping more middle-class workers in the 40p tax rate
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-dd5762671828` → predicted **code_gen**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nJames Ward (pictured) was left with a severed finger and wounds all over of his body after being attacked at his home by axe-wielding intruders . A father-of-six
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- factual: 2.5
- code_debug: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-e62540507d17` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nLucy Mecklenburgh shot to fame as one of the postergirls of The Only Way Is Essex - and she's not the only family member making waves in the modelling world. Luc
```

**All scorer scores:**

- logic: 8.5 ← WINS
- summarization: 3.5
- factual: 2.5
- sentiment: 1.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0

---

### `cnndm-e785b8a0f375` → predicted **code_gen**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nAhead of another weekend in the Barclays Premier League, Sportsmail brings you the latest squad news, odds and stats on every top flight fixture as it breaks. Si
```

**All scorer scores:**

- code_gen: 6.5 ← WINS
- logic: 4.0
- factual: 2.5
- summarization: 2.5
- math: 2.0
- code_debug: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-fc5ee5d1eef1` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nMost mothers dream of their children's birthdays, graduations and weddings. But Jodie Barden has the heartbreaking task of arranging both her daughters' funerals
```

**All scorer scores:**

- logic: 7.5 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-0aeeb467858c` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nJordan's significance results partly from its strategic location at the crossroads of what Christians, Jews and Muslims call the Holy Land.\nIt is a k
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-0d6acd081be9` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nThe summertime routes were due to end in late September but due to poor seat sales, the airline is stopping the service at the end of August.\nThe air
```

**All scorer scores:**

- math: 4.0 ← WINS
- factual: 2.5
- summarization: 2.5
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-1267a3e46a87` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nMedia playback is not supported on this device\nBut the tagline - Alter Your Reality - could easily be used to promote the fight between Conor McGrego
```

**All scorer scores:**

- logic: 6.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-18c005d22465` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nThe admission comes a day after mSpy told BBC News it had not been hacked and no data had been stolen.\nIt has also emerged that the UK's Information 
```

**All scorer scores:**

- logic: 3.0 ← WINS
- summarization: 3.0
- factual: 2.5
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-30a7d0844d65` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nIain Duncan Smith, Leave campaigner and another ex-Conservative leader, said: "You can't claim democracy when you want it and reject it when you don'
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-318460a69cc3` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nHer previous attempt to change the law was defeated in parliament but she said the public now had better awareness of the issue.\nThe Lothians MSP, wh
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-375606908925` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nIn part the answer might be that they are not as solid as we have been led to believe. After all, the man who headed an inquiry into the future safet
```

**All scorer scores:**

- logic: 3.5 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-3c48f6156432` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nPhilippe Bianchi told France Info radio his son, who remains in a coma, had shown "no significant progress" since crashing into a recovery vehicle la
```

**All scorer scores:**

- math: 6.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-55e11e2be361` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nPolice said Kinga Pelc from the east Belfast area died in hospital on Saturday.\nA man in his 20s remains in a critical condition in hospital followin
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 2.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-5821186df69b` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nAlmost 80,000 passengers and crew have sailed into Belfast this year on 43 different tourist ships.\nOne of the big draws is the new Titanic Visitor C
```

**All scorer scores:**

- math: 4.0 ← WINS
- logic: 3.0
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-59ad3c679b08` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nFe gysylltodd teulu Elen Jones, 36 oed a'i mab wyth oed, Lewis Rhys Jones gyda'r heddlu i ddweud nad yw'r ddau wedi cael eu gweld ers dydd Mawrth 17 
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 2.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-640ff6dc792f` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nBroadband suppliers will now have to show upfront and monthly costs, without separating out line rental prices, according to the changes brought in b
```

**All scorer scores:**

- math: 6.0 ← WINS
- summarization: 3.0
- sentiment: 2.5
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0

---

### `xsum-6e87dec0b3b5` → predicted **code_gen**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nMr Jammeh has been given until noon on Friday to leave office or be forced out by UN-backed regional forces.\nTroops have been told to halt their adva
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- factual: 2.5
- math: 2.0
- code_debug: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-74f962bf42f3` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nIt is hoped it will help recognise the "vital role" of those educating and caring for children aged up to seven.\nEducation Minister Huw Lewis said th
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- sentiment: 2.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0

---

### `xsum-7b8fcdf79219` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nThe 32-year old, who has won 12 caps for his country, has been signed to provide injury cover.\nOspreys tight heads Dmitri Arhip and Ma'afu Fia are cu
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-7ce8e5e2d202` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nLike-for-like sales were up by 4.9% during the quarter compared with a year earlier, in part due to the continued success of meal deals.\nThe company 
```

**All scorer scores:**

- math: 6.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-7eda9fc5477b` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nJason James, 41, of Charles Road, Torquay, previously admitted manslaughter at Winchester Crown Court.\nDave "Chewie" Coxon, from Torquay died after b
```

**All scorer scores:**

- logic: 3.0 ← WINS
- summarization: 3.0
- factual: 2.5
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-7f1476b64dab` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nIt's the fifth coin portrait to have been created during the Queen's reign.\nIt was unveiled in a special ceremony in London and coins carrying the ne
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 2.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-85a7dc54ca4a` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nColin Hughes was attacked in Old Swan in September 2010 as he went to investigate a noise at the door of his  home.\nTwo men, aged 18 and 21, from Old
```

**All scorer scores:**

- logic: 3.0 ← WINS
- summarization: 2.5
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-86350cd25ceb` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nThe Times Educational Supplement (TES) says measures to guard against grade inflation were not appropriate for this year's English exams.\nTeaching un
```

**All scorer scores:**

- logic: 3.0 ← WINS
- summarization: 3.0
- factual: 1.5
- math: 1.2
- code_debug: 0.0
- code_gen: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-a527c03256b3` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nThe plans include a new multi-story car park and knocking down the 1937 arrivals building.\nThe building, which has already lost two floors due to saf
```

**All scorer scores:**

- logic: 3.5 ← WINS
- summarization: 3.0
- factual: 2.5
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-b876a715c4ae` → predicted **code_gen**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nThe Mosque and Imams National Advisory Board (MINAB) recognises the problem of abuse in the after school classes.\nAhmed Beg, from the board, said: "W
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-b89148a7c262` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nAfter a five-year approval process, the humble Yakka Skink - a secretive lizard known to hide under rocks and inside hollow logs - and the Ornamental
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- factual: 1.5
- code_debug: 0.0
- code_gen: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-bad825fdec1e` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nLeeds, 15th in the Championship table, looked vulnerable at the start of the season, particularly at set-pieces.\nBut central defender Bartley has for
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.0
- factual: 2.5
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-cc22ef6d798f` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nMedia playback is not supported on this device\nSilverstone has been home to the race every year since 1987.\nHowever, the British Racing Drivers' Club
```

**All scorer scores:**

- math: 6.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-cd41948b8761` → predicted **code_gen**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nIn return Venezuela accused Chile of "inadmissible interventionism" and a "lack of diplomatic circumspection".\nBraulio Jatar was taken into custody o
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-d88794a29080` → predicted **math**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nIn 2015/16, 93.3% of youngsters were in a "positive destination" three months after leaving high school. That was up from 93% in 2014/15.\nThe Scottis
```

**All scorer scores:**

- math: 6.0 ← WINS
- summarization: 3.0
- sentiment: 2.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0

---

### `xsum-e262a350ab5d` → predicted **code_gen**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nThe wager on Murray - the competition's 5/2 second favourite, behind Novak Djokovic - would net a £175,000 return, including the stake.\nLadbrokes bel
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 2.5
- math: 2.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-eb2dcccc9a1d` → predicted **code_gen**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nLaunched last year, the widely unpopular programme saw presenters mingle with a live audience and included funny home-video clips.\nThe BBC said it wi
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-f360ef467de0` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nMSC Oscar is shorter in length than the previous holder the CSCL Globe, which docked in Felixstowe in January, but can carry 124 more containers.\nIt 
```

**All scorer scores:**

- logic: 5.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-f976d74cd4c9` → predicted **logic**
**Source:** `data/eval/training-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nThe 54-year-old has spent seven years overseeing the Premier League side's youth set-up, and has a full Uefa coaching A licence.\n"The time was right 
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-0486495421ef` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\nJohn Carver says he has the hardest job in football as head coach of Newcastle United with thousands of supporters planning to boycott Sunday's 
```

**All scorer scores:**

- logic: 6.0 ← WINS
- summarization: 5.0
- factual: 4.0
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-210df8a60d12` → predicted **code_debug**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\nThe devastated mother of a schoolgirl killed by a speeding driver who was on drugs had ‘dark thoughts’ about committing suicide at the spot wher
```

**All scorer scores:**

- code_debug: 5.0 ← WINS
- code_gen: 3.0
- summarization: 3.0
- factual: 2.5
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-2952aae90884` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\n(CNN)My name is Mark Goodacre, and I am a professor of New Testament and Christian origins in the Department of Religious Studies at Duke University. I was serie
```

**All scorer scores:**

- code_gen: 3.0 ← WINS
- summarization: 3.0
- factual: 2.0
- code_debug: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-2f5dd4d87f13` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nRenowned fashion designer Collette Dinnigan has placed her luxurious Paddington home on the market after splashing out on a stylish waterfront home in one of Syd
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.0
- factual: 2.5
- math: 2.0
- code_gen: 1.5
- code_debug: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-3a5d8699d8e7` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\nA petition has been launched asking for Sunrise's Samantha Armytage to apologise for comments she made on-air last month, dubbed by some viewers
```

**All scorer scores:**

- logic: 6.0 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-4424af9fc635` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nA 21-year-old man with depression has made the most gut-wrenching final goodbye video to his family, telling them not to blame themselves and reassuring them the
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-477da7599d5d` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nOne man has curated a spooky collection of nine old dolls, which which he claims each have their own personalities. Ian Rogers, 36, became hooked on all things p
```

**All scorer scores:**

- logic: 5.5 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-55f1bc64d6f6` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\n(The Hollywood Reporter)Good news arrived Thursday for "Fifty Shades of Grey" fans. Universal announced Thursday that the sequel to the box offi
```

**All scorer scores:**

- logic: 4.0 ← WINS
- code_gen: 3.0
- summarization: 3.0
- factual: 2.5
- math: 2.0
- sentiment: 1.5
- code_debug: 0.0
- ner: 0.0

---

### `cnndm-56d5598910da` → predicted **math**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nMagistrates at Rotherham Magistrates Court took just 15 minutes to find Ms Gaynor not guilty . A mother-of-two was locked in a police cells for six hours and put
```

**All scorer scores:**

- math: 6.0 ← WINS
- summarization: 3.0
- factual: 2.5
- code_gen: 1.5
- code_debug: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-5b0d314d0950` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\nHe's a Chelsea legend after 13 trophy-laden years at the club and it seems that Frank Lampard can't keep away from that part of west London. The
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-5fe1c36afce3` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nA shocking 'resurrection ceremony' for a  two-year-old dead boy at a Texas church has been caught on camera. In the clip capturing the attempted resurrection, th
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.0
- code_gen: 1.5
- factual: 1.5
- code_debug: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-601853000dc2` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nThe morning after the row the night before, Lewis Hamilton and Nico Rosberg barely exchanged a word in the first-class cabin of their Emirates flight to Dubai, e
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- factual: 1.5
- code_debug: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-6156f22cd820` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nA teacher who was suspended for allowing her class to write 'get well soon' letters to a convicted police killer claims the children wanted to send notes to him.
```

**All scorer scores:**

- code_gen: 8.0 ← WINS
- summarization: 3.0
- factual: 2.5
- code_debug: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-632c18406956` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\nThe girlfriend of an Arizona State University football player who has been suspended after she accused him of abuse has now publicly recanted he
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.0
- sentiment: 2.5
- math: 2.0
- code_gen: 1.5
- code_debug: 0.0
- factual: 0.0
- ner: 0.0

---

### `cnndm-668e01d98740` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nSomething odd is going on this year in Goa - and it has to do with the rows of empty sunbeds. With the rouble in trouble, Russians are freezing at home, rather t
```

**All scorer scores:**

- code_gen: 6.5 ← WINS
- summarization: 3.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-6baabb1905ed` → predicted **math**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\n(CNN)Hillary Clinton finally answered the question we've all been asking for years: Will she run for president in 2016? With official news of he
```

**All scorer scores:**

- math: 8.0 ← WINS
- summarization: 3.0
- factual: 1.5
- code_debug: 0.0
- code_gen: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-787ca907aff4` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\nLiverpool are showing interest in Borussia Dortmund striker Ciro Immobile. The Italy international has not had the best of times in the Bundesli
```

**All scorer scores:**

- logic: 3.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-8757e85dca14` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nWashington (CNN)An off-duty member of the Uniformed Division of Secret Service was arrested Friday in Washington and charged with first-degree attempted burglary
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-a38173241200` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nSao Paulo, Brazil (CNN)Brazilian supermodel Gisele Bundchen sashayed down the catwalk at Sao Paulo Fashion Week on Wednesday night in an emotional farewell to th
```

**All scorer scores:**

- logic: 5.5 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-ae68a98a6142` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\nManchester United's thumping 4-2 derby victory over City is the latest vindication for the methods of manager Louis van Gaal. The Dutchman spent
```

**All scorer scores:**

- logic: 5.5 ← WINS
- summarization: 3.0
- factual: 2.5
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-c6990f11336a` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nIt's been a good month for Andy Murray. If tying the knot with childhood sweetheart Kim Sears wasn't enough, Murray has now witnessed two Barcelona victories in 
```

**All scorer scores:**

- code_gen: 3.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-cf6ed92ca7f4` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\nWealth:Sheikh Mohammed bin Rashid Al Maktoum and his wife Princess Haya of Jordan Royal Ascot race meeting, plans to build six-storey car park .
```

**All scorer scores:**

- logic: 8.0 ← WINS
- math: 4.0
- summarization: 3.0
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-d481ac31f42a` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nYaya Toure was accused by Jamie Carragher of ducking out of the way of Jason Puncheon's free-kick which killed off Manchester City's hopes of salvaging something
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-daed4487c907` → predicted **math**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nPatterns appearing on both the very large and very small scale are rare in nature. But researchers have found such a pattern in two apparently unrelated places -
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-e79be9501e28` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nCarlos Tevez talks about being as free as a bird, back to the days when he was banging in the goals in Argentina for Boca Juniors. A big hit on the famous steepe
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-ea530635e35c` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nLabour leader Harold Wilson secretly lobbied the BBC to change the time of popular comedy Steptoe and Son on the night of the 1964 election because he feared wor
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-ead496dcd277` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\n(CNN)For many Girl Scout troops it is officially cookie season. I feel guilty saying no to the sweet, enthusiastic girls standing outside my grocery store who us
```

**All scorer scores:**

- logic: 5.5 ← WINS
- math: 4.0
- summarization: 3.0
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-055e45b6c826` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article in exactly 2 sentences:\n\nUnited will be in the Champions League and boss Jose Mourinho said executive vice-chairman Ed Woodward has had his targets "for more than two m
```

**All scorer scores:**

- code_gen: 6.5 ← WINS
- summarization: 3.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-19da9a3aed23` → predicted **math**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article in exactly 2 sentences:\n\nUnder US law, 40% of the corn harvest must be used to make biofuel, a quota which the UN says could contribute to a food crisis around the worl
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- logic: 2.0
- factual: 1.5
- code_debug: 0.0
- code_gen: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-22ba7c77dcfa` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize as exactly 3 bullet points:\n\nWayne Madsen (70) batted well before the innings stalled in the twilight, as the pink ball swung more.\nMarchant de Lange (3-82) and Timm van der Gugten (3-88) we
```

**All scorer scores:**

- logic: 3.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-3c99323634b2` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize as exactly 3 bullet points:\n\nHamilton is 33 points behind the German with 100 available in the last four races, the first of which is the US Grand Prix in Austin, Texas on Sunday.\nThe Briton
```

**All scorer scores:**

- logic: 7.0 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-3cbdaa0a9a1d` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article in exactly 2 sentences:\n\n20 May 2016 Last updated at 12:40 BST\nThey adopted him and he came into the studio for a live interview with his owner who has written a book a
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 2.5
- math: 0.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-3f06754efad2` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize as exactly 3 bullet points:\n\nSpecial Report: The Technology of Business\nThe workplaces that build Africa's future\nIs teleworking driving us crazy?\nThe tech getting disabled people into work\n
```

**All scorer scores:**

- logic: 3.5 ← WINS
- summarization: 3.0
- code_gen: 1.5
- code_debug: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-41c2e3e5a80c` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article in exactly 2 sentences:\n\nJohn Coyle tried to rob a Scotmid store in Glasgow's Easterhouse, but left empty-handed after the worker shouted to her mother who also worked 
```

**All scorer scores:**

- logic: 3.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-4f20ac8ad2a7` → predicted **math**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article:\n\nMeanwhile, the big four of Djokovic, Federer, Nadal and Murray show no signs yet of retiring, while others such as Wawrinka, Berdych, and Nishikori also provide compe
```

**All scorer scores:**

- math: 6.0 ← WINS
- factual: 4.5
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-77ccd665c98a` → predicted **math**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article:\n\nThe scandal began in May when the payslips of top managers at the state insurance company were leaked to the media, showing they were receiving very generous salaries
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-84d6b7121a1a` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article:\n\nAs part of the Welsh Government plans, councils will need to have a "public toilet strategy".\nRob Poultney of Criccieth, Gwynedd who has Crohn's disease, said it was 
```

**All scorer scores:**

- logic: 6.0 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-8c87fd94e4f3` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article:\n\nUefa's control, ethics and disciplinary body imposed the punishment for "acts of violence against the referee".\nLennon was sent to the stand during last month's Europ
```

**All scorer scores:**

- code_gen: 6.5 ← WINS
- summarization: 3.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-9375adfaa828` → predicted **math**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article:\n\nBolivians were asked to vote on a change to the country's constitution. Presently, the president and vice-president are limited to two consecutive five-year terms.\nMr
```

**All scorer scores:**

- logic: 4.0
- math: 4.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-a431dda38a5a` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article in exactly 2 sentences:\n\nSteven Hirschorn said his son Ollie was upset when he lost his lion, Liley, at Hughenden Manor in Buckinghamshire.\nHowever, the toy was returne
```

**All scorer scores:**

- logic: 7.0 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-b0d11278b56c` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize as exactly 3 bullet points:\n\nMorton are well placed to secure a Championship play-off spot after Tuesday's 1-0 win at Raith Rovers.\nThey were one of only two sides to earn a point at Ibrox i
```

**All scorer scores:**

- logic: 3.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-b52f8c57a8e7` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize as exactly 3 bullet points:\n\nIt represents one of the biggest tie-ups between video gaming and a major sports league.\nEventually all of the 30 NBA teams will have an e-sports division, but i
```

**All scorer scores:**

- code_gen: 9.5 ← WINS
- summarization: 3.0
- logic: 2.0
- code_debug: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-bd16977b7903` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article in exactly 2 sentences:\n\nArthur Cave, 15, was found with multiple injuries on the underpass of Ovingdean Gap in July. He died later at the Royal Sussex County Hospital.
```

**All scorer scores:**

- logic: 4.5 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-c86b1bfacd3c` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article in exactly 2 sentences:\n\nThe club beat Derby County 6-2 in a memorable Championship clash on 30 April 2005.\n"The buzz in the pubs that night was brilliant," says Dave K
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-d5a40eb490c2` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article:\n\nFifa delegates are voting on their new president, but the incumbent Mr Blatter has refused to withdraw from the contest, despite the arrest of senior colleagues on ma
```

**All scorer scores:**

- logic: 6.0 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-e03a90fe67af` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize as exactly 3 bullet points:\n\nThe Japanese company posted a 9.9bn yen ($97m; Â£57m) deficit for the April-to-June months, compared with an 8.6bn yen profit for the same period a year earlier.
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-e45a067fc4e8` → predicted **math**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article:\n\nSamuel Ward scored a goal either side of half-time to set up the victory, before Henry Weir made sure of the result in the final quarter.\nJackson went close to scorin
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 2.5
- logic: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-e898f70696ad` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize as exactly 3 bullet points:\n\nAfter no play was possible on the first day, Stevens led Kent to 389-7 after Essex's decision to bowl first.\nThey had slipped to 208-6 with Essex seamer Matt Dix
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- factual: 2.5
- code_debug: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-e97d51fd9cbd` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article in exactly 2 sentences:\n\nMedia playback is not supported on this device\nEleven months on from their first Scottish Cup triumph in 114 years, the feel-good factor around
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-f0f8e5bb1dde` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize as exactly 3 bullet points:\n\nMedia playback is not supported on this device\nThe Sweden striker, who is out of contract after four years at Paris St-Germain, could follow former manager Jose 
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-f103c06d5ff5` → predicted **code_gen**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article:\n\nHaving resumed on 54, with his team 251-6, the 23-year-old played patiently before reaching his ton from 260 balls and eventually ending on 118 not out.\nAndy Hodd (44
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- logic: 4.0
- summarization: 3.0
- sentiment: 2.5
- factual: 1.5
- code_debug: 0.0
- math: 0.0
- ner: 0.0

---

### `xsum-f5fee34ad9af` → predicted **code_debug**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article in exactly 2 sentences:\n\nThe Dungannon rider suffered chest and pelvic injuries in the crash on 12 May.\nHis condition was described as stable on 16 May and he was moved
```

**All scorer scores:**

- code_debug: 5.0 ← WINS
- code_gen: 3.0
- summarization: 3.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-f8baaf2eec62` → predicted **logic**
**Source:** `data/eval/training-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article:\n\nHuw Irranca-Davies, who has announced he is leaving the shadow frontbench, told Radio Wales his party needed to "consistently re-invigorate" itself.\nHe said the party
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-ef3d0b8ec1b2` → predicted **logic**
**Source:** `data/eval/training-v3.json`
**Prompt (first 200 chars):**
```
Summarize: Thousands of animals, many of them endangered, are part of the count which is required by law as part of the zoo's licence.\nImportant details about each and every individual are noted down 
```

**All scorer scores:**

- logic: 3.0 ← WINS
- summarization: 2.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-4de93c2c77b9` → predicted **code_gen**
**Source:** `data/eval/validation-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nLondon (CNN)It wasn't messrs Clooney, Pitt and their nine accomplices who sailed down an elevator shaft and cracked open dozens of safety deposit boxes at a Lond
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-55eeb56c4ab1` → predicted **code_gen**
**Source:** `data/eval/validation-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nNew York (CNN)Jake Tapper is the next anchor of CNN's Sunday morning political interview program "State of the Union." CNN announced Tapper's promotion on Friday
```

**All scorer scores:**

- code_gen: 3.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-7b8fae9522c8` → predicted **ner**
**Source:** `data/eval/validation-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nFrightened: Diana Doyle was plagued with cold calls . A firm linked to the sale of NHS patient data is offering details of eating disorder sufferers for just 12p
```

**All scorer scores:**

- ner: 4.5 ← WINS
- logic: 3.0
- summarization: 3.0
- code_gen: 1.5
- code_debug: 0.0
- factual: 0.0
- math: 0.0
- sentiment: 0.0

---

### `cnndm-b86be0f8b23a` → predicted **logic**
**Source:** `data/eval/validation-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nNatasha Jonas, the first female boxer to represent Great Britain in an Olympic Games, has announced her retirement from the sport. The Liverpool-born 30-year-old
```

**All scorer scores:**

- logic: 3.5 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-d9871d5c10d2` → predicted **math**
**Source:** `data/eval/validation-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nYouTube is set to offer an ad free, subscription-based service for the first time. The plan was revealed in an email sent out to YouTube Partners. It will offer 
```

**All scorer scores:**

- math: 4.0 ← WINS
- code_gen: 3.0
- summarization: 3.0
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-e81a1adc2985` → predicted **logic**
**Source:** `data/eval/validation-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\n(CNN)South America's Atacama Desert, one of the driest places on Earth, resembles some of the faraway planets monitored by giant telescopes there. The lack of hu
```

**All scorer scores:**

- logic: 7.0 ← WINS
- summarization: 3.5
- factual: 3.0
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-f50c54aa3be2` → predicted **logic**
**Source:** `data/eval/validation-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nFor Chris Brugger the cost of staying alive is $16,000 every three weeks. This is what the New South Wales leukaemia patient must pay for a new drug treatment th
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-53b26c1e9fda` → predicted **math**
**Source:** `data/eval/validation-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nThe ex-Cabinet Minister lost his Yeovil seat in May's general election after 14 years as the constituency member.\nMr Laws, 49, will serve as executiv
```

**All scorer scores:**

- math: 6.0 ← WINS
- summarization: 3.0
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-742a9d0daced` → predicted **logic**
**Source:** `data/eval/validation-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nIf you were sold PPI via your credit card then the differences can amount to thousands of pounds.\nThis is because of the way some credit card provide
```

**All scorer scores:**

- logic: 6.0 ← WINS
- summarization: 3.0
- factual: 2.5
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-f8b4c165bb39` → predicted **math**
**Source:** `data/eval/validation-v1.json`
**Prompt (first 200 chars):**
```
Summarize the following article in 1-2 sentences:\n\nThey want President Dmitry Medvedev to be confronted over a perceived failure to protect business against corruption.\nIn a Sunday Times letter, they 
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-109866f74227` → predicted **math**
**Source:** `data/eval/validation-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nThe general in charge of the U.S. Army Recruiting Command has warned that America’s growing obesity epidemic ‘is becoming a national security issue.’ Major Gener
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-3d003f5a95f4` → predicted **math**
**Source:** `data/eval/validation-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\n(CNN)Most airline pilots have an above average ability to compartmentalize personal problems. The cockpit is our "safe" place. The flight deck i
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-5295ecfbba73` → predicted **logic**
**Source:** `data/eval/validation-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\nTwo best friends who shed an impressive amount of weight and scored top modelling careers prove that with a little determination and self-belief
```

**All scorer scores:**

- logic: 3.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-61d3d723ba41` → predicted **logic**
**Source:** `data/eval/validation-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nCrawling commando-style underneath an open-sided Land Rover, I came virtually nose to nose with a 4,500lb horned rhino. I tried to slow my breathing as she looke
```

**All scorer scores:**

- logic: 5.0 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-7807adea88e6` → predicted **logic**
**Source:** `data/eval/validation-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nA mother who is accused of killing her children and then stuffing their bodies in a freezer was removed from court after confronting the father during a parental
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-82e7a647c228` → predicted **logic**
**Source:** `data/eval/validation-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article:\n\nNot all turtles can swim, said Florida wildlife officials this week after beachgoers tried to throw baby tortoises in the ocean. There were at least three report
```

**All scorer scores:**

- logic: 5.5 ← WINS
- summarization: 3.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-b14e55289a15` → predicted **logic**
**Source:** `data/eval/validation-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\nBournemouth manager Eddie Howe expects the club's fans to play a crucial role as his side close in on a first-ever promotion to the Barclays Pre
```

**All scorer scores:**

- logic: 5.0 ← WINS
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-b3e795dcbf7e` → predicted **logic**
**Source:** `data/eval/validation-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\nA wife has told how she has put her husband on a lifelong sex ban and tracks his every move, after discovering he had cheated on her after takin
```

**All scorer scores:**

- logic: 6.0 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `cnndm-d5e0ca8d1710` → predicted **logic**
**Source:** `data/eval/validation-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following news article in 2-3 sentences:\n\nSimon Wood, 38, from Oldham won Masterchef last Friday . Wannabe chefs with their eye on a career as the next Gordon Ramsay or Jamie Oliver shou
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.0
- factual: 2.5
- math: 2.0
- code_debug: 0.0
- code_gen: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-4397329e0182` → predicted **logic**
**Source:** `data/eval/validation-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article in exactly 2 sentences:\n\nThe former Celtic defender has left his role as a youth coach at Everton to take the job.\nHe replaces Terry Butcher, who was sacked following H
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-745b07016381` → predicted **math**
**Source:** `data/eval/validation-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article:\n\nExports fell 2.1% compared with October, German's Federal Statistical Office reported, while imports rose 1.5%.\nMeanwhile, factory production fell by 0.1% from Octobe
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-b21e5f00195c` → predicted **logic**
**Source:** `data/eval/validation-v2.json`
**Prompt (first 200 chars):**
```
Summarize the following article in exactly 2 sentences:\n\n"The truth is that Taylor Swift and I are together, and we're very happy," the actor told the Hollywood Reporter.\n"That's the truth," he contin
```

**All scorer scores:**

- logic: 4.0 ← WINS
- summarization: 3.5
- factual: 2.5
- code_debug: 0.0
- code_gen: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-cca09e3a4000` → predicted **math**
**Source:** `data/eval/validation-v2.json`
**Prompt (first 200 chars):**
```
Summarize as exactly 3 bullet points:\n\nIt investigated online agents Booking.com, Expedia and hotel group InterContinental Hotels (IHG).\nThe probe initially suspected the deals infringed competition l
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.0
- logic: 2.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `complexity_eval_40.json_q36` → predicted **math**
**Source:** `data/eval/tests/complexity_eval_40.json`
**Prompt (first 200 chars):**
```
On December 5, 2022, scientists at the National Ignition Facility (NIF) at Lawrence Livermore National Laboratory achieved a historic milestone: a fusion reaction that produced more energy than it con
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.5
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `complexity_eval_40.json_q37` → predicted **code_gen**
**Source:** `data/eval/tests/complexity_eval_40.json`
**Prompt (first 200 chars):**
```
LEGAL BRIEF - PLAINTIFFS' ARGUMENT (Artists vs. Stability AI, US District Court for the Northern District of California):\n\nThe plaintiffs, representing a class of visual artists, allege that Stability
```

**All scorer scores:**

- code_gen: 8.0 ← WINS
- summarization: 7.5
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `complexity_eval_candidates.json_q67` → predicted **code_debug**
**Source:** `data/eval/tests/complexity_eval_candidates.json`
**Prompt (first 200 chars):**
```
What was the SS Robert Coryndon?
```

**All scorer scores:**

- code_debug: 0.0 ← WINS
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0
- summarization: 0.0

---

### `complexity_eval_candidates.json_q69` → predicted **code_gen**
**Source:** `data/eval/tests/complexity_eval_candidates.json`
**Prompt (first 200 chars):**
```
Please give me a short bulleted list of the major achievements Taylor Swift has achieved.
```

**All scorer scores:**

- code_gen: 1.5 ← WINS
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0
- summarization: 0.0

---

### `complexity_eval_candidates.json_q72` → predicted **code_gen**
**Source:** `data/eval/tests/complexity_eval_candidates.json`
**Prompt (first 200 chars):**
```
Dundee Sheriff Court heard the 16-year-old boy had attended a party in Dundee last November and left with three friends for another address.\nThe teen, who cannot be named, held the blade to driver Moh
```

**All scorer scores:**

- code_gen: 5.0 ← WINS
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0
- summarization: 0.0

---

### `complexity_eval_candidates.json_q77` → predicted **code_gen**
**Source:** `data/eval/tests/complexity_eval_candidates.json`
**Prompt (first 200 chars):**
```
Nia Griffith said her party offers "investment in Wales, versus cuts from Westminster" by the Tories.\nShe urged people to "imagine the price" if there was a Conservative government in Wales as well.\n"
```

**All scorer scores:**

- code_gen: 1.5 ← WINS
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0
- summarization: 0.0

---

### `complexity_eval_candidates.json_q82` → predicted **math**
**Source:** `data/eval/tests/complexity_eval_candidates.json`
**Prompt (first 200 chars):**
```
The Met Office said the reading had been registered at Heathrow - breaking the previous record set in 2006.\nA level 3 "heatwave action" heat-health alert has been declared for all parts of England.\nBu
```

**All scorer scores:**

- math: 2.0 ← WINS
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0
- summarization: 0.0

---

### `complexity_eval_candidates.json_q84` → predicted **math**
**Source:** `data/eval/tests/complexity_eval_candidates.json`
**Prompt (first 200 chars):**
```
Last year's runner-up will bid to become the first top weight to triumph since Red Rum in 1974.\nA sell-out crowd of 70,000 is expected at Aintree Racecourse on Merseyside.\nVieux Lion Rouge and Definit
```

**All scorer scores:**

- math: 2.0 ← WINS
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0
- summarization: 0.0

---

### `long_summarization_01` → predicted **logic**
**Source:** `data/eval/tests/eval_longform_20.json`
**Prompt (first 200 chars):**
```
Read the following text carefully, then provide a detailed summary that covers ALL major points mentioned. Do NOT shorten or condense — capture every important concept, argument, and conclusion.\n\n"The
```

**All scorer scores:**

- logic: 8.5 ← WINS
- summarization: 5.0
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_v14_test_20.json_q16` → predicted **math**
**Source:** `data/eval/tests/eval_v14_test_20.json`
**Prompt (first 200 chars):**
```
On December 5, 2022, scientists at the National Ignition Facility (NIF) at Lawrence Livermore National Laboratory achieved a historic milestone: a fusion reaction that produced more energy than it con
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 3.5
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `eval_v14_remaining_20.json_q18` → predicted **code_gen**
**Source:** `data/eval/tests/eval_v14_remaining_20.json`
**Prompt (first 200 chars):**
```
LEGAL BRIEF - PLAINTIFFS' ARGUMENT (Artists vs. Stability AI, US District Court for the Northern District of California):\n\nThe plaintiffs, representing a class of visual artists, allege that Stability
```

**All scorer scores:**

- code_gen: 8.0 ← WINS
- summarization: 7.5
- code_debug: 0.0
- factual: 0.0
- logic: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `timeout_long_summary` → predicted **factual**
**Source:** `data/eval/tests/eval_v14_timeout_stress_19.json`
**Prompt (first 200 chars):**
```
Summarize the following text in 2-3 sentences:\n\nThe history of computing is a rich tapestry of innovation that spans centuries. From the abacus to mechanical calculators, from Babbage's Analytical Eng
```

**All scorer scores:**

- factual: 3.5 ← WINS
- code_gen: 3.0
- summarization: 3.0
- math: 2.0
- code_debug: 0.0
- logic: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `fireworks_eval_20.json_q14` → predicted **math**
**Source:** `data/eval/tests/fireworks_eval_20.json`
**Prompt (first 200 chars):**
```
A new study published in Nature Communications Earth and Environment has analyzed 40 years of satellite data to quantify Arctic sea ice loss. The study found that September sea ice extent (the annual 
```

**All scorer scores:**

- math: 4.0 ← WINS
- summarization: 4.0
- logic: 1.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- ner: 0.0
- sentiment: 0.0

---

### `xsum-ef3d0b8ec1b2` → predicted **logic**
**Source:** `data/eval/summarization_combined_25.json`
**Prompt (first 200 chars):**
```
Summarize: Thousands of animals, many of them endangered, are part of the count which is required by law as part of the zoo's licence.\nImportant details about each and every individual are noted down 
```

**All scorer scores:**

- logic: 3.0 ← WINS
- summarization: 2.5
- code_debug: 0.0
- code_gen: 0.0
- factual: 0.0
- math: 0.0
- ner: 0.0
- sentiment: 0.0

---

## Correctly Classified Items

(487 items correctly classified as summarization)

- `q_88` (from `input/cx_300.json`)
- `q_89` (from `input/cx_300.json`)
- `q_92` (from `input/cx_300.json`)
- `q_93` (from `input/cx_300.json`)
- `q_95` (from `input/cx_300.json`)
- `q_96` (from `input/cx_300.json`)
- `q_98` (from `input/cx_300.json`)
- `q_99` (from `input/cx_300.json`)
- `q_188` (from `input/cx_300.json`)
- `q_189` (from `input/cx_300.json`)
- `q_192` (from `input/cx_300.json`)
- `q_193` (from `input/cx_300.json`)
- `q_195` (from `input/cx_300.json`)
- `q_196` (from `input/cx_300.json`)
- `q_198` (from `input/cx_300.json`)
- `q_199` (from `input/cx_300.json`)
- `cx_14` (from `input/complexity_40.json`)
- `cx_15` (from `input/complexity_40.json`)
- `cx_24` (from `input/complexity_40.json`)
- `cx_33` (from `input/complexity_40.json`)
- `eval_hard_218.json_q88` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q89` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q92` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q93` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q95` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q96` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q98` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q99` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q188` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q189` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q192` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q193` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q195` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q196` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q198` (from `data/eval/primary/eval_hard_218.json`)
- `eval_hard_218.json_q199` (from `data/eval/primary/eval_hard_218.json`)
- `eval_60_medium_hard.json_q53` (from `data/eval/primary/eval_60_medium_hard.json`)
- `eval_60_medium_hard.json_q54` (from `data/eval/primary/eval_60_medium_hard.json`)
- `eval_60_medium_hard.json_q55` (from `data/eval/primary/eval_60_medium_hard.json`)
- `eval_60_medium_hard.json_q58` (from `data/eval/primary/eval_60_medium_hard.json`)
- `eval_60_medium_hard.json_q59` (from `data/eval/primary/eval_60_medium_hard.json`)
- `eval_clean_val.json_q6626` (from `data/eval/primary/eval_clean_val.json`)
- `eval_clean_val.json_q7663` (from `data/eval/primary/eval_clean_val.json`)
- `cnndm-01a52f33c5a1` (from `data/eval/training-v1.json`)
- `cnndm-07a9f47471a3` (from `data/eval/training-v1.json`)
- `cnndm-0a97c4261532` (from `data/eval/training-v1.json`)
- `cnndm-0eeaaf84dd09` (from `data/eval/training-v1.json`)
- `cnndm-137520e6643d` (from `data/eval/training-v1.json`)
- `cnndm-1662dce610f5` (from `data/eval/training-v1.json`)
- `cnndm-1732775c8deb` (from `data/eval/training-v1.json`)
- `cnndm-1bab7919b4a9` (from `data/eval/training-v1.json`)
- `cnndm-1bdffef168e4` (from `data/eval/training-v1.json`)
- `cnndm-243f465ba33e` (from `data/eval/training-v1.json`)
- `cnndm-2689694e796f` (from `data/eval/training-v1.json`)
- `cnndm-29e7c06d6627` (from `data/eval/training-v1.json`)
- `cnndm-2d82eaa24541` (from `data/eval/training-v1.json`)
- `cnndm-32aedf1a39aa` (from `data/eval/training-v1.json`)
- `cnndm-36321d06591d` (from `data/eval/training-v1.json`)
- `cnndm-3cba4a9f438d` (from `data/eval/training-v1.json`)
- `cnndm-3fb1378312fe` (from `data/eval/training-v1.json`)
- `cnndm-3fc26e671734` (from `data/eval/training-v1.json`)
- `cnndm-44906c4ca0d3` (from `data/eval/training-v1.json`)
- `cnndm-457e29224492` (from `data/eval/training-v1.json`)
- `cnndm-47b737befd74` (from `data/eval/training-v1.json`)
- `cnndm-4abab25deb40` (from `data/eval/training-v1.json`)
- `cnndm-4f5944c5d432` (from `data/eval/training-v1.json`)
- `cnndm-50c7894b484d` (from `data/eval/training-v1.json`)
- `cnndm-56eeea00f48c` (from `data/eval/training-v1.json`)
- `cnndm-58ff1d69eadf` (from `data/eval/training-v1.json`)
- `cnndm-621c364a98ea` (from `data/eval/training-v1.json`)
- `cnndm-65502fd26815` (from `data/eval/training-v1.json`)
- `cnndm-675734f8d953` (from `data/eval/training-v1.json`)
- `cnndm-6798b0ff9094` (from `data/eval/training-v1.json`)
- `cnndm-6f14ac98a968` (from `data/eval/training-v1.json`)
- `cnndm-6f9b0359217e` (from `data/eval/training-v1.json`)
- `cnndm-71409f2f8da6` (from `data/eval/training-v1.json`)
- `cnndm-732f990d8ae4` (from `data/eval/training-v1.json`)
- `cnndm-781d55df6256` (from `data/eval/training-v1.json`)
- `cnndm-7a45e6951fb4` (from `data/eval/training-v1.json`)
- `cnndm-7c4ef733bbb9` (from `data/eval/training-v1.json`)
- `cnndm-8041d676993e` (from `data/eval/training-v1.json`)
- `cnndm-8074ac5744d4` (from `data/eval/training-v1.json`)
- `cnndm-83c1e4f62158` (from `data/eval/training-v1.json`)
- `cnndm-85db81fd3b97` (from `data/eval/training-v1.json`)
- `cnndm-86ae034adace` (from `data/eval/training-v1.json`)
- `cnndm-884650548e65` (from `data/eval/training-v1.json`)
- `cnndm-8c5e3648ef62` (from `data/eval/training-v1.json`)
- `cnndm-9097d8d445e5` (from `data/eval/training-v1.json`)
- `cnndm-92cc1c9f4eff` (from `data/eval/training-v1.json`)
- `cnndm-94fef269abf3` (from `data/eval/training-v1.json`)
- `cnndm-996c8beae5b4` (from `data/eval/training-v1.json`)
- `cnndm-9bc7c362f729` (from `data/eval/training-v1.json`)
- `cnndm-9c0fc8005f9f` (from `data/eval/training-v1.json`)
- `cnndm-9c3be143c79b` (from `data/eval/training-v1.json`)
- `cnndm-9dbde297f5e5` (from `data/eval/training-v1.json`)
- `cnndm-a43d51f3554c` (from `data/eval/training-v1.json`)
- `cnndm-a744b37c64e2` (from `data/eval/training-v1.json`)
- `cnndm-ad81c01b5896` (from `data/eval/training-v1.json`)
- `cnndm-aeccda7b9b58` (from `data/eval/training-v1.json`)
- `cnndm-b18c7c9052da` (from `data/eval/training-v1.json`)
- `cnndm-b54798bebfc5` (from `data/eval/training-v1.json`)
- `cnndm-bbc55e82a885` (from `data/eval/training-v1.json`)
- `cnndm-c51a8a3f2a47` (from `data/eval/training-v1.json`)
- `cnndm-c8fe307f00dc` (from `data/eval/training-v1.json`)
- `cnndm-c9d4986bc65d` (from `data/eval/training-v1.json`)
- `cnndm-ccc40cf79d0e` (from `data/eval/training-v1.json`)
- `cnndm-cd4a8501d83d` (from `data/eval/training-v1.json`)
- `cnndm-cf56561c3be5` (from `data/eval/training-v1.json`)
- `cnndm-cfcd15b18c8b` (from `data/eval/training-v1.json`)
- `cnndm-d07c2a0bffbc` (from `data/eval/training-v1.json`)
- `cnndm-da60d3239277` (from `data/eval/training-v1.json`)
- `cnndm-dc2612ad80a9` (from `data/eval/training-v1.json`)
- `cnndm-deed2a362b48` (from `data/eval/training-v1.json`)
- `cnndm-e080575ed225` (from `data/eval/training-v1.json`)
- `cnndm-e2959d3508d9` (from `data/eval/training-v1.json`)
- `cnndm-e9cde9512ac5` (from `data/eval/training-v1.json`)
- `cnndm-ea99e59c399f` (from `data/eval/training-v1.json`)
- `cnndm-f297cf31cba4` (from `data/eval/training-v1.json`)
- `cnndm-f514ff79e7a5` (from `data/eval/training-v1.json`)
- `cnndm-f85bec195a88` (from `data/eval/training-v1.json`)
- `cnndm-fb4b0f5285f0` (from `data/eval/training-v1.json`)
- `cnndm-fe05fe7dfb98` (from `data/eval/training-v1.json`)
- `xsum-05f4c8cd318e` (from `data/eval/training-v1.json`)
- `xsum-090a5c5611a7` (from `data/eval/training-v1.json`)
- `xsum-1eb7caa8fc95` (from `data/eval/training-v1.json`)
- `xsum-287fe343c74a` (from `data/eval/training-v1.json`)
- `xsum-32a58d114eda` (from `data/eval/training-v1.json`)
- `xsum-338c57779bd0` (from `data/eval/training-v1.json`)
- `xsum-367cd49334b8` (from `data/eval/training-v1.json`)
- `xsum-38e2a9a5bd1a` (from `data/eval/training-v1.json`)
- `xsum-39e561b7449d` (from `data/eval/training-v1.json`)
- `xsum-3bf1418452d5` (from `data/eval/training-v1.json`)
- `xsum-3cad679f321e` (from `data/eval/training-v1.json`)
- `xsum-40c8925df884` (from `data/eval/training-v1.json`)
- `xsum-42be0aa28682` (from `data/eval/training-v1.json`)
- `xsum-432d6beb3c7d` (from `data/eval/training-v1.json`)
- `xsum-4443e749029d` (from `data/eval/training-v1.json`)
- `xsum-48fc85d3e968` (from `data/eval/training-v1.json`)
- `xsum-4d63fc6c27f4` (from `data/eval/training-v1.json`)
- `xsum-4f53185f0128` (from `data/eval/training-v1.json`)
- `xsum-5417f691dd1c` (from `data/eval/training-v1.json`)
- `xsum-59a35df0d3f4` (from `data/eval/training-v1.json`)
- `xsum-5a2df6a52e5e` (from `data/eval/training-v1.json`)
- `xsum-5b84b9cc3907` (from `data/eval/training-v1.json`)
- `xsum-643b072b4c2b` (from `data/eval/training-v1.json`)
- `xsum-6be69b1951ec` (from `data/eval/training-v1.json`)
- `xsum-6c3813356c6f` (from `data/eval/training-v1.json`)
- `xsum-709fa4fcbc72` (from `data/eval/training-v1.json`)
- `xsum-71af96c010be` (from `data/eval/training-v1.json`)
- `xsum-76625fef2769` (from `data/eval/training-v1.json`)
- `xsum-7ccf017242d5` (from `data/eval/training-v1.json`)
- `xsum-7d3641479a32` (from `data/eval/training-v1.json`)
- `xsum-835348f5bd6d` (from `data/eval/training-v1.json`)
- `xsum-878c6439c93d` (from `data/eval/training-v1.json`)
- `xsum-881b61bdf2a3` (from `data/eval/training-v1.json`)
- `xsum-8aed84c42b81` (from `data/eval/training-v1.json`)
- `xsum-8feb5c5e87d3` (from `data/eval/training-v1.json`)
- `xsum-905ffc1b88fb` (from `data/eval/training-v1.json`)
- `xsum-929921a566e0` (from `data/eval/training-v1.json`)
- `xsum-96154246e8dc` (from `data/eval/training-v1.json`)
- `xsum-963089700fd2` (from `data/eval/training-v1.json`)
- `xsum-9789f1472fba` (from `data/eval/training-v1.json`)
- `xsum-983059bda99b` (from `data/eval/training-v1.json`)
- `xsum-9aed27aaa63e` (from `data/eval/training-v1.json`)
- `xsum-9f6a4f298682` (from `data/eval/training-v1.json`)
- `xsum-a597cc7e7510` (from `data/eval/training-v1.json`)
- `xsum-a96df15d3e1e` (from `data/eval/training-v1.json`)
- `xsum-ac9436672959` (from `data/eval/training-v1.json`)
- `xsum-aedadce1f20a` (from `data/eval/training-v1.json`)
- `xsum-b1aa7e08e06a` (from `data/eval/training-v1.json`)
- `xsum-b4c42de3b234` (from `data/eval/training-v1.json`)
- `xsum-bada88ee49eb` (from `data/eval/training-v1.json`)
- `xsum-bc93116c4992` (from `data/eval/training-v1.json`)
- `xsum-bf1f0b94d353` (from `data/eval/training-v1.json`)
- `xsum-c3c52270bbd3` (from `data/eval/training-v1.json`)
- `xsum-c41d9d2a833f` (from `data/eval/training-v1.json`)
- `xsum-c4fd42cf73da` (from `data/eval/training-v1.json`)
- `xsum-c859a84a02cd` (from `data/eval/training-v1.json`)
- `xsum-cb8eb80a858c` (from `data/eval/training-v1.json`)
- `xsum-ce98d1f027cf` (from `data/eval/training-v1.json`)
- `xsum-d0fee044f5aa` (from `data/eval/training-v1.json`)
- `xsum-d2d1ae40cab1` (from `data/eval/training-v1.json`)
- `xsum-d73567fa5191` (from `data/eval/training-v1.json`)
- `xsum-e255583cf9f7` (from `data/eval/training-v1.json`)
- `xsum-e32feac48537` (from `data/eval/training-v1.json`)
- `xsum-ebe35ba3f136` (from `data/eval/training-v1.json`)
- `xsum-ec80f3f1109e` (from `data/eval/training-v1.json`)
- `xsum-f1f313b99391` (from `data/eval/training-v1.json`)
- `xsum-f2f6967f2a66` (from `data/eval/training-v1.json`)
- `xsum-fabcf7628ec6` (from `data/eval/training-v1.json`)
- `cnndm-064448831a14` (from `data/eval/training-v2.json`)
- `cnndm-142e760c8c7b` (from `data/eval/training-v2.json`)
- `cnndm-1729c3c05b52` (from `data/eval/training-v2.json`)
- `cnndm-1a791ae83b69` (from `data/eval/training-v2.json`)
- `cnndm-1bbf8af16d53` (from `data/eval/training-v2.json`)
- `cnndm-1bfe0c0858ab` (from `data/eval/training-v2.json`)
- `cnndm-2121021756de` (from `data/eval/training-v2.json`)
- `cnndm-2177d78e2b84` (from `data/eval/training-v2.json`)
- `cnndm-23ac422e1b05` (from `data/eval/training-v2.json`)
- `cnndm-25ffd8ffed46` (from `data/eval/training-v2.json`)
- `cnndm-288bd109baba` (from `data/eval/training-v2.json`)
- `cnndm-2fc1dd7ef748` (from `data/eval/training-v2.json`)
- `cnndm-338c9d04ec37` (from `data/eval/training-v2.json`)
- `cnndm-35e565193a36` (from `data/eval/training-v2.json`)
- `cnndm-39594f30c292` (from `data/eval/training-v2.json`)
- `cnndm-3ebaed55ad88` (from `data/eval/training-v2.json`)
- `cnndm-3edbaac5b4cf` (from `data/eval/training-v2.json`)
- `cnndm-4138d4e97dd7` (from `data/eval/training-v2.json`)
- `cnndm-435963d90248` (from `data/eval/training-v2.json`)
- `cnndm-49985c79453f` (from `data/eval/training-v2.json`)
- `cnndm-4c673f22bc9d` (from `data/eval/training-v2.json`)
- `cnndm-4cdab3e87818` (from `data/eval/training-v2.json`)
- `cnndm-50c7894b484d` (from `data/eval/training-v2.json`)
- `cnndm-512822ace2b9` (from `data/eval/training-v2.json`)
- `cnndm-56ba1bce53bf` (from `data/eval/training-v2.json`)
- `cnndm-58740f4fdff3` (from `data/eval/training-v2.json`)
- `cnndm-5874b2fba8bd` (from `data/eval/training-v2.json`)
- `cnndm-59a7756a0d9b` (from `data/eval/training-v2.json`)
- `cnndm-5b91da6e2f76` (from `data/eval/training-v2.json`)
- `cnndm-5dff24eeae98` (from `data/eval/training-v2.json`)
- `cnndm-5e82270bf283` (from `data/eval/training-v2.json`)
- `cnndm-6243614c943d` (from `data/eval/training-v2.json`)
- `cnndm-6545c8d178cd` (from `data/eval/training-v2.json`)
- `cnndm-6b274f8b49c5` (from `data/eval/training-v2.json`)
- `cnndm-7776ea328bb2` (from `data/eval/training-v2.json`)
- `cnndm-7c3676bd6494` (from `data/eval/training-v2.json`)
- `cnndm-7ced5c633120` (from `data/eval/training-v2.json`)
- `cnndm-8105d410e547` (from `data/eval/training-v2.json`)
- `cnndm-84e65a8d15af` (from `data/eval/training-v2.json`)
- `cnndm-8b80bcb62c0f` (from `data/eval/training-v2.json`)
- `cnndm-8be8b0cfd9c0` (from `data/eval/training-v2.json`)
- `cnndm-8f9845ca8766` (from `data/eval/training-v2.json`)
- `cnndm-91f49d4d1d44` (from `data/eval/training-v2.json`)
- `cnndm-92cd318a26db` (from `data/eval/training-v2.json`)
- `cnndm-955efd974493` (from `data/eval/training-v2.json`)
- `cnndm-9cb14122989c` (from `data/eval/training-v2.json`)
- `cnndm-9f3e9ce4285b` (from `data/eval/training-v2.json`)
- `cnndm-9f7f88dd3428` (from `data/eval/training-v2.json`)
- `cnndm-a477c6e94bdc` (from `data/eval/training-v2.json`)
- `cnndm-a892d147261b` (from `data/eval/training-v2.json`)
- `cnndm-aad5d433f4aa` (from `data/eval/training-v2.json`)
- `cnndm-b040d416390b` (from `data/eval/training-v2.json`)
- `cnndm-b05e54033468` (from `data/eval/training-v2.json`)
- `cnndm-b2e279a3dc7f` (from `data/eval/training-v2.json`)
- `cnndm-b37b6fe53dd0` (from `data/eval/training-v2.json`)
- `cnndm-b470540063b8` (from `data/eval/training-v2.json`)
- `cnndm-bd542f72f37b` (from `data/eval/training-v2.json`)
- `cnndm-c328f47ebe0c` (from `data/eval/training-v2.json`)
- `cnndm-c58b3999488a` (from `data/eval/training-v2.json`)
- `cnndm-cc3524c544a3` (from `data/eval/training-v2.json`)
- `cnndm-ce3e9a9e2604` (from `data/eval/training-v2.json`)
- `cnndm-d5745b5b0a6f` (from `data/eval/training-v2.json`)
- `cnndm-d930a556c88f` (from `data/eval/training-v2.json`)
- `cnndm-d96cbfad5b39` (from `data/eval/training-v2.json`)
- `cnndm-ddf4c06034e7` (from `data/eval/training-v2.json`)
- `cnndm-e36f868273ee` (from `data/eval/training-v2.json`)
- `cnndm-e5a77edb7048` (from `data/eval/training-v2.json`)
- `cnndm-eba3593771a9` (from `data/eval/training-v2.json`)
- `cnndm-ed760108e362` (from `data/eval/training-v2.json`)
- `cnndm-ed8ca120f895` (from `data/eval/training-v2.json`)
- `cnndm-eee7095c40e5` (from `data/eval/training-v2.json`)
- `cnndm-f5be2b32204e` (from `data/eval/training-v2.json`)
- `cnndm-f7c0a9fb9e77` (from `data/eval/training-v2.json`)
- `xsum-0aaf49256d88` (from `data/eval/training-v2.json`)
- `xsum-0ec4694d4043` (from `data/eval/training-v2.json`)
- `xsum-15ef329e33b6` (from `data/eval/training-v2.json`)
- `xsum-1b0194c32b0a` (from `data/eval/training-v2.json`)
- `xsum-1cc628c9cd15` (from `data/eval/training-v2.json`)
- `xsum-214c4088fc78` (from `data/eval/training-v2.json`)
- `xsum-22269d2c8abf` (from `data/eval/training-v2.json`)
- `xsum-26500b209925` (from `data/eval/training-v2.json`)
- `xsum-27e924a375c9` (from `data/eval/training-v2.json`)
- `xsum-29e307748d59` (from `data/eval/training-v2.json`)
- `xsum-2ab1d21ef36c` (from `data/eval/training-v2.json`)
- `xsum-2bdf52dceb22` (from `data/eval/training-v2.json`)
- `xsum-340ba8e77862` (from `data/eval/training-v2.json`)
- `xsum-34d1f2974f5a` (from `data/eval/training-v2.json`)
- `xsum-350b049881cb` (from `data/eval/training-v2.json`)
- `xsum-36fbe932d862` (from `data/eval/training-v2.json`)
- `xsum-3bdd32086459` (from `data/eval/training-v2.json`)
- `xsum-3bf1418452d5` (from `data/eval/training-v2.json`)
- `xsum-3ec75e8c5611` (from `data/eval/training-v2.json`)
- `xsum-3f05222a0381` (from `data/eval/training-v2.json`)
- `xsum-44a8051ddc46` (from `data/eval/training-v2.json`)
- `xsum-45f2aaaa6795` (from `data/eval/training-v2.json`)
- `xsum-460acd0e45a8` (from `data/eval/training-v2.json`)
- `xsum-50b9a54ccc2a` (from `data/eval/training-v2.json`)
- `xsum-57e173a653ac` (from `data/eval/training-v2.json`)
- `xsum-634d829f8b05` (from `data/eval/training-v2.json`)
- `xsum-66ed8c6e4cff` (from `data/eval/training-v2.json`)
- `xsum-68cd1e7b4557` (from `data/eval/training-v2.json`)
- `xsum-6e3d27fad540` (from `data/eval/training-v2.json`)
- `xsum-6e5d466382b9` (from `data/eval/training-v2.json`)
- `xsum-70c494f19653` (from `data/eval/training-v2.json`)
- `xsum-730db085a2d8` (from `data/eval/training-v2.json`)
- `xsum-77cee7407129` (from `data/eval/training-v2.json`)
- `xsum-7f9f8e8fad3e` (from `data/eval/training-v2.json`)
- `xsum-7fa3baa94974` (from `data/eval/training-v2.json`)
- `xsum-811eaefbe04c` (from `data/eval/training-v2.json`)
- `xsum-835d88fc1b67` (from `data/eval/training-v2.json`)
- `xsum-883fdd96bcf4` (from `data/eval/training-v2.json`)
- `xsum-88f7c855f76e` (from `data/eval/training-v2.json`)
- `xsum-9568d7174160` (from `data/eval/training-v2.json`)
- `xsum-a0505dc40ffb` (from `data/eval/training-v2.json`)
- `xsum-a42b6c448c2d` (from `data/eval/training-v2.json`)
- `xsum-a59a3b01aeb9` (from `data/eval/training-v2.json`)
- `xsum-a5cbaf2638ab` (from `data/eval/training-v2.json`)
- `xsum-ae82a375cba4` (from `data/eval/training-v2.json`)
- `xsum-b2558e6d6aba` (from `data/eval/training-v2.json`)
- `xsum-b37ebd524fec` (from `data/eval/training-v2.json`)
- `xsum-b3aa25ea4548` (from `data/eval/training-v2.json`)
- `xsum-b3cdf48dfc2f` (from `data/eval/training-v2.json`)
- `xsum-bc0ecfa7bda8` (from `data/eval/training-v2.json`)
- `xsum-bca5e6037c5b` (from `data/eval/training-v2.json`)
- `xsum-be30e07a0704` (from `data/eval/training-v2.json`)
- `xsum-ca4058c78829` (from `data/eval/training-v2.json`)
- `xsum-caad0d0f98b0` (from `data/eval/training-v2.json`)
- `xsum-cb0ccce3fbd5` (from `data/eval/training-v2.json`)
- `xsum-cb7d771158e6` (from `data/eval/training-v2.json`)
- `xsum-cffd4166adce` (from `data/eval/training-v2.json`)
- `xsum-d1ceb43d74c5` (from `data/eval/training-v2.json`)
- `xsum-d3f3bc6aa4fe` (from `data/eval/training-v2.json`)
- `xsum-d9e9542de0ef` (from `data/eval/training-v2.json`)
- `xsum-dc8b27b24145` (from `data/eval/training-v2.json`)
- `xsum-dd47aedbfc9c` (from `data/eval/training-v2.json`)
- `xsum-df7a6c183997` (from `data/eval/training-v2.json`)
- `xsum-e135f0f25766` (from `data/eval/training-v2.json`)
- `xsum-e1a81f585193` (from `data/eval/training-v2.json`)
- `xsum-e5287a847b61` (from `data/eval/training-v2.json`)
- `xsum-e65a4a37544b` (from `data/eval/training-v2.json`)
- `xsum-e75319a5e2b8` (from `data/eval/training-v2.json`)
- `xsum-e924fb2cc876` (from `data/eval/training-v2.json`)
- `xsum-eace8cf73116` (from `data/eval/training-v2.json`)
- `xsum-ecd163fb37a4` (from `data/eval/training-v2.json`)
- `xsum-f5d1afdc7be8` (from `data/eval/training-v2.json`)
- `xsum-f82307e93de2` (from `data/eval/training-v2.json`)
- `xsum-fdcfc6350e5d` (from `data/eval/training-v2.json`)
- `xsum-1f0e69fa8897` (from `data/eval/training-v3.json`)
- `xsum-260eec64733b` (from `data/eval/training-v3.json`)
- `xsum-4c2076d75e1c` (from `data/eval/training-v3.json`)
- `xsum-5ead3b4e019b` (from `data/eval/training-v3.json`)
- `xsum-60f2ba73e141` (from `data/eval/training-v3.json`)
- `xsum-726724dafa9e` (from `data/eval/training-v3.json`)
- `xsum-75d704a754da` (from `data/eval/training-v3.json`)
- `xsum-7756c474bf25` (from `data/eval/training-v3.json`)
- `xsum-79d08afd57e1` (from `data/eval/training-v3.json`)
- `xsum-83e2b8250a8c` (from `data/eval/training-v3.json`)
- `xsum-862f366208a4` (from `data/eval/training-v3.json`)
- `xsum-975d27e940d2` (from `data/eval/training-v3.json`)
- `xsum-9f77a14788c7` (from `data/eval/training-v3.json`)
- `xsum-bb6aee03a3db` (from `data/eval/training-v3.json`)
- `xsum-bd87969dec13` (from `data/eval/training-v3.json`)
- `xsum-c181680bb7bc` (from `data/eval/training-v3.json`)
- `xsum-dccce4944b71` (from `data/eval/training-v3.json`)
- `xsum-f0baddd56cc9` (from `data/eval/training-v3.json`)
- `cnndm-04737f04b93d` (from `data/eval/validation-v1.json`)
- `cnndm-05379d905b22` (from `data/eval/validation-v1.json`)
- `cnndm-0971726f34de` (from `data/eval/validation-v1.json`)
- `cnndm-0f3a4d4aff0b` (from `data/eval/validation-v1.json`)
- `cnndm-3225fefc9c19` (from `data/eval/validation-v1.json`)
- `cnndm-46f911e4afb9` (from `data/eval/validation-v1.json`)
- `cnndm-68f236895ad4` (from `data/eval/validation-v1.json`)
- `cnndm-70f86ed09115` (from `data/eval/validation-v1.json`)
- `cnndm-7ef83b211f30` (from `data/eval/validation-v1.json`)
- `cnndm-8c811f9cb9f9` (from `data/eval/validation-v1.json`)
- `cnndm-8d3e48bbc093` (from `data/eval/validation-v1.json`)
- `cnndm-aa82d04c1915` (from `data/eval/validation-v1.json`)
- `cnndm-b37b6fe53dd0` (from `data/eval/validation-v1.json`)
- `cnndm-bdf664de935e` (from `data/eval/validation-v1.json`)
- `cnndm-c230e33dbef8` (from `data/eval/validation-v1.json`)
- `cnndm-c935518e86c7` (from `data/eval/validation-v1.json`)
- `cnndm-c94b67afc062` (from `data/eval/validation-v1.json`)
- `cnndm-d15c4a8a1082` (from `data/eval/validation-v1.json`)
- `cnndm-e1249289f99e` (from `data/eval/validation-v1.json`)
- `cnndm-e2a25afb6b9c` (from `data/eval/validation-v1.json`)
- `cnndm-e3564498678c` (from `data/eval/validation-v1.json`)
- `cnndm-f05ad320f46f` (from `data/eval/validation-v1.json`)
- `xsum-1d9a4fc9e97d` (from `data/eval/validation-v1.json`)
- `xsum-2b0af516adb6` (from `data/eval/validation-v1.json`)
- `xsum-37c7e3e166b7` (from `data/eval/validation-v1.json`)
- `xsum-4974dc6f1fd7` (from `data/eval/validation-v1.json`)
- `xsum-71986a3f2aaf` (from `data/eval/validation-v1.json`)
- `xsum-80b6185f7958` (from `data/eval/validation-v1.json`)
- `xsum-8c58a038b3f9` (from `data/eval/validation-v1.json`)
- `xsum-94223451fd61` (from `data/eval/validation-v1.json`)
- `xsum-997dbf83210c` (from `data/eval/validation-v1.json`)
- `xsum-a1160196759d` (from `data/eval/validation-v1.json`)
- `xsum-a1c1a380ceb5` (from `data/eval/validation-v1.json`)
- `xsum-b9dafa7874b3` (from `data/eval/validation-v1.json`)
- `xsum-c8c780b98469` (from `data/eval/validation-v1.json`)
- `xsum-cf1f111650be` (from `data/eval/validation-v1.json`)
- `xsum-d2dc1bd62033` (from `data/eval/validation-v1.json`)
- `xsum-da3e79668487` (from `data/eval/validation-v1.json`)
- `xsum-dda757a4451b` (from `data/eval/validation-v1.json`)
- `xsum-ed20dcd4bcd7` (from `data/eval/validation-v1.json`)
- `cnndm-051687a091be` (from `data/eval/validation-v2.json`)
- `cnndm-0745d0c8d7a8` (from `data/eval/validation-v2.json`)
- `cnndm-15fbc244d52c` (from `data/eval/validation-v2.json`)
- `cnndm-4203e5decb7e` (from `data/eval/validation-v2.json`)
- `cnndm-5afae018c4e6` (from `data/eval/validation-v2.json`)
- `cnndm-64ee6d21aba6` (from `data/eval/validation-v2.json`)
- `cnndm-6606756f9f5c` (from `data/eval/validation-v2.json`)
- `cnndm-6a44e2b116b6` (from `data/eval/validation-v2.json`)
- `cnndm-863367e031d1` (from `data/eval/validation-v2.json`)
- `cnndm-8702529bed56` (from `data/eval/validation-v2.json`)
- `cnndm-b50368a1eae7` (from `data/eval/validation-v2.json`)
- `cnndm-c2932b02862e` (from `data/eval/validation-v2.json`)
- `cnndm-c6dfd45f9d8d` (from `data/eval/validation-v2.json`)
- `cnndm-ce9d53ed74ba` (from `data/eval/validation-v2.json`)
- `cnndm-e74b69c37547` (from `data/eval/validation-v2.json`)
- `cnndm-eb80e2bf62d4` (from `data/eval/validation-v2.json`)
- `xsum-0d99e6a36aea` (from `data/eval/validation-v2.json`)
- `xsum-270fa1a896dc` (from `data/eval/validation-v2.json`)
- `xsum-39e7a66aff4b` (from `data/eval/validation-v2.json`)
- `xsum-3b91aa95e7c1` (from `data/eval/validation-v2.json`)
- `xsum-3d2a1edfc4d9` (from `data/eval/validation-v2.json`)
- `xsum-435fe7b7f9af` (from `data/eval/validation-v2.json`)
- `xsum-4b1185208ec8` (from `data/eval/validation-v2.json`)
- `xsum-4c68aeb63dc2` (from `data/eval/validation-v2.json`)
- `xsum-4d43a9069599` (from `data/eval/validation-v2.json`)
- `xsum-508b4c8d0fa5` (from `data/eval/validation-v2.json`)
- `xsum-5bfff451c02f` (from `data/eval/validation-v2.json`)
- `xsum-635dbf467fa7` (from `data/eval/validation-v2.json`)
- `xsum-7b18ff3f271b` (from `data/eval/validation-v2.json`)
- `xsum-94ea8df3875a` (from `data/eval/validation-v2.json`)
- `xsum-9bdbd2bec934` (from `data/eval/validation-v2.json`)
- `xsum-a63fd5f227c6` (from `data/eval/validation-v2.json`)
- `xsum-db2cb4f797f2` (from `data/eval/validation-v2.json`)
- `xsum-edd6e2c51517` (from `data/eval/validation-v2.json`)
- `xsum-f317b569351b` (from `data/eval/validation-v2.json`)
- `xsum-f466d642faa5` (from `data/eval/validation-v2.json`)
- `xsum-ff69ad5de4ae` (from `data/eval/validation-v2.json`)
- `xsum-00ac9888e2f5` (from `data/eval/validation-v3.json`)
- `xsum-11a19a39f4d0` (from `data/eval/validation-v3.json`)
- `xsum-2054f538d8ea` (from `data/eval/validation-v3.json`)
- `xsum-407b80861d2f` (from `data/eval/validation-v3.json`)
- `xsum-c5fc1996f68d` (from `data/eval/validation-v3.json`)
- `xsum-d824a5cab529` (from `data/eval/validation-v3.json`)
- `complexity_eval_40.json_q35` (from `data/eval/tests/complexity_eval_40.json`)
- `complexity_eval_40.json_q38` (from `data/eval/tests/complexity_eval_40.json`)
- `complexity_eval_40.json_q39` (from `data/eval/tests/complexity_eval_40.json`)
- `complexity_eval_candidates.json_q68` (from `data/eval/tests/complexity_eval_candidates.json`)
- `complexity_eval_candidates.json_q70` (from `data/eval/tests/complexity_eval_candidates.json`)
- `complexity_eval_candidates.json_q71` (from `data/eval/tests/complexity_eval_candidates.json`)
- `complexity_eval_candidates.json_q73` (from `data/eval/tests/complexity_eval_candidates.json`)
- `complexity_eval_candidates.json_q74` (from `data/eval/tests/complexity_eval_candidates.json`)
- `complexity_eval_candidates.json_q75` (from `data/eval/tests/complexity_eval_candidates.json`)
- `complexity_eval_candidates.json_q76` (from `data/eval/tests/complexity_eval_candidates.json`)
- `complexity_eval_candidates.json_q78` (from `data/eval/tests/complexity_eval_candidates.json`)
- `complexity_eval_candidates.json_q79` (from `data/eval/tests/complexity_eval_candidates.json`)
- `complexity_eval_candidates.json_q80` (from `data/eval/tests/complexity_eval_candidates.json`)
- `complexity_eval_candidates.json_q81` (from `data/eval/tests/complexity_eval_candidates.json`)
- `complexity_eval_candidates.json_q83` (from `data/eval/tests/complexity_eval_candidates.json`)
- `complexity_eval_candidates.json_q85` (from `data/eval/tests/complexity_eval_candidates.json`)
- `complexity_eval_candidates.json_q86` (from `data/eval/tests/complexity_eval_candidates.json`)
- `medium_summarization_01` (from `data/eval/tests/eval_longform_20.json`)
- `eval_v14_test_20.json_q0` (from `data/eval/tests/eval_v14_test_20.json`)
- `eval_v14_test_20.json_q7` (from `data/eval/tests/eval_v14_test_20.json`)
- `eval_v14_remaining_20.json_q19` (from `data/eval/tests/eval_v14_remaining_20.json`)
- `control_summary_fusion` (from `data/eval/tests/eval_v14_timeout_stress_19.json`)
- `fireworks_eval_20.json_q12` (from `data/eval/tests/fireworks_eval_20.json`)
- `fireworks_eval_20.json_q13` (from `data/eval/tests/fireworks_eval_20.json`)
- `fireworks_eval_20.json_q15` (from `data/eval/tests/fireworks_eval_20.json`)
- `xsum-1f0e69fa8897` (from `data/eval/summarization_combined_25.json`)
- `xsum-260eec64733b` (from `data/eval/summarization_combined_25.json`)
- `xsum-4c2076d75e1c` (from `data/eval/summarization_combined_25.json`)
- `xsum-5ead3b4e019b` (from `data/eval/summarization_combined_25.json`)
- `xsum-60f2ba73e141` (from `data/eval/summarization_combined_25.json`)
- `xsum-726724dafa9e` (from `data/eval/summarization_combined_25.json`)
- `xsum-75d704a754da` (from `data/eval/summarization_combined_25.json`)
- `xsum-7756c474bf25` (from `data/eval/summarization_combined_25.json`)
- `xsum-79d08afd57e1` (from `data/eval/summarization_combined_25.json`)
- `xsum-83e2b8250a8c` (from `data/eval/summarization_combined_25.json`)
- `xsum-862f366208a4` (from `data/eval/summarization_combined_25.json`)
- `xsum-975d27e940d2` (from `data/eval/summarization_combined_25.json`)
- `xsum-9f77a14788c7` (from `data/eval/summarization_combined_25.json`)
- `xsum-bb6aee03a3db` (from `data/eval/summarization_combined_25.json`)
- `xsum-bd87969dec13` (from `data/eval/summarization_combined_25.json`)
- `xsum-c181680bb7bc` (from `data/eval/summarization_combined_25.json`)
- `xsum-dccce4944b71` (from `data/eval/summarization_combined_25.json`)
- `xsum-f0baddd56cc9` (from `data/eval/summarization_combined_25.json`)
- `xsum-00ac9888e2f5` (from `data/eval/summarization_combined_25.json`)
- `xsum-11a19a39f4d0` (from `data/eval/summarization_combined_25.json`)
- `xsum-2054f538d8ea` (from `data/eval/summarization_combined_25.json`)
- `xsum-407b80861d2f` (from `data/eval/summarization_combined_25.json`)
- `xsum-c5fc1996f68d` (from `data/eval/summarization_combined_25.json`)
- `xsum-d824a5cab529` (from `data/eval/summarization_combined_25.json`)

## Insights & Root Cause Analysis

### Summarization → logic
- Count: 70
- Avg summarization score: 3.293
- Avg logic score (winner): 4.607
- Avg score gap: 1.314

  Likely trigger: Constraint patterns, paragraph breaks, named entities
  in narrative prose being interpreted as logic puzzles.

### Summarization → math
- Count: 49
- Avg summarization score: 3.000
- Avg math score (winner): 4.367
- Avg score gap: 1.367

  Likely trigger: Numbers in the prompt triggering math scorer, or
  arithmetic operations in narrative context.

### Summarization → code_gen
- Count: 40
- Avg summarization score: 3.525
- Avg code_gen score (winner): 5.325
- Avg score gap: 1.800

  Likely trigger: Code-related keywords in technical summarization prompts.
