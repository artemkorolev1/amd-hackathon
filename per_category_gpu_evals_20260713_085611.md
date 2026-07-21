# Comprehensive Per-Category GPU Eval Report

- **Model**: qwen2.5-coder-1.5b-instruct
- **Model path**: models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
- **Date**: 2026-07-13 08:56:11
- **GPU**: RTX A4000 8GB, N_GPU_LAYERS=-1

## Overall Summary

| Metric | Value |
|---|---|
| Total Questions | 366 |
| Correct | 278 |
| **Accuracy** | **75.96%** |
| 84.2% Gate | ❌ FAIL |
| Avg Time/Question | 2696.7 ms |
| Max Time/Question | 9430.8 ms |
| Total Time | 987.0 s |

## Per-Category Breakdown

| Category | Total | Correct | Accuracy | Avg Time (ms) |
|---|---|---|---|---|
| ✅ code_debug | 32 | 31 | 96.9% | 1036 |
| ✅ code_gen | 36 | 32 | 88.9% | 1742 |
| ❌ factual | 58 | 34 | 58.6% | 825 |
| ✅ general | 1 | 1 | 100.0% | 256 |
| ✅ logic | 40 | 38 | 95.0% | 6240 |
| ❌ math | 94 | 76 | 80.9% | 5340 |
| ❌ ner | 31 | 24 | 77.4% | 1541 |
| ❌ sentiment | 37 | 28 | 75.7% | 43 |
| ❌ summarization | 37 | 14 | 37.8% | 1136 |

## Per-Source Breakdown

| Source | Total | Correct | Accuracy |
|---|---|---|---|
| build-A-40 | 18 | 18 | 100.0% |
| build-B-40 | 18 | 18 | 100.0% |
| claude-code-hard-v1 | 100 | 72 | 72.0% |
| complexity_40 | 11 | 10 | 90.9% |
| dev_40 | 8 | 8 | 100.0% |
| eval_mini_10 | 1 | 1 | 100.0% |
| gsm8k | 19 | 11 | 57.9% |
| heldout_40 | 16 | 16 | 100.0% |
| humanevalpack | 19 | 18 | 94.7% |
| logiqa | 10 | 9 | 90.0% |
| mbpp | 19 | 15 | 78.9% |
| nq_open | 19 | 5 | 26.3% |
| sst2 | 25 | 24 | 96.0% |
| syn-026 | 1 | 1 | 100.0% |
| syn-027 | 1 | 1 | 100.0% |
| syn-028 | 1 | 1 | 100.0% |
| syn-035 | 1 | 1 | 100.0% |
| syn-040 | 1 | 1 | 100.0% |
| syn-041 | 1 | 1 | 100.0% |
| syn-044 | 1 | 1 | 100.0% |
| syn-046 | 1 | 1 | 100.0% |
| syn-049 | 1 | 1 | 100.0% |
| syn-057 | 1 | 1 | 100.0% |
| syn-063 | 1 | 1 | 100.0% |
| syn-064 | 1 | 1 | 100.0% |
| syn-065 | 1 | 1 | 100.0% |
| syn-066 | 1 | 1 | 100.0% |
| syn-067 | 1 | 1 | 100.0% |
| syn-069 | 1 | 1 | 100.0% |
| syn-070 | 1 | 1 | 100.0% |
| syn-072 | 1 | 1 | 100.0% |
| tweetner7 | 18 | 15 | 83.3% |
| validation-v3 | 12 | 4 | 33.3% |
| wnut2017 | 1 | 0 | 0.0% |
| xsum | 25 | 7 | 28.0% |
| zebra | 9 | 9 | 100.0% |

## Per-Difficulty Breakdown

| Difficulty | Total | Correct | Accuracy |
|---|---|---|---|
| easy | 59 | 40 | 67.8% |
| hard | 192 | 141 | 73.4% |
| medium | 31 | 22 | 71.0% |
| unspecified | 84 | 75 | 89.3% |

## Failures (88)

| # | Task ID | Category | Expected | Got (snippet) | Reason |
|---|---|---|---|---|---|
| 1 | q-00276998 | factual | The rise of archosaurs (early dinosaur ancestors) to dominance in terrestrial ec | If the Siberian Traps eruptions had released only one-tenth of their actual CO₂  | expected: The rise of archosaurs (early dinosaur ancestors)  |
| 2 | q-037995ed | factual | Even with the electron's charge measured, there would be no evidence that the el | If Millikan's oil drop experiment had been completed in 1896, the critical piece | numeric mismatch 1896.0 vs 1.0; numeric mismatch 1897.0 vs 8 |
| 3 | q-00f945db | factual | The concept of a single Mitochondrial Eve would not exist as understood, because | If mitochondrial DNA were inherited from both parents (biparentally) instead of  | expected: The concept of a single Mitochondrial Eve would no |
| 4 | q-053efbd8 | factual | NASA's Apollo program would almost certainly have continued beyond Apollo 17, wi | If the Soviet Union had successfully landed a cosmonaut on the Moon in August 19 | expected: NASA's Apollo program would almost certainly have  |
| 5 | q-023d19fb | logic | 625 | Let's solve this logic puzzle step by step:

1. **Identify the clues:**
   - The | expected: 625, got: Let's solve this logic puzzle step by st |
| 6 | q-04c83bf7 | math | b) 1/33 | To solve this problem, we need to calculate the probability that both cards draw | expected: b) 1/33, got: To solve this problem, we need to ca |
| 7 | q-0189fb15 | math | d) 624 | To solve this problem, we need to follow these steps:

1. **Determine the number | expected: d) 624, got: To solve this problem, we need to fol |
| 8 | q-04ad17c1 | math | a) 886 | To solve this problem, we need to find the smallest positive integer \( N \) tha | expected: a) 886, got: To solve this problem, we need to fin |
| 9 | q-0534499d | math | b) ₹2,000 | To solve this problem, we need to follow these steps:

1. **Determine the marked | expected: b) ₹2,000, got: To solve this problem, we need to  |
| 10 | q-029c2a17 | math | a) 8156 | To solve this problem, we need to consider the constraints and calculate the num | expected: a) 8156, got: To solve this problem, we need to co |
| 11 | q-00c9910f | math | a) 8:19 | To solve this problem, we need to calculate the volumes of the smaller cone (top | expected: a) 8:19, got: To solve this problem, we need to ca |
| 12 | q-055436d8 | math | c) 58 | Let's solve this step by step:

1. **Define the variables:**
   - Let \( F \) be | expected: c) 58, got: Let's solve this step by step:

1. **D |
| 13 | q-05b8eb47 | ner | PERSON: Maria Fernandez, Jensen Huang, Christopher Danely; ORGANIZATION: Goldman | Goldman Sachs
Maria Fernandez
NVIDIA Corporation
NVDA
GTC 2025 conference
Jensen | expected: PERSON: Maria Fernandez, Jensen Huang, Christopher |
| 14 | q-02c80855 | ner | DISEASE: non-small cell lung cancer (NSCLC); GENE/MUTATION: STK11, KEAP1; PROTEI | ORG: Keytruda, Platinum-based chemotherapy, pembrolizumab, NSCLC, STK11, KEAP1,  | expected: DISEASE: non-small cell lung cancer (NSCLC); GENE/ |
| 15 | q-00bb8092 | ner | PERSON: Jonathan Gray, Harold Patterson; ORGANIZATION: Blackstone Group, MedLine | Blackstone Group (ORG)
MedLine Holdings (ORG)
Jonathan Gray (PER)
Chicago (LOC)
 | expected: PERSON: Jonathan Gray, Harold Patterson; ORGANIZAT |
| 16 | q-04a3772a | sentiment | neutral (hedging) | positive | expected: neutral (hedging), got: positive |
| 17 | q-02539796 | sentiment | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| 18 | q-030b8cd2 | sentiment | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| 19 | q-05767fa7 | sentiment | negative (dismissive) | positive | expected: negative (dismissive), got: positive |
| 20 | q-014b7b9b | sentiment | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| 21 | q-030f1bed | sentiment | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| 22 | q-056d29ef | sentiment | neutral (dry wit) | positive | expected: neutral (dry wit), got: positive |
| 23 | q-0277a050 | sentiment | neutral (hedging) | positive | expected: neutral (hedging), got: positive |
| 24 | q-01aac354 | summarization | Apple framed its USB-C adoption as a voluntary innovation choice, but the EU cit | Apple's decision to adopt USB-C for iPhone 15 is driven by innovation and user e | expected: Apple framed its USB-C adoption as a voluntary inn |
| 25 | q-002c3bdf | summarization | The diplomat expressed grave concern over military movements near a demarcation  | The Permanent Representative of the Federated States to the United Nations Secur | expected: The diplomat expressed grave concern over military |
| 26 | q-04b891f1 | summarization | Study A found moderate red wine consumption associated with 30% lower cardiovasc | Study A suggests that moderate red wine consumption can reduce cardiovascular ev | expected: Study A found moderate red wine consumption associ |
| 27 | q-011a76cf | summarization | Boeing's 2024 safety report touts improved culture and transparency, but was pub | The report highlights Boeing's renewed focus on safety culture following the 737 | expected: Boeing's 2024 safety report touts improved culture |
| 28 | q-04410ce6 | summarization | Meta claims improved election interference detection with 40,000 safety staff an | The three sources provide conflicting information about the effectiveness of Fac | expected: Meta claims improved election interference detecti |
| 29 | hep-a97abd078878 | code_debug | if l == sorted(l) or l == sorted(l, reverse=True):
        return True
    retur | ```python
def monotonic(l: list) -> bool:
    return all(x <= y for x, y in zip( | expected: if l == sorted(l) or l == sorted(l, reverse=True): |
| 30 | mbpp-0182a06f56ab | code_gen | def get_Char(strr):  
    summ = 0
    for i in range(len(strr)): 
        su | ```python
def sum_characters(s):
    return sum(ord(c) for c in s)
``` | expected: def get_Char(strr):  
    summ = 0
    for i in  |
| 31 | mbpp-11ee5a57f917 | code_gen | from operator import itemgetter 
def index_minimum(test_list):
  res = min(tes | ```python
def find_min_index_record(records):
    # Your code here
``` | expected: from operator import itemgetter 
def index_minimu |
| 32 | mbpp-4f81d2404a78 | code_gen | def compute_Last_Digit(A,B): 
    variable = 1
    if (A == B): 
        retu | ```python
def last_digit_factorial(a, b):
    # Your code here
``` | expected: def compute_Last_Digit(A,B): 
    variable = 1
  |
| 33 | mbpp-7c6c70d4d316 | code_gen | import re
def text_match_string(text):
        patterns = '^\w+'
        if r | ```python
def match_word_at_start(text):
    # Your code here
``` | expected: import re
def text_match_string(text):
        p |
| 34 | nq-0c39ac5763fb | factual | a biocidal effect of metals | The oligodynamic effect is a phenomenon that describes the ability of certain ce | expected: a biocidal effect of metals, got: The oligodynamic |
| 35 | nq-1186bd67ee61 | factual | Chhattisgarh | The state located in the centre of India is Uttar Pradesh. | expected: Chhattisgarh, got: The state located in the centre |
| 36 | nq-2a7776c7d218 | factual | New Orleans | The butchers in the slaughterhouse cases lived in the city of Chicago, Illinois. | expected: New Orleans, got: The butchers in the slaughterhou |
| 37 | nq-2ad2fa80c023 | factual | Inner Circle | The theme song for the TV show "Cops" is performed by the band The Police. | expected: Inner Circle, got: The theme song for the TV show  |
| 38 | nq-2fab2b52f064 | factual | in the Gospel of Luke | The Lord's Prayer is found in the Bible in the Book of Matthew, Chapter 6, verse | expected: in the Gospel of Luke, got: The Lord's Prayer is f |
| 39 | nq-3faaeadf61b6 | factual | John Young | It's a long long way to Pasadena was written by the American poet and novelist W | expected: John Young, got: It's a long long way to Pasadena  |
| 40 | nq-60c012cff21c | factual | Jacob Tremblay | The movie "The Wonder" stars Taylor Swift as Auggie. | expected: Jacob Tremblay, got: The movie "The Wonder" stars  |
| 41 | nq-664b95f22952 | factual | commercial at | The at symbol is also known as the "at" symbol. | expected: commercial at, got: The at symbol is also known as |
| 42 | nq-77cee4f7ee4e | factual | Mary Rose Foster | Bette Midler portrayed the character of Rose in the 1990 film "The Rose." | expected: Mary Rose Foster, got: Bette Midler portrayed the  |
| 43 | nq-af387eb22a53 | factual | Max Shippee | Gram is played by actress Tessa Thompson in the TV series "The Young and the Res | expected: Max Shippee, got: Gram is played by actress Tessa  |
| 44 | nq-c8c977e94f10 | factual | Cathy Dennis and Rob Davis | The lyrics to "Cant Get You Out of My Head" were written by the American singer- | expected: Cathy Dennis and Rob Davis, got: The lyrics to "Ca |
| 45 | nq-de9ab68dac46 | factual | Lisa Stelly | The girl in Green Day's 21 Guns is named Maggie. | expected: Lisa Stelly, got: The girl in Green Day's 21 Guns  |
| 46 | nq-f17349362327 | factual | a normally inaccessible mini-game | The Hot Coffee Mod is a mod for the game San Andreas that allows players to cust | expected: a normally inaccessible mini-game, got: The Hot Co |
| 47 | nq-f37bafda2714 | factual | Sir Edmund Barton | The first Australian Prime Minister elected was Sir Robert Menzies, who was elec | expected: Sir Edmund Barton, got: The first Australian Prime |
| 48 | logiqa-f838ff5557d2 | logic | 0. The sales volume only partially reflects the popularity of the book. | To solve this logic puzzle, we need to analyze the given argument and identify a | expected: 0. The sales volume only partially reflects the po |
| 49 | gsm8k-1fc947ca5061 | math | 9 | To solve this problem, we need to follow these steps:

1. Calculate the number o | expected: 9, got: To solve this problem, we need to follow t |
| 50 | gsm8k-28aec708c304 | math | 45 | To solve this problem, we need to calculate the distance John travels in each se | expected: 45, got: To solve this problem, we need to calcula |
| 51 | gsm8k-3891af574c8c | math | 294 | To solve this problem, we need to follow these steps:

1. Calculate the number o | expected: 294, got: To solve this problem, we need to follow |
| 52 | gsm8k-3df88cbb4023 | math | 60 | To solve this problem, we need to determine how much money Bailey started with.  | expected: 60, got: To solve this problem, we need to determi |
| 53 | gsm8k-487d10cb2fb1 | math | 36 | To solve this problem, we need to follow these steps:

1. Determine the number o | expected: 36, got: To solve this problem, we need to follow  |
| 54 | gsm8k-6a26459b58de | math | 25000 | To solve this problem, we need to calculate Marcy's annual pension for the first | expected: 25000, got: To solve this problem, we need to calc |
| 55 | gsm8k-8c62b8200b4d | math | 60 | To solve this problem, we need to determine the square footage of each level of  | expected: 60, got: To solve this problem, we need to determi |
| 56 | gsm8k-f3a142df6b12 | math | 17 | To solve this problem, we need to follow these steps:

1. Calculate the number o | expected: 17, got: To solve this problem, we need to follow  |
| 57 | tweet7-368eb95230b5 | ner | product: {@WhatsApp@}
product: {@WhatsApp@} | PERSON
ORG
LOC
DATE | expected: product: {@WhatsApp@}
product: {@WhatsApp@}, got:  |
| 58 | tweet7-55107ff863ae | ner | person: {@Israel Adesanya@}
person: Jan
event: Ufc259 | LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
 | expected: person: {@Israel Adesanya@}
person: Jan
event: Ufc |
| 59 | tweet7-73e661fcb028 | ner | creative_work: sacred choral Christmas music
group: Choir of Kings College
event | PERSON: {{USERNAME}}
ORG: Choir of Kings College
LOC: one room, other room
DATE: | expected: creative_work: sacred choral Christmas music
group |
| 60 | wnut17-48701c713d02 | ner | product: & gt ; * The soldier was killed when another avalanche hit an army barr | ORG: Sonmarg
LOC: northern area of Sonmarg
LOC: army barracks | expected: product: & gt ; * The soldier was killed when anot |
| 61 | sst2-9c925eff46d4 | sentiment | Sentiment: POSITIVE | negative | expected: Sentiment: POSITIVE, got: negative |
| 62 | xsum-1f0e69fa8897 | summarization | Former US President Bill Clinton is introduced to the McGuinness family by Sinn  | A news article was published on March 23, 2017, at 14:55 GMT. | expected: Former US President Bill Clinton is introduced to  |
| 63 | xsum-260eec64733b | summarization | Peterborough United defender Miles Addison has signed a new one-month contract w | Graham Westley's Posh team has made just one appearance since joining the club,  | expected: Peterborough United defender Miles Addison has sig |
| 64 | xsum-4c2076d75e1c | summarization | A family of five, including three young children, had to be rescued from a Somer | Two crews and a hovercraft rescued two adults and children from Uphill beach at  | expected: A family of five, including three young children,  |
| 65 | xsum-5ead3b4e019b | summarization | An artist's impression of how an expanded primary school in Addlestone will look | Surrey County Council plans to double its Sayes Court capacity by September 2015 | expected: An artist's impression of how an expanded primary  |
| 66 | xsum-726724dafa9e | summarization | Cardiff Devils lost 6-1 away to Nottingham Panthers in the Elite League, less th | Panthers secured a crucial victory against Devils, closing the gap to four point | expected: Cardiff Devils lost 6-1 away to Nottingham Panther |
| 67 | xsum-75d704a754da | summarization | A 19-year-old man has been charged after a cat was killed and others were badly  | In March and April, air rifle attacks in Cranleigh, Guildford, and Woking result | expected: A 19-year-old man has been charged after a cat was |
| 68 | xsum-7756c474bf25 | summarization | A man is being held in Manchester on suspicion of terrorism offences. | Police arrested a suspect on suspicion of terrorism offences in Manchester. | expected: A man is being held in Manchester on suspicion of  |
| 69 | xsum-83e2b8250a8c | summarization | Five people have been injured following a crash between a bin lorry, a bus and t | A bus accident occurred on the Edinburgh bypass between Baberton and Dreghorn at | expected: Five people have been injured following a crash be |
| 70 | xsum-975d27e940d2 | summarization | A man has appeared in court over an acid attack that left two people with "life- | Two men, aged 23 and 24, were charged with wounding with intent to do grievous b | expected: A man has appeared in court over an acid attack th |
| 71 | xsum-bb6aee03a3db | summarization | Leicester City striker Gary Taylor-Fletcher has joined Sheffield Wednesday on an | Taylor-Fletcher has signed a new one-year deal with the Premier League newcomers | expected: Leicester City striker Gary Taylor-Fletcher has jo |
| 72 | xsum-c181680bb7bc | summarization | Disabled pensioner Alan Barnes has said he is "ready to move on with his life" a | A 67-year-old man has begun house hunting after being attacked in January and ha | expected: Disabled pensioner Alan Barnes has said he is "rea |
| 73 | xsum-dccce4944b71 | summarization | A buzzard has been rescued after becoming trapped between fencing and a wall in  | A buzzard stuck in a wire fence in Bethesda was freed by RSPCA Inspector Mike Pu | expected: A buzzard has been rescued after becoming trapped  |
| 74 | xsum-ef3d0b8ec1b2 | summarization | Keepers at Chester Zoo are making sure every creature, from the biggest elephant | The text describes a zoo's requirement to keep track of all its animals, includi | expected: Keepers at Chester Zoo are making sure every creat |
| 75 | gsm8k-4eee572fb01e | math | 64 | To solve this problem, we need to calculate the total cost of the glasses Kylar  | expected: 64, got: To solve this problem, we need to calcula |
| 76 | gsm8k-9bedc2a22490 | math | 40 | To solve this problem, we need to follow these steps:

1. Determine the number o | expected: 40, got: To solve this problem, we need to follow  |
| 77 | gsm8k-ed7a5c520f64 | math | 57500 | To solve this problem, we need to calculate Jill's total annual salary by multip | expected: 57500, got: To solve this problem, we need to calc |
| 78 | xsum-00ac9888e2f5 | summarization | A man who sparked a drug-fuelled roof-top siege after breaking up with his partn | Ben Frost barricaded himself into his girlfriend's flat in Princetown on Dartmoo | expected: A man who sparked a drug-fuelled roof-top siege af |
| 79 | xsum-2054f538d8ea | summarization | Eastleigh have confirmed the loan signing of Portsmouth striker Matt Tubbs. | Tubbs has joined the Spitfires on a deal until the end of the season, and he cou | expected: Eastleigh have confirmed the loan signing of Ports |
| 80 | xsum-407b80861d2f | summarization | Children across Britain linked up with Tim Peake on the International Space Stat | British astronaut Leah Williams spent six weeks in space, but took time out to s | expected: Children across Britain linked up with Tim Peake o |
| 81 | xsum-c5fc1996f68d | summarization | League Two side York City have signed Hull City midfielder Matt Dixon on an 18-m | The 21-year-old made his debut for the Tigers in the League Cup first-round tie  | expected: League Two side York City have signed Hull City mi |
| 82 | xsum-d824a5cab529 | summarization | Four hillwalkers who got into difficulties during severe weather on Ben Lomond h | Two people were rescued from a high-altitude mountain in Scotland, while others  | expected: Four hillwalkers who got into difficulties during  |
| 83 | nq-0daf4903f2dd | factual | E-8s senior chief petty officer | The ranks in the United States Navy are divided into three main categories: juni | expected: E-8s senior chief petty officer, got: The ranks in |
| 84 | nq-27eee714ec33 | factual | Anakin Skywalker | The character Darth Vader is under the mask of Emperor Palpatine. | expected: Anakin Skywalker, got: The character Darth Vader i |
| 85 | nq-37ce106128e2 | factual | Donna | Eric ends up with a woman named Sarah in the 70s show "That's All Right." | expected: Donna, got: Eric ends up with a woman named Sarah  |
| 86 | nq-617ba3b5f165 | factual | 1976 | The last time the Vikings were in the NFC Championship was in 2014. | numeric mismatch 2014.0 vs 1976.0 |
| 87 | nq-844ea0cf827e | factual | Gene Kelly | The song "Good Morning, Good Morning" was sung by the British rock band The Beat | expected: Gene Kelly, got: The song "Good Morning, Good Morn |
| 88 | cx_18 | factual | Gabriel Garcia Marquez | The novel 'One Hundred Years of Solitude' was written by Gabriel García Márquez. | expected: Gabriel Garcia Marquez, got: The novel 'One Hundre |

### code_debug Failures (1)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| hep-a97abd078878 | if l == sorted(l) or l == sorted(l, reverse=True):
        r | ```python
def monotonic(l: list) -> bool:
    return all(x < | expected: if l == sorted(l) or l == sorted(l, reve |

### code_gen Failures (4)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| mbpp-0182a06f56ab | def get_Char(strr):  
    summ = 0
    for i in range(len( | ```python
def sum_characters(s):
    return sum(ord(c) for c | expected: def get_Char(strr):  
    summ = 0
    |
| mbpp-11ee5a57f917 | from operator import itemgetter 
def index_minimum(test_lis | ```python
def find_min_index_record(records):
    # Your cod | expected: from operator import itemgetter 
def in |
| mbpp-4f81d2404a78 | def compute_Last_Digit(A,B): 
    variable = 1
    if (A = | ```python
def last_digit_factorial(a, b):
    # Your code he | expected: def compute_Last_Digit(A,B): 
    varia |
| mbpp-7c6c70d4d316 | import re
def text_match_string(text):
        patterns =  | ```python
def match_word_at_start(text):
    # Your code her | expected: import re
def text_match_string(text): |

### factual Failures (24)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-00276998 | The rise of archosaurs (early dinosaur ancestors) to dominan | If the Siberian Traps eruptions had released only one-tenth  | expected: The rise of archosaurs (early dinosaur a |
| q-037995ed | Even with the electron's charge measured, there would be no  | If Millikan's oil drop experiment had been completed in 1896 | numeric mismatch 1896.0 vs 1.0; numeric mismatch 1 |
| q-00f945db | The concept of a single Mitochondrial Eve would not exist as | If mitochondrial DNA were inherited from both parents (bipar | expected: The concept of a single Mitochondrial Ev |
| q-053efbd8 | NASA's Apollo program would almost certainly have continued  | If the Soviet Union had successfully landed a cosmonaut on t | expected: NASA's Apollo program would almost certa |
| nq-0c39ac5763fb | a biocidal effect of metals | The oligodynamic effect is a phenomenon that describes the a | expected: a biocidal effect of metals, got: The ol |
| nq-1186bd67ee61 | Chhattisgarh | The state located in the centre of India is Uttar Pradesh. | expected: Chhattisgarh, got: The state located in  |
| nq-2a7776c7d218 | New Orleans | The butchers in the slaughterhouse cases lived in the city o | expected: New Orleans, got: The butchers in the sl |
| nq-2ad2fa80c023 | Inner Circle | The theme song for the TV show "Cops" is performed by the ba | expected: Inner Circle, got: The theme song for th |
| nq-2fab2b52f064 | in the Gospel of Luke | The Lord's Prayer is found in the Bible in the Book of Matth | expected: in the Gospel of Luke, got: The Lord's P |
| nq-3faaeadf61b6 | John Young | It's a long long way to Pasadena was written by the American | expected: John Young, got: It's a long long way to |
| nq-60c012cff21c | Jacob Tremblay | The movie "The Wonder" stars Taylor Swift as Auggie. | expected: Jacob Tremblay, got: The movie "The Wond |
| nq-664b95f22952 | commercial at | The at symbol is also known as the "at" symbol. | expected: commercial at, got: The at symbol is als |
| nq-77cee4f7ee4e | Mary Rose Foster | Bette Midler portrayed the character of Rose in the 1990 fil | expected: Mary Rose Foster, got: Bette Midler port |
| nq-af387eb22a53 | Max Shippee | Gram is played by actress Tessa Thompson in the TV series "T | expected: Max Shippee, got: Gram is played by actr |
| nq-c8c977e94f10 | Cathy Dennis and Rob Davis | The lyrics to "Cant Get You Out of My Head" were written by  | expected: Cathy Dennis and Rob Davis, got: The lyr |
| nq-de9ab68dac46 | Lisa Stelly | The girl in Green Day's 21 Guns is named Maggie. | expected: Lisa Stelly, got: The girl in Green Day' |
| nq-f17349362327 | a normally inaccessible mini-game | The Hot Coffee Mod is a mod for the game San Andreas that al | expected: a normally inaccessible mini-game, got:  |
| nq-f37bafda2714 | Sir Edmund Barton | The first Australian Prime Minister elected was Sir Robert M | expected: Sir Edmund Barton, got: The first Austra |
| nq-0daf4903f2dd | E-8s senior chief petty officer | The ranks in the United States Navy are divided into three m | expected: E-8s senior chief petty officer, got: Th |
| nq-27eee714ec33 | Anakin Skywalker | The character Darth Vader is under the mask of Emperor Palpa | expected: Anakin Skywalker, got: The character Dar |
| nq-37ce106128e2 | Donna | Eric ends up with a woman named Sarah in the 70s show "That' | expected: Donna, got: Eric ends up with a woman na |
| nq-617ba3b5f165 | 1976 | The last time the Vikings were in the NFC Championship was i | numeric mismatch 2014.0 vs 1976.0 |
| nq-844ea0cf827e | Gene Kelly | The song "Good Morning, Good Morning" was sung by the Britis | expected: Gene Kelly, got: The song "Good Morning, |
| cx_18 | Gabriel Garcia Marquez | The novel 'One Hundred Years of Solitude' was written by Gab | expected: Gabriel Garcia Marquez, got: The novel ' |

### logic Failures (2)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-023d19fb | 625 | Let's solve this logic puzzle step by step:

1. **Identify t | expected: 625, got: Let's solve this logic puzzle  |
| logiqa-f838ff5557d2 | 0. The sales volume only partially reflects the popularity o | To solve this logic puzzle, we need to analyze the given arg | expected: 0. The sales volume only partially refle |

### math Failures (18)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-04c83bf7 | b) 1/33 | To solve this problem, we need to calculate the probability  | expected: b) 1/33, got: To solve this problem, we  |
| q-0189fb15 | d) 624 | To solve this problem, we need to follow these steps:

1. ** | expected: d) 624, got: To solve this problem, we n |
| q-04ad17c1 | a) 886 | To solve this problem, we need to find the smallest positive | expected: a) 886, got: To solve this problem, we n |
| q-0534499d | b) ₹2,000 | To solve this problem, we need to follow these steps:

1. ** | expected: b) ₹2,000, got: To solve this problem, w |
| q-029c2a17 | a) 8156 | To solve this problem, we need to consider the constraints a | expected: a) 8156, got: To solve this problem, we  |
| q-00c9910f | a) 8:19 | To solve this problem, we need to calculate the volumes of t | expected: a) 8:19, got: To solve this problem, we  |
| q-055436d8 | c) 58 | Let's solve this step by step:

1. **Define the variables:** | expected: c) 58, got: Let's solve this step by ste |
| gsm8k-1fc947ca5061 | 9 | To solve this problem, we need to follow these steps:

1. Ca | expected: 9, got: To solve this problem, we need t |
| gsm8k-28aec708c304 | 45 | To solve this problem, we need to calculate the distance Joh | expected: 45, got: To solve this problem, we need  |
| gsm8k-3891af574c8c | 294 | To solve this problem, we need to follow these steps:

1. Ca | expected: 294, got: To solve this problem, we need |
| gsm8k-3df88cbb4023 | 60 | To solve this problem, we need to determine how much money B | expected: 60, got: To solve this problem, we need  |
| gsm8k-487d10cb2fb1 | 36 | To solve this problem, we need to follow these steps:

1. De | expected: 36, got: To solve this problem, we need  |
| gsm8k-6a26459b58de | 25000 | To solve this problem, we need to calculate Marcy's annual p | expected: 25000, got: To solve this problem, we ne |
| gsm8k-8c62b8200b4d | 60 | To solve this problem, we need to determine the square foota | expected: 60, got: To solve this problem, we need  |
| gsm8k-f3a142df6b12 | 17 | To solve this problem, we need to follow these steps:

1. Ca | expected: 17, got: To solve this problem, we need  |
| gsm8k-4eee572fb01e | 64 | To solve this problem, we need to calculate the total cost o | expected: 64, got: To solve this problem, we need  |
| gsm8k-9bedc2a22490 | 40 | To solve this problem, we need to follow these steps:

1. De | expected: 40, got: To solve this problem, we need  |
| gsm8k-ed7a5c520f64 | 57500 | To solve this problem, we need to calculate Jill's total ann | expected: 57500, got: To solve this problem, we ne |

### ner Failures (7)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-05b8eb47 | PERSON: Maria Fernandez, Jensen Huang, Christopher Danely; O | Goldman Sachs
Maria Fernandez
NVIDIA Corporation
NVDA
GTC 20 | expected: PERSON: Maria Fernandez, Jensen Huang, C |
| q-02c80855 | DISEASE: non-small cell lung cancer (NSCLC); GENE/MUTATION:  | ORG: Keytruda, Platinum-based chemotherapy, pembrolizumab, N | expected: DISEASE: non-small cell lung cancer (NSC |
| q-00bb8092 | PERSON: Jonathan Gray, Harold Patterson; ORGANIZATION: Black | Blackstone Group (ORG)
MedLine Holdings (ORG)
Jonathan Gray  | expected: PERSON: Jonathan Gray, Harold Patterson; |
| tweet7-368eb95230b5 | product: {@WhatsApp@}
product: {@WhatsApp@} | PERSON
ORG
LOC
DATE | expected: product: {@WhatsApp@}
product: {@WhatsAp |
| tweet7-55107ff863ae | person: {@Israel Adesanya@}
person: Jan
event: Ufc259 | LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
LOC
 | expected: person: {@Israel Adesanya@}
person: Jan
 |
| tweet7-73e661fcb028 | creative_work: sacred choral Christmas music
group: Choir of | PERSON: {{USERNAME}}
ORG: Choir of Kings College
LOC: one ro | expected: creative_work: sacred choral Christmas m |
| wnut17-48701c713d02 | product: & gt ; * The soldier was killed when another avalan | ORG: Sonmarg
LOC: northern area of Sonmarg
LOC: army barrack | expected: product: & gt ; * The soldier was killed |

### sentiment Failures (9)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-04a3772a | neutral (hedging) | positive | expected: neutral (hedging), got: positive |
| q-02539796 | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| q-030b8cd2 | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| q-05767fa7 | negative (dismissive) | positive | expected: negative (dismissive), got: positive |
| q-014b7b9b | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| q-030f1bed | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| q-056d29ef | neutral (dry wit) | positive | expected: neutral (dry wit), got: positive |
| q-0277a050 | neutral (hedging) | positive | expected: neutral (hedging), got: positive |
| sst2-9c925eff46d4 | Sentiment: POSITIVE | negative | expected: Sentiment: POSITIVE, got: negative |

### summarization Failures (23)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-01aac354 | Apple framed its USB-C adoption as a voluntary innovation ch | Apple's decision to adopt USB-C for iPhone 15 is driven by i | expected: Apple framed its USB-C adoption as a vol |
| q-002c3bdf | The diplomat expressed grave concern over military movements | The Permanent Representative of the Federated States to the  | expected: The diplomat expressed grave concern ove |
| q-04b891f1 | Study A found moderate red wine consumption associated with  | Study A suggests that moderate red wine consumption can redu | expected: Study A found moderate red wine consumpt |
| q-011a76cf | Boeing's 2024 safety report touts improved culture and trans | The report highlights Boeing's renewed focus on safety cultu | expected: Boeing's 2024 safety report touts improv |
| q-04410ce6 | Meta claims improved election interference detection with 40 | The three sources provide conflicting information about the  | expected: Meta claims improved election interferen |
| xsum-1f0e69fa8897 | Former US President Bill Clinton is introduced to the McGuin | A news article was published on March 23, 2017, at 14:55 GMT | expected: Former US President Bill Clinton is intr |
| xsum-260eec64733b | Peterborough United defender Miles Addison has signed a new  | Graham Westley's Posh team has made just one appearance sinc | expected: Peterborough United defender Miles Addis |
| xsum-4c2076d75e1c | A family of five, including three young children, had to be  | Two crews and a hovercraft rescued two adults and children f | expected: A family of five, including three young  |
| xsum-5ead3b4e019b | An artist's impression of how an expanded primary school in  | Surrey County Council plans to double its Sayes Court capaci | expected: An artist's impression of how an expande |
| xsum-726724dafa9e | Cardiff Devils lost 6-1 away to Nottingham Panthers in the E | Panthers secured a crucial victory against Devils, closing t | expected: Cardiff Devils lost 6-1 away to Nottingh |
| xsum-75d704a754da | A 19-year-old man has been charged after a cat was killed an | In March and April, air rifle attacks in Cranleigh, Guildfor | expected: A 19-year-old man has been charged after |
| xsum-7756c474bf25 | A man is being held in Manchester on suspicion of terrorism  | Police arrested a suspect on suspicion of terrorism offences | expected: A man is being held in Manchester on sus |
| xsum-83e2b8250a8c | Five people have been injured following a crash between a bi | A bus accident occurred on the Edinburgh bypass between Babe | expected: Five people have been injured following  |
| xsum-975d27e940d2 | A man has appeared in court over an acid attack that left tw | Two men, aged 23 and 24, were charged with wounding with int | expected: A man has appeared in court over an acid |
| xsum-bb6aee03a3db | Leicester City striker Gary Taylor-Fletcher has joined Sheff | Taylor-Fletcher has signed a new one-year deal with the Prem | expected: Leicester City striker Gary Taylor-Fletc |
| xsum-c181680bb7bc | Disabled pensioner Alan Barnes has said he is "ready to move | A 67-year-old man has begun house hunting after being attack | expected: Disabled pensioner Alan Barnes has said  |
| xsum-dccce4944b71 | A buzzard has been rescued after becoming trapped between fe | A buzzard stuck in a wire fence in Bethesda was freed by RSP | expected: A buzzard has been rescued after becomin |
| xsum-ef3d0b8ec1b2 | Keepers at Chester Zoo are making sure every creature, from  | The text describes a zoo's requirement to keep track of all  | expected: Keepers at Chester Zoo are making sure e |
| xsum-00ac9888e2f5 | A man who sparked a drug-fuelled roof-top siege after breaki | Ben Frost barricaded himself into his girlfriend's flat in P | expected: A man who sparked a drug-fuelled roof-to |
| xsum-2054f538d8ea | Eastleigh have confirmed the loan signing of Portsmouth stri | Tubbs has joined the Spitfires on a deal until the end of th | expected: Eastleigh have confirmed the loan signin |
| xsum-407b80861d2f | Children across Britain linked up with Tim Peake on the Inte | British astronaut Leah Williams spent six weeks in space, bu | expected: Children across Britain linked up with T |
| xsum-c5fc1996f68d | League Two side York City have signed Hull City midfielder M | The 21-year-old made his debut for the Tigers in the League  | expected: League Two side York City have signed Hu |
| xsum-d824a5cab529 | Four hillwalkers who got into difficulties during severe wea | Two people were rescued from a high-altitude mountain in Sco | expected: Four hillwalkers who got into difficulti |
