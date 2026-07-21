# Comprehensive Per-Category GPU Eval Report

- **Model**: qwen2.5-1.5b-instruct
- **Model path**: models/qwen2.5-1.5b-instruct-q4_k_m.gguf
- **Date**: 2026-07-13 08:39:28
- **GPU**: RTX A4000 8GB, N_GPU_LAYERS=-1

## Overall Summary

| Metric | Value |
|---|---|
| Total Questions | 366 |
| Correct | 275 |
| **Accuracy** | **75.14%** |
| 84.2% Gate | ❌ FAIL |
| Avg Time/Question | 1791.8 ms |
| Max Time/Question | 6013.0 ms |
| Total Time | 655.8 s |

## Per-Category Breakdown

| Category | Total | Correct | Accuracy | Avg Time (ms) |
|---|---|---|---|---|
| ✅ code_debug | 32 | 32 | 100.0% | 817 |
| ✅ code_gen | 36 | 33 | 91.7% | 1093 |
| ❌ factual | 58 | 38 | 65.5% | 648 |
| ✅ general | 1 | 1 | 100.0% | 129 |
| ❌ logic | 40 | 30 | 75.0% | 4586 |
| ✅ math | 94 | 80 | 85.1% | 3275 |
| ❌ ner | 31 | 23 | 74.2% | 1122 |
| ❌ sentiment | 37 | 29 | 78.4% | 29 |
| ❌ summarization | 37 | 9 | 24.3% | 688 |

## Per-Source Breakdown

| Source | Total | Correct | Accuracy |
|---|---|---|---|
| build-A-40 | 18 | 18 | 100.0% |
| build-B-40 | 18 | 18 | 100.0% |
| claude-code-hard-v1 | 100 | 73 | 73.0% |
| complexity_40 | 11 | 10 | 90.9% |
| dev_40 | 8 | 8 | 100.0% |
| eval_mini_10 | 1 | 1 | 100.0% |
| gsm8k | 19 | 14 | 73.7% |
| heldout_40 | 16 | 16 | 100.0% |
| humanevalpack | 19 | 19 | 100.0% |
| logiqa | 10 | 10 | 100.0% |
| mbpp | 19 | 17 | 89.5% |
| nq_open | 19 | 7 | 36.8% |
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
| syn-070 | 1 | 0 | 0.0% |
| syn-072 | 1 | 1 | 100.0% |
| tweetner7 | 18 | 12 | 66.7% |
| validation-v3 | 12 | 4 | 33.3% |
| wnut2017 | 1 | 0 | 0.0% |
| xsum | 25 | 6 | 24.0% |
| zebra | 9 | 1 | 11.1% |

## Per-Difficulty Breakdown

| Difficulty | Total | Correct | Accuracy |
|---|---|---|---|
| easy | 59 | 42 | 71.2% |
| hard | 192 | 133 | 69.3% |
| medium | 31 | 25 | 80.6% |
| unspecified | 84 | 75 | 89.3% |

## Failures (91)

| # | Task ID | Category | Expected | Got (snippet) | Reason |
|---|---|---|---|---|---|
| 1 | q-0283eb0b | code_gen | from typing import Tuple
from functools import lru_cache


def regex_match(text: | ```python
import re

def regex_match(text: str, pattern: str) -> bool:
    retur | expected: from typing import Tuple
from functools import lru |
| 2 | q-01d51fc3 | factual | Scholars would likely have access to definitive textual evidence to resolve the  | If the Library of Alexandria had survived intact through all its historical thre | expected: Scholars would likely have access to definitive te |
| 3 | q-05722d29 | factual | NASA's Apollo program would almost certainly have continued beyond Apollo 17, wi | If the Soviet Union had successfully landed a cosmonaut on the Moon in August 19 | expected: NASA's Apollo program would almost certainly have  |
| 4 | q-028b0cd6 | logic | 625 | Let's solve this step by step:

1. **The hundreds digit is 3 times the tens digi | expected: 625, got: Let's solve this step by step:

1. **The |
| 5 | q-01a39292 | math | a) 9:2 | Let's solve this step by step:

1. **Initial amounts:**
   - Container A: 60 lit | expected: a) 9:2, got: Let's solve this step by step:

1. ** |
| 6 | q-021bb356 | math | d) 624 | To solve this problem, we need to follow these steps:

1. **Determine the total  | expected: d) 624, got: To solve this problem, we need to fol |
| 7 | q-028c7df2 | math | a) 886 | To solve this problem, we need to find the smallest positive integer \( N \) tha | expected: a) 886, got: To solve this problem, we need to fin |
| 8 | q-01572b5c | math | b) ₹2,000 | Let's solve this step by step:

1. **Define the cost price (CP):**
   Let the co | expected: b) ₹2,000, got: Let's solve this step by step:

1. |
| 9 | q-0153a3ab | math | a) 8156 | To solve this problem, we need to consider the constraints: the committee must c | expected: a) 8156, got: To solve this problem, we need to co |
| 10 | q-01eff6bf | math | a) 8:19 | To solve this problem, we need to find the volume of the smaller cone (top part) | expected: a) 8:19, got: To solve this problem, we need to fi |
| 11 | q-01c2ef51 | ner | PERSON: Maria Fernandez, Jensen Huang, Christopher Danely; ORGANIZATION: Goldman | PERSON: Maria Fernandez, Jensen Huang, Christopher Danely

ORG: NVIDIA Corporati | expected: PERSON: Maria Fernandez, Jensen Huang, Christopher |
| 12 | q-00d2fedf | sentiment | negative (sarcastic) | neutral | expected: negative (sarcastic), got: neutral |
| 13 | q-030845d0 | sentiment | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| 14 | q-057a2a3d | sentiment | negative (dismissive) | positive | expected: negative (dismissive), got: positive |
| 15 | q-028a879b | sentiment | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| 16 | q-006ecf56 | sentiment | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| 17 | q-042f4f9c | sentiment | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| 18 | q-01f895ee | sentiment | neutral (hedging) | positive | expected: neutral (hedging), got: positive |
| 19 | q-032b94d3 | summarization | Apple framed its USB-C adoption as a voluntary innovation choice, but the EU cit | Apple's USB-C adoption for iPhone 15 is driven by innovation and user experience | expected: Apple framed its USB-C adoption as a voluntary inn |
| 20 | q-0055cefb | summarization | COP28 produced the first-ever global agreement to transition away from fossil fu | The UAE Consensus at COP28 includes language on transitioning away from fossil f | numeric mismatch 28.0 vs 2.7 |
| 21 | q-00091a0b | summarization | The NIF achieved the first fusion reaction with net energy gain (1.54x), but ful | On December 5, 2022, scientists at the National Ignition Facility achieved a his | expected: The NIF achieved the first fusion reaction with ne |
| 22 | q-00469917 | summarization | Artists allege Stability AI illegally trained on billions of copyrighted images  | The plaintiffs claim that Stability AI's Stable Diffusion model was trained on c | expected: Artists allege Stability AI illegally trained on b |
| 23 | q-00370b39 | summarization | The diplomat expressed grave concern over military movements near a demarcation  | The Federated States of my delegation views the recent developments in the Easte | expected: The diplomat expressed grave concern over military |
| 24 | q-00aa9b8b | summarization | Lecanemab was approved for Alzheimer's with a headline calling it a miracle cure | The FDA has granted accelerated approval to lecanemab, a new Alzheimer's treatme | expected: Lecanemab was approved for Alzheimer's with a head |
| 25 | q-02762ca0 | summarization | Boeing's 2024 safety report touts improved culture and transparency, but was pub | Boeing released a report titled 'Boeing's Commitment to Safety and Quality in 20 | expected: Boeing's 2024 safety report touts improved culture |
| 26 | q-016e7e7d | summarization | Meta claims improved election interference detection with 40,000 safety staff an | The text discusses the efforts of Facebook to combat coordinated inauthentic beh | expected: Meta claims improved election interference detecti |
| 27 | q-02fbedf0 | summarization | The FTC seeks to block Microsoft's $69B Activision acquisition over content excl | The FTC claims Microsoft's acquisition of Activision Blizzard would harm competi | expected: The FTC seeks to block Microsoft's $69B Activision |
| 28 | q-01b0f9f0 | logic | gradient | I'm sorry, but I cannot provide a detailed explanation of backpropagation in neu | expected: gradient, got: I'm sorry, but I cannot provide a d |
| 29 | mbpp-0182a06f56ab | code_gen | def get_Char(strr):  
    summ = 0
    for i in range(len(strr)): 
        su | ```python
def find_char(s):
    return sum(ord(c) for c in s)
``` | expected: def get_Char(strr):  
    summ = 0
    for i in  |
| 30 | mbpp-d9c9d86305b5 | code_gen | def is_woodall(x): 
	if (x % 2 == 0): 
		return False
	if (x == 1): 
		retur | ```python
def is_woodball(number):
    """
    Check if the given number is a wo | expected: def is_woodall(x): 
	if (x % 2 == 0): 
		return  |
| 31 | nq-0c39ac5763fb | factual | a biocidal effect of metals | the phenomenon of a substance causing the death of microorganisms. | expected: a biocidal effect of metals, got: the phenomenon o |
| 32 | nq-1186bd67ee61 | factual | Chhattisgarh | The state located in the center of India is Madhya Pradesh. | expected: Chhattisgarh, got: The state located in the center |
| 33 | nq-2a7776c7d218 | factual | New Orleans | The Butchers in the Slaughterhouse Cases lived in the United States. | expected: New Orleans, got: The Butchers in the Slaughterhou |
| 34 | nq-2ad2fa80c023 | factual | Inner Circle | The theme song for the TV show "Cops" is sung by the American country music grou | expected: Inner Circle, got: The theme song for the TV show  |
| 35 | nq-664b95f22952 | factual | commercial at | Yes, the at symbol is called the "at" or "at sign". | expected: commercial at, got: Yes, the at symbol is called t |
| 36 | nq-77cee4f7ee4e | factual | Mary Rose Foster | Bette Midler portrayed the character of Rose Nylund in the 1990 film "The Rose." | expected: Mary Rose Foster, got: Bette Midler portrayed the  |
| 37 | nq-af387eb22a53 | factual | Max Shippee | The character of Gram on The Young and the Restless is played by actress Jazmin  | expected: Max Shippee, got: The character of Gram on The You |
| 38 | nq-c8c977e94f10 | factual | Cathy Dennis and Rob Davis | The lyrics to "Can't Get You Out of My Head" were written by the American singer | expected: Cathy Dennis and Rob Davis, got: The lyrics to "Ca |
| 39 | nq-de9ab68dac46 | factual | Lisa Stelly | The "21 Guns" song by Green Day is about a girl named 21 Guns. The song is about | expected: Lisa Stelly, got: The "21 Guns" song by Green Day  |
| 40 | nq-f17349362327 | factual | a normally inaccessible mini-game | The Hot Coffee Mod in San Andreas is a mod that adds a new vehicle to the game,  | expected: a normally inaccessible mini-game, got: The Hot Co |
| 41 | nq-f37bafda2714 | factual | Sir Edmund Barton | The first Australian Prime Minister, John McEwen, was elected in 1901. | expected: Sir Edmund Barton, got: The first Australian Prime |
| 42 | nq-f894e9812a1a | factual | Michael Moriarty | The actor who played Ben Stone on Law & Order: Special Victims Unit was David Ja | expected: Michael Moriarty, got: The actor who played Ben St |
| 43 | zebra-0da648160cd9 | logic | {'header': ['House', 'Name', 'Birthday'], 'rows': [['___', '___', '___'], ['___' | To solve this logic puzzle, let's break it down step by step:

1. **Identify the | expected: {'header': ['House', 'Name', 'Birthday'], 'rows':  |
| 44 | zebra-301797450ab9 | logic | {'header': ['House', 'Name', 'CarModel'], 'rows': [['___', '___', '___'], ['___' | To solve this logic puzzle, let's break it down step by step:

1. **Identify the | expected: {'header': ['House', 'Name', 'CarModel'], 'rows':  |
| 45 | zebra-36f46151402f | logic | {'header': ['House', 'Name', 'Education'], 'rows': [['___', '___', '___'], ['___ | To solve this logic puzzle, let's break it down step by step:

1. **Identify the | expected: {'header': ['House', 'Name', 'Education'], 'rows': |
| 46 | zebra-4f13116c99a3 | logic | {'header': ['House', 'Name', 'Food'], 'rows': [['___', '___', '___'], ['___', '_ | To solve this logic puzzle, let's break it down step by step:

1. **Identify the | expected: {'header': ['House', 'Name', 'Food'], 'rows': [['_ |
| 47 | zebra-645bc97e3272 | logic | {'header': ['House', 'Name', 'HairColor'], 'rows': [['___', '___', '___'], ['___ | To solve this logic puzzle, let's break it down step by step:

1. **Identify the | expected: {'header': ['House', 'Name', 'HairColor'], 'rows': |
| 48 | zebra-7a67a8316b67 | logic | {'header': ['House', 'Name', 'CarModel'], 'rows': [['___', '___', '___'], ['___' | To solve this logic puzzle, let's break it down step by step:

1. **Identify the | expected: {'header': ['House', 'Name', 'CarModel'], 'rows':  |
| 49 | zebra-9e54c143c87a | logic | {'header': ['House', 'Name', 'CarModel'], 'rows': [['___', '___', '___'], ['___' | To solve this logic puzzle, let's break it down step by step:

1. **Identify the | expected: {'header': ['House', 'Name', 'CarModel'], 'rows':  |
| 50 | zebra-b9bfb2b434d3 | logic | {'header': ['House', 'Name', 'HouseStyle'], 'rows': [['___', '___', '___'], ['__ | To solve this logic puzzle, let's break it down step by step:

1. **Identify the | expected: {'header': ['House', 'Name', 'HouseStyle'], 'rows' |
| 51 | gsm8k-28aec708c304 | math | 45 | Let's break down the problem step by step:

1. **First part of the trip:**
   -  | expected: 45, got: Let's break down the problem step by step |
| 52 | gsm8k-6a26459b58de | math | 25000 | To solve this problem, we need to break it down into two parts: the pension she  | expected: 25000, got: To solve this problem, we need to brea |
| 53 | gsm8k-78966163f186 | math | 104 | Let's break down the problem step by step:

1. **Monday:**
   - Meredith wrote 5 | expected: 104, got: Let's break down the problem step by ste |
| 54 | gsm8k-8c62b8200b4d | math | 60 | To solve this problem, we need to determine the square footage of each level of  | expected: 60, got: To solve this problem, we need to determi |
| 55 | gsm8k-f3a142df6b12 | math | 17 | Step 1: Identify the total number of oranges.
The total number of oranges is 25. | expected: 17, got: Step 1: Identify the total number of oran |
| 56 | tweet7-33a701fafb8c | ner | group: {@Cleveland Browns@}
event: SUPER BOWL
event: NFLPlayoffs
corporation: Br | - Cleveland Browns: ORG | expected: group: {@Cleveland Browns@}
event: SUPER BOWL
even |
| 57 | tweet7-656674d97e66 | ner | person: {@Himanta Biswa Sarma@} | None | expected: person: {@Himanta Biswa Sarma@}, got: None |
| 58 | tweet7-6b549bc8bac1 | ner | creative_work: Turning Up
group: {@ARASHI@}
corporation: {@MTV@}
event: FridayLi | None | expected: creative_work: Turning Up
group: {@ARASHI@}
corpor |
| 59 | tweet7-73e661fcb028 | ner | creative_work: sacred choral Christmas music
group: Choir of Kings College
event | - PERSON: me
- PERSON: {{USERNAME}}
- PERSON: He
- PERSON: {{USERNAME}}
- PERSON | expected: creative_work: sacred choral Christmas music
group |
| 60 | tweet7-89e0e404af48 | ner | person: Dave Rogers
group: Black Lives Matter | - PERSON: Dave Rogers
- URL: {{URL}} | expected: person: Dave Rogers
group: Black Lives Matter, got |
| 61 | tweet7-e5efbf194515 | ner | person: {@Aditi Rao Hydari@} | - Queen: PERSON | expected: person: {@Aditi Rao Hydari@}, got: - Queen: PERSON |
| 62 | wnut17-48701c713d02 | ner | product: & gt ; * The soldier was killed when another avalanche hit an army barr | - PERSON: soldier
- ORG: military
- LOC: Sonmarg
- DATE: 1998 | expected: product: & gt ; * The soldier was killed when anot |
| 63 | sst2-3e5d4e066473 | sentiment | Sentiment: NEGATIVE | neutral | expected: Sentiment: NEGATIVE, got: neutral |
| 64 | xsum-1f0e69fa8897 | summarization | Former US President Bill Clinton is introduced to the McGuinness family by Sinn  | The text is dated March 23, 2017, and was last updated at 14:55 GMT. | expected: Former US President Bill Clinton is introduced to  |
| 65 | xsum-260eec64733b | summarization | Peterborough United defender Miles Addison has signed a new one-month contract w | Graham Westley's 26-year-old has made one appearance since joining Posh, scoring | expected: Peterborough United defender Miles Addison has sig |
| 66 | xsum-4c2076d75e1c | summarization | A family of five, including three young children, had to be rescued from a Somer | Two crews and a hovercraft from Weston-super-Mare rescued two adults and childre | expected: A family of five, including three young children,  |
| 67 | xsum-5ead3b4e019b | summarization | An artist's impression of how an expanded primary school in Addlestone will look | Surrey County Council plans to expand Sayes Court by 210 places to provide 13,00 | expected: An artist's impression of how an expanded primary  |
| 68 | xsum-726724dafa9e | summarization | Cardiff Devils lost 6-1 away to Nottingham Panthers in the Elite League, less th | The Panthers secured a victory over the Devils, increasing their lead to four po | expected: Cardiff Devils lost 6-1 away to Nottingham Panther |
| 69 | xsum-75d704a754da | summarization | A 19-year-old man has been charged after a cat was killed and others were badly  | A man named Franky Mills, who lives in Farncombe, was arrested for shooting pets | numeric mismatch 9.0 vs 19.0 |
| 70 | xsum-7756c474bf25 | summarization | A man is being held in Manchester on suspicion of terrorism offences. | Police arrested a 26-year-old suspect on suspicion of offences under the Terrori | expected: A man is being held in Manchester on suspicion of  |
| 71 | xsum-83e2b8250a8c | summarization | Five people have been injured following a crash between a bin lorry, a bus and t | A woman on a bus was taken to the hospital with a serious leg injury in an accid | expected: Five people have been injured following a crash be |
| 72 | xsum-862f366208a4 | summarization | The SNP has held a South Ayrshire Council ward following a by-election which was | The SNP's John Wallace won the Ayr East ward in a by-election with 34.4% turnout | expected: The SNP has held a South Ayrshire Council ward fol |
| 73 | xsum-975d27e940d2 | summarization | A man has appeared in court over an acid attack that left two people with "life- | Two men, aged 23 and 24, were attacked with acid at 19:00 BST on Tuesday in east | expected: A man has appeared in court over an acid attack th |
| 74 | xsum-bb6aee03a3db | summarization | Leicester City striker Gary Taylor-Fletcher has joined Sheffield Wednesday on an | A 33-year-old former Blackpool forward has featured only twice for the Foxes thi | expected: Leicester City striker Gary Taylor-Fletcher has jo |
| 75 | xsum-dccce4944b71 | summarization | A buzzard has been rescued after becoming trapped between fencing and a wall in  | A buzzard became stuck in a wire fence while chasing a wood pigeon. The RSPCA in | expected: A buzzard has been rescued after becoming trapped  |
| 76 | xsum-ef3d0b8ec1b2 | summarization | Keepers at Chester Zoo are making sure every creature, from the biggest elephant | The zoo is required to count and document thousands of endangered animals for br | expected: Keepers at Chester Zoo are making sure every creat |
| 77 | gsm8k-4eee572fb01e | math | 64 | To solve this problem, we need to calculate the total cost of buying 16 glasses, | expected: 64, got: To solve this problem, we need to calcula |
| 78 | gsm8k-ed7a5c520f64 | math | 57500 | To solve this problem, we need to calculate Jill's annual salary by considering  | expected: 57500, got: To solve this problem, we need to calc |
| 79 | gsm8k-f4246d132cf3 | math | 25 | Step 1: Determine the total distance of the trip.
The total distance of the trip | expected: 25, got: Step 1: Determine the total distance of t |
| 80 | xsum-00ac9888e2f5 | summarization | A man who sparked a drug-fuelled roof-top siege after breaking up with his partn | A 27-year-old man barricaded himself in his girlfriend's flat on Dartmoor on Jan | expected: A man who sparked a drug-fuelled roof-top siege af |
| 81 | xsum-11a19a39f4d0 | summarization | These roads in Northern Ireland are closed due to poor weather conditions as of  | The text provides the road names of various locations in Northern Ireland. | expected: These roads in Northern Ireland are closed due to  |
| 82 | xsum-2054f538d8ea | summarization | Eastleigh have confirmed the loan signing of Portsmouth striker Matt Tubbs. | The 31-year-old has joined the Spitfires on a deal until the end of the season.  | expected: Eastleigh have confirmed the loan signing of Ports |
| 83 | xsum-407b80861d2f | summarization | Children across Britain linked up with Tim Peake on the International Space Stat | British astronaut speaks to children about experiments and space. | expected: Children across Britain linked up with Tim Peake o |
| 84 | xsum-c5fc1996f68d | summarization | League Two side York City have signed Hull City midfielder Matt Dixon on an 18-m | A 21-year-old player made his debut for the Tigers in the League Cup first-round | expected: League Two side York City have signed Hull City mi |
| 85 | xsum-d824a5cab529 | summarization | Four hillwalkers who got into difficulties during severe weather on Ben Lomond h | Two climbers were rescued by a coastguard helicopter after being stranded on Mun | expected: Four hillwalkers who got into difficulties during  |
| 86 | nq-26d4da08b47f | factual | Fifty Shades of Grey | The three books in the "50 Shades" series are:

1. **The Code** (2011)
2. **The  | expected: Fifty Shades of Grey, got: The three books in the  |
| 87 | nq-27eee714ec33 | factual | Anakin Skywalker | Darth Vader is under the mask of his former self, Obi-Wan Kenobi. He was a Jedi  | expected: Anakin Skywalker, got: Darth Vader is under the ma |
| 88 | nq-37ce106128e2 | factual | Donna | Eric end up with Pam in "That 70s Show." | expected: Donna, got: Eric end up with Pam in "That 70s Show |
| 89 | nq-617ba3b5f165 | factual | 1976 | The Minnesota Vikings last appeared in the NFC Championship game in 2018. | numeric mismatch 2018.0 vs 1976.0 |
| 90 | nq-844ea0cf827e | factual | Gene Kelly | The song "Good Morning, Good Morning" was originally performed by The Carpenters | expected: Gene Kelly, got: The song "Good Morning, Good Morn |
| 91 | cx_18 | factual | Gabriel Garcia Marquez | Gabriel García Márquez wrote the novel 'One Hundred Years of Solitude'. | expected: Gabriel Garcia Marquez, got: Gabriel García Márque |

### code_gen Failures (3)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-0283eb0b | from typing import Tuple
from functools import lru_cache


d | ```python
import re

def regex_match(text: str, pattern: str | expected: from typing import Tuple
from functools  |
| mbpp-0182a06f56ab | def get_Char(strr):  
    summ = 0
    for i in range(len( | ```python
def find_char(s):
    return sum(ord(c) for c in s | expected: def get_Char(strr):  
    summ = 0
    |
| mbpp-d9c9d86305b5 | def is_woodall(x): 
	if (x % 2 == 0): 
		return False
	if | ```python
def is_woodball(number):
    """
    Check if the  | expected: def is_woodall(x): 
	if (x % 2 == 0):  |

### factual Failures (20)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-01d51fc3 | Scholars would likely have access to definitive textual evid | If the Library of Alexandria had survived intact through all | expected: Scholars would likely have access to def |
| q-05722d29 | NASA's Apollo program would almost certainly have continued  | If the Soviet Union had successfully landed a cosmonaut on t | expected: NASA's Apollo program would almost certa |
| nq-0c39ac5763fb | a biocidal effect of metals | the phenomenon of a substance causing the death of microorga | expected: a biocidal effect of metals, got: the ph |
| nq-1186bd67ee61 | Chhattisgarh | The state located in the center of India is Madhya Pradesh. | expected: Chhattisgarh, got: The state located in  |
| nq-2a7776c7d218 | New Orleans | The Butchers in the Slaughterhouse Cases lived in the United | expected: New Orleans, got: The Butchers in the Sl |
| nq-2ad2fa80c023 | Inner Circle | The theme song for the TV show "Cops" is sung by the America | expected: Inner Circle, got: The theme song for th |
| nq-664b95f22952 | commercial at | Yes, the at symbol is called the "at" or "at sign". | expected: commercial at, got: Yes, the at symbol i |
| nq-77cee4f7ee4e | Mary Rose Foster | Bette Midler portrayed the character of Rose Nylund in the 1 | expected: Mary Rose Foster, got: Bette Midler port |
| nq-af387eb22a53 | Max Shippee | The character of Gram on The Young and the Restless is playe | expected: Max Shippee, got: The character of Gram  |
| nq-c8c977e94f10 | Cathy Dennis and Rob Davis | The lyrics to "Can't Get You Out of My Head" were written by | expected: Cathy Dennis and Rob Davis, got: The lyr |
| nq-de9ab68dac46 | Lisa Stelly | The "21 Guns" song by Green Day is about a girl named 21 Gun | expected: Lisa Stelly, got: The "21 Guns" song by  |
| nq-f17349362327 | a normally inaccessible mini-game | The Hot Coffee Mod in San Andreas is a mod that adds a new v | expected: a normally inaccessible mini-game, got:  |
| nq-f37bafda2714 | Sir Edmund Barton | The first Australian Prime Minister, John McEwen, was electe | expected: Sir Edmund Barton, got: The first Austra |
| nq-f894e9812a1a | Michael Moriarty | The actor who played Ben Stone on Law & Order: Special Victi | expected: Michael Moriarty, got: The actor who pla |
| nq-26d4da08b47f | Fifty Shades of Grey | The three books in the "50 Shades" series are:

1. **The Cod | expected: Fifty Shades of Grey, got: The three boo |
| nq-27eee714ec33 | Anakin Skywalker | Darth Vader is under the mask of his former self, Obi-Wan Ke | expected: Anakin Skywalker, got: Darth Vader is un |
| nq-37ce106128e2 | Donna | Eric end up with Pam in "That 70s Show." | expected: Donna, got: Eric end up with Pam in "Tha |
| nq-617ba3b5f165 | 1976 | The Minnesota Vikings last appeared in the NFC Championship  | numeric mismatch 2018.0 vs 1976.0 |
| nq-844ea0cf827e | Gene Kelly | The song "Good Morning, Good Morning" was originally perform | expected: Gene Kelly, got: The song "Good Morning, |
| cx_18 | Gabriel Garcia Marquez | Gabriel García Márquez wrote the novel 'One Hundred Years of | expected: Gabriel Garcia Marquez, got: Gabriel Gar |

### logic Failures (10)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-028b0cd6 | 625 | Let's solve this step by step:

1. **The hundreds digit is 3 | expected: 625, got: Let's solve this step by step: |
| q-01b0f9f0 | gradient | I'm sorry, but I cannot provide a detailed explanation of ba | expected: gradient, got: I'm sorry, but I cannot p |
| zebra-0da648160cd9 | {'header': ['House', 'Name', 'Birthday'], 'rows': [['___', ' | To solve this logic puzzle, let's break it down step by step | expected: {'header': ['House', 'Name', 'Birthday'] |
| zebra-301797450ab9 | {'header': ['House', 'Name', 'CarModel'], 'rows': [['___', ' | To solve this logic puzzle, let's break it down step by step | expected: {'header': ['House', 'Name', 'CarModel'] |
| zebra-36f46151402f | {'header': ['House', 'Name', 'Education'], 'rows': [['___',  | To solve this logic puzzle, let's break it down step by step | expected: {'header': ['House', 'Name', 'Education' |
| zebra-4f13116c99a3 | {'header': ['House', 'Name', 'Food'], 'rows': [['___', '___' | To solve this logic puzzle, let's break it down step by step | expected: {'header': ['House', 'Name', 'Food'], 'r |
| zebra-645bc97e3272 | {'header': ['House', 'Name', 'HairColor'], 'rows': [['___',  | To solve this logic puzzle, let's break it down step by step | expected: {'header': ['House', 'Name', 'HairColor' |
| zebra-7a67a8316b67 | {'header': ['House', 'Name', 'CarModel'], 'rows': [['___', ' | To solve this logic puzzle, let's break it down step by step | expected: {'header': ['House', 'Name', 'CarModel'] |
| zebra-9e54c143c87a | {'header': ['House', 'Name', 'CarModel'], 'rows': [['___', ' | To solve this logic puzzle, let's break it down step by step | expected: {'header': ['House', 'Name', 'CarModel'] |
| zebra-b9bfb2b434d3 | {'header': ['House', 'Name', 'HouseStyle'], 'rows': [['___', | To solve this logic puzzle, let's break it down step by step | expected: {'header': ['House', 'Name', 'HouseStyle |

### math Failures (14)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-01a39292 | a) 9:2 | Let's solve this step by step:

1. **Initial amounts:**
   - | expected: a) 9:2, got: Let's solve this step by st |
| q-021bb356 | d) 624 | To solve this problem, we need to follow these steps:

1. ** | expected: d) 624, got: To solve this problem, we n |
| q-028c7df2 | a) 886 | To solve this problem, we need to find the smallest positive | expected: a) 886, got: To solve this problem, we n |
| q-01572b5c | b) ₹2,000 | Let's solve this step by step:

1. **Define the cost price ( | expected: b) ₹2,000, got: Let's solve this step by |
| q-0153a3ab | a) 8156 | To solve this problem, we need to consider the constraints:  | expected: a) 8156, got: To solve this problem, we  |
| q-01eff6bf | a) 8:19 | To solve this problem, we need to find the volume of the sma | expected: a) 8:19, got: To solve this problem, we  |
| gsm8k-28aec708c304 | 45 | Let's break down the problem step by step:

1. **First part  | expected: 45, got: Let's break down the problem st |
| gsm8k-6a26459b58de | 25000 | To solve this problem, we need to break it down into two par | expected: 25000, got: To solve this problem, we ne |
| gsm8k-78966163f186 | 104 | Let's break down the problem step by step:

1. **Monday:**
  | expected: 104, got: Let's break down the problem s |
| gsm8k-8c62b8200b4d | 60 | To solve this problem, we need to determine the square foota | expected: 60, got: To solve this problem, we need  |
| gsm8k-f3a142df6b12 | 17 | Step 1: Identify the total number of oranges.
The total numb | expected: 17, got: Step 1: Identify the total numb |
| gsm8k-4eee572fb01e | 64 | To solve this problem, we need to calculate the total cost o | expected: 64, got: To solve this problem, we need  |
| gsm8k-ed7a5c520f64 | 57500 | To solve this problem, we need to calculate Jill's annual sa | expected: 57500, got: To solve this problem, we ne |
| gsm8k-f4246d132cf3 | 25 | Step 1: Determine the total distance of the trip.
The total  | expected: 25, got: Step 1: Determine the total dis |

### ner Failures (8)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-01c2ef51 | PERSON: Maria Fernandez, Jensen Huang, Christopher Danely; O | PERSON: Maria Fernandez, Jensen Huang, Christopher Danely

O | expected: PERSON: Maria Fernandez, Jensen Huang, C |
| tweet7-33a701fafb8c | group: {@Cleveland Browns@}
event: SUPER BOWL
event: NFLPlay | - Cleveland Browns: ORG | expected: group: {@Cleveland Browns@}
event: SUPER |
| tweet7-656674d97e66 | person: {@Himanta Biswa Sarma@} | None | expected: person: {@Himanta Biswa Sarma@}, got: No |
| tweet7-6b549bc8bac1 | creative_work: Turning Up
group: {@ARASHI@}
corporation: {@M | None | expected: creative_work: Turning Up
group: {@ARASH |
| tweet7-73e661fcb028 | creative_work: sacred choral Christmas music
group: Choir of | - PERSON: me
- PERSON: {{USERNAME}}
- PERSON: He
- PERSON: { | expected: creative_work: sacred choral Christmas m |
| tweet7-89e0e404af48 | person: Dave Rogers
group: Black Lives Matter | - PERSON: Dave Rogers
- URL: {{URL}} | expected: person: Dave Rogers
group: Black Lives M |
| tweet7-e5efbf194515 | person: {@Aditi Rao Hydari@} | - Queen: PERSON | expected: person: {@Aditi Rao Hydari@}, got: - Que |
| wnut17-48701c713d02 | product: & gt ; * The soldier was killed when another avalan | - PERSON: soldier
- ORG: military
- LOC: Sonmarg
- DATE: 199 | expected: product: & gt ; * The soldier was killed |

### sentiment Failures (8)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-00d2fedf | negative (sarcastic) | neutral | expected: negative (sarcastic), got: neutral |
| q-030845d0 | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| q-057a2a3d | negative (dismissive) | positive | expected: negative (dismissive), got: positive |
| q-028a879b | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| q-006ecf56 | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| q-042f4f9c | negative (sarcastic) | positive | expected: negative (sarcastic), got: positive |
| q-01f895ee | neutral (hedging) | positive | expected: neutral (hedging), got: positive |
| sst2-3e5d4e066473 | Sentiment: NEGATIVE | neutral | expected: Sentiment: NEGATIVE, got: neutral |

### summarization Failures (28)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-032b94d3 | Apple framed its USB-C adoption as a voluntary innovation ch | Apple's USB-C adoption for iPhone 15 is driven by innovation | expected: Apple framed its USB-C adoption as a vol |
| q-0055cefb | COP28 produced the first-ever global agreement to transition | The UAE Consensus at COP28 includes language on transitionin | numeric mismatch 28.0 vs 2.7 |
| q-00091a0b | The NIF achieved the first fusion reaction with net energy g | On December 5, 2022, scientists at the National Ignition Fac | expected: The NIF achieved the first fusion reacti |
| q-00469917 | Artists allege Stability AI illegally trained on billions of | The plaintiffs claim that Stability AI's Stable Diffusion mo | expected: Artists allege Stability AI illegally tr |
| q-00370b39 | The diplomat expressed grave concern over military movements | The Federated States of my delegation views the recent devel | expected: The diplomat expressed grave concern ove |
| q-00aa9b8b | Lecanemab was approved for Alzheimer's with a headline calli | The FDA has granted accelerated approval to lecanemab, a new | expected: Lecanemab was approved for Alzheimer's w |
| q-02762ca0 | Boeing's 2024 safety report touts improved culture and trans | Boeing released a report titled 'Boeing's Commitment to Safe | expected: Boeing's 2024 safety report touts improv |
| q-016e7e7d | Meta claims improved election interference detection with 40 | The text discusses the efforts of Facebook to combat coordin | expected: Meta claims improved election interferen |
| q-02fbedf0 | The FTC seeks to block Microsoft's $69B Activision acquisiti | The FTC claims Microsoft's acquisition of Activision Blizzar | expected: The FTC seeks to block Microsoft's $69B  |
| xsum-1f0e69fa8897 | Former US President Bill Clinton is introduced to the McGuin | The text is dated March 23, 2017, and was last updated at 14 | expected: Former US President Bill Clinton is intr |
| xsum-260eec64733b | Peterborough United defender Miles Addison has signed a new  | Graham Westley's 26-year-old has made one appearance since j | expected: Peterborough United defender Miles Addis |
| xsum-4c2076d75e1c | A family of five, including three young children, had to be  | Two crews and a hovercraft from Weston-super-Mare rescued tw | expected: A family of five, including three young  |
| xsum-5ead3b4e019b | An artist's impression of how an expanded primary school in  | Surrey County Council plans to expand Sayes Court by 210 pla | expected: An artist's impression of how an expande |
| xsum-726724dafa9e | Cardiff Devils lost 6-1 away to Nottingham Panthers in the E | The Panthers secured a victory over the Devils, increasing t | expected: Cardiff Devils lost 6-1 away to Nottingh |
| xsum-75d704a754da | A 19-year-old man has been charged after a cat was killed an | A man named Franky Mills, who lives in Farncombe, was arrest | numeric mismatch 9.0 vs 19.0 |
| xsum-7756c474bf25 | A man is being held in Manchester on suspicion of terrorism  | Police arrested a 26-year-old suspect on suspicion of offenc | expected: A man is being held in Manchester on sus |
| xsum-83e2b8250a8c | Five people have been injured following a crash between a bi | A woman on a bus was taken to the hospital with a serious le | expected: Five people have been injured following  |
| xsum-862f366208a4 | The SNP has held a South Ayrshire Council ward following a b | The SNP's John Wallace won the Ayr East ward in a by-electio | expected: The SNP has held a South Ayrshire Counci |
| xsum-975d27e940d2 | A man has appeared in court over an acid attack that left tw | Two men, aged 23 and 24, were attacked with acid at 19:00 BS | expected: A man has appeared in court over an acid |
| xsum-bb6aee03a3db | Leicester City striker Gary Taylor-Fletcher has joined Sheff | A 33-year-old former Blackpool forward has featured only twi | expected: Leicester City striker Gary Taylor-Fletc |
| xsum-dccce4944b71 | A buzzard has been rescued after becoming trapped between fe | A buzzard became stuck in a wire fence while chasing a wood  | expected: A buzzard has been rescued after becomin |
| xsum-ef3d0b8ec1b2 | Keepers at Chester Zoo are making sure every creature, from  | The zoo is required to count and document thousands of endan | expected: Keepers at Chester Zoo are making sure e |
| xsum-00ac9888e2f5 | A man who sparked a drug-fuelled roof-top siege after breaki | A 27-year-old man barricaded himself in his girlfriend's fla | expected: A man who sparked a drug-fuelled roof-to |
| xsum-11a19a39f4d0 | These roads in Northern Ireland are closed due to poor weath | The text provides the road names of various locations in Nor | expected: These roads in Northern Ireland are clos |
| xsum-2054f538d8ea | Eastleigh have confirmed the loan signing of Portsmouth stri | The 31-year-old has joined the Spitfires on a deal until the | expected: Eastleigh have confirmed the loan signin |
| xsum-407b80861d2f | Children across Britain linked up with Tim Peake on the Inte | British astronaut speaks to children about experiments and s | expected: Children across Britain linked up with T |
| xsum-c5fc1996f68d | League Two side York City have signed Hull City midfielder M | A 21-year-old player made his debut for the Tigers in the Le | expected: League Two side York City have signed Hu |
| xsum-d824a5cab529 | Four hillwalkers who got into difficulties during severe wea | Two climbers were rescued by a coastguard helicopter after b | expected: Four hillwalkers who got into difficulti |
