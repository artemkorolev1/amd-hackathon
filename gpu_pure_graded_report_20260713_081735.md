# GPU Pure Run — Graded Report

**Generated:** 2026-07-13 08:17:35
**Results:** `gpu_run_pure_results_detailed.json`
**Ground Truth:** `data/eval/factual_combined_80.json`

---

## Overall Accuracy

| Metric | Value |
|--------|-------|
| Total Questions | 58 |
| Correct | 30 |
| Incorrect | 28 |
| **Overall Accuracy** | **30/58 (51.7%)** |
| 84.2% Gate | ❌ FAILED |

## Per-Category Accuracy (by Model's Predicted Category)

| Category | Total | Correct | Accuracy |
|----------|-------|---------|----------|
| code_debug | 1 | 1 | 100.0% ✅ |
| code_gen | 1 | 0 | 0.0% ❌ |
| factual | 25 | 15 | 60.0% ❌ |
| logic | 15 | 4 | 26.7% ❌ |
| math | 8 | 4 | 50.0% ❌ |
| ner | 7 | 5 | 71.4% ❌ |
| sentiment | 1 | 1 | 100.0% ✅ |

## Per-Source Accuracy

| Source | Total | Correct | Accuracy |
|--------|-------|---------|----------|
| build-A-40 | 4 | 4 | 100.0% |
| build-B-40 | 4 | 4 | 100.0% |
| complexity_40 | 6 | 5 | 83.3% |
| dev_40 | 3 | 3 | 100.0% |
| eval_hard_218 | 13 | 7 | 53.8% |
| eval_mini_10 | 1 | 1 | 100.0% |
| heldout_40 | 2 | 2 | 100.0% |
| training-v3 | 19 | 4 | 21.1% |
| validation-v3 | 6 | 0 | 0.0% |

## Confusion Matrix (Predicted Category vs Correctness)

| Predicted Category | Total | Correct | Incorrect | Accuracy |
|--------------------|-------|---------|-----------|----------|
| code_debug | 1 | 1 | 0 | 100.0% |
| code_gen | 1 | 0 | 1 | 0.0% |
| factual | 25 | 15 | 10 | 60.0% |
| logic | 15 | 4 | 11 | 26.7% |
| math | 8 | 4 | 4 | 50.0% |
| ner | 7 | 5 | 2 | 71.4% |
| sentiment | 1 | 1 | 0 | 100.0% |

### Confusion: How Predicted Categories Relate to Ground-Truth Sources

| Task ID | Predicted Category | Ground Truth Source | Correct? |
|---------|-------------------|--------------------|----------|
| B-fact-1 | math | build-B-40 | ✅ |
| B-fact-2 | ner | build-B-40 | ✅ |
| B-fact-3 | factual | build-B-40 | ✅ |
| B-fact-4 | math | build-B-40 | ✅ |
| cx_04 | factual | complexity_40 | ✅ |
| cx_05 | factual | complexity_40 | ✅ |
| cx_18 | logic | complexity_40 | ❌ |
| cx_19 | factual | complexity_40 | ✅ |
| cx_27 | ner | complexity_40 | ✅ |
| cx_34 | factual | complexity_40 | ✅ |
| fact-1 | factual | dev_40 | ✅ |
| fact-2 | factual | dev_40 | ✅ |
| fact-4 | logic | dev_40 | ✅ |
| factual_25 | math | eval_hard_218 | ✅ |
| factual_26 | factual | eval_hard_218 | ✅ |
| factual_27 | factual | eval_hard_218 | ❌ |
| factual_28 | math | eval_hard_218 | ❌ |
| factual_29 | sentiment | eval_hard_218 | ✅ |
| factual_30 | factual | eval_hard_218 | ✅ |
| factual_31 | factual | eval_hard_218 | ✅ |
| factual_32 | math | eval_hard_218 | ❌ |
| factual_33 | code_debug | eval_hard_218 | ✅ |
| factual_34 | ner | eval_hard_218 | ❌ |
| factual_35 | math | eval_hard_218 | ❌ |
| factual_36 | ner | eval_hard_218 | ✅ |
| factual_37 | code_gen | eval_hard_218 | ❌ |
| nq-0c39ac5763fb | factual | training-v3 | ❌ |
| nq-0daf4903f2dd | factual | validation-v3 | ❌ |
| nq-1186bd67ee61 | factual | training-v3 | ❌ |
| nq-225aa775d644 | math | training-v3 | ❌ |
| nq-26d4da08b47f | factual | validation-v3 | ❌ |
| nq-27eee714ec33 | logic | validation-v3 | ❌ |
| nq-2a7776c7d218 | factual | training-v3 | ❌ |
| nq-2ad2fa80c023 | logic | training-v3 | ❌ |
| nq-2fab2b52f064 | factual | training-v3 | ❌ |
| nq-37ce106128e2 | logic | validation-v3 | ❌ |
| nq-3faaeadf61b6 | logic | training-v3 | ❌ |
| nq-60c012cff21c | logic | training-v3 | ✅ |
| nq-617ba3b5f165 | ner | validation-v3 | ❌ |
| nq-664b95f22952 | factual | training-v3 | ❌ |
| nq-77cee4f7ee4e | logic | training-v3 | ❌ |
| nq-844ea0cf827e | logic | validation-v3 | ❌ |
| nq-af387eb22a53 | logic | training-v3 | ❌ |
| nq-ba6a28152193 | ner | training-v3 | ✅ |
| nq-c8c977e94f10 | logic | training-v3 | ❌ |
| nq-cee5214ca17f | factual | training-v3 | ✅ |
| nq-dcc557178b19 | logic | training-v3 | ✅ |
| nq-de9ab68dac46 | logic | training-v3 | ❌ |
| nq-f17349362327 | factual | training-v3 | ❌ |
| nq-f37bafda2714 | factual | training-v3 | ❌ |
| nq-f894e9812a1a | logic | training-v3 | ❌ |
| tc8 | factual | eval_mini_10 | ✅ |
| xfact-101 | factual | build-A-40 | ✅ |
| xfact-102 | factual | build-A-40 | ✅ |
| xfact-103 | logic | build-A-40 | ✅ |
| xfact-104 | factual | build-A-40 | ✅ |
| xfact-11 | math | heldout_40 | ✅ |
| xfact-9 | ner | heldout_40 | ✅ |

## Failure Analysis

**Total Failures:** 28/58

### Failure by Predicted Category

| Category | Failures |
|----------|----------|
| code_gen | 1 |
| factual | 10 |
| logic | 11 |
| math | 4 |
| ner | 2 |

### Failure by Source

| Source | Failures |
|--------|----------|
| complexity_40 | 1 |
| eval_hard_218 | 6 |
| training-v3 | 15 |
| validation-v3 | 6 |

### Detailed Failure Patterns

- **cx_18** (pred_cat=logic, src=complexity_40)
  - Expected: `Gabriel Garcia Marquez`
  - Got: `The novel 'One Hundred Years of Solitude' was written by Gabriel García Márquez.`
  - Reason: expected: Gabriel Garcia Marquez, got: The novel 'One Hundred Years of Solitude' was written by Gabriel García Márquez.
- **factual_27** (pred_cat=factual, src=eval_hard_218)
  - Expected: `The rise of archosaurs (early dinosaur ancestors) to dominance in terrestrial ecosystems would likely not have occurred,`
  - Got: `If the Siberian Traps eruptions had released only one-tenth of their actual CO₂ output, the archosaurs (dinosaurs and cr`
  - Reason: expected: The rise of archosaurs (early dinosaur ancestors) to dominance in terrestrial ecosystems would likely not have occurred,, got: If the Siberian Traps eruptions had released only one-tenth of their actual CO₂ output, the archosaurs (dinosaurs and cr
- **factual_28** (pred_cat=math, src=eval_hard_218)
  - Expected: `They would not have detected the CMB because Earth's atmosphere is opaque at 1 mm (sub-millimeter radiation is absorbed `
  - Got: `0`
  - Reason: expected: They would not have detected the CMB because Earth's atmosphere is opaque at 1 mm (sub-millimeter radiation is absorbed , got: 0
- **factual_32** (pred_cat=math, src=eval_hard_218)
  - Expected: `The Amazon dieback would release massive amounts of stored carbon (150-200 billion tons) into the atmosphere through bio`
  - Got: `Reduced rainfall`
  - Reason: expected: The Amazon dieback would release massive amounts of stored carbon (150-200 billion tons) into the atmosphere through bio, got: Reduced rainfall
- **factual_34** (pred_cat=ner, src=eval_hard_218)
  - Expected: `Scholars would likely have access to definitive textual evidence to resolve the Homeric Question, because the Library of`
  - Got: `If the Library of Alexandria had survived intact through all its historical threats, our understanding of the Homeric Qu`
  - Reason: expected: Scholars would likely have access to definitive textual evidence to resolve the Homeric Question, because the Library of, got: If the Library of Alexandria had survived intact through all its historical threats, our understanding of the Homeric Qu
- **factual_35** (pred_cat=math, src=eval_hard_218)
  - Expected: `The concept of a single Mitochondrial Eve would not exist as understood, because biparental inheritance would allow reco`
  - Got: `0`
  - Reason: expected: The concept of a single Mitochondrial Eve would not exist as understood, because biparental inheritance would allow reco, got: 0
- **factual_37** (pred_cat=code_gen, src=eval_hard_218)
  - Expected: `NASA's Apollo program would almost certainly have continued beyond Apollo 17, with missions 18, 19, and 20 proceeding as`
  - Got: `The post-1969 trajectory would likely have been significantly more focused on international collaboration and less relia`
  - Reason: expected: NASA's Apollo program would almost certainly have continued beyond Apollo 17, with missions 18, 19, and 20 proceeding as, got: The post-1969 trajectory would likely have been significantly more focused on international collaboration and less relia
- **nq-0c39ac5763fb** (pred_cat=factual, src=training-v3)
  - Expected: `a biocidal effect of metals`
  - Got: `the ability of certain substances to kill or inhibit the growth of other organisms.`
  - Reason: expected: a biocidal effect of metals, got: the ability of certain substances to kill or inhibit the growth of other organisms.
- **nq-0daf4903f2dd** (pred_cat=factual, src=validation-v3)
  - Expected: `E-8s senior chief petty officer`
  - Got: `The US Navy ranks include: Vice Admiral, Admiral, Captain, Commander, Lieutenant Commander, Lieutenant, Midshipman.`
  - Reason: expected: E-8s senior chief petty officer, got: The US Navy ranks include: Vice Admiral, Admiral, Captain, Commander, Lieutenant Commander, Lieutenant, Midshipman.
- **nq-1186bd67ee61** (pred_cat=factual, src=training-v3)
  - Expected: `Chhattisgarh`
  - Got: `Uttar Pradesh is located in the center of India.`
  - Reason: expected: Chhattisgarh, got: Uttar Pradesh is located in the center of India.
- **nq-225aa775d644** (pred_cat=math, src=training-v3)
  - Expected: `one`
  - Got: `10`
  - Reason: expected: one, got: 10
- **nq-26d4da08b47f** (pred_cat=factual, src=validation-v3)
  - Expected: `Fifty Shades of Grey`
  - Got: `The three books in the "50 Shades" series are "The Code of Flesh," "The Silence of the Lambs," and "The Rule of None."`
  - Reason: expected: Fifty Shades of Grey, got: The three books in the "50 Shades" series are "The Code of Flesh," "The Silence of the Lambs," and "The Rule of None."
- **nq-27eee714ec33** (pred_cat=logic, src=validation-v3)
  - Expected: `Anakin Skywalker`
  - Got: `Darth Vader is under the mask of his former master, Darth Sidious. Sidious used Vader's body to manipulate events and ga`
  - Reason: expected: Anakin Skywalker, got: Darth Vader is under the mask of his former master, Darth Sidious. Sidious used Vader's body to manipulate events and ga
- **nq-2a7776c7d218** (pred_cat=factual, src=training-v3)
  - Expected: `New Orleans`
  - Got: `The Butchers in the Slaughterhouse Cases lived in the United States.`
  - Reason: expected: New Orleans, got: The Butchers in the Slaughterhouse Cases lived in the United States.
- **nq-2ad2fa80c023** (pred_cat=logic, src=training-v3)
  - Expected: `Inner Circle`
  - Got: `The theme song for the TV show "Cops" is sung by the American country music group The Oak Ridge Boys.`
  - Reason: expected: Inner Circle, got: The theme song for the TV show "Cops" is sung by the American country music group The Oak Ridge Boys.
- **nq-2fab2b52f064** (pred_cat=factual, src=training-v3)
  - Expected: `in the Gospel of Luke`
  - Got: `Lord's Prayer is found in the New Testament, specifically in Matthew 6:9-13.`
  - Reason: expected: in the Gospel of Luke, got: Lord's Prayer is found in the New Testament, specifically in Matthew 6:9-13.
- **nq-37ce106128e2** (pred_cat=logic, src=validation-v3)
  - Expected: `Donna`
  - Got: `In the 1970s sitcom "That '70s Show," Eric is the main character who is a high school student. The show revolves around `
  - Reason: expected: Donna, got: In the 1970s sitcom "That '70s Show," Eric is the main character who is a high school student. The show revolves around 
- **nq-3faaeadf61b6** (pred_cat=logic, src=training-v3)
  - Expected: `John Young`
  - Got: `It's a Long Long Way to Pasadena was written by Bob Dylan.`
  - Reason: expected: John Young, got: It's a Long Long Way to Pasadena was written by Bob Dylan.
- **nq-617ba3b5f165** (pred_cat=ner, src=validation-v3)
  - Expected: `1976`
  - Got: `The last time the Vikings were in the NFC Championship was in 2015.`
  - Reason: numeric mismatch 2015.0 vs 1976.0
- **nq-664b95f22952** (pred_cat=factual, src=training-v3)
  - Expected: `commercial at`
  - Got: `Yes, it is called the at symbol.`
  - Reason: expected: commercial at, got: Yes, it is called the at symbol.
- **nq-77cee4f7ee4e** (pred_cat=logic, src=training-v3)
  - Expected: `Mary Rose Foster`
  - Got: `In the musical "The Rose," Betty Midler portrayed the character of Rose.`
  - Reason: expected: Mary Rose Foster, got: In the musical "The Rose," Betty Midler portrayed the character of Rose.
- **nq-844ea0cf827e** (pred_cat=logic, src=validation-v3)
  - Expected: `Gene Kelly`
  - Got: `The song "Good Morning, Good Morning" was originally performed by the American country music duo The Statler Brothers. H`
  - Reason: expected: Gene Kelly, got: The song "Good Morning, Good Morning" was originally performed by the American country music duo The Statler Brothers. H
- **nq-af387eb22a53** (pred_cat=logic, src=training-v3)
  - Expected: `Max Shippee`
  - Got: `The character of Gram on The Young and the Restless is played by actress Kaitlyn Dever. She joined the cast in 2018 and `
  - Reason: expected: Max Shippee, got: The character of Gram on The Young and the Restless is played by actress Kaitlyn Dever. She joined the cast in 2018 and 
- **nq-c8c977e94f10** (pred_cat=logic, src=training-v3)
  - Expected: `Cathy Dennis and Rob Davis`
  - Got: `The song "Can't Get You Out of My Head" was written by the American singer-songwriter and musician, James Morrison.`
  - Reason: expected: Cathy Dennis and Rob Davis, got: The song "Can't Get You Out of My Head" was written by the American singer-songwriter and musician, James Morrison.
- **nq-de9ab68dac46** (pred_cat=logic, src=training-v3)
  - Expected: `Lisa Stelly`
  - Got: `The song "21 Guns" by Green Day features a female vocalist, but she is not named in the song lyrics. The song is perform`
  - Reason: expected: Lisa Stelly, got: The song "21 Guns" by Green Day features a female vocalist, but she is not named in the song lyrics. The song is perform
- **nq-f17349362327** (pred_cat=factual, src=training-v3)
  - Expected: `a normally inaccessible mini-game`
  - Got: `The Hot Coffee mod in San Andreas allows players to obtain coffee by shooting it from vending machines.`
  - Reason: expected: a normally inaccessible mini-game, got: The Hot Coffee mod in San Andreas allows players to obtain coffee by shooting it from vending machines.
- **nq-f37bafda2714** (pred_cat=factual, src=training-v3)
  - Expected: `Sir Edmund Barton`
  - Got: `The first Australian Prime Minister, John McEwen, was elected in 1901.`
  - Reason: expected: Sir Edmund Barton, got: The first Australian Prime Minister, John McEwen, was elected in 1901.
- **nq-f894e9812a1a** (pred_cat=logic, src=training-v3)
  - Expected: `Michael Moriarty`
  - Got: `The actor who played Ben Stone on Law & Order: Special Victims Unit was David Janssen. He portrayed the character for th`
  - Reason: expected: Michael Moriarty, got: The actor who played Ben Stone on Law & Order: Special Victims Unit was David Janssen. He portrayed the character for th

## Per-Question Detail

| # | Task ID | Pred Category | Source | Expected | Got | Correct | Reason |
|---|---------|--------------|--------|----------|-----|---------|--------|
| 1 | B-fact-1 | math | build-B-40 |  | Mercury, 88 Earth days | ✅ | Non-empty answer (no gold expected) |
| 2 | B-fact-2 | ner | build-B-40 |  | Tokyo and Nagasaki | ✅ | Non-empty answer (no gold expected) |
| 3 | B-fact-3 | factual | build-B-40 |  | Mercury (Hg) is highly valued in jewelry and electronics. | ✅ | Non-empty answer (no gold expected) |
| 4 | B-fact-4 | math | build-B-40 |  | Kilimandzhui, 5,895 meters | ✅ | Non-empty answer (no gold expected) |
| 5 | cx_04 | factual | complexity_40 | Jupiter | The largest planet in our solar system is Jupiter. | ✅ | Passed |
| 6 | cx_05 | factual | complexity_40 | Ottawa | The capital city of Canada is Ottawa. | ✅ | Passed |
| 7 | cx_18 | logic | complexity_40 | Gabriel Garcia Marquez | The novel 'One Hundred Years of Solitude' was written by Gabriel García Márquez. | ❌ | expected: Gabriel Garcia Marquez, got: T |
| 8 | cx_19 | factual | complexity_40 | Titan | Titan. | ✅ | Passed |
| 9 | cx_27 | ner | complexity_40 | 1989 | The Berlin Wall fell on November 9, 1989. | ✅ | Passed |
| 10 | cx_34 | factual | complexity_40 | Sputnik | Sputnik 1. | ✅ | Passed |
| 11 | fact-1 | factual | dev_40 |  | The capital city of Australia is Canberra. | ✅ | Non-empty answer (no gold expected) |
| 12 | fact-2 | factual | dev_40 |  | Iron. | ✅ | Non-empty answer (no gold expected) |
| 13 | fact-4 | logic | dev_40 |  | The novel "One Hundred Years of Solitude" was written by Gabriel García Márquez. | ✅ | Non-empty answer (no gold expected) |
| 14 | factual_25 | math | eval_hard_218 | Its radius (or physical diameter/size). The transit method measures the fraction | Minimum mass | ✅ | Passed |
| 15 | factual_26 | factual | eval_hard_218 | Mendeleev would have arranged elements by atomic number instead of atomic weight | If Moseley had conducted his X-ray spectroscopy experiments and published the at | ✅ | Passed |
| 16 | factual_27 | factual | eval_hard_218 | The rise of archosaurs (early dinosaur ancestors) to dominance in terrestrial ec | If the Siberian Traps eruptions had released only one-tenth of their actual CO₂  | ❌ | expected: The rise of archosaurs (early  |
| 17 | factual_28 | math | eval_hard_218 | They would not have detected the CMB because Earth's atmosphere is opaque at 1 m | 0 | ❌ | expected: They would not have detected t |
| 18 | factual_29 | sentiment | eval_hard_218 | Arabic translations during the Islamic Golden Age would have been less critical, | Less critical | ✅ | Passed |
| 19 | factual_30 | factual | eval_hard_218 | Proteins in the snailfish's cells would become overly rigid and malfunction (the | If a Mariana hadal snailfish were rapidly brought from its natural depth of 8,00 | ✅ | Passed |
| 20 | factual_31 | factual | eval_hard_218 | Even with the electron's charge measured, there would be no evidence that the el | If Millikan's oil drop experiment had been completed in 1896, the critical piece | ✅ | Passed |
| 21 | factual_32 | math | eval_hard_218 | The Amazon dieback would release massive amounts of stored carbon (150-200 billi | Reduced rainfall | ❌ | expected: The Amazon dieback would relea |
| 22 | factual_33 | code_debug | eval_hard_218 | Penicillin resistance would likely have emerged much later, possibly in the 1950 | Here's the bug identified and the fix:  **Bug:** The text states Fleming discove | ✅ | Passed |
| 23 | factual_34 | ner | eval_hard_218 | Scholars would likely have access to definitive textual evidence to resolve the  | If the Library of Alexandria had survived intact through all its historical thre | ❌ | expected: Scholars would likely have acc |
| 24 | factual_35 | math | eval_hard_218 | The concept of a single Mitochondrial Eve would not exist as understood, because | 0 | ❌ | expected: The concept of a single Mitoch |
| 25 | factual_36 | ner | eval_hard_218 | A fractured landscape of incompatible networks would have emerged, with no singl | If the ARPANET had been decommissioned in 1980 (before the TCP/IP transition), t | ✅ | Passed |
| 26 | factual_37 | code_gen | eval_hard_218 | NASA's Apollo program would almost certainly have continued beyond Apollo 17, wi | The post-1969 trajectory would likely have been significantly more focused on in | ❌ | expected: NASA's Apollo program would al |
| 27 | nq-0c39ac5763fb | factual | training-v3 | a biocidal effect of metals | the ability of certain substances to kill or inhibit the growth of other organis | ❌ | expected: a biocidal effect of metals, g |
| 28 | nq-0daf4903f2dd | factual | validation-v3 | E-8s senior chief petty officer | The US Navy ranks include: Vice Admiral, Admiral, Captain, Commander, Lieutenant | ❌ | expected: E-8s senior chief petty office |
| 29 | nq-1186bd67ee61 | factual | training-v3 | Chhattisgarh | Uttar Pradesh is located in the center of India. | ❌ | expected: Chhattisgarh, got: Uttar Prade |
| 30 | nq-225aa775d644 | math | training-v3 | one | 10 | ❌ | expected: one, got: 10 |
| 31 | nq-26d4da08b47f | factual | validation-v3 | Fifty Shades of Grey | The three books in the "50 Shades" series are "The Code of Flesh," "The Silence  | ❌ | expected: Fifty Shades of Grey, got: The |
| 32 | nq-27eee714ec33 | logic | validation-v3 | Anakin Skywalker | Darth Vader is under the mask of his former master, Darth Sidious. Sidious used  | ❌ | expected: Anakin Skywalker, got: Darth V |
| 33 | nq-2a7776c7d218 | factual | training-v3 | New Orleans | The Butchers in the Slaughterhouse Cases lived in the United States. | ❌ | expected: New Orleans, got: The Butchers |
| 34 | nq-2ad2fa80c023 | logic | training-v3 | Inner Circle | The theme song for the TV show "Cops" is sung by the American country music grou | ❌ | expected: Inner Circle, got: The theme s |
| 35 | nq-2fab2b52f064 | factual | training-v3 | in the Gospel of Luke | Lord's Prayer is found in the New Testament, specifically in Matthew 6:9-13. | ❌ | expected: in the Gospel of Luke, got: Lo |
| 36 | nq-37ce106128e2 | logic | validation-v3 | Donna | In the 1970s sitcom "That '70s Show," Eric is the main character who is a high s | ❌ | expected: Donna, got: In the 1970s sitco |
| 37 | nq-3faaeadf61b6 | logic | training-v3 | John Young | It's a Long Long Way to Pasadena was written by Bob Dylan. | ❌ | expected: John Young, got: It's a Long L |
| 38 | nq-60c012cff21c | logic | training-v3 | Jacob Tremblay | The character of Auggie in the movie "The Wonder" is played by Jacob Tremblay. | ✅ | Passed |
| 39 | nq-617ba3b5f165 | ner | validation-v3 | 1976 | The last time the Vikings were in the NFC Championship was in 2015. | ❌ | numeric mismatch 2015.0 vs 1976.0 |
| 40 | nq-664b95f22952 | factual | training-v3 | commercial at | Yes, it is called the at symbol. | ❌ | expected: commercial at, got: Yes, it is |
| 41 | nq-77cee4f7ee4e | logic | training-v3 | Mary Rose Foster | In the musical "The Rose," Betty Midler portrayed the character of Rose. | ❌ | expected: Mary Rose Foster, got: In the  |
| 42 | nq-844ea0cf827e | logic | validation-v3 | Gene Kelly | The song "Good Morning, Good Morning" was originally performed by the American c | ❌ | expected: Gene Kelly, got: The song "Goo |
| 43 | nq-af387eb22a53 | logic | training-v3 | Max Shippee | The character of Gram on The Young and the Restless is played by actress Kaitlyn | ❌ | expected: Max Shippee, got: The characte |
| 44 | nq-ba6a28152193 | ner | training-v3 | Exodus | The Ten Commandments are located in the first five books of the Bible, specifica | ✅ | Passed |
| 45 | nq-c8c977e94f10 | logic | training-v3 | Cathy Dennis and Rob Davis | The song "Can't Get You Out of My Head" was written by the American singer-songw | ❌ | expected: Cathy Dennis and Rob Davis, go |
| 46 | nq-cee5214ca17f | factual | training-v3 | December 1985 | The South Asian Association for Regional Cooperation (SAARC) was formed on Augus | ✅ | Passed |
| 47 | nq-dcc557178b19 | logic | training-v3 | James I | The ruler of England in 1616 was King James I of England. | ✅ | Passed |
| 48 | nq-de9ab68dac46 | logic | training-v3 | Lisa Stelly | The song "21 Guns" by Green Day features a female vocalist, but she is not named | ❌ | expected: Lisa Stelly, got: The song "21 |
| 49 | nq-f17349362327 | factual | training-v3 | a normally inaccessible mini-game | The Hot Coffee mod in San Andreas allows players to obtain coffee by shooting it | ❌ | expected: a normally inaccessible mini-g |
| 50 | nq-f37bafda2714 | factual | training-v3 | Sir Edmund Barton | The first Australian Prime Minister, John McEwen, was elected in 1901. | ❌ | expected: Sir Edmund Barton, got: The fi |
| 51 | nq-f894e9812a1a | logic | training-v3 | Michael Moriarty | The actor who played Ben Stone on Law & Order: Special Victims Unit was David Ja | ❌ | expected: Michael Moriarty, got: The act |
| 52 | tc8 | factual | eval_mini_10 | Paris | Paris. | ✅ | Passed |
| 53 | xfact-101 | factual | build-A-40 |  | The chemical symbol for liquid metal at room temperature is Ag. | ✅ | Non-empty answer (no gold expected) |
| 54 | xfact-102 | factual | build-A-40 |  | Mercury is the smallest planet in our solar system. | ✅ | Non-empty answer (no gold expected) |
| 55 | xfact-103 | logic | build-A-40 |  | The novel 'Brave New World' was written by Aldous Huxley. It was published in 19 | ✅ | Non-empty answer (no gold expected) |
| 56 | xfact-104 | factual | build-A-40 |  | The tallest mountain in Africa is Mount Kilimanjaro, which is approximately 5,89 | ✅ | Non-empty answer (no gold expected) |
| 57 | xfact-11 | math | heldout_40 |  | Mount Tambora | ✅ | Non-empty answer (no gold expected) |
| 58 | xfact-9 | ner | heldout_40 |  | The first person to lead an expedition that reached the South Pole was Sir Ernes | ✅ | Non-empty answer (no gold expected) |
