#!/usr/bin/env python3
"""Build combined summarization eval set."""
import json

with open("data/eval/training-v3.json") as f:
    train = json.load(f)
with open("data/eval/validation-v3.json") as f:
    valid = json.load(f)

summ = ([q for q in train if q.get("category") == "summarization"]
        + [q for q in valid if q.get("category") == "summarization"])

print(f"Summarization eval set: {len([q for q in train if q.get('category')=='summarization'])} train + "
      f"{len([q for q in valid if q.get('category')=='summarization'])} valid = {len(summ)} questions")

with open("data/eval/summarization_combined_25.json", "w") as f:
    json.dump(summ, f, indent=2)
print("Saved to data/eval/summarization_combined_25.json")
