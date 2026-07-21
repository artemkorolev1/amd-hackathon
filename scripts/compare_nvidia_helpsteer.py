#!/usr/bin/env python3
"""Compare NVIDIA prompt-task-and-complexity-classifier against HelpSteer human labels."""
import sys, os, json, gzip, io, time, random

sys.path.insert(0, '/home/artem/dev/amd-hackathon')
os.chdir('/home/artem/dev/amd-hackathon')

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from safetensors.torch import load_file

print("Loading NVIDIA model...")

class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * mask, 1) / mask.sum(1).clamp(min=1e-9)

class MulticlassHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)
    def forward(self, x):
        return self.fc(x)

class NVIDIAClassifier(nn.Module):
    def __init__(self, target_sizes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
        self.target_sizes = target_sizes
        h = self.backbone.config.hidden_size
        self.heads = nn.ModuleList([MulticlassHead(h, sz) for sz in target_sizes.values()])
        for i in range(len(self.heads)):
            self.add_module(f"head_{i}", self.heads[i])
        self.pool = MeanPooling()
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        return [head(pooled) for head in self.heads]

# Load config directly (custom model, no model_type key for AutoConfig)
print("Loading model config...")
cache = "/home/artem/.cache/huggingface/hub/models--nvidia--prompt-task-and-complexity-classifier/snapshots"
snap = os.listdir(cache)[0]
config_path = os.path.join(cache, snap, "config.json")
with open(config_path) as f:
    config_dict = json.load(f)

config = config_dict  # plain dict, not AutoConfig
tokenizer = AutoTokenizer.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
model = NVIDIAClassifier(config["target_sizes"])

model.load_state_dict(load_file(os.path.join(cache, snap, "model.safetensors")), strict=False)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Load HelpSteer validation
print("\nLoading HelpSteer validation...")
with gzip.open('/home/artem/dev/amd-hackathon-shared/prompt_data/sources/helpsteer_validation.jsonl.gz') as f:
    items = [json.loads(line.decode('utf-8')) for line in f]
print(f"Loaded {len(items)} items")

# Sample balanced across complexity levels
random.seed(42)
cplx_groups = {}
for item in items:
    cplx_groups.setdefault(item['complexity'], []).append(item)
sampled = []
for c in sorted(cplx_groups.keys()):
    pool = cplx_groups[c]
    sampled.extend(random.sample(pool, min(40, len(pool))))
    print(f"  Level {c}: sampled {min(40, len(pool))} of {len(pool)}")

# Run inference
HEAD_NAMES = list(config["target_sizes"].keys())
weights_map = config["weights_map"]
divisor_map = config["divisor_map"]

results = []
batch_size = 16
t0 = time.time()

with torch.no_grad():
    for i in range(0, len(sampled), batch_size):
        batch = sampled[i:i+batch_size]
        texts = [item['prompt'] for item in batch]
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items() if k != 'token_type_ids'}
        logits_list = model(**inputs)
        
        for j, item in enumerate(batch):
            entry = {'human_complexity': item['complexity']}
            for h_idx, (h_name, logits) in enumerate(zip(HEAD_NAMES, logits_list)):
                probs = torch.softmax(logits, dim=-1)[j]
                pred = probs.argmax().item()
                entry[f"{h_name}"] = pred
                entry[f"{h_name}_conf"] = round(float(probs[pred]), 4)
                # Only compute weight for heads that are in weights_map
                if h_name in weights_map:
                    w = weights_map[h_name][pred]
                    d = divisor_map.get(h_name, 1)
                    entry[f"{h_name}_weight"] = w / d
                else:
                    entry[f"{h_name}_weight"] = 0.0
            # Sum only the weights from weighted heads
            weighted_heads = [h for h in HEAD_NAMES if h in weights_map]
            pred_cplx = sum(entry.get(f"{h}_weight", 0) for h in weighted_heads)
            entry['predicted_complexity'] = round(pred_cplx, 4)
            results.append(entry)

print(f"\nClassified {len(sampled)} items in {time.time()-t0:.0f}s")
out = '/home/artem/dev/amd-hackathon-shared/nvidia_helpsteer_comparison.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out}")

# ANALYSIS
correct = 0
cm = [[0]*5 for _ in range(5)]
for r in results:
    pc = r['predicted_complexity']
    pl = min(int(pc), 4) if pc >= 0 else 0
    actual = r['human_complexity']
    cm[actual][pl] += 1
    if actual == pl:
        correct += 1

print(f"\nAccuracy (exact match): {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
print("Confusion matrix (row=human, col=predicted):")
print(f"{'':>8s} {'0':>6s} {'1':>6s} {'2':>6s} {'3':>6s} {'4':>6s}")
for i in range(5):
    row = f"  {i} (n={sum(cm[i]):>3d})"
    for j in range(5):
        row += f" {cm[i][j]:>5d} "
    row += f" {cm[i][i]/max(sum(cm[i]),1)*100:.0f}%"
    print(row)

within1 = sum(1 for r in results if abs(r['human_complexity'] - min(int(r['predicted_complexity']),4)) <= 1)
print(f"Accuracy within ±1: {within1}/{len(results)} = {within1/len(results)*100:.1f}%")
