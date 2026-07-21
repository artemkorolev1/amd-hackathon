#!/usr/bin/env python3
"""
Generate the Colab notebook for QLoRA fine-tuning via Unsloth.
"""
import json, os

# Convert training data to Qwen chat format and create zip
import zipfile, io

DATA_DIR = os.path.expanduser("~/dev/amd-hackathon/lora_data")
OUTPUT_DIR = os.path.expanduser("~/dev/amd-hackathon/")

def to_chat_format(prompt: str, response: str) -> str:
    """Format as Qwen2.5 chat template."""
    return json.dumps({
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    })

# Create zip
zip_path = os.path.join(OUTPUT_DIR, "lora_training_data.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        cat = fname.replace(".jsonl", "")
        # Convert to chat format
        items = []
        for line in open(os.path.join(DATA_DIR, fname)):
            item = json.loads(line)
            items.append(to_chat_format(item["prompt"], item["response"]))
        chat_data = "\n".join(items)
        zf.writestr(f"{cat}.jsonl", chat_data.encode("utf-8"))
    
    # Also include raw JSONL for reference
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".jsonl"):
            zf.write(os.path.join(DATA_DIR, fname), f"raw/{fname}")

print(f"Created {zip_path} ({os.path.getsize(zip_path)/1024:.0f} KB)")

# Now generate the Colab notebook
NOTEBOOK = r"""{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {"id": "title"},
   "source": [
    "# QLoRA Fine-Tuning: 8 Adapters for AMD Hackathon\n",
    "\n",
    "**Model:** Qwen2.5-1.5B-Instruct  \n",
    "**Method:** QLoRA via Unsloth  \n",
    "**GPU:** T4 (16GB VRAM) — included in Colab free tier\n",
    "\n",
    "Trains 8 separate LoRA adapters for: factual, math, sentiment, ner, summarization, code_gen, code_debug, logic"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install dependencies\n",
    "!pip install unsloth datasets transformers accelerate -q\n",
    "!pip install bitsandbytes -q"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Mount Google Drive to save adapters\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "ADAPTER_DIR = '/content/drive/MyDrive/amd_hackathon_adapters/'\n",
    "!mkdir -p $ADAPTER_DIR"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Upload training data zip\n",
    "from google.colab import files\n",
    "print('Upload lora_training_data.zip from your local machine...')\n",
    "uploaded = files.upload()\n",
    "!unzip -q lora_training_data.zip -d /content/training_data/\n",
    "!ls /content/training_data/*.jsonl | head -20"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load Base Model (Qwen2.5-1.5B-Instruct)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from unsloth import FastLanguageModel\n",
    "from datasets import Dataset\n",
    "from trl import SFTTrainer\n",
    "from transformers import TrainingArguments\n",
    "import json, os, gc\n",
    "\n",
    "MAX_SEQ_LENGTH = 1024\n",
    "MODEL_NAME = \"Qwen/Qwen2.5-1.5B-Instruct\"\n",
    "\n",
    "CATEGORIES = [\n",
    "    ('factual', 1),\n",
    "    ('math', 2),\n",
    "    ('sentiment', 1),\n",
    "    ('ner', 2),\n",
    "    ('summarization', 2),\n",
    "    ('code_gen', 2),\n",
    "    ('code_debug', 2),\n",
    "    ('logic', 2),\n",
    "]\n",
    "# (name, epochs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_chat_data(path):\n",
    "    \"\"\"Load chat-format JSONL.\"\"\"\n",
    "    with open(path) as f:\n",
    "        return [json.loads(line) for line in f]\n",
    "\n",
    "def formatting_func(example):\n",
    "    \"\"\"Format chat messages for training.\"\"\"\n",
    "    messages = example['messages']\n",
    "    text = tokenizer.apply_chat_template(messages, tokenize=False)\n",
    "    return {'text': text}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_adapter(category, epochs=2):\n",
    "    \"\"\"Train one LoRA adapter and save to Drive.\"\"\"\n",
    "    global model, tokenizer\n",
    "    \n",
    "    print(f\"\\n{'='*60}\")\n",
    "    print(f\"Training adapter for: {category}\")\n",
    "    print(f\"{'='*60}\")\n",
    "    \n",
    "    # Load data\n",
    "    data_path = f'/content/training_data/{category}.jsonl'\n",
    "    if not os.path.exists(data_path):\n",
    "        print(f\"Data not found: {data_path}\")\n",
    "        return False\n",
    "    \n",
    "    raw_data = load_chat_data(data_path)\n",
    "    print(f\"Loaded {len(raw_data)} training examples\")\n",
    "    \n",
    "    # Create dataset\n",
    "    dataset = Dataset.from_list(raw_data)\n",
    "    dataset = dataset.map(formatting_func)\n",
    "    \n",
    "    # Set up training\n",
    "    trainer = SFTTrainer(\n",
    "        model=model,\n",
    "        tokenizer=tokenizer,\n",
    "        train_dataset=dataset,\n",
    "        dataset_text_field='text',\n",
    "        max_seq_length=MAX_SEQ_LENGTH,\n",
    "        args=TrainingArguments(\n",
    "            per_device_train_batch_size=4,\n",
    "            gradient_accumulation_steps=4,\n",
    "            num_train_epochs=epochs,\n",
    "            learning_rate=2e-4,\n",
    "            fp16=not torch.cuda.is_bf16_supported(),\n",
    "            bf16=torch.cuda.is_bf16_supported(),\n",
    "            logging_steps=10,\n",
    "            output_dir=f'/content/checkpoints_{category}',\n",
    "            save_strategy='no',\n",
    "            report_to='none',\n",
    "            optim='adamw_8bit',\n",
    "            weight_decay=0.01,\n",
    "            lr_scheduler_type='linear',\n",
    "            warmup_steps=10,\n",
    "        ),\n",
    "    )\n",
    "    \n",
    "    # Train\n",
    "    trainer.train()\n",
    "    \n",
    "    # Save adapter\n",
    "    adapter_path = os.path.join(ADAPTER_DIR, category)\n",
    "    model.save_pretrained(adapter_path)\n",
    "    tokenizer.save_pretrained(adapter_path)\n",
    "    print(f\"Adapter saved to: {adapter_path}\")\n",
    "    \n",
    "    # Cleanup\n",
    "    del trainer\n",
    "    gc.collect()\n",
    "    torch.cuda.empty_cache()\n",
    "    return True"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Initialize Model & Train All Adapters\n",
    "\n",
    "The model is loaded ONCE. For each adapter, we apply LoRA weights and train independently.\n",
    "**Important:** The adapter weights accumulate — each category trains on top of the previous one.\n",
    "This is intentional: each adapter specializes for its task. If you want truly independent adapters, reload the base model between categories (see alternative below)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load base model in 4-bit\n",
    "print(\"Loading Qwen2.5-1.5B-Instruct...\")\n",
    "model, tokenizer = FastLanguageModel.from_pretrained(\n",
    "    model_name=MODEL_NAME,\n",
    "    max_seq_length=MAX_SEQ_LENGTH,\n",
    "    dtype=None,\n",
    "    load_in_4bit=True,\n",
    "    device_map='auto',\n",
    ")\n",
    "print(\"Model loaded.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Apply LoRA configuration (same for all adapters)\n",
    "model = FastLanguageModel.get_peft_model(\n",
    "    model,\n",
    "    r=16,\n",
    "    target_modules=[\n",
    "        'q_proj', 'k_proj', 'v_proj', 'o_proj',\n",
    "        'gate_proj', 'up_proj', 'down_proj',\n",
    "    ],\n",
    "    lora_alpha=16,\n",
    "    lora_dropout=0,\n",
    "    bias='none',\n",
    "    use_gradient_checkpointing='unsloth',\n",
    "    random_state=42,\n",
    "    max_seq_length=MAX_SEQ_LENGTH,\n",
    ")\n",
    "print(f\"Trainable params: {model.num_parameters(only_trainable=True):,}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train all adapters sequentially\n",
    "for category, epochs in CATEGORIES:\n",
    "    train_adapter(category, epochs)\n",
    "    \n",
    "print(f\"\\nAll adapters saved to: {ADAPTER_DIR}\")\n",
    "print(f\"Contents: {os.listdir(ADAPTER_DIR)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Alternative: Independent Adapters (Reload Base Between Categories)\n",
    "\n",
    "If you want each adapter to be truly independent (not building on previous fine-tuning), reload the base model for each category. This takes longer but produces independent adapters.\n",
    "\n",
    "Uncomment and run this cell instead of the one above."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# OPTIONAL: Independent adapters (reload base model each time)\n",
    "# for category, epochs in CATEGORIES:\n",
    "#     print(f\"Loading base model for {category}...\")\n",
    "#     model, tokenizer = FastLanguageModel.from_pretrained(\n",
    "#         model_name=MODEL_NAME,\n",
    "#         max_seq_length=MAX_SEQ_LENGTH,\n",
    "#         dtype=None,\n",
    "#         load_in_4bit=True,\n",
    "#         device_map='auto',\n",
    "#     )\n",
    "#     model = FastLanguageModel.get_peft_model(\n",
    "#         model, r=16, target_modules=[...],\n",
    "#         lora_alpha=16, lora_dropout=0, bias='none',\n",
    "#         use_gradient_checkpointing='unsloth', random_state=42,\n",
    "#         max_seq_length=MAX_SEQ_LENGTH,\n",
    "#     )\n",
    "#     train_adapter(category, epochs)\n",
    "#     # Cleanup\n",
    "#     del model, tokenizer\n",
    "#     gc.collect()\n",
    "#     torch.cuda.empty_cache()\n",
    "#     print()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Verify Saved Adapters\n",
    "\n",
    "Check that all adapters were saved correctly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!echo \"=== Saved Adapters ===\"\n",
    "!ls -la $ADAPTER_DIR\n",
    "!echo \"\"\n",
    "for cat, _ in CATEGORIES:\n",
    "    path = f'{ADAPTER_DIR}{cat}'\n",
    "    if cat == 'factual':\n",
    "        !echo \"Sample: $path\"\n",
    "        !ls -la $path | head -5"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Quick Test: Generate with Adapter\n",
    "\n",
    "Test one adapter by generating a response."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def test_adapter(category, prompt_text):\n",
    "    \"\"\"Run inference with a specific adapter.\"\"\"\n",
    "    print(f\"Testing adapter: {category}\")\n",
    "    print(f\"Prompt: {prompt_text[:100]}...\")\n",
    "    \n",
    "    # Load fresh base + adapter\n",
    "    test_model, test_tokenizer = FastLanguageModel.from_pretrained(\n",
    "        model_name=MODEL_NAME,\n",
    "        max_seq_length=MAX_SEQ_LENGTH,\n",
    "        dtype=None,\n",
    "        load_in_4bit=True,\n",
    "    )\n",
    "    \n",
    "    # Load adapter\n",
    "    from peft import PeftModel\n",
    "    adapter_path = f'{ADAPTER_DIR}{category}'\n",
    "    test_model = PeftModel.from_pretrained(test_model, adapter_path)\n",
    "    \n",
    "    # Format prompt\n",
    "    messages = [{\"role\": \"user\", \"content\": prompt_text}]\n",
    "    prompt = test_tokenizer.apply_chat_template(messages, tokenize=False)\n",
    "    \n",
    "    # Generate\n",
    "    inputs = test_tokenizer([prompt], return_tensors='pt').to('cuda')\n",
    "    outputs = test_model.generate(\n",
    "        **inputs,\n",
    "        max_new_tokens=256,\n",
    "        temperature=0.1,\n",
    "        do_sample=True,\n",
    "    )\n",
    "    result = test_tokenizer.decode(outputs[0], skip_special_tokens=True)\n",
    "    # Extract assistant reply\n",
    "    if 'assistant' in result:\n",
    "        result = result.split('assistant')[-1].strip()\n",
    "    print(f\"Output: {result[:200]}\")\n",
    "    return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test a few adapters\n",
    "test_adapter('sentiment', 'The movie was terrible but the acting was decent')\n",
    "print()\n",
    "test_adapter('math', 'Natalia sold 48 clips. She sold half as many in May. How many total?')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Optional: Merge Adapter into Base Model\n",
    "\n",
    "Use this if you want to convert the adapter into a standalone model (no PEFT dependency at inference)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# OPTIONAL: Merge adapter into base model and save as full model\n",
    "# category = 'sentiment'\n",
    "# merged_model = model.merge_and_unload()\n",
    "# merge_path = f'{ADAPTER_DIR}merged_{category}'\n",
    "# merged_model.save_pretrained(merge_path)\n",
    "# tokenizer.save_pretrained(merge_path)\n",
    "# print(f'Merged model saved to: {merge_path}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Done! All 8 LoRA adapters trained and saved to Google Drive.\")\n",
    "print(f\"\\nNext steps:\")\n",
    "print(f\"1. Download adapters from {ADAPTER_DIR}\")\n",
    "print(f\"2. Integrate into pipeline: pick adapter based on S2 category\")\n",
    "print(f\"3. Test with full eval pipeline\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
"""

# Write the notebook
nb_path = os.path.join(OUTPUT_DIR, "unsloth_lora_training.ipynb")
with open(nb_path, "w") as f:
    f.write(NOTEBOOK)

print(f"Created {nb_path}")
print(f"Created {zip_path}")
print()
print("=== NEXT STEPS ===")
print(f"1. Copy {zip_path} to your local machine")
print(f"2. Upload to Google Colab via the notebook's 'Upload data' cell")
print(f"3. Run the notebook on a T4 runtime (Edit → Notebook settings → T4 GPU)")
print(f"4. Download adapters from Google Drive after training")
print(f"5. Integrate into pipeline: adapter = selected from the 8 based on S2 category")
