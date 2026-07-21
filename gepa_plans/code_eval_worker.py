#!/usr/bin/env python3
"""Worker: loads a GGUF model and runs inference on a batch of prompts."""
import sys, json, os, time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    model_path = sys.argv[1]
    tasks_path = sys.argv[2]

    with open(tasks_path) as f:
        tasks = json.load(f)

    from llama_cpp import Llama
    llm = Llama(
        model_path=model_path,
        n_ctx=1024,
        n_gpu_layers=-1,
        verbose=False,
        seed=42,
    )

    results = []
    for t in tasks:
        try:
            output = llm(
                t["prompt"],
                max_tokens=512,
                temperature=0.0,
                stop=["\n\n", "```\n\n"],
                echo=False,
            )
            generated = output["choices"][0]["text"].strip()
        except Exception as e:
            generated = f"__ERROR__:{e}"

        results.append({
            "generated": generated,
        })

    # Print only the JSON result
    print(json.dumps(results))

if __name__ == "__main__":
    main()
