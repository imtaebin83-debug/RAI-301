"""Bounded planner subprocess, based on the official Qwen2.5 model-card API.

Weights belong in custom/models/Qwen2.5-1.5B-Instruct. The parent process enforces
a 45 second wall-clock limit including model loading. This helper never downloads
models or uses hidden scenario data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def generate(payload: dict) -> str:
    default = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-1.5B-Instruct"
    model_name = Path(os.getenv("HAIC_LOCAL_PLANNER_MODEL", str(default))).expanduser()
    if not (model_name / "config.json").is_file():
        raise RuntimeError("local planner weights missing: " + str(model_name))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(str(model_name), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_name),
        torch_dtype=torch.float32, local_files_only=True).eval()
    messages = [{"role": "system", "content": payload["system"]},
                {"role": "user", "content": payload["instruction"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    print(generate(json.load(sys.stdin)))
