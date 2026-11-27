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

_MODEL = None


def generate(payload: dict) -> str:
    global _MODEL
    if _MODEL is None:
        _MODEL = LocalModel()
    return _MODEL.generate(payload)


class LocalModel:
    def __init__(self):
        default = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-1.5B-Instruct"
        model_name = Path(os.getenv("HAIC_LOCAL_PLANNER_MODEL", str(default))).expanduser()
        if not (model_name / "config.json").is_file():
            raise RuntimeError("local planner weights missing: " + str(model_name))
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_name), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(str(model_name),
            torch_dtype=torch.float32, local_files_only=True).eval()

    def generate(self, payload):
        import torch
        messages = [{"role": "system", "content": payload["system"]},
                    {"role": "user", "content": payload["instruction"]}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt")
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=320, do_sample=False,
                                        pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)[0].strip()


def serve():
    for line in sys.stdin:
        try:
            if len(line) > 32000:
                raise ValueError('planner request is too large')
            response = {'answer': generate(json.loads(line))}
        except Exception as error:
            response = {'error': f'{type(error).__name__}: {error}'}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ['--worker']:
        serve()
    else:
        print(generate(json.load(sys.stdin)))
