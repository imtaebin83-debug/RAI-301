"""Reproduce the exact CPU planner/VLM weights used in validation."""
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = [
    ('Qwen/Qwen2.5-1.5B-Instruct','989aa7980e4cf806f80c7fef2b1adb7bc71aa306'),
    ('HuggingFaceTB/SmolVLM-500M-Instruct','753c1752602b5e739af7a7fa6947a635051897a3'),
]


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'models'
    for repo,revision in MODELS:
        snapshot_download(repo_id=repo,revision=revision,local_dir=root/repo.split('/')[-1],
            allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','README.md','LICENSE*'])
