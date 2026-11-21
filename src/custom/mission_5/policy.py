"""Load explicit team-trained policy artifacts; no untrained teacher fallback."""
import os
from pathlib import Path


def default_checkpoint():
    return Path(__file__).resolve().parent/'keyposes.json'


def load_policy(checkpoint,device='cpu'):
    checkpoint=Path(checkpoint)
    if checkpoint.suffix.lower()=='.json':
        from .keypose_policy import Policy
    else:
        from .bc.policy import Policy
    return Policy(checkpoint,device=device)


def create_policy():
    return load_policy(os.getenv('M5_CHECKPOINT',str(default_checkpoint())),
                       os.getenv('M5_DEVICE','cpu'))
