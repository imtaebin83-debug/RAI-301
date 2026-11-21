"""Export a compact inference checkpoint, retaining training provenance."""
import argparse
import hashlib
from pathlib import Path
import torch

from .policy import Policy


def export_checkpoint(source, destination):
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():
        raise ValueError('Export destination must differ from the training checkpoint')
    saved=torch.load(source,map_location='cpu',weights_only=True)
    # Validate architecture, whitelist and normalization before copying weights.
    Policy(source,'cpu')
    keys=('format','model','config','state_mean','state_std','fps','action_frame','input_keys',
          'manifest_sha256','epoch','best_val','collection_camera_metadata','official_task_success','max_batches')
    compact={key:saved[key] for key in keys if key in saved}
    compact['source_checkpoint_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary=destination.with_suffix(destination.suffix+'.tmp')
    torch.save(compact,temporary)
    Policy(temporary,'cpu')
    temporary.replace(destination)
    return compact['source_checkpoint_sha256']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(export_checkpoint(args.checkpoint,args.out))
