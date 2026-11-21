"""Copy collection sessions to a new directory, preserving provenance and originals."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .prepare import audit_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    sources = sorted({f.resolve() for folder in args.sessions for f in folder.glob("episode_*.h5")})
    if not sources:
        p.error("No episode HDF5 files found")
    # Validate all source episodes before creating the destination.
    for source in sources:
        audit_episode(source)
    args.out.mkdir(parents=True, exist_ok=False)
    rows, seen = [], set()
    for source in sources:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        target = args.out / f"episode_{len(rows):06d}.h5"
        shutil.copy2(source, target)
        rows.append({"file": target.name, "source": str(source), "sha256": digest})
    (args.out / "provenance.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"copied_episodes": len(rows), "out": str(args.out.resolve())}))


if __name__ == "__main__":
    main()
