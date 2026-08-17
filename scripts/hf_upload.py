"""Sync large binary artifacts to the HF dataset grainsilo/digital-minds-sprint.

The git repo holds code/JSON/findings; the HF dataset holds the big tensors,
laid out as lenses/{model}/{j,r}lens.pt and emotion-vectors/{model}/
{vectors,manifold}.pt. This script diffs local artifacts against the repo tree
and uploads only what's missing (pass --force to re-upload everything).
Resume checkpoints (ckpt_*.pt) and activation caches are deliberately not
synced — they're regenerable scratch (see p1/.gitignore for rebuild notes).

Auth: a write token in $HF_HOME/token (or `hf auth login`).
Usage: uv run --project p1 python scripts/hf_upload.py [--dry] [--force]
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "grainsilo/digital-minds-sprint"
MODELS = ["llama31-8b", "qwen25-7b", "qwen3-4b", "qwen25-32b"]


def mapping():
    out = []
    for m in MODELS:
        for lens in ("jlens", "rlens"):
            out.append((ROOT / "lenses" / "results" / m / f"{lens}.pt",
                        f"lenses/{m}/{lens}.pt"))
        for art in ("vectors", "manifold"):
            out.append((ROOT / "p1" / "results" / "stage2" / m / f"{art}.pt",
                        f"emotion-vectors/{m}/{art}.pt"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    have = set(api.list_repo_files(REPO, repo_type="dataset"))
    for local, remote in mapping():
        if not local.exists():
            print(f"skip (no local): {remote}")
            continue
        if remote in have and not args.force:
            print(f"skip (uploaded): {remote}")
            continue
        print(f"upload: {remote} ({local.stat().st_size / 1e6:.0f} MB)")
        if not args.dry:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                            repo_id=REPO, repo_type="dataset",
                            commit_message=f"add {remote}")


if __name__ == "__main__":
    main()
