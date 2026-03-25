#!/usr/bin/env python3
"""Download heartwise/VasoVision (or another gated HF model) using api_key.json."""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def load_key(path: str = "api_key.json") -> str:
    with open(path) as f:
        data = json.load(f)
    key = data.get("HUGGING_FACE_API_KEY")
    if not key:
        raise KeyError(f"HUGGING_FACE_API_KEY not found in {path}")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description="Download gated HuggingFace model using api_key.json")
    parser.add_argument(
        "--repo",
        default="heartwise/VasoVision",
        help="Repo ID (default: heartwise/VasoVision)",
    )
    parser.add_argument(
        "--local-dir",
        default="weights/VasoVision",
        help="Local directory to save the model (default: weights/VasoVision)",
    )
    parser.add_argument(
        "--api-key-path",
        default="api_key.json",
        help="Path to JSON file with HUGGING_FACE_API_KEY",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local_dir already exists",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    api_path = root / args.api_key_path
    local_dir = root / args.local_dir

    if not api_path.exists():
        print(f"Error: {api_path} not found", file=sys.stderr)
        sys.exit(1)

    def can_use(p: Path) -> bool:
        if p.exists():
            test = p / ".write_test"
            try:
                test.write_text("")
                test.unlink()
                return True
            except OSError:
                return False
        try:
            p.mkdir(parents=True, exist_ok=True)
            (p / ".write_test").write_text("")
            (p / ".write_test").unlink()
            return True
        except OSError:
            return False

    if not can_use(local_dir):
        fallback = root / "downloaded_VasoVision"
        if can_use(fallback):
            print(
                f"Cannot write to {local_dir} (e.g. Docker left weights/ owned by root).\n"
                f"Downloading to {fallback} instead.",
                file=sys.stderr,
            )
            local_dir = fallback
        else:
            print(
                f"Error: Cannot write to {local_dir}.\n"
                f"Fix ownership then re-run:\n  sudo chown -R $(whoami) {local_dir.parent}",
                file=sys.stderr,
            )
            sys.exit(1)

    token = load_key(str(api_path))
    used_fallback = local_dir != (root / args.local_dir)
    print(f"Downloading {args.repo} to {local_dir}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(local_dir),
        repo_type="model",
        token=token,
        force_download=args.force,
    )
    print(f"Done. Model saved under {local_dir}")
    if used_fallback:
        print(
            "To use with external_validation, fix ownership then move:\n"
            "  sudo chown -R $(whoami) weights\n"
            "  mkdir -p weights/VasoVision && mv downloaded_VasoVision/* weights/VasoVision/"
        )


if __name__ == "__main__":
    main()
