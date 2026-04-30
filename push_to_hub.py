"""Upload the local FallacyHunter RoBERTa checkpoint to the Hugging Face Hub.

Usage:
    python push_to_hub.py --repo-id <username>/<model-name>

Authentication is read from the HF_TOKEN environment variable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


DEFAULT_MODEL_DIRS = [Path("backend/models/roberta-classifier"), Path("fallacy_model")]


def load_dotenv_token() -> str | None:
    for env_path in (Path(".env"), Path(__file__).resolve().with_name(".env")):
        if not env_path.is_file():
            continue

        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
                continue

            key, value = stripped_line.split("=", 1)
            if key.strip() != "HF_TOKEN":
                continue

            return value.strip().strip('"').strip("'")

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the FallacyHunter RoBERTa classifier to the Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face repository in the form <namespace>/<repo_name>",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIRS[0],
        help="Local directory containing the model files",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repository as private if it does not already exist",
    )
    return parser.parse_args()


def require_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = load_dotenv_token()
    if not token:
        raise SystemExit("HF_TOKEN is not set. Export HF_TOKEN or add it to .env before running this script.")
    return token


def main() -> int:
    args = parse_args()
    token = require_token()

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        for candidate_dir in DEFAULT_MODEL_DIRS:
            resolved_candidate = candidate_dir.resolve()
            if resolved_candidate.is_dir():
                model_dir = resolved_candidate
                break
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {model_dir}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, private=args.private, token=token, exist_ok=True)

    upload_folder(
        repo_id=args.repo_id,
        folder_path=str(model_dir),
        path_in_repo="",
        token=token,
        commit_message="Upload FallacyHunter RoBERTa classifier",
    )

    print(f"Uploaded {model_dir} to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())