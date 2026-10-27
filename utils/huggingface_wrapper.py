"""Versioned Hugging Face downloads and atomic model-folder uploads."""

import argparse
import fnmatch
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Sequence

from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger(__name__)

# Released checkpoints verified through model_info; custom repositories remain configurable.
DEFAULT_REVISIONS = {
    "heartwise/deepcoro_clip_stenosis": "bb3f5e4785eed5f493d20a5406feed40cace84cf",
    "heartwise/deepcoro_clip_mace": "7be030d1c7b62e41f45a4dd35adff9dae21d884f",
}


class HuggingFaceWrapper:
    def __init__(self, hugging_face_api_key: str | None = None):
        # None delegates authentication to HF_TOKEN or the SDK's saved login.
        self.hugging_face_api_key = hugging_face_api_key
        self.api = HfApi(token=hugging_face_api_key)

    def get_model(
        self,
        repo_id: str,
        local_dir: str | Path | None = None,
        revision: str | None = None,
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        allow_patterns: str | Sequence[str] | None = None,
    ) -> str:
        """Resolve a snapshot, resuming incomplete downloads through the SDK.

        A directory's existence does not prove it contains a complete checkpoint.
        SDK metadata handles validation and reuse of already downloaded files.
        Pin ``revision`` to a commit hash for reproducible inference.
        """
        effective_revision = revision or DEFAULT_REVISIONS.get(repo_id) or "main"
        patterns = [allow_patterns] if isinstance(allow_patterns, str) else list(allow_patterns or [])
        # The SDK may return an existing local directory after a metadata/network
        # failure. Verify against Hub metadata online, or our completion manifest
        # offline, so a partial folder never becomes a successful download.
        if not local_files_only:
            info = self.api.model_info(repo_id, revision=effective_revision)
            expected = [item.rfilename for item in info.siblings
                        if not patterns or any(fnmatch.fnmatchcase(item.rfilename, pat) for pat in patterns)]
            if not expected:
                raise ValueError(f"No model files match the requested patterns in {repo_id}")
        resolved = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            cache_dir=cache_dir,
            revision=effective_revision,
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
            token=self.hugging_face_api_key,
        )
        root = Path(resolved)
        selector = json.dumps([repo_id, effective_revision, sorted(patterns)])
        manifest = root / ".cache" / "deepcoro" / (hashlib.sha256(selector.encode()).hexdigest() + ".json")
        if local_files_only:
            if not manifest.is_file():
                raise ValueError("Offline model folder has no completion manifest; run this download online first")
            completed = json.loads(manifest.read_text())
        else:
            missing = [name for name in expected if not (root / name).is_file()]
            if missing:
                raise FileNotFoundError(f"Incomplete snapshot for {repo_id}: {missing}")
            completed = {name: (root / name).stat().st_size for name in expected}
        if not completed or any(not (root / name).is_file() or (root / name).stat().st_size != size
                                for name, size in completed.items()):
            raise FileNotFoundError(f"Incomplete or changed model snapshot: {root}")
        if not local_files_only:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=manifest.parent, delete=False) as stream:
                json.dump(completed, stream, sort_keys=True)
                temporary = Path(stream.name)
            temporary.replace(manifest)
        return str(root)

    def upload_model(
        self,
        repo_id: str,
        local_dir: str | Path,
        commit_message: str = "Update model",
        *,
        private: bool = True,
        revision: str | None = None,
        allow_patterns: str | Sequence[str] | None = None,
    ):
        """Upload an explicitly selected model folder in one commit.

        New repositories are private by default; the privacy of an existing
        repository is left unchanged. SDK errors propagate to the caller.
        """
        folder = Path(local_dir)
        if not folder.is_dir():
            raise FileNotFoundError(f"Model folder does not exist: {folder}")
        if not any(path.is_file() for path in folder.rglob("*")):
            raise ValueError(f"Model folder is empty: {folder}")
        self.api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        return self.api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(folder),
            commit_message=commit_message,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=[".git/**", ".cache/**", "api_key*.json", ".env", ".env.*"],
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and cache a Hugging Face model")
    parser.add_argument("repo_id")
    parser.add_argument("--local-dir", default="./pretrained_models")
    parser.add_argument("--revision", help="Commit hash, tag, or branch")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--token", help="Prefer HF_TOKEN or saved SDK login")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    local_dir = Path(args.local_dir) / args.repo_id.replace("/", "_")
    result = HuggingFaceWrapper(args.token).get_model(
        args.repo_id, local_dir, revision=args.revision,
        cache_dir=args.cache_dir, local_files_only=args.offline,
    )
    logger.info("Model resolved to %s", result)


if __name__ == "__main__":
    main()
