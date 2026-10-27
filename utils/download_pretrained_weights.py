import argparse
import hashlib
import os
from pathlib import Path
from typing import Optional, Sequence


def download_pretrained_weights(
    local_dir: str,
    hugging_face_api_key: str,
    repo_id: str,
    revision: Optional[str] = None,
) -> None:
    """
    Download pretrained weights from Hugging Face, optionally at a pinned revision.
    """
    if __package__:
        from .huggingface_wrapper import HuggingFaceWrapper
    else:
        from huggingface_wrapper import HuggingFaceWrapper

    huggingface_wrapper = HuggingFaceWrapper(hugging_face_api_key)
    result_dir = huggingface_wrapper.get_model(repo_id, local_dir, revision=revision)
    print(f"✓ Successfully downloaded model to: {result_dir}")


PRETRAINED_MODELS = {
    # build selector -> (repo_id, local folder under <project_root>/weights)
    "stenosis": (
        "heartwise/deepcoro_clip_stenosis",
        "deepcoro_clip_generic",
    ),
    "mace": (
        "heartwise/deepcoro_clip_mace",
        "deepcoro_clip_mace",
    ),
    "cathef": (
        "heartwise/CathEF_CLIP",
        "CathEF_CLIP",
    ),
    "deeprv": (
        "heartwise/DeepRV_CLIP",
        "DeepRV_CLIP",
    ),
}

# Immutable HF commit + checkpoint sha256 for models whose outputs were verified against a
# reference run. The build fails if the repo content differs, instead of silently loading a
# different checkpoint (torch.load(weights_only=False) runs code stored in the file).
PINNED_MODELS = {
    "cathef": {
        "revision": "df9165e77566dd2419981ac6b8ca9dffc077f391",
        "sha256": {"best_model_epoch_43.pt": "006b03dd912a56d15648d58c089f9d891a6b64341ca90b44314f159d4ba85c19"},
    },
    "deeprv": {
        "revision": "2d9a4847982bac5bdd140bcc552cf9fbec465552",
        "sha256": {"best_model_epoch_4.pt": "9e0c9dc43b0554b16483de2968c43be5e6954b3142cb9d28e397e524cd4d373d"},
    },
}


def verify_checksums(local_dir: Path, expected: dict[str, str]) -> None:
    """Raise if any expected file is missing or its sha256 differs."""
    for relative_path, expected_sha in expected.items():
        path = Path(local_dir) / relative_path
        if not path.is_file():
            raise RuntimeError(f"Expected checkpoint {path} is missing after download")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha:
            raise RuntimeError(
                f"Checksum mismatch for {path}: expected {expected_sha}, got {digest.hexdigest()}"
            )


def parse_model_selection(value: str) -> list[str]:
    """Parse and validate a comma-separated model selection."""
    selected_models = []
    for raw_name in value.split(","):
        model_name = raw_name.strip().lower()
        if not model_name:
            raise ValueError("Model selection cannot contain an empty value")
        if model_name not in PRETRAINED_MODELS:
            supported = ", ".join(PRETRAINED_MODELS)
            raise ValueError(
                f"Unknown model '{model_name}'. Supported models: {supported}"
            )
        if model_name not in selected_models:
            selected_models.append(model_name)
    return selected_models


def download_selected_models(
    model_names: Sequence[str],
    hugging_face_api_key: str,
    project_root: Path,
) -> None:
    """Download only the explicitly selected pretrained models."""
    for model_name in model_names:
        repo_id, local_name = PRETRAINED_MODELS[model_name]
        pinned = PINNED_MODELS.get(model_name, {})
        local_dir = project_root / "weights" / local_name
        try:
            download_pretrained_weights(
                repo_id=repo_id,
                local_dir=str(local_dir),
                hugging_face_api_key=hugging_face_api_key,
                revision=pinned.get("revision"),
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to download requested model '{model_name}' from '{repo_id}'. "
                "Verify that HUGGING_FACE_API_KEY is valid and approved for this "
                "gated repository."
            ) from error
        if pinned.get("sha256"):
            verify_checksums(local_dir, pinned["sha256"])


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download selected DeepCORO-CLIP pretrained weights"
    )
    parser.add_argument(
        "--models",
        required=True,
        help=(
            "Comma-separated models to download. Supported values: "
            "stenosis, mace, cathef, deeprv."
        ),
    )
    args = parser.parse_args(argv)

    try:
        model_names = parse_model_selection(args.models)
    except ValueError as error:
        parser.error(str(error))

    if __package__:
        from .files_handler import read_api_key
    else:
        from files_handler import read_api_key

    try:
        hugging_face_api_key = read_api_key("api_key.json").get("HUGGING_FACE_API_KEY")
    except FileNotFoundError:
        # The Hub SDK also honors HF_TOKEN and the saved hf auth login.
        hugging_face_api_key = os.environ.get("HUGGING_FACE_API_KEY")
    project_root = Path(__file__).resolve().parent.parent
    download_selected_models(model_names, hugging_face_api_key, project_root)


if __name__ == "__main__":
    main()
