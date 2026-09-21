import argparse
from pathlib import Path
from typing import Optional, Sequence


def download_pretrained_weights(
    local_dir: str,
    hugging_face_api_key: str,
    repo_id: str,
) -> None:
    """
    Download pretrained weights from Hugging Face.
    """
    if __package__:
        from .huggingface_wrapper import HuggingFaceWrapper
    else:
        from huggingface_wrapper import HuggingFaceWrapper

    huggingface_wrapper = HuggingFaceWrapper(hugging_face_api_key)
    result_dir = huggingface_wrapper.get_model(repo_id, local_dir)
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
}


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
        try:
            download_pretrained_weights(
                repo_id=repo_id,
                local_dir=str(project_root / "weights" / local_name),
                hugging_face_api_key=hugging_face_api_key,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to download requested model '{model_name}' from '{repo_id}'. "
                "Verify that HUGGING_FACE_API_KEY is valid and approved for this "
                "gated repository."
            ) from error


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download selected DeepCORO-CLIP pretrained weights"
    )
    parser.add_argument(
        "--models",
        required=True,
        help=(
            "Comma-separated models to download. Supported values: "
            "stenosis, mace."
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

    hugging_face_api_key = read_api_key("api_key.json")["HUGGING_FACE_API_KEY"]
    project_root = Path(__file__).resolve().parent.parent
    download_selected_models(model_names, hugging_face_api_key, project_root)


if __name__ == "__main__":
    main()
