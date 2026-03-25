from pathlib import Path

from files_handler import read_api_key
from huggingface_wrapper import HuggingFaceWrapper


def download_pretrained_weights(
    local_dir: str, 
    hugging_face_api_key: str,
    repo_id: str
) -> None:
    """
    Download pretrained weights from Hugging Face.
    """
    huggingface_wrapper = HuggingFaceWrapper(hugging_face_api_key)
    result_dir = huggingface_wrapper.get_model(repo_id, local_dir)
    print(f"✓ Successfully downloaded model to: {result_dir}")


if __name__ == "__main__":
    hugging_face_api_key = read_api_key("api_key.json")["HUGGING_FACE_API_KEY"]
    project_root = Path(__file__).resolve().parent.parent
    download_pretrained_weights(
        repo_id="heartwise/deepcoro_clip_stenosis", 
        local_dir=str(project_root / "weights" / "deepcoro_clip_generic"),
        hugging_face_api_key=hugging_face_api_key
    )
