#!/usr/bin/env python3
"""Script to download models and tokenizers from HuggingFace Hub.

This script provides a convenient way to pre-download HuggingFace models
and tokenizers for offline use or to ensure they are cached locally.
"""

import argparse
import os
import sys
from pathlib import Path
import logging
from huggingface_hub import snapshot_download, HfApi

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HuggingFaceWrapper:
    def __init__(self, hugging_face_api_key):
        self.hugging_face_api_key = hugging_face_api_key
        self.api = HfApi()

    @staticmethod
    def get_model(repo_id, local_dir, hugging_face_api_key):           
        # Download repo from HuggingFace
        print(f"Checking if {repo_id} already exists in {local_dir}")
        if os.path.exists(local_dir):
            print(f"{repo_id} already exists in {local_dir}")
            return local_dir
        
        print(f"{local_dir} does not exist, creating it")
        os.makedirs(local_dir, exist_ok=True)
        print(f"Downloading {repo_id} to {local_dir}")
        local_dir = snapshot_download(
            repo_id=repo_id, 
            local_dir=local_dir, 
            repo_type="model", 
            token=hugging_face_api_key
        )
        
        print(f"{repo_id} downloaded to {local_dir}")

        return local_dir

    def upload_model(self, repo_id, local_dir, commit_message="Update model"):
        """
        Upload a model to Hugging Face.
        
        :param repo_id: The ID of the repository (e.g., 'username/repo-name')
        :param local_dir: The local directory containing the model files
        :param commit_message: The commit message for this update
        """
        try:
            # Ensure the repository exists (create if it doesn't)
            try:
                self.api.create_repo(repo_id=repo_id, token=self.hugging_face_api_key, exist_ok=True)
            except Exception as e:
                print(f"Note: {e}")

            # Upload all files in the directory
            for root, _, files in os.walk(local_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    repo_path = os.path.relpath(file_path, local_dir)
                    
                    self.api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        token=self.hugging_face_api_key
                    )
                    print(f"Uploaded {file} to {repo_id}")

            # Create a commit with all the uploaded files
            self.api.create_commit(
                repo_id=repo_id,
                operations="push",
                commit_message=commit_message,
                token=self.hugging_face_api_key
            )
            
            print(f"Successfully uploaded and committed model to {repo_id}")
        except Exception as e:
            print(f"An error occurred while uploading the model: {e}")


def main():
    """Main function to handle command line arguments and execute downloads."""
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models using snapshot_download",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download the heartwise model
  python utils/download_hf_model.py heartwise/deepcoro_clip_stenosis
  
  # Download to specific directory
  python utils/download_hf_model.py --local-dir ./pretrained_models heartwise/deepcoro_clip_stenosis
  
  # Download with token
  python utils/download_hf_model.py --token YOUR_TOKEN heartwise/deepcoro_clip_stenosis
        """
    )
    
    parser.add_argument(
        "repo_id",
        help="Repository ID to download from HuggingFace Hub (e.g., heartwise/deepcoro_clip_stenosis)"
    )
    
    parser.add_argument(
        "--local-dir",
        type=str,
        default="./pretrained_models",
        help="Local directory to download the model to (default: ./pretrained_models)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token for private models"
    )
    
    args = parser.parse_args()
    
    try:
        # Create local directory path
        model_name = args.repo_id.replace("/", "_")
        local_dir = os.path.join(args.local_dir, model_name)
        
        logger.info(f"Downloading {args.repo_id} to {local_dir}")
        
        # Download the model
        result_dir = HuggingFaceWrapper.get_model(
            repo_id=args.repo_id,
            local_dir=local_dir,
            hugging_face_api_key=args.token
        )
        
        logger.info(f"✓ Successfully downloaded model to: {result_dir}")
        
    except Exception as e:
        logger.error(f"✗ Failed to download model: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()