import os
import cv2
import sys
import yaml
import json
import shutil
import logging
import pydicom
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import (
    Future, 
    ProcessPoolExecutor, 
    as_completed
)

from tqdm import tqdm
from pathlib import Path
from typing import Optional

from utils.dicoms import convert_dicom_to_avi
from utils.external_validation_constants import (
    BINARY_COLUMNS,
    REGRESSION_COLUMNS
)

from heartwise_statplots.utils import HuggingFaceWrapper
from heartwise_statplots.utils.api import load_api_keys
from heartwise_statplots.metrics import MetricsComputer, ClassificationMetrics



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)


# Add the parent directory to the system path using pathlib
orion_path = Path(__file__).parent / 'Orion'
sys.path.append(str(orion_path))
from orion.utils.video_training_and_eval import perform_inference


def get_model_weights(
    model_name: str, 
    hugging_face_api_key: str
)->str:
    """
    Retrieves the local path to the specified model's weights from Hugging Face.

    This function uses the `HuggingFaceWrapper` to download the model weights
    from the Hugging Face repository and stores them in a designated local directory.

    Args:
        model_name (str): The name of the model to retrieve weights for.
        hugging_face_api_key (str): The API key for authenticating with Hugging Face.

    Returns:
        str: The local file system path where the model weights are stored.
    """    
    return HuggingFaceWrapper.get_model(
        repo_id=f"heartwise/{model_name}",
        local_dir=os.path.join("weights", model_name),
        hugging_face_api_key=hugging_face_api_key
    )


def setup_orion_config(
    args: dict, 
    default_model_config: dict[str, str]
)->dict:
    """
    Sets up the Orion configuration for a given model.

    Args:
        args (HearWiseArgs): The command-line arguments.
        default_model_config (dict): The default model configuration.

    Returns:
        dict: The updated Orion configuration.
    """
    try:
        with open(args.config_path) as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {args.config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise            
    config['output_dir'] = args.output_folder
    config['model_path'] = args.model_path
    config['data_filename'] = args.data_path
    config['batch_size'] = args.batch_size
    config['num_workers'] = args.num_workers
    config['device'] = args.model_device
    config.update(default_model_config)
    return config


def get_model_path(
    model: str, 
    hugging_face_api_key: str
)->str:
    """
    Retrieves the local path to the specified model's weights from Hugging Face.

    Args:
        model (str): The name of the model to retrieve weights for.
        hugging_face_api_key (str): The API key for authenticating with Hugging Face.

    Returns:
        str: The local file system path where the model weights are stored.
    """
    # Get model weights
    model_weights_path: str = get_model_weights(model, hugging_face_api_key)
    pt_file: str = next((f for f in os.listdir(model_weights_path) if f.endswith('.pt')), None)
    if not pt_file:
        raise ValueError("No .pt file found in the directory")    
    return os.path.join(model_weights_path, pt_file)


def compute_metrics(df_predictions_inference: pd.DataFrame)->dict:
    """
    Computes classification metrics (AUC, AUPRC, F1 Score) based on predictions and true labels.

    Args:
        df_predictions_inference (pd.DataFrame): DataFrame containing 'y_hat' for predictions and 'y_true' for true labels.

    Returns:
        dict: A dictionary containing the computed metrics.
    """    
    y_pred: np.ndarray = df_predictions_inference['y_hat'].to_numpy().astype(np.float64)
    y_true: np.ndarray = df_predictions_inference['y_true'].to_numpy().astype(np.int64)
    metrics = MetricsComputer.compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        metrics=[
            ClassificationMetrics.AUC, 
            ClassificationMetrics.AUPRC,
            ClassificationMetrics.SENSITIVITY,
            ClassificationMetrics.SPECIFICITY,
            ClassificationMetrics.F1_SCORE
        ],
        bootstrap=True,
        n_iterations=1000
    )
    return metrics

def preprocess_dataset(
    df: pd.DataFrame,
    binary_columns: list[str]=BINARY_COLUMNS,
)->pd.DataFrame:
    """
    Preprocesses the dataset for the Orion model.

    Args:
        df (pd.DataFrame): The input dataframe.
        binary_columns (list[str]): The binary columns to process.
        regression_columns (list[str]): The regression columns to process.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    # Process binary columns
    for column in binary_columns:
        if column in df.columns:
            if column.endswith('_calcif_binary'):
                # For calcification columns: replace "none" with 0, others with 1
                df[column] = df[column].map({'none': 0, 'mild': 1, 'moderate': 1, 'severe': 1})
            elif column.endswith('_cto') or column.endswith('_thrombus'):
                # For CTO and thrombus columns: replace False with 0, True with 1
                df[column] = df[column].map({False: 0, True: 1})

    # Identify and remove empty columns
    empty_columns = []
    for column in df.columns.tolist():
        value_counts = df[column].value_counts()
        if value_counts.empty:
            empty_columns.append(column)
            print(f"EMPTY COLUMN: {column}")
        else:
            print(f"{column}:")
            print(value_counts)
        print("--------------------------------")

    # Remove empty columns
    if empty_columns:
        print(f"\nRemoving {len(empty_columns)} empty columns: {empty_columns}")
        df = df.drop(columns=empty_columns)
        print(f"Dataframe shape after removing empty columns: {df.shape}")
    else:
        print("\nNo empty columns found.")

    for column in df.columns.tolist():
        print(df[column].value_counts())
        print("--------------------------------")
    
    df['Split'] = 'val'
    
    return df


def main(args: dict):
    try:     
        # Define tmp dir
        tmp_dir: Path = Path(args['tmp_dir'])
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Preprocess dataset
        df = pd.read_csv(args.data_path)
        df_preprocessed = preprocess_dataset(df)

        # Add validation after preprocessing
        if df_preprocessed.empty:
            raise ValueError("No data remaining after preprocessing")

        if 'DICOMPath' not in df_preprocessed.columns:
            raise ValueError("DICOMPath column not found in dataframe")

        # Keep only columns of interest
        # columns_to_keep = ['DICOMPath', 'CathReport_MRN', 'StudyInstanceUID'] + BINARY_COLUMNS + REGRESSION_COLUMNS
        # df_preprocessed = df_preprocessed[columns_to_keep]
        # use debug dataframe for now
        df_preprocessed = pd.read_csv('data/debug_wDicomPath_val.csv')


        # Convert DICOM videos to AVI
        logging.info("Check for DICOM files and convert to AVI")
        dicom_filepaths: list[str] = df_preprocessed['DICOMPath'].tolist()
        logger.info(f"{len(dicom_filepaths)} DICOM files found in input dataframe.")

        # Initialize conversion tracking variables
        converted_count: int = 0
        rows_to_discard: list[int] = []
        filepath_mapping: dict[str, str] = {}
        # Process files sequentially
        for dicom_filepath in tqdm(dicom_filepaths, desc="Converting DICOM to AVI"):
            try:
                avi_filepath: str = convert_dicom_to_avi(input_path=dicom_filepath, output_path=str(tmp_dir))
                if avi_filepath:
                    filepath_mapping[dicom_filepath] = avi_filepath
                    converted_count += 1
                else:
                    # More efficient way to find the index
                    mask = df_preprocessed['DICOMPath'] == dicom_filepath
                    failed_indices = df_preprocessed[mask].index.tolist()
                    rows_to_discard.extend(failed_indices)
            except Exception as e:
                logger.warning(f"Failed to convert {dicom_filepath}: {e}")
                mask = df_preprocessed['DICOMPath'] == dicom_filepath
                failed_indices = df_preprocessed[mask].index.tolist()
                rows_to_discard.extend(failed_indices)


        # Process results and clean up dataframe
        logger.info(f"Starting with {len(df_preprocessed)} rows")
        logger.info(f"Successfully converted {converted_count}/{len(dicom_filepaths)} files")
        logger.info(f"Failed conversions: {len(rows_to_discard)}")
        logger.info(f"Converted {converted_count} DICOM files to AVI format in {tmp_dir}")
        df_preprocessed['FileName'] = df_preprocessed['DICOMPath'].map(filepath_mapping)
        df_preprocessed.drop(rows_to_discard, inplace=True)
        df_preprocessed.reset_index(drop=True, inplace=True)
        logger.info(f"Discarded {len(rows_to_discard)} rows - new dataframe length: {len(df_preprocessed)}")

        # Save new input dataframe
        logger.info(f"Saving preprocessed dataframe to {tmp_dir / 'df_preprocessed.csv'}")
        df_preprocessed.to_csv(tmp_dir / "df_preprocessed.csv", index=False, sep='α')
        args.data_path_processed = tmp_dir / "df_preprocessed.csv"
        
        # Download model weights
        # Load API key
        api_keys: dict = load_api_keys(args.hugging_face_api_key_path)
        hugging_face_api_key: str = api_keys.get('HUGGING_FACE_API_KEY')
        if not hugging_face_api_key:
            logger.error("HUGGING_FACE_API_KEY not found in the provided API keys file.")
            raise KeyError("HUGGING_FACE_API_KEY not found.")
        model_path = get_model_path(
            model='orion-deeprv', 
            hugging_face_api_key=hugging_face_api_key
        )
        args.model_path = model_path
        
        # Remove tmp dir
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    except Exception as e:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        logger.exception(f"An error occurred during execution: {e}")
        sys.exit(1)     















if __name__ == "__main__":
    args = {
        'hugging_face_api_key_path': 'api_key.yaml',
        'data_path': 'data/concatenated_final_20250703_181738.csv',
        'output_folder': 'results',
        'tmp_dir': '/app/tmp',  # Make this configurable
        'model_path': '',
        'batch_size': 1,
        'num_workers': 1,
        'model_device': 'cuda'
    }
    main(args)