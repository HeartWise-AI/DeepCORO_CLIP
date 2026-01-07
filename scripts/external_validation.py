import os
import cv2
import sys
import yaml
import json
import shutil
import logging
import pydicom
import subprocess

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

from utils.dicoms import process_dicom_video
from utils.external_validation_constants import (
    MODEL_MAPPING,
    BINARY_COLUMNS,
    REGRESSION_COLUMNS
)

from heartwise_statplots.utils import HuggingFaceWrapper
from heartwise_statplots.utils.api import load_api_keys
from heartwise_statplots.metrics import MetricsComputer, ClassificationMetrics

# Add Orion directory to the system path using pathlib
orion_path = Path(__file__).parent.parent / 'Orion'
sys.path.append(str(orion_path))

from orion.utils.video_training_and_eval import perform_inference


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


def get_model_weights(
    model_name: str,
    hugging_face_api_key: str,
    force_download: bool = False,
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
    if not os.path.exists(os.path.join("weights", model_name)) or force_download:
        return HuggingFaceWrapper.get_model(
            repo_id=f"heartwise/{model_name}",
            local_dir=os.path.join("weights", model_name),
            hugging_face_api_key=hugging_face_api_key
        )
        
    return os.path.join("weights", model_name)


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
    config = {}        
    config['output_dir'] = args['output_folder']
    config['model_path'] = args['model_path']
    config['data_filename'] = args['data_path']
    config['batch_size'] = args['batch_size']
    config['num_workers'] = args['num_workers']
    config['device'] = args['model_device']
    config.update(default_model_config)
    
    return config


def get_model_path(
    model: str, 
    hugging_face_api_key: str,
    force_download: bool = False,
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
    model_weights_path: str = get_model_weights(
        model_name=model, 
        hugging_face_api_key=hugging_face_api_key, 
        force_download=force_download
    )
    
    # Search recursively for .pt files in the model directory and subdirectories
    pt_file = None
    for root, dirs, files in os.walk(model_weights_path):
        for file in files:
            if file.endswith('.pt'):
                pt_file = os.path.join(root, file)
                break
        if pt_file:
            break
    
    if not pt_file:
        raise ValueError(f"No .pt file found in {model_weights_path} or its subdirectories")    
    
    return pt_file


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
    binary_columns: list[str]=BINARY_COLUMNS
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

    # Process regression columns
    stenosis_columns = [col for col in df.columns.tolist() if col.endswith('_stenosis')]
    for column in stenosis_columns:
        df[f'{column}_binary'] = (df[column] > 70).astype(int)

    # Rename columns EXAM_ID and StudyInstanceUID
    df.rename(columns={
        'ss_patient_id': 'Patient_ID',
        'ss_event_cath_id': 'StudyInstanceUID'
        }, inplace=True)

    # Identify and remove empty columns
    empty_columns = []
    for column in df.columns.tolist():
        value_counts = df[column].value_counts()
        if value_counts.empty:
            empty_columns.append(column)
            print(f"EMPTY COLUMN: {column}")
            print('--------------------------------')

    # Remove empty columns
    if empty_columns:
        print(f"\nRemoving {len(empty_columns)} empty columns: {empty_columns}")
        df = df.drop(columns=empty_columns)
        print(f"Dataframe shape after removing empty columns: {df.shape}")
    else:
        print("\nNo empty columns found.")
    
    return df


def process_single_dicom(args_tuple):
    """
    Wrapper function for process_dicom_video to be used with multiprocessing.
    
    Args:
        args_tuple: Tuple containing (dicom_filepath, output_path)
    
    Returns:
        Tuple: (dicom_filepath, avi_filepath, acquisition_time, acquisition_datetime, success)
    """
    # Import logger inside the function to avoid multiprocessing issues
    import logging
    logger = logging.getLogger(__name__)
    
    dicom_filepath, output_path = args_tuple
    try:
        avi_filepath, series_times = process_dicom_video(
            input_path=dicom_filepath, 
            output_path=output_path
        )
        success = avi_filepath is not None
        return dicom_filepath, avi_filepath, series_times, success
    except Exception as e:
        logger.warning(f"Failed to convert {dicom_filepath}: {e}")
        return dicom_filepath, None, None, False


def clean_vaso_info_dataframe(
    df_vaso_info: pd.DataFrame
)->pd.DataFrame:
    """
    Process the vaso info dataframe.
    """
    columns_to_discard = [col for col in df_vaso_info.columns.tolist() if 'pred_' in col]
    df_vaso_info = df_vaso_info.drop(columns=columns_to_discard)
    
    rename_columns_map = {
        'contrast_agent_class': 'contrast_agent',
        'main_structure_class': 'main_structure',
        'stent_presence_class': 'stent_presence',
        'dominance_class': 'dominance'
    }
    df_vaso_info = df_vaso_info.rename(columns=rename_columns_map)
    
    return df_vaso_info

def assign_procedure_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign procedure status based on PCI (stent) presence and timing.
    
    This function creates three mutually-exclusive status categories:
    - "PCI": Current procedure has stent placement
    - "POST_PCI": Current procedure is after a previous PCI in the same study/artery
    - "diagnostic": Diagnostic procedure with no previous PCI
    
    Args:
        df: Input DataFrame with stent_presence_class and related columns
    
    Returns:
        DataFrame with added 'status' column
    """
    logger.info("Assigning procedure status based on PCI timing...")
    
    df_copy = df.copy()
    
    # ── 1. Ensure the column exists up front ────────────────────────────────────────
    df_copy["status"] = "unknown"          # will be overwritten below

    # ── 2. Convenience flags ────────────────────────────────────────────────────────
    df_copy["is_pci"] = df_copy["stent_presence"].eq(1)

    # cumulative "has PCI already been seen *earlier* in this study AND artery?"
    group_cols = ["StudyInstanceUID"]
    df_copy["pci_seen_before"] = (
        df_copy
        .groupby(group_cols, sort=False)["is_pci"]
        .transform(lambda x: x.cumsum().shift(fill_value=0))
        .astype(bool)
    )

    # ── 3. Build the three mutually-exclusive conditions ───────────────────────────
    cond_pci        = df_copy["is_pci"]
    cond_post_pci   = (~cond_pci
                       & df_copy["pci_seen_before"]
                       & df_copy["contrast_agent"].eq(1))

    cond_diagnostic = ~cond_pci & ~df_copy["pci_seen_before"]

    # ── 4. Final assignment (vectorised) ───────────────────────────────────────────
    df_copy.loc[cond_pci,        "status"] = "PCI"
    df_copy.loc[cond_post_pci,   "status"] = "POST_PCI"
    df_copy.loc[cond_diagnostic, "status"] = "diagnostic"

    # ── 5. (Optional) tidy-up helper columns ───────────────────────────────────────
    df_copy.drop(columns=["is_pci", "pci_seen_before"], inplace=True)

    # Log status distribution
    status_counts = df_copy["status"].value_counts()
    logger.info(f"Status distribution: {status_counts.to_dict()}")
    
    return df_copy


def main(args: dict):
    try:     
        # Define tmp dir
        tmp_dir: Path = Path(args['tmp_dir'])
        print(f"tmp_dir: {tmp_dir}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocess dataset
        df = pd.read_csv(args['data_path'])
        df_preprocessed = preprocess_dataset(df)
        print(df_preprocessed.columns.tolist())
        # Add validation after preprocessing
        if df_preprocessed.empty:
            raise ValueError("No data remaining after preprocessing")

        if 'DICOMPath' not in df_preprocessed.columns:
            raise ValueError("DICOMPath column not found in dataframe")

        if not args['debug']:
            # Convert DICOM videos to AVI using multiprocessing
            logging.info("Check for DICOM files and convert to AVI")
            dicom_filepaths: list[str] = df_preprocessed['DICOMPath'].tolist()
            logger.info(f"{len(dicom_filepaths)} DICOM files found in input dataframe.")

            # Initialize conversion tracking variables
            converted_count: int = 0
            rows_to_discard: list[int] = []
            filepath_mapping: dict[str, str] = {}
            series_times_mapping: dict[str, str] = {}
            
            # Prepare arguments for multiprocessing
            process_args = [(dicom_filepath, str(tmp_dir)) for dicom_filepath in dicom_filepaths]
            
            # Process files in parallel using multiprocessing.Pool
            num_workers = min(mp.cpu_count(), len(dicom_filepaths), 8)
            logger.info(f"Using {num_workers} workers for parallel processing")
            
            with mp.Pool(processes=num_workers) as pool:
                # Use imap for progress tracking
                results = list(tqdm(
                    pool.imap(process_single_dicom, process_args),
                    total=len(process_args),
                    desc="Converting DICOM to AVI"
                ))
            
            # Process results
            for dicom_filepath, avi_filepath, series_times, success in results:
                if success and avi_filepath:
                    filepath_mapping[dicom_filepath] = avi_filepath
                    converted_count += 1
                    series_times_mapping[dicom_filepath] = series_times
                else:
                    # Find the index for failed conversions
                    mask = df_preprocessed['DICOMPath'] == dicom_filepath
                    failed_indices = df_preprocessed[mask].index.tolist()
                    rows_to_discard.extend(failed_indices)

            # Process results and clean up dataframe
            logger.info(f"Starting with {len(df_preprocessed)} rows")
            logger.info(f"Successfully converted {converted_count}/{len(dicom_filepaths)} files")
            logger.info(f"Failed conversions: {len(rows_to_discard)}")
            logger.info(f"Converted {converted_count} DICOM files to AVI format in {tmp_dir}")
            df_preprocessed['Split'] = 'inference' # Required to run Orion and VasoVision inference
            df_preprocessed['FileName'] = df_preprocessed['DICOMPath'].map(filepath_mapping)
            df_preprocessed['SeriesTimes'] = df_preprocessed['DICOMPath'].map(series_times_mapping)
            df_preprocessed.drop(rows_to_discard, inplace=True)
            df_preprocessed.reset_index(drop=True, inplace=True)
            logger.info(f"Discarded {len(rows_to_discard)} rows - new dataframe length: {len(df_preprocessed)}")

            # Save new input dataframe
            logger.info(f"Saving preprocessed dataframe to {tmp_dir / 'df_preprocessed.csv'}")
            df_preprocessed.to_csv(tmp_dir / "df_preprocessed.csv", index=False, sep='α')
            
        else:
            df_preprocessed = pd.read_csv(tmp_dir / "df_preprocessed.csv", sep='α')

        args['data_path'] = tmp_dir / "df_preprocessed.csv"
        
        # Download model weights
        # Load API key
        api_keys: dict = load_api_keys(args['hugging_face_api_key_path'])
        hugging_face_api_key: str = api_keys.get('HUGGING_FACE_API_KEY')
        if not hugging_face_api_key:
            logger.error("HUGGING_FACE_API_KEY not found in the provided API keys file.")
            raise KeyError("HUGGING_FACE_API_KEY not found.")
        
        # Initialize VasoVision
        vaso_vision_hugging_face_model_name: str = MODEL_MAPPING['vaso_vision']['hugging_face_model_name']
        args['model_path'] = get_model_path(
            model=vaso_vision_hugging_face_model_name, 
            hugging_face_api_key=hugging_face_api_key,
            force_download=False
        )
        
        # Setup orion config for VisionVision
        vaso_vision_orion_config: dict = setup_orion_config(
            args=args,
            default_model_config=MODEL_MAPPING['vaso_vision']['config']
        )        
        
        if not args['debug']:
            df_vaso_info: pd.DataFrame = perform_inference(
                config=vaso_vision_orion_config,
                split='inference',
                log_wandb=False
            )
        
            df_vaso_info.rename(
                columns={'filename': 'FileName'}, 
                inplace=True
            )
        
            # Save vaso info dataframe
            df_vaso_info.to_csv(tmp_dir / "df_vaso_info.csv", index=False, sep='α')
            
        else:
            df_vaso_info = pd.read_csv(tmp_dir / "df_vaso_info.csv", sep='α')

        
        # process vaso info dataframe
        df_vaso_info = clean_vaso_info_dataframe(df_vaso_info)

        df_preprocessed = pd.merge(
            df_preprocessed, df_vaso_info, on='FileName', how='left'
        )
        print(df_preprocessed.columns.tolist())
        
        # Get procedure status
        df_preprocessed = df_preprocessed.sort_values(['StudyInstanceUID', 'SeriesTimes']) # Sort by study instance uid and series times first

        # Keep rows only if main_structure is right or left dominant
        df_preprocessed = df_preprocessed[
            (
                (df_preprocessed['main_structure'].astype(int) == vaso_vision_orion_config['labels_map']['main_structure']['Right Coronary']) | 
                (df_preprocessed['main_structure'].astype(int) == vaso_vision_orion_config['labels_map']['main_structure']['Left Coronary'])
            )        
        ]

        # Assign procedure status
        df_preprocessed = assign_procedure_status(df=df_preprocessed)

        # Keep rows only if contrast agent detected and status is diagnostic
        df_preprocessed = df_preprocessed[
            (
                df_preprocessed['contrast_agent'] == vaso_vision_orion_config['labels_map']['contrast_agent']['yes']
            ) & (
                df_preprocessed['status'] == 'diagnostic'
            )
        ]

        # Add split column
        df_preprocessed['Split'] = 'val'

        # Save dataframe
        df_preprocessed.to_csv(tmp_dir / "df_preprocessed_filtered.csv", index=False, sep='α')
        
           
        # Initialize deepcoro_clip
        deepcoro_clip_hugging_face_model_name: str = MODEL_MAPPING['deepcoro_clip']['hugging_face_model_name']
        args['model_path'] = get_model_path(
            model=deepcoro_clip_hugging_face_model_name, 
            hugging_face_api_key=hugging_face_api_key,
            force_download=False
        )
        
        print(f"model_path: {args['model_path']}")
                        
        # Run bash command to run deepcoro_clip inference
        bash_command = f"bash scripts/runner.sh --use_wandb false --base_config config/linear_probing/stenosis/docker_base_config.yaml --run_mode val --selected_gpus 0"
        logger.info(f"Executing deepcoro_clip inference command: {bash_command}")
        try:
            result = subprocess.run(bash_command, shell=True, check=True, capture_output=True, text=True)
            logger.info("deepcoro_clip inference completed successfully")
            if result.stdout:
                logger.info(f"Command output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"deepcoro_clip inference failed with return code {e.returncode}")
            if e.stdout:
                logger.error(f"Command stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"Command stderr: {e.stderr}")
            raise RuntimeError(f"deepcoro_clip inference failed: {e}") from e

    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")
        sys.exit(1)     


if __name__ == "__main__":
    args = {
        'hugging_face_api_key_path': 'api_key.json',
        'data_path': 'data/data_cto_calcif_stn_thr_docker_cleaned.csv',
        'output_folder': 'results',
        'tmp_dir': '/app/tmp',
        'model_path': '',
        'batch_size': 12,
        'num_workers': 12,
        'model_device': 'cuda',
        'debug': False
    }
    main(args)
