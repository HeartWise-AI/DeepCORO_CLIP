import os
import re
import sys
import yaml
import shutil
import logging
import subprocess

import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from typing import Optional

from utils.dicoms import process_dicom_video
from utils.external_validation_constants import (
    MODEL_MAPPING,
    BINARY_COLUMNS
)
from utils.orion_runtime_patches import apply_orion_runtime_patches

from heartwise_statplots.metrics import MetricsComputer, ClassificationMetrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
orion_path = PROJECT_ROOT / 'Orion'
sys.path.append(str(orion_path))

import orion.utils.video_training_and_eval as orion_video_training_and_eval

apply_orion_runtime_patches()


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

DOCKER_BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "linear_probing" / "stenosis" / "docker_base_config.yaml"


def resolve_output_dir(output_folder: str) -> Path:
    output_dir = Path(output_folder)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir


def mirror_to_workspace_tmp(source_path: Path) -> Path:
    """
    Mirror generated CSVs into PROJECT_ROOT/tmp for downstream configs that
    use relative paths such as tmp/df_preprocessed_filtered.csv.
    """
    workspace_tmp_dir = PROJECT_ROOT / "tmp"
    workspace_tmp_dir.mkdir(parents=True, exist_ok=True)
    mirrored_path = workspace_tmp_dir / source_path.name
    mirrored_path.write_bytes(source_path.read_bytes())
    return mirrored_path


def export_csv_artifacts(results_dir: Path, labeled_roots: list[tuple[str, Path]]) -> None:
    """
    Copy all generated CSV files into results_dir/csv_artifacts so a host mount
    can collect every pipeline CSV from one place.
    """
    artifacts_dir = results_dir / "csv_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    exported_count = 0

    for label, root in labeled_roots:
        if not root.exists():
            continue

        csv_paths = [root] if root.is_file() and root.suffix == ".csv" else root.rglob("*.csv")
        for csv_path in csv_paths:
            csv_path = Path(csv_path)
            if artifacts_dir in csv_path.parents:
                continue

            if root.is_file():
                flattened_name = f"{label}__{csv_path.name}"
            else:
                relative_parts = csv_path.relative_to(root).parts
                flattened_name = "__".join((label, *relative_parts))

            destination_path = artifacts_dir / flattened_name
            if csv_path.resolve() == destination_path.resolve():
                continue

            shutil.copy2(csv_path, destination_path)
            exported_count += 1

    logger.info("Exported %d CSV artifacts to %s", exported_count, artifacts_dir)


def get_deepcoro_target_labels(config_path: Path = DOCKER_BASE_CONFIG_PATH) -> list[str]:
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)
    return list(config.get("target_label", []))


def resolve_deepcoro_run_mode(
    df: pd.DataFrame,
    config_path: Path = DOCKER_BASE_CONFIG_PATH,
) -> str:
    """
    Default the Docker workflow to inference mode. Set DEEPCORO_RUN_MODE=auto
    to restore schema-based mode selection, or set DEEPCORO_RUN_MODE explicitly
    to force a supported mode.
    """
    requested_mode = os.environ.get("DEEPCORO_RUN_MODE", "inference").strip().lower()
    supported_modes = {"inference", "val", "test", "train", "auto"}
    if requested_mode not in supported_modes:
        logger.warning(
            "Unsupported DEEPCORO_RUN_MODE=%r; defaulting to inference.",
            requested_mode,
        )
        return "inference"

    if requested_mode != "auto":
        logger.info("Using DeepCORO run mode: %s", requested_mode)
        return requested_mode

    target_labels = get_deepcoro_target_labels(config_path)
    if not target_labels:
        logger.warning("No DeepCORO target labels found in %s; using inference mode.", config_path)
        return "inference"

    missing_columns = [label for label in target_labels if label not in df.columns]
    if missing_columns:
        logger.warning(
            "DeepCORO validation labels are incomplete; switching to inference mode. "
            "Missing %d target columns. Examples: %s",
            len(missing_columns),
            missing_columns[:5],
        )
        return "inference"

    null_columns = [label for label in target_labels if df[label].isna().any()]
    if null_columns:
        logger.warning(
            "DeepCORO validation labels contain null values after filtering; switching to inference mode. "
            "Affected target columns: %d. Examples: %s",
            len(null_columns),
            null_columns[:5],
        )
        return "inference"

    return "val"


def resolve_model_device(requested_device: str) -> str:
    """
    Resolve the requested model device to a usable runtime device.
    """
    if requested_device != 'cuda':
        return requested_device

    try:
        import torch
    except ImportError:
        logger.warning("PyTorch is unavailable while resolving CUDA; keeping requested device 'cuda'.")
        return requested_device

    if torch.cuda.is_available():
        return requested_device

    logger.warning("CUDA requested but unavailable at runtime; falling back to CPU.")
    return 'cpu'


def _model_dir_has_pt(model_dir: str) -> bool:
    for _root, _dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".pt"):
                return True
    return False


def get_model_weights(model_name: str) -> str:
    """
    Returns the local path to the specified model's weights.
    Raises FileNotFoundError if weights are not found (they must be downloaded at build time).
    """
    candidate_dirs = [
        PROJECT_ROOT / "weights" / model_name,
        PROJECT_ROOT / ("downloaded_" + model_name),
    ]

    # Older local/dev builds used a generic pretrained_models directory.
    if model_name == "deepcoro_clip_generic":
        candidate_dirs.append(PROJECT_ROOT / "pretrained_models")

    for candidate_dir in candidate_dirs:
        if candidate_dir.exists() and _model_dir_has_pt(str(candidate_dir)):
            return str(candidate_dir)

    raise FileNotFoundError(
        f"Model weights for '{model_name}' not found in: "
        f"{', '.join(str(path) for path in candidate_dirs)}. "
        f"Weights must be downloaded at Docker build time."
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
    config = {}        
    config['output_dir'] = args['output_folder']
    config['model_path'] = args['model_path']
    config['data_filename'] = args['data_path']
    config['batch_size'] = args['batch_size']
    config['num_workers'] = args['num_workers']
    config['device'] = resolve_model_device(args['model_device'])
    config.update(default_model_config)
    
    return config


def get_model_path(model: str) -> str:
    """
    Returns the local path to the specified model's .pt file.
    Raises FileNotFoundError if weights are not found.
    """
    model_weights_path: str = get_model_weights(model_name=model)
    
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
        fallback = PROJECT_ROOT / ("downloaded_" + model)
        raise ValueError(
            f"No .pt file found in {model_weights_path} or its subdirectories. "
            f"Ensure the model is in that folder or in {fallback} (e.g. run scripts/download_vasovision.py)."
        )
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


def _parse_stenosis_value(value) -> Optional[float]:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    matches = re.findall(r"\d+(?:\.\d+)?", text)
    if not matches:
        return np.nan

    values = [float(match) for match in matches]
    return max(values)


def validate_input_dataframe(df: pd.DataFrame) -> None:
    required_columns = {"DICOMPath"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required input columns: {missing_columns}")

    if df.empty:
        raise ValueError("Input dataframe is empty")

    if df["DICOMPath"].isna().any():
        missing_count = int(df["DICOMPath"].isna().sum())
        raise ValueError(f"DICOMPath contains {missing_count} null values")

    dicom_paths = df["DICOMPath"].astype(str).str.strip()
    empty_mask = dicom_paths.eq("")
    if empty_mask.any():
        raise ValueError(f"DICOMPath contains {int(empty_mask.sum())} empty paths")

    missing_paths = [path for path in dicom_paths.unique() if not Path(path).exists()]
    if missing_paths:
        sample = missing_paths[:5]
        raise FileNotFoundError(
            f"{len(missing_paths)} DICOM paths do not exist. Examples: {sample}"
        )

    duplicate_paths = int(dicom_paths.duplicated().sum())
    if duplicate_paths:
        logger.warning(f"Input dataframe contains {duplicate_paths} duplicated DICOMPath values")


def validate_converted_dataframe(df: pd.DataFrame) -> None:
    required_columns = ["FileName", "Split", "SeriesTimes", "StudyInstanceUID", "DICOMPath"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Converted dataframe missing required columns: {missing_columns}")

    if df.empty:
        raise RuntimeError("No rows remain after DICOM conversion")

    if df["FileName"].isna().any():
        missing_count = int(df["FileName"].isna().sum())
        raise ValueError(f"FileName contains {missing_count} null values after DICOM conversion")

    missing_avi = [path for path in df["FileName"].astype(str).unique() if not Path(path).exists()]
    if missing_avi:
        sample = missing_avi[:5]
        raise FileNotFoundError(
            f"{len(missing_avi)} converted AVI paths do not exist. Examples: {sample}"
        )


def validate_vaso_merge(df: pd.DataFrame) -> None:
    required_columns = ["main_structure", "contrast_agent", "stent_presence"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"VasoVision merge missing expected columns: {missing_columns}")

    missing_predictions = int(df["main_structure"].isna().sum())
    if missing_predictions:
        sample = df.loc[df["main_structure"].isna(), "FileName"].head(5).tolist()
        raise ValueError(
            f"VasoVision predictions are missing for {missing_predictions} rows after merge. "
            f"Examples: {sample}"
        )

    if df.empty:
        raise RuntimeError("No rows remain after merging VasoVision predictions")

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
                # If already numeric, keep as-is (0/1/NaN); otherwise map from strings
                if df[column].dropna().apply(lambda x: isinstance(x, (int, float, np.integer, np.floating))).all():
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                else:
                    df[column] = df[column].map({'none': 0, 'mild': 1, 'moderate': 1, 'severe': 1})
            elif column.endswith('_cto') or column.endswith('_thrombus'):
                # For CTO and thrombus columns: replace False with 0, True with 1
                df[column] = df[column].map({False: 0, True: 1})

    # Process regression columns
    stenosis_columns = [col for col in df.columns.tolist() if col.endswith('_stenosis')]
    for column in stenosis_columns:
        original_non_null = int(df[column].notna().sum())
        df[column] = df[column].apply(_parse_stenosis_value)
        parsed_non_null = int(df[column].notna().sum())
        failed_to_parse = original_non_null - parsed_non_null
        if failed_to_parse > 0:
            logger.warning(
                f"Column {column}: failed to parse {failed_to_parse} non-null stenosis values to numeric"
            )
        df[f'{column}_binary'] = df[column].fillna(-np.inf).gt(70).astype(int)

    # Rename columns EXAM_ID and StudyInstanceUID
    rename_map = {'ss_patient_id': 'Patient_ID'}
    if 'StudyInstanceUID' not in df.columns:
        rename_map['ss_event_cath_id'] = 'StudyInstanceUID'
    df.rename(columns=rename_map, inplace=True)

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
    Wrapper for process_dicom_video for multiprocessing.

    Returns:
        (dicom_filepath, avi_filepath, series_times, success, reason)
        reason: "video" | "image" | "error"
    """
    import logging
    logger = logging.getLogger(__name__)

    dicom_filepath, output_path = args_tuple
    try:
        avi_filepath, series_times, reason = process_dicom_video(
            input_path=dicom_filepath,
            output_path=output_path
        )
        success = avi_filepath is not None
        return dicom_filepath, avi_filepath, series_times, success, reason
    except Exception as e:
        logger.warning(f"Failed to convert {dicom_filepath}: {e}")
        return dicom_filepath, None, None, False, "error"


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
        results_dir = resolve_output_dir(args['output_folder'])
        print(f"tmp_dir: {tmp_dir}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocess dataset
        df = pd.read_csv(args['data_path'])
        validate_input_dataframe(df)
        df_preprocessed = preprocess_dataset(df)
        print(df_preprocessed.columns.tolist())
        # Add validation after preprocessing
        if df_preprocessed.empty:
            raise ValueError("No data remaining after preprocessing")

        if 'DICOMPath' not in df_preprocessed.columns:
            raise ValueError("DICOMPath column not found in dataframe")
        if 'StudyInstanceUID' not in df_preprocessed.columns:
            raise ValueError("StudyInstanceUID column not found after preprocessing")

        if not args['debug']:
            # Convert DICOM videos to AVI using multiprocessing
            logging.info("Check for DICOM files and convert to AVI")
            dicom_filepaths: list[str] = df_preprocessed['DICOMPath'].tolist()
            logger.info(f"{len(dicom_filepaths)} DICOM files found in input dataframe.")

            converted_count = 0
            count_video = 0
            count_image = 0
            count_error = 0
            rows_to_discard: list[int] = []
            filepath_mapping: dict[str, str] = {}
            series_times_mapping: dict[str, str] = {}

            process_args = [(dicom_filepath, str(tmp_dir)) for dicom_filepath in dicom_filepaths]
            num_workers = min(mp.cpu_count(), len(dicom_filepaths), 8)
            logger.info(f"Using {num_workers} workers for parallel processing")

            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_single_dicom, process_args),
                    total=len(process_args),
                    desc="Converting DICOM to AVI"
                ))

            for dicom_filepath, avi_filepath, series_times, success, reason in results:
                if reason == "video":
                    count_video += 1
                elif reason == "image":
                    count_image += 1
                else:
                    count_error += 1
                if success and avi_filepath:
                    filepath_mapping[dicom_filepath] = avi_filepath
                    converted_count += 1
                    series_times_mapping[dicom_filepath] = series_times
                else:
                    mask = df_preprocessed['DICOMPath'] == dicom_filepath
                    rows_to_discard.extend(df_preprocessed[mask].index.tolist())

            logger.info(f"Starting with {len(df_preprocessed)} rows")
            print(
                f"DICOM counts: {count_video} videos (converted), "
                f"{count_image} single-frame images (skipped), {count_error} other errors"
            )
            logger.info(f"DICOM counts: {count_video} videos (converted), {count_image} single-frame images (skipped), {count_error} other errors")
            logger.info(f"Successfully converted {converted_count}/{len(dicom_filepaths)} files")
            logger.info(f"Failed conversions: {len(rows_to_discard)}")
            logger.info(f"Converted {converted_count} DICOM files to AVI format in {tmp_dir}")
            if converted_count == 0:
                raise RuntimeError("All DICOM to AVI conversions failed")
            df_preprocessed['Split'] = 'inference' # Required to run Orion and VasoVision inference
            df_preprocessed['FileName'] = df_preprocessed['DICOMPath'].map(filepath_mapping)
            df_preprocessed['SeriesTimes'] = df_preprocessed['DICOMPath'].map(series_times_mapping)
            df_preprocessed.drop(rows_to_discard, inplace=True)
            df_preprocessed.reset_index(drop=True, inplace=True)
            logger.info(f"Discarded {len(rows_to_discard)} rows - new dataframe length: {len(df_preprocessed)}")
            validate_converted_dataframe(df_preprocessed)

            # Save new input dataframe
            logger.info(f"Saving preprocessed dataframe to {tmp_dir / 'df_preprocessed.csv'}")
            preprocessed_csv = tmp_dir / "df_preprocessed.csv"
            df_preprocessed.to_csv(preprocessed_csv, index=False, sep='α')
            mirror_to_workspace_tmp(preprocessed_csv)
            
        else:
            df_preprocessed = pd.read_csv(tmp_dir / "df_preprocessed.csv", sep='α')
            validate_converted_dataframe(df_preprocessed)

        args['data_path'] = tmp_dir / "df_preprocessed.csv"

        # Initialize VasoVision
        vaso_vision_hugging_face_model_name: str = MODEL_MAPPING['vaso_vision']['hugging_face_model_name']
        args['model_path'] = get_model_path(model=vaso_vision_hugging_face_model_name)
        
        # Setup orion config for VisionVision
        vaso_vision_orion_config: dict = setup_orion_config(
            args=args,
            default_model_config=MODEL_MAPPING['vaso_vision']['config']
        )        
        
        if not args['debug']:
            df_vaso_info: pd.DataFrame = orion_video_training_and_eval.perform_inference(
                config=vaso_vision_orion_config,
                split='inference',
                log_wandb=False
            )
        
            df_vaso_info.rename(
                columns={'filename': 'FileName'}, 
                inplace=True
            )
        
            # Save vaso info dataframe
            vaso_info_csv = tmp_dir / "df_vaso_info.csv"
            df_vaso_info.to_csv(vaso_info_csv, index=False, sep='α')
            mirror_to_workspace_tmp(vaso_info_csv)
            
        else:
            df_vaso_info = pd.read_csv(tmp_dir / "df_vaso_info.csv", sep='α')

        
        # process vaso info dataframe
        df_vaso_info = clean_vaso_info_dataframe(df_vaso_info)
        if 'FileName' not in df_vaso_info.columns:
            raise ValueError("VasoVision output does not contain FileName")

        df_preprocessed = pd.merge(
            df_preprocessed, df_vaso_info, on='FileName', how='left'
        )
        validate_vaso_merge(df_preprocessed)
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
        if df_preprocessed.empty:
            raise RuntimeError("No rows remain after VasoVision filtering and diagnostic-status selection")

        deepcoro_run_mode = resolve_deepcoro_run_mode(df_preprocessed)
        df_preprocessed['Split'] = deepcoro_run_mode

        # Save dataframe
        filtered_csv = tmp_dir / "df_preprocessed_filtered.csv"
        df_preprocessed.to_csv(filtered_csv, index=False, sep='α')
        mirror_to_workspace_tmp(filtered_csv)
        
           
        # Initialize deepcoro_clip
        deepcoro_clip_hugging_face_model_name: str = MODEL_MAPPING['deepcoro_clip']['hugging_face_model_name']
        args['model_path'] = get_model_path(model=deepcoro_clip_hugging_face_model_name)
        
        print(f"model_path: {args['model_path']}")
                        
        # Run bash command to run deepcoro_clip inference
        bash_command = (
            "bash scripts/runner.sh "
            "--use_wandb false "
            "--base_config config/linear_probing/stenosis/docker_base_config.yaml "
            f"--run_mode {deepcoro_run_mode} "
            "--selected_gpus 0"
        )
        logger.info(f"Executing deepcoro_clip inference command: {bash_command}")
        try:
            result = subprocess.run(bash_command, shell=True, check=True, capture_output=True, text=True)
            logger.info("deepcoro_clip inference completed successfully")
            if result.stdout:
                logger.info(f"Command output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            details = []
            if e.stdout:
                details.append(f"stdout:\n{e.stdout}")
            if e.stderr:
                details.append(f"stderr:\n{e.stderr}")
            detail_text = "\n".join(details)
            message = f"deepcoro_clip inference failed with return code {e.returncode}"
            if detail_text:
                message = f"{message}\n{detail_text}"
            raise RuntimeError(message) from e

        export_csv_artifacts(
            results_dir=results_dir,
            labeled_roots=[
                ("tmp", tmp_dir),
                ("results", results_dir),
            ],
        )

    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")
        sys.exit(1)     


if __name__ == "__main__":
    data_path = os.environ.get("EXTERNAL_VALIDATION_DATA_PATH")
    if not data_path:
        raise ValueError("EXTERNAL_VALIDATION_DATA_PATH is not set")

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"EXTERNAL_VALIDATION_DATA_PATH does not exist: {data_path}")

    
    args = {
        'data_path': data_path,
        'output_folder': 'results',
        'tmp_dir': '/app/tmp',
        'model_path': '',
        'batch_size': 12,
        'num_workers': 12,
        'model_device': 'cuda',
        'debug': False
    }
    main(args)
