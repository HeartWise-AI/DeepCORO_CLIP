import cv2
import pydicom
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from utils.external_validation_constants import DICOM_TAGS


def process_dicom_video(
    input_path: str,
    output_path: str
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Converts DICOM videos to AVI format and extracts acquisition time information.

    Returns:
        (avi_path, series_times, kind) with kind one of "video", "image", "error".
        "image" when the DICOM is single-frame (2D array); "error" on other failure.
    """
    ds: pydicom.Dataset = pydicom.dcmread(input_path)

    series_times: Optional[str] = None
    try:
        series_times = float(ds[DICOM_TAGS['series_times']].value)
    except Exception as e:
        print(f"Warning: Could not extract acquisition time from {input_path}: {e}")
        return None, None, "error"

    output_filename: Path = Path(input_path).stem + '.avi'
    output_path_str: str = str(Path(output_path) / output_filename)

    if Path(output_path_str).exists():
        return output_path_str, series_times, "video"

    video: np.ndarray = ds.pixel_array

    if len(video.shape) != 3:
        print(f"Error: Extracted video's shape is not 3D: {video.shape} -  {input_path}")
        return None, series_times, "image"

    frame_height: int = ds[DICOM_TAGS['frame_height']].value
    frame_width: int = ds[DICOM_TAGS['frame_width']].value

    if frame_height != video.shape[1]:
        print(f"Error: Dicom video height {frame_height} does not match extracted video's shape: {video.shape[1]} -  {input_path}")
        return None, series_times, "error"

    if frame_width != video.shape[2]:
        print(f"Error: Dicom video width {frame_width} does not match extracted video's shape: {video.shape[2]} -  {input_path}")
        return None, series_times, "error"

    fps: float = 30.0
    if DICOM_TAGS['frame_rate'] in ds:
        fps = float(ds[DICOM_TAGS['frame_rate']].value)

    try:
        photometrics: str = ds.PhotometricInterpretation
        if photometrics not in ['MONOCHROME1', 'MONOCHROME2', 'RGB']:
            print(f"Error: Unsupported Photometric Interpretation: {photometrics} - with shape {video.shape}")
            return None, series_times, "error"
    except Exception:
        print(f"Error in reading {input_path}")
        return None, series_times, "error"

    fourcc: int = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    out: cv2.VideoWriter = cv2.VideoWriter(output_path_str, fourcc, fps, (frame_width, frame_height))

    conversion_fn: int = cv2.COLOR_GRAY2BGR if photometrics in ('MONOCHROME1', 'MONOCHROME2') else cv2.COLOR_RGB2BGR
    for frame in ds.pixel_array:
        frame_bgr: np.ndarray = cv2.cvtColor(frame, conversion_fn)
        out.write(frame_bgr)

    out.release()
    return output_path_str, series_times, "video"