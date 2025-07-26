import cv2
import pydicom
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from utils.external_validation_constants import DICOM_TAGS


def process_dicom_video(
    input_path: str, 
    output_path: str
)->Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Converts DICOM videos to AVI format and extracts acquisition time information.
    
    Args:
        input_path (str): The path to the DICOM file.
        output_path (str): The path to save the AVI file.
    
    Returns:
        Tuple[Optional[str], Optional[str], Optional[str]]: A tuple containing 
            (avi_path, acquisition_time, acquisition_datetime).
            avi_path: The path to the converted AVI file, or None if conversion failed.
            acquisition_time: The acquisition time as a string (HHMMSS.FFFFFF), or None if not available.
            acquisition_datetime: The acquisition datetime as a string (YYYYMMDDHHMMSS.FFFFFF), or None if not available.
    """
    # Read DICOM file
    ds: pydicom.Dataset = pydicom.dcmread(input_path)

    # Extract acquisition time and datetime
    series_times: Optional[str] = None
    
    try:
        series_times = str(ds[DICOM_TAGS['series_times']].value)
    except Exception as e:
        print(f"Warning: Could not extract acquisition time from {input_path}: {e}")
        return None, None
    
    output_filename: Path = Path(input_path).stem + '.avi'
    output_path: str = str(Path(output_path) / output_filename)
    
    if Path(output_path).exists():
        return output_path, series_times

    # get pixel array
    video: np.ndarray = ds.pixel_array
    
    # Insure extracted array is 3D
    if len(video.shape) != 3:
        print(ValueError(f"Extracted video`shape is not 3D: {video.shape} -  {input_path}")) 
        return None, series_times
    
    # get frame height and width
    frame_height: int = ds[DICOM_TAGS['frame_height']].value
    frame_width: int = ds[DICOM_TAGS['frame_width']].value
    
    # Insure consistence between dicom info and extracted video
    if frame_height != video.shape[1]:
        print(ValueError(f"Dicom video height {frame_height} does not match extracted video`shape: {video.shape[1]} -  {input_path}")) 
        return None, series_times
    
    if frame_width != video.shape[2]:
        print(ValueError(f"Dicom video width {frame_width} does not match extracted video`shape: {video.shape[2]} -  {input_path}")) 
        return None, series_times
    
    # Extract FPS; ensure the DICOM tag exists
    fps: float = 30.0  # Default FPS if not specified
    if DICOM_TAGS['frame_rate'] in ds:
        fps = float(ds[DICOM_TAGS['frame_rate']].value)      
        
    try:
        photometrics: str = ds.PhotometricInterpretation
        if photometrics not in ['MONOCHROME1', 'MONOCHROME2', 'RGB']:
            print(ValueError(f"Unsupported Photometric Interpretation: {photometrics} - with shape{video.shape}"))
            return None, series_times
    except:
        print(f"Error in reading {input_path}")
        return None, series_times
        
    # Create video writer
    fourcc: int = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    out: cv2.VideoWriter = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
    conversion_fn: int = cv2.COLOR_GRAY2BGR if photometrics == 'MONOCHROME1' or photometrics == 'MONOCHROME2' else cv2.COLOR_RGB2BGR    
    for frame in ds.pixel_array:
        frame: np.ndarray = cv2.cvtColor(frame, conversion_fn)
        out.write(frame)
    
    # Release video writer
    out.release()

    return output_path, series_times