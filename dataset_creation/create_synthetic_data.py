import pandas as pd
import numpy as np
import random
from pathlib import Path

def create_synthetic_test_data():
    """Create synthetic test data with various stenosis patterns"""

    # Define vessel columns
    vessel_columns = [
        'left_main_stenosis', 'prox_lad_stenosis', 'mid_lad_stenosis', 'dist_lad_stenosis',
        'D1_stenosis', 'D2_stenosis', 'lcx_stenosis', 'dist_lcx_stenosis',
        'om1_stenosis', 'om2_stenosis', 'bx_stenosis', 'lvp_stenosis',
        'prox_rca_stenosis', 'mid_rca_stenosis', 'dist_rca_stenosis',
        'pda_stenosis', 'posterolateral_stenosis'
    ]

    # Create sample studies with different patterns
    studies = []

    # Study 1: Critical LAD with mild RCA disease
    studies.append({
        'StudyInstanceUID': 'study_1',
        'SeriesInstanceUID': 'series_1_left',
        'ViewAngle': 'left',
        'prox_lad_stenosis': 95.0,
        'dist_lad_stenosis': 30.0,
        'prox_rca_stenosis': 30.0,
        'dominance': 'right',
        **{col: np.nan for col in vessel_columns if col not in ['prox_lad_stenosis', 'dist_lad_stenosis', 'prox_rca_stenosis']}
    })
    studies.append({
        'StudyInstanceUID': 'study_1',
        'SeriesInstanceUID': 'series_1_right',
        'ViewAngle': 'right',
        'prox_lad_stenosis': 95.0,
        'dist_lad_stenosis': 30.0,
        'prox_rca_stenosis': 30.0,
        'dominance': 'right',
        **{col: np.nan for col in vessel_columns if col not in ['prox_lad_stenosis', 'dist_lad_stenosis', 'prox_rca_stenosis']}
    })

    # Study 2: Multi-vessel disease
    studies.append({
        'StudyInstanceUID': 'study_2',
        'SeriesInstanceUID': 'series_2_left',
        'ViewAngle': 'left',
        'left_main_stenosis': 50.0,
        'prox_lad_stenosis': 70.0,
        'lcx_stenosis': 80.0,
        'om1_stenosis': 60.0,
        'prox_rca_stenosis': 90.0,
        'dominance': 'right',
        **{col: 0.0 for col in vessel_columns if col not in ['left_main_stenosis', 'prox_lad_stenosis', 'lcx_stenosis', 'om1_stenosis', 'prox_rca_stenosis']}
    })

    # Study 3: Normal coronaries
    studies.append({
        'StudyInstanceUID': 'study_3',
        'SeriesInstanceUID': 'series_3_left',
        'ViewAngle': 'left',
        'dominance': 'right',
        **{col: 0.0 for col in vessel_columns}
    })

    # Study 4: LCx territory disease
    studies.append({
        'StudyInstanceUID': 'study_4',
        'SeriesInstanceUID': 'series_4_left',
        'ViewAngle': 'left',
        'lcx_stenosis': 85.0,
        'om1_stenosis': 75.0,
        'om2_stenosis': 40.0,
        'dominance': 'left',
        **{col: 0.0 for col in vessel_columns if col not in ['lcx_stenosis', 'om1_stenosis', 'om2_stenosis']}
    })

    # Study 5: Mixed with unknowns
    studies.append({
        'StudyInstanceUID': 'study_5',
        'SeriesInstanceUID': 'series_5_left',
        'ViewAngle': 'left',
        'prox_lad_stenosis': 100.0,  # Total occlusion
        'mid_lad_stenosis': np.nan,
        'dist_lad_stenosis': np.nan,
        'lcx_stenosis': 25.0,
        'prox_rca_stenosis': 60.0,
        'dominance': 'right',
        **{col: np.nan for col in vessel_columns if col not in ['prox_lad_stenosis', 'mid_lad_stenosis', 'dist_lad_stenosis', 'lcx_stenosis', 'prox_rca_stenosis']}
    })

    df = pd.DataFrame(studies)

    # Add required columns
    df['video_file_path'] = df['SeriesInstanceUID'].apply(lambda x: f'/fake/path/{x}.mp4')
    df['report_status'] = 'positive'

    return df

if __name__ == '__main__':
    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    # Generate synthetic data
    df = create_synthetic_test_data()

    # Save to CSV
    output_path = output_dir / 'synthetic_input.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved synthetic test data to {output_path}")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df[['StudyInstanceUID', 'SeriesInstanceUID', 'prox_lad_stenosis', 'lcx_stenosis', 'prox_rca_stenosis']].head())