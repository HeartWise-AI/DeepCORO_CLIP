"""
Vessel Constants Module for DeepCORO CLIP Analysis

This module defines vessel groupings and constants used across the analysis pipeline.
Created to resolve circular import issues between data_aggregation and plot_metrics modules.
"""

# --- Core vessel groupings ---
RCA_VESSELS = ["prox_rca_stenosis", "mid_rca_stenosis", "dist_rca_stenosis"]
NON_RCA_VESSELS = [
    "left_main_stenosis", "prox_lad_stenosis", "mid_lad_stenosis", "dist_lad_stenosis",
    "D1_stenosis", "D2_stenosis", "prox_lcx_stenosis", "dist_lcx_stenosis",
    "om1_stenosis", "om2_stenosis", "bx_stenosis", "lvp_stenosis"
]
RIGHT_DOMINANCE_DEPENDENT_VESSELS = ["pda_stenosis", "posterolateral_stenosis"]
LEFT_DOMINANCE_DEPENDENT_VESSELS = ["pda_stenosis", "lvp_stenosis"]

# --- Coronary Area vessel definitions for plotting ---
LEFT_CORONARY_DOMINANCE_VESSELS = NON_RCA_VESSELS + LEFT_DOMINANCE_DEPENDENT_VESSELS  # All left-side vessels
RIGHT_CORONARY_DOMINANCE_VESSELS = RCA_VESSELS + RIGHT_DOMINANCE_DEPENDENT_VESSELS   # All right-side vessels

# --- Helper function ---
def mode(lst):
    """Return the most common element using pandas Series.mode(). If there are ties, just take the first value."""
    import pandas as pd
    import numpy as np
    
    if not lst:
        return None
    
    # Convert to pandas Series for optimized mode calculation
    series = pd.Series(lst).dropna()
    if series.empty:
        return None
    
    mode_result = series.mode()
    # If there are ties (multiple modes), just take the first one
    return mode_result.iloc[0] if not mode_result.empty else None 