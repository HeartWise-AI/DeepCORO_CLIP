

from enum import Enum

# Angio classes
class AngioClasses(Enum):
    CATHETER = 0
    DIST_LAD = 1  # Distal Left Anterior Descending
    DIST_LCX = 2  # Distal Left Circumflex
    DIST_RCA = 3  # Distal Right Coronary Artery
    GUIDEWIRE = 4
    LAD = 5        # Left Anterior Descending
    LCX = 6        # Left Circumflex
    LEFTMAIN = 7   # Left Main Coronary Artery
    MID_LAD = 8    # Mid Left Anterior Descending
    MID_RCA = 9    # Mid Right Coronary Artery
    OBSTRUCTION = 10
    PACEMAKER = 11
    PDA = 12       # Posterior Descending Artery
    POSTEROLATERAL = 13
    PROX_RCA = 14  # Proximal Right Coronary Artery
    STENOSIS = 15
    STENT = 16
    STERNOTOMY = 17
    VALVE = 18
    

DICOM_TAGS = {
    'frame_height': (0x028, 0x0011),
    'frame_width': (0x028, 0x0010),
    'frame_rate': (0x08, 0x2144),
    'series_times': (0x0008, 0x0031)
}

REGRESSION_COLUMNS = [
    'prox_rca_stenosis',
    'mid_rca_stenosis',
    'dist_rca_stenosis',
    'pda_stenosis',
    'posterolateral_stenosis',
    'left_main_stenosis',
    'prox_lad_stenosis',
    'mid_lad_stenosis',
    'dist_lad_stenosis',
    'D1_stenosis',
    'D2_stenosis',
    'prox_lcx_stenosis',
    'mid_lcx_stenosis',
    'dist_lcx_stenosis',
    'om1_stenosis',
    'om2_stenosis',
    'bx_stenosis',
    'lvp_stenosis'
]

BINARY_COLUMNS = [
    'prox_rca_calcif_binary',
    'mid_rca_calcif_binary', 
    'dist_rca_calcif_binary',
    'pda_calcif_binary',
    'posterolateral_calcif_binary',
    'left_main_calcif_binary',
    'prox_lad_calcif_binary',
    'mid_lad_calcif_binary',
    'dist_lad_calcif_binary',
    'D1_calcif_binary',
    'D2_calcif_binary',
    'prox_lcx_calcif_binary',
    'mid_lcx_calcif_binary',
    'dist_lcx_calcif_binary',
    'om1_calcif_binary',
    'om2_calcif_binary',
    # CTO columns
    'prox_rca_cto',
    'mid_rca_cto',
    'dist_rca_cto',
    'pda_cto',
    'posterolateral_cto',
    'left_main_cto',
    'prox_lad_cto',
    'mid_lad_cto',
    'dist_lad_cto',
    'D1_cto',
    'D2_cto',
    'prox_lcx_cto',
    'mid_lcx_cto',
    'dist_lcx_cto',
    'om1_cto',
    'om2_cto',
    # Thrombus columns
    'prox_rca_thrombus',
    'mid_rca_thrombus',
    'dist_rca_thrombus',
    'pda_thrombus',
    'posterolateral_thrombus',
    'left_main_thrombus',
    'prox_lad_thrombus',
    'mid_lad_thrombus',
    'dist_lad_thrombus',
    'D1_thrombus',
    'D2_thrombus',
    'prox_lcx_thrombus',
    'mid_lcx_thrombus',
    'dist_lcx_thrombus',
    'om1_thrombus',
    'om2_thrombus'
]


MODEL_MAPPING = {
    'vaso_vision': {
        'hugging_face_model_name': 'VasoVision', 
        'config' : {
            'frames': 72, 
            'resize': 256,
            'num_classes': 1,
            'head_structure': {
                'contrast_agent': 1,
                'main_structure': 12,
                'stent_presence': 1,
                'dominance': 1
            },
            'labels_map': {
                'contrast_agent': {
                    'yes': 1,
                    'no': 0
                },
                'main_structure': {
                    'Left Coronary': 0,
                    'Right Coronary': 1,
                    'Other': 2,
                    'Graft': 3,
                    'Catheter': 4,
                    'Femoral': 5,
                    'LV': 6,
                    'TAVR': 7,
                    'Aorta': 8,
                    'Radial': 9,
                    'TEE probe': 10,
                    'Pigtail': 11
                },
                'stent_presence': {
                    'present': 1,
                    'absent': 0
                },
                'dominance': {
                    'right_dominant': 0,
                    'left_dominant': 1
                }
            },
            'model_name': 'x3d_m',
            'task': 'classification',
            'resume': False,
            'mean': [112.24039459228516,112.24039459228516, 112.24039459228516],
            'std': [39.012229919433594,39.012229919433594,39.012229919433594],
            'period': 2,
            'root': ".",
            'apply_mask': False,
            'view_count': None,
            'datapoint_loc_label': 'FileName'
        }
    },
    'deepcoro_clip': {
        'hugging_face_model_name': 'deepcoro_clip_generic' 
    }
}
