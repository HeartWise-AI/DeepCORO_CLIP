#!/usr/bin/env python3
"""
SCRIPT 1: Merge predictions with labels and keep ONLY matched rows (4828)
Sauvegarde un CSV avec SEULEMENT les 4828 lignes qui ont été merged
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def extract_first_basename(x):
    """Extract filename from path or list"""
    if pd.isna(x):
        return None
    try:
        if isinstance(x, str) and x.startswith("["):
            import ast
            lst = ast.literal_eval(x)
            if isinstance(lst, (list, tuple)) and len(lst) > 0:
                p = lst[0]
            else:
                p = None
        else:
            p = x
    except Exception:
        try:
            s = str(x).strip()
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1]
            parts = [p.strip().strip("'\"") for p in s.split(',') if p.strip()]
            p = parts[0] if parts else None
        except Exception:
            p = None
    if p is None:
        return None
    try:
        return os.path.basename(str(p))
    except:
        return str(p)

# ===== CONFIGURATION =====
predictions_csv = "/volume/Deep-Coro-final-branch/outputs/DeepCORO_video_linear_probing/DeepCORO_video_linear_probing_multiview_improved_cls_token/k2o4zy24_20251016-002318/predictions/inference_predictions_epoch_-1.csv"  # ← À CHANGER
labels_csv = "/volume/Deep-Coro-final-branch/data/CTO_THROMBUS_STENOSIS_70_CALCIF_inference.csv"            # ← À CHANGER
output_dir = "/volume/Deep-Coro-final-branch/mes_scripts/new"
# ========================

os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("SCRIPT 1: MERGE PREDICTIONS + LABELS → Garder 4828 lignes matched")
print("="*70)

# 1. Lire predictions
print("\n[1/4] Lecture predictions.csv...")
preds = pd.read_csv(predictions_csv, sep=',')
preds.columns = preds.columns.str.strip()
print(f"  → {len(preds)} lignes, {len(preds.columns)} colonnes")

# 2. Construire video_key pour predictions
print("\n[2/4] Construction video_key (predictions)...")
preds_key_pref = ['video_name', 'FileName', 'DICOMPath', 'dicom_id', 'file', 'path']
for col in preds_key_pref:
    if col in preds.columns:
        preds['video_key'] = preds[col].apply(extract_first_basename)
        non_null = int(preds['video_key'].dropna().shape[0])
        print(f"  → video_key depuis '{col}': {non_null} clés non-null")
        break
else:
    preds['video_key'] = None
    print(f"  ⚠ Aucune colonne détectée")

# 3. Lire labels et merge
print("\n[3/4] Lecture labels.csv...")
labels = pd.read_csv(labels_csv, sep='α')
labels.columns = labels.columns.str.strip()
print(f"  → {len(labels)} lignes, {len(labels.columns)} colonnes")

print("\n[3b/4] Construction video_key (labels)...")
label_pref = ['FileName', 'FileName.1', 'DICOMPath', 'dicom_id', 'dicom_id_x', 'dicom_id_y']
for col in label_pref:
    if col in labels.columns:
        labels['video_key'] = labels[col].apply(extract_first_basename)
        non_null = int(labels['video_key'].dropna().shape[0])
        print(f"  → video_key depuis '{col}': {non_null} clés non-null")
        break
else:
    labels['video_key'] = None
    print(f"  ⚠ Aucune colonne détectée")

# 4. MERGE avec 'inner' pour garder SEULEMENT les lignes matchées
print("\n[4/4] MERGE (inner join) sur video_key...")
print(f"  Avant merge: predictions={len(preds)}, labels={len(labels)}")

# Identifier colonnes _true dans labels
pred_bases = [c[:-5] for c in preds.columns if c.endswith('_pred')]
true_cols_needed = []
for base in pred_bases:
    if base in labels.columns or (base + '_true') in labels.columns:
        true_cols_needed.append(base)

if not true_cols_needed:
    print("  ⚠ ERREUR: Aucune colonne _true trouvée dans labels!")
    exit(1)

# Préparer colonnes labels à merger
labels_subset = labels[['video_key'] + true_cols_needed].copy()
labels_subset.columns = ['video_key'] + [c + '_true' for c in true_cols_needed]

# MERGE avec inner (seulement lignes matchées)
merged = preds.merge(labels_subset, on='video_key', how='inner')

print(f"  Après merge (inner): {len(merged)} lignes")
print(f"  ✓ Seulement les {len(merged)} lignes avec labels sont conservées!")

# 5. Sauvegarde
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"merged_clean_{ts}.csv")
merged.to_csv(output_path, index=False)
print(f"\n✓ CSV sauvegardé: {output_path}")
print(f"  → {len(merged)} lignes avec données complètes\n")

print("="*70)
print("Utilise ce fichier pour le SCRIPT 2 (analyse)")
print("="*70)