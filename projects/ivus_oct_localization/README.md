# IVUS / OCT localisation on coronary angiography

Goal: find every study with intravascular imaging (IVUS / OCT), the vessel imaged, and the angiogram cine(s)
where the imaging catheter is visible. Clocks of the IVUS/OCT consoles and the X-ray system are not synced
(IVUS ~+1-2 h, OCT +/-1 h drift), so nothing here uses timestamps.

## Pipeline (run from repo root, `PYTHONPATH=.pyoverlay` for the encoder)
1. `build_cohort.py` — study labels (DICOM Modality IVUS/OCT, report `IVUS` flag, `*_ivusrealise`, `*_ivusarea`,
   free text) + year-matched PCI controls; patient-level split; video list.
2. `vessel_labels.py` — vessel per study: IVUS DICOM `ImageComments` > report segment > report line mentioning
   IVUS/OCT > PCI-vessel fallback.
3. `extract_embeddings.py` — DeepCORO-CLIP (mViT, 8av1xygm epoch 7) mean+max patch-token embeddings per video.
4. `recording_classifier.py` — Stage A: XA vs IVUS-recording vs OCT-recording (OOF 99.99% on 2024 known modality);
   recovers IVUS/OCT recordings in 2022-2023 where Modality is blank.
5. `train_mil.py` — study-level attention MIL (weak labels). Used only to propose candidates for review.
6. `instance_model.py` + `label_sheets.py` + `add_labels.py` — active-learning loop: visually review top cines,
   train per-cine logistic regression "imaging catheter in coronary" (verified positives vs verified + control negatives).
7. `assemble_results.py` — final tables in `data/results/`.
8. `scan_dicom_modalities.py` — (slow, background) header scan of study folders to find IVUS/OCT objects missing
   from the parquet (2023, 2025 and positive studies of other years) -> `data/dicom_intravascular_scan.csv`.

## Outputs (`data/results/`)
- `ivus_oct_studies.csv`: one row per (study, modality): evidence (DICOM objects, recording videos, report),
  `vessel_final` + `vessel_final_source`, `best_cine`, `best_cine_score`, `cine_found` (score >= 0.5).
- `ivus_oct_cines.csv`: top-3 candidate cines per (study, modality) with catheter score, view, angles,
  visual-review label and vessel when reviewed.

## Performance (2026-10-01)
- OCT per-cine model: CV AUROC 0.91 on reviewed cines; held-out review of 30 unseen studies: precision 9/13,
  recall 9/13 at 0.5. 59% of OCT studies with a confirmed recording have a cine >= 0.5.
- IVUS per-cine model: CV AUROC 0.90 but only 29 reviewed cines; IVUS pullbacks are recorded under fluoro
  (no contrast), are visually subtle and often not archived -> only 19% of confirmed IVUS studies have a cine >= 0.5.
- Image-based vessel model (top cines -> vessel, 5-fold patient CV): OCT 0.77, IVUS 0.63 accuracy
  (LM vs LAD is the main confusion). Used only when no report/DICOM/visual vessel is available.
