# Labelbox: deepcoro-ivus-oct

Video-level classification of angiography cines for intravascular imaging (OCT vs IVUS) and the vessel imaged.

- Package: /media/data1/datasets/DeepCoro/labelbox_ivus_oct (videos/, manifest.csv, data_rows.json, copies of these scripts)
- 1,908 studies / 6,091 cines: 1,708 IVUS/OCT-evidence studies (2017-2025) x top-3 candidate cines per modality,
  + 200 control PCI studies x 2 hardest negatives. Shuffled; annotators see only the video.
- Format matches earlier DeepCORO projects: 512x512, 15 fps, H.264 mp4, global radio classifications.
- Global key ivoct_<studyhash>_<filehash>; metadata study_key groups cines of a study. UIDs/paths stay in manifest.csv.

Steps
1. `pip install labelbox; export LABELBOX_API_KEY=...`
2. `python upload_to_labelbox.py /media/data1/datasets/DeepCoro/labelbox_ivus_oct` (dataset + ontology + project + batches of 1,000)
3. After labelling, export NDJSON and run `python parse_export.py <export.ndjson>` -> per-cine labels + study-level
   modality (OCT / IVUS / both / none) and vessels.

Ontology (ontology.json): intravascular_imaging {OCT, IVUS, imaging catheter type uncertain, other device only, none,
not interpretable}; vessel {LM, LAD, diagonal, LCX, OM/ramus, RCA, PDA/PLV, graft, uncertain}; left_main_traversed;
acquisition_phase {positioning, pullback/flush, uncertain}.
