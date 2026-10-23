# Labelbox classifier, October 2026

## Coronary-system update

Nidal's October 11 email in “Projet ivus OCT” proposed a Left coronary answer
when LAD versus LCX is unclear or wires occupy both branches. Robert agreed on
October 12. The live ontology now includes a `left_coronary` vessel choice and
an additional `coronary_system` question for imaging and other-device cines.
All original question/option IDs and human branch answers are preserved.

The deterministic remap sends LM/LAD/diagonal/LCX/OM-ramus to left coronary,
RCA to right coronary, graft to graft, and PDA/PLV or uncertain to uncertain.
Missing answers stay missing. PDA/PLV is not assigned right without dominance
or branch-origin evidence. Left coronary does not imply left-main traversal.
The remapped export contains 270 left, 65 right, 214 uncertain, 3 graft and
260 missing answers; original vessel and source label IDs remain available.

The versioned classifier uses vessel answers from all device cines for the
coronary-system head. Three classes are supported (left, right, uncertain);
graft has insufficient support. Validation accuracy: 68.7% (n=99, including
one unsupported graft). Test accuracy: 74.6% (n=67), macro F1 0.777.
Left-coronary test precision/recall: 88.9%/64.9% (37 examples).
Right-coronary test precision/recall: 81.8%/100% (9 examples).
These are review suggestions, not calibrated or clinically validated outputs.

Config: `config/ivus_oct_coronary_system.json`.
[Updated W&B run](https://wandb.ai/mhi_ai/deepcoro-ivus-oct/runs/aq8ifkrp).
NAS: `/media/data1/models/DeepCoro_CLIP/ivus_oct_coronary_system_20261023/`.
The current live human answers match the saved October 19 export exactly.
The live-export importer supports the SDK's stream metadata response and resolves
classification IDs independently of display labels. See `labelbox_sync.json`
for the verified import job and status. The remapped human export preserves
the distinction between existing human answers and model-assisted predictions.

```bash
python projects/ivus_oct_localization/train_labelbox.py --config config/ivus_oct_coronary_system.json
python projects/ivus_oct_localization/labelbox/sync_classifier.py --config config/ivus_oct_coronary_system.json --upload
```

The original model and its historical evaluation follow.

Run from the repository root:

```bash
python projects/ivus_oct_localization/train_labelbox.py --config config/ivus_oct_labelbox.json
python projects/ivus_oct_localization/labelbox/sync_classifier.py
python projects/ivus_oct_localization/test_labelbox_classifier.py
```

The completed output directory is protected against accidental retraining. Set a
new output path and run name in a copied configuration for a new experiment.

The source is the saved October 19 Labelbox export: 812 records, 769 with an
imaging answer and 43 skipped. Human labels are unreviewed. Frozen DeepCORO-CLIP
mViT mean/max patch tokens cover all 6,091 manifest cines. Linear classifier heads
use balanced logistic regression and train-only feature scaling. Four regularization
values are selected by validation macro F1. The original cohort split is preserved
and verified to be patient-disjoint. There is no encoder finetuning.

Secondary questions use human-labelled imaging cines only. Missing answers are
excluded. Classes with fewer than five training examples are unsupported; this
includes IVUS (one example), LM and OM/ramus. Left-main traversal has only one
supported class and is omitted. No IVUS subtype performance can be estimated.

## Results

[W&B run](https://wandb.ai/mhi_ai/deepcoro-ivus-oct/runs/1v3g8ox5),
model artifact `deepcoro-ivus-oct-labelbox-20261019:v0`.

| Head | Validation n / accuracy | Test n / accuracy | Test macro F1 |
| --- | --- | --- | --- |
| Imaging catheter, yes/no | 126 / 89.7% | 91 / 91.2% | 0.793 |
| Imaging question, five supported classes | 135 / 63.7% | 96 / 71.9% | 0.518 |
| Vessel, conditional on imaging | 16 / 81.3% | 13 / 84.6% | 0.783 |
| Pullback vs positioning | 16 / 87.5% | 12 / 91.7% | 0.916 |

Catheter detection test AUROC is 0.819, AUPRC 0.592; precision and recall are both
0.636 (7/11). Vessel test labels contain only LAD and uncertain; this does not
validate RCA/LCX/LM performance. Test sizes are small and candidate-enriched.
Patient separation applies to the classifier split; the encoder's historical
pretraining cohort has not been audited for overlap.

NAS outputs: `/media/data1/models/DeepCoro_CLIP/ivus_oct_labelbox_20261019/`.
Contains classifier, config, source exports and study report, source hashes,
metrics, validation/test inference, all-cine inference and reannotation payload.
Dataset inference copies are in
`/media/data1/datasets/DeepCoro/labelbox_ivus_oct/classifier_20261019/`.

## Labelbox reannotation

The review queue includes 5,279 unopened and 43 skipped cines (5,322 total).
Prepared suggestions are model assistance, requiring human review. Confidence
scores are raw balanced-classifier probabilities, not calibrated confidence.
Left-main traversal suggestions are omitted; other secondary questions appear
only when the imaging suggestion is OCT or imaging-uncertain.

To export current human labels and submit model-assisted predictions:

```bash
# Configure LABELBOX_API_KEY in the session, then:
python projects/ivus_oct_localization/labelbox/sync_classifier.py --upload
```

This refreshes the export and excludes rows with any current human imaging answer
before uploading. It does not overwrite human labels. The initial run has **not**
uploaded to Labelbox: the session has no Labelbox API credential. If new labels
are present in the fresh export, train a versioned follow-up run from that export.
