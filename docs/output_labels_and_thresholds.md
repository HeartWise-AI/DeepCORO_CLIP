# Output labels and decision thresholds (stenosis checkpoint)

This page describes what the public stenosis checkpoint predicts and which cut-offs turn its outputs into yes / no
calls. It applies to `zcb8cu0l_20260213-033432/best_model_epoch_11.pt` in
[heartwise/deepcoro_clip_generic](https://huggingface.co/heartwise/deepcoro_clip_generic) (formerly
`deepcoro_clip_stenosis`), run with `config/linear_probing/stenosis/docker_base_config.yaml` or
`config/inference/zcb8cu0l_inference_config.yaml`.

The thresholds are also available as a file: [`config/inference/zcb8cu0l_thresholds.json`](../config/inference/zcb8cu0l_thresholds.json).

Research use only. The model is not a medical device.

## Outputs

One study (up to 10 cine runs) gives 90 values: 5 outputs for each of 18 coronary segments. The prediction file has one
column per output, named `<output>_pred`.

| Output | Type | Meaning |
|---|---|---|
| `<segment>_stenosis` | regression | Estimated percent diameter stenosis, on the 0 to 100 scale of the angiography report |
| `<segment>_stenosis_binary` | probability | Probability of a significant stenosis (definition below) |
| `<segment>_calcif_binary` | probability | Probability of moderate or severe calcification |
| `<segment>_cto` | probability | Probability of a chronic total occlusion |
| `<segment>_thrombus` | probability | Probability of thrombus |

Segments: `prox_rca`, `mid_rca`, `dist_rca`, `pda`, `posterolateral`, `left_main`, `prox_lad`, `mid_lad`, `dist_lad`,
`D1`, `D2`, `prox_lcx`, `mid_lcx`, `dist_lcx`, `om1`, `om2`, `bx` (ramus), `lvp` (posterior left ventricular branch).

Binary outputs are sigmoid probabilities between 0 and 1. The regression output is not clipped and can fall below 0
or above 100.

## Label definitions used in training

Labels come from the structured angiography report of the operator (visual assessment), one value per segment and study.

- **Significant stenosis (`*_stenosis_binary`)**: report stenosis of **70% or more**; for the **left main, 50% or more**.
- **Calcification (`*_calcif_binary`)**: moderate or severe calcification, or a calcified nodule. None or mild is negative.
- **CTO (`*_cto`)** and **thrombus (`*_thrombus`)**: the finding is recorded for that segment in the report.

## Decision thresholds

**Rule: a finding is called positive when the probability is strictly greater than the threshold of that output.**

The probabilities are not calibrated and the findings are uncommon, so the thresholds are low (stenosis:
0.0014 to 0.1754). Two cut-offs that look natural do not work:

- **Do not use 0.5 on the probability.** On the internal test set it gives a mean sensitivity of 0.18
  (specificity 0.99).
- **Do not use 70 on the regression output.** The percent estimate is compressed toward low values; a cut-off of 70
  (50 for the left main) gives a mean sensitivity of 0.08 (specificity 1.00). Use the regression
  output as a continuous estimate and the probability output for the yes / no call.

These are the thresholds used in the HeartWise deployment of this checkpoint. They follow the rule implemented in
`compute_best_threshold` (`utils/metrics.py`): the point of the ROC curve with the highest geometric mean of
sensitivity and specificity, on internal Montreal Heart Institute data. These are
balanced operating points: sensitivity is favoured and the positive predictive value is low. Choose another point if
your use needs high specificity.

### Significant stenosis

Sensitivity, specificity and positive predictive value (PPV) are for the listed threshold on the internal held-out test
set (3,194 studies, labels from the report).

| Segment | Output column | Label cut-off | Threshold | Positives / 3,194 | AUROC | Sensitivity | Specificity | PPV |
|---|---|---|---|---|---|---|---|---|
| Proximal RCA | `prox_rca_stenosis_binary_pred` | ≥ 70% | 0.1079 | 462 | 0.91 | 0.84 | 0.83 | 0.46 |
| Mid RCA | `mid_rca_stenosis_binary_pred` | ≥ 70% | 0.1284 | 431 | 0.92 | 0.84 | 0.83 | 0.43 |
| Distal RCA | `dist_rca_stenosis_binary_pred` | ≥ 70% | 0.0327 | 215 | 0.90 | 0.92 | 0.71 | 0.18 |
| Posterior descending artery | `pda_stenosis_binary_pred` | ≥ 70% | 0.0299 | 146 | 0.82 | 0.84 | 0.62 | 0.10 |
| Posterolateral branch | `posterolateral_stenosis_binary_pred` | ≥ 70% | 0.0209 | 107 | 0.82 | 0.80 | 0.72 | 0.09 |
| Left main | `left_main_stenosis_binary_pred` | ≥ 50% | 0.0546 | 271 | 0.93 | 0.86 | 0.84 | 0.33 |
| Proximal LAD | `prox_lad_stenosis_binary_pred` | ≥ 70% | 0.1145 | 489 | 0.90 | 0.84 | 0.78 | 0.41 |
| Mid LAD | `mid_lad_stenosis_binary_pred` | ≥ 70% | 0.1754 | 589 | 0.88 | 0.79 | 0.79 | 0.46 |
| Distal LAD | `dist_lad_stenosis_binary_pred` | ≥ 70% | 0.0373 | 102 | 0.77 | 0.65 | 0.76 | 0.08 |
| First diagonal | `D1_stenosis_binary_pred` | ≥ 70% | 0.1308 | 326 | 0.77 | 0.69 | 0.69 | 0.20 |
| Second diagonal | `D2_stenosis_binary_pred` | ≥ 70% | 0.0568 | 198 | 0.77 | 0.72 | 0.67 | 0.13 |
| Proximal LCx | `prox_lcx_stenosis_binary_pred` | ≥ 70% | 0.0830 | 351 | 0.89 | 0.85 | 0.76 | 0.31 |
| Mid LCx | `mid_lcx_stenosis_binary_pred` | ≥ 70% | 0.0597 | 219 | 0.84 | 0.83 | 0.73 | 0.19 |
| Distal LCx | `dist_lcx_stenosis_binary_pred` | ≥ 70% | 0.0105 | 49 | 0.79 | 0.78 | 0.61 | 0.03 |
| First obtuse marginal | `om1_stenosis_binary_pred` | ≥ 70% | 0.0723 | 182 | 0.79 | 0.64 | 0.75 | 0.13 |
| Second obtuse marginal | `om2_stenosis_binary_pred` | ≥ 70% | 0.0767 | 201 | 0.83 | 0.81 | 0.70 | 0.15 |
| Ramus | `bx_stenosis_binary_pred` | ≥ 70% | 0.0299 | 105 | 0.80 | 0.83 | 0.66 | 0.08 |
| Posterior left ventricular branch (LVP) | `lvp_stenosis_binary_pred` | ≥ 70% | 0.0014 | 17 | 0.93 | 1.00 | 0.72 | 0.02 |
| **Mean of 18** | | | | | 0.85 | 0.81 | 0.73 | 0.21 |

### Calcification, CTO and thrombus

These outputs are exploratory: positives are few, so the estimates are imprecise. Performance is shown only for
outputs with at least 10 positive studies in the test set. "Not derived" means there were too few positive cases to
set a threshold; the deployed value is a 0.5 placeholder that almost never fires, and the output should be treated as
not supported.

**Calcification**

| Segment | Output column | Threshold | Positives / 3,194 | AUROC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| Proximal RCA | `prox_rca_calcif_binary_pred` | 0.0376 | 166 | 0.85 | 0.88 | 0.72 |
| Mid RCA | `mid_rca_calcif_binary_pred` | 0.0302 | 99 | 0.88 | 0.90 | 0.73 |
| Distal RCA | `dist_rca_calcif_binary_pred` | 0.0171 | 43 | 0.92 | 0.88 | 0.83 |
| Posterior descending artery | `pda_calcif_binary_pred` | 0.0066 | 15 | 0.89 | 0.80 | 0.80 |
| Posterolateral branch | `posterolateral_calcif_binary_pred` | 0.0012 | 10 | 0.85 | 1.00 | 0.62 |
| Left main | `left_main_calcif_binary_pred` | 0.0230 | 116 | 0.88 | 0.83 | 0.77 |
| Proximal LAD | `prox_lad_calcif_binary_pred` | 0.0729 | 214 | 0.80 | 0.69 | 0.77 |
| Mid LAD | `mid_lad_calcif_binary_pred` | 0.0894 | 168 | 0.83 | 0.72 | 0.78 |
| Distal LAD | `dist_lad_calcif_binary_pred` | 0.0026 | 10 | 0.84 | 0.80 | 0.72 |
| First diagonal | `D1_calcif_binary_pred` | 0.0206 | 41 | 0.81 | 0.78 | 0.76 |
| Second diagonal | `D2_calcif_binary_pred` | 0.0065 | 30 | 0.87 | 0.97 | 0.64 |
| Proximal LCx | `prox_lcx_calcif_binary_pred` | 0.0207 | 103 | 0.87 | 0.86 | 0.74 |
| Mid LCx | `mid_lcx_calcif_binary_pred` | 0.0062 | 26 | 0.85 | 0.92 | 0.69 |
| Distal LCx | `dist_lcx_calcif_binary_pred` | 0.0050 | 5 | n/a | n/a | n/a |
| First obtuse marginal | `om1_calcif_binary_pred` | 0.0104 | 21 | 0.88 | 0.81 | 0.79 |
| Second obtuse marginal | `om2_calcif_binary_pred` | 0.0091 | 21 | 0.88 | 0.90 | 0.70 |
| Ramus | `bx_calcif_binary_pred` | 0.0029 | 20 | 0.82 | 0.85 | 0.61 |
| Posterior left ventricular branch (LVP) | `lvp_calcif_binary_pred` | not derived (0.5 placeholder) | 2 | n/a | n/a | n/a |

**Chronic total occlusion**

| Segment | Output column | Threshold | Positives / 3,194 | AUROC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| Proximal RCA | `prox_rca_cto_pred` | 0.0134 | 89 | 0.96 | 0.93 | 0.86 |
| Mid RCA | `mid_rca_cto_pred` | 0.0225 | 57 | 0.95 | 0.91 | 0.85 |
| Distal RCA | `dist_rca_cto_pred` | 0.0043 | 18 | 0.94 | 1.00 | 0.80 |
| Posterior descending artery | `pda_cto_pred` | 0.0031 | 12 | 0.82 | 0.67 | 0.75 |
| Posterolateral branch | `posterolateral_cto_pred` | 0.0034 | 13 | 0.89 | 0.77 | 0.81 |
| Left main | `left_main_cto_pred` | not derived (0.5 placeholder) | 3 | n/a | n/a | n/a |
| Proximal LAD | `prox_lad_cto_pred` | 0.0043 | 22 | 0.94 | 0.95 | 0.78 |
| Mid LAD | `mid_lad_cto_pred` | 0.0159 | 50 | 0.93 | 0.90 | 0.82 |
| Distal LAD | `dist_lad_cto_pred` | 0.0017 | 9 | n/a | n/a | n/a |
| First diagonal | `D1_cto_pred` | 0.0030 | 14 | 0.83 | 0.93 | 0.56 |
| Second diagonal | `D2_cto_pred` | 0.0009 | 7 | n/a | n/a | n/a |
| Proximal LCx | `prox_lcx_cto_pred` | 0.0083 | 34 | 0.92 | 0.91 | 0.78 |
| Mid LCx | `mid_lcx_cto_pred` | 0.0112 | 22 | 0.86 | 0.73 | 0.83 |
| Distal LCx | `dist_lcx_cto_pred` | 0.0011 | 4 | n/a | n/a | n/a |
| First obtuse marginal | `om1_cto_pred` | 0.0019 | 9 | n/a | n/a | n/a |
| Second obtuse marginal | `om2_cto_pred` | 0.0086 | 12 | 0.76 | 0.67 | 0.81 |
| Ramus | `bx_cto_pred` | 0.0012 | 3 | n/a | n/a | n/a |
| Posterior left ventricular branch (LVP) | `lvp_cto_pred` | not derived (0.5 placeholder) | 4 | n/a | n/a | n/a |

**Thrombus**

| Segment | Output column | Threshold | Positives / 3,194 | AUROC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| Proximal RCA | `prox_rca_thrombus_pred` | 0.0109 | 18 | 0.92 | 0.89 | 0.84 |
| Mid RCA | `mid_rca_thrombus_pred` | 0.0126 | 16 | 0.95 | 0.81 | 0.89 |
| Distal RCA | `dist_rca_thrombus_pred` | 0.0058 | 10 | 0.96 | 1.00 | 0.81 |
| Posterior descending artery | `pda_thrombus_pred` | not derived (0.5 placeholder) | 1 | n/a | n/a | n/a |
| Posterolateral branch | `posterolateral_thrombus_pred` | 0.0017 | 3 | n/a | n/a | n/a |
| Left main | `left_main_thrombus_pred` | not derived (0.5 placeholder) | 4 | n/a | n/a | n/a |
| Proximal LAD | `prox_lad_thrombus_pred` | 0.0086 | 15 | 0.92 | 0.80 | 0.86 |
| Mid LAD | `mid_lad_thrombus_pred` | 0.0038 | 21 | 0.93 | 0.90 | 0.73 |
| Distal LAD | `dist_lad_thrombus_pred` | not derived (0.5 placeholder) | 2 | n/a | n/a | n/a |
| First diagonal | `D1_thrombus_pred` | not derived (0.5 placeholder) | 1 | n/a | n/a | n/a |
| Second diagonal | `D2_thrombus_pred` | not derived (0.5 placeholder) | 1 | n/a | n/a | n/a |
| Proximal LCx | `prox_lcx_thrombus_pred` | 0.0003 | 3 | n/a | n/a | n/a |
| Mid LCx | `mid_lcx_thrombus_pred` | 0.0028 | 5 | n/a | n/a | n/a |
| Distal LCx | `dist_lcx_thrombus_pred` | not derived (0.5 placeholder) | 0 | n/a | n/a | n/a |
| First obtuse marginal | `om1_thrombus_pred` | not derived (0.5 placeholder) | 1 | n/a | n/a | n/a |
| Second obtuse marginal | `om2_thrombus_pred` | not derived (0.5 placeholder) | 2 | n/a | n/a | n/a |
| Ramus | `bx_thrombus_pred` | not derived (0.5 placeholder) | 1 | n/a | n/a | n/a |
| Posterior left ventricular branch (LVP) | `lvp_thrombus_pred` | not derived (0.5 placeholder) | 0 | n/a | n/a | n/a |

## Limits of these numbers

- Single centre (Montreal Heart Institute); the reference is the operator's visual report, not quantitative coronary
  angiography or a core laboratory.
- The split used to derive the thresholds was not recorded with them. On the test set they lie close to the
  test-optimal points, so the sensitivity and specificity above may be slightly optimistic.
- The test-set figures were obtained with up to 12 diagnostic runs per study and no view filtering. The Docker
  pipeline of this repository filters runs with VasoVision and uses up to 10; expect small differences.
- Thresholds depend on prevalence, acquisition and case mix. **For another site, re-derive or at least check them on a
  local labelled calibration set that is separate from the evaluation set**, and report threshold-free metrics
  (AUROC, AUPRC, correlation with percent stenosis) next to the thresholded ones.
