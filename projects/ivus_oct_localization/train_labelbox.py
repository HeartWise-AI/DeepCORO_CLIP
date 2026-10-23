"""Train frozen DeepCORO-CLIP classifier heads from exported human labels.

Run from the repository root: python projects/ivus_oct_localization/train_labelbox.py
Patient identifiers, label exports and predictions stay on NAS. Missing/uncertain
answers are never converted into negative ground truth. Rare classes are explicitly
reported as unsupported. Predictions are suggestions for human review.
"""
import argparse
import glob
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_features(manifest, patterns):
    """Read shards one at a time, retaining only manifest rows with valid decoding."""
    lookup = {p: i for i, p in enumerate(manifest.FileName)}
    if len(lookup) != len(manifest):
        raise ValueError('Manifest FileName must be unique')
    features = np.full((len(manifest), 1024), np.nan, dtype=np.float32)
    sources = []
    for file in sorted({f for pat in patterns for f in glob.glob(pat)}):
        with np.load(file, allow_pickle=True) as shard:
            paths = shard['paths']
            keep = [(i, lookup[p]) for i, p in enumerate(paths)
                    if p in lookup and shard['ok'][i]]
            if keep:
                source, target = np.array(keep).T
                values = np.concatenate([shard['mean'][source], shard['max'][source]], axis=1)
                features[target] = values
            sources.append({'path': file, 'matched_valid_rows': len(keep)})
        print('Loaded', file, len(keep), flush=True)
    return features, sources


def metrics(y, probabilities, classes):
    predictions = classes[probabilities.argmax(axis=1)]
    result = {'n': len(y), 'accuracy': accuracy_score(y, predictions),
              'macro_f1': f1_score(y, predictions, average='macro', zero_division=0),
              'report': classification_report(y, predictions, output_dict=True, zero_division=0)}
    for i, cls in enumerate(classes):
        target = (y == cls).astype(int)
        if len(np.unique(target)) == 2:
            result[f'auroc/{cls}'] = roc_auc_score(target, probabilities[:, i])
            result[f'auprc/{cls}'] = average_precision_score(target, probabilities[:, i])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/ivus_oct_labelbox.json')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.json').exists():
        raise FileExistsError(f'Completed run already exists: {out}')
    manifest = pd.read_csv(cfg['manifest'], keep_default_na=False)
    labels = pd.read_csv(cfg['labels'], keep_default_na=False)
    if labels.global_key.duplicated().any():
        raise ValueError('Resolve duplicate or conflicting labels before training')
    cohort = pd.read_csv(cfg['cohort'], dtype={'Patient_ID': str})
    if cohort.Patient_ID.isna().any() or cohort.groupby('Patient_ID').split.nunique().max() != 1:
        raise ValueError('Missing patient identifiers or patient leakage')
    labels = labels.merge(cohort[['StudyInstanceUID', 'Patient_ID', 'split']],
                          on='StudyInstanceUID', validate='many_to_one')
    if len(labels) != len(pd.read_csv(cfg['labels'])) or labels.split.isna().any():
        raise ValueError('Unmapped labels')
    X, sources = load_features(manifest, cfg['embedding_globs'])
    valid = np.isfinite(X).all(axis=1)
    indices = pd.Series(np.arange(len(manifest)), index=manifest.global_key)
    label_indices = indices.loc[labels.global_key].to_numpy()
    if not valid[label_indices].all():
        raise ValueError('Labelled rows with missing or invalid embeddings')
    L = X[label_indices]
    labels.to_csv(out / 'labels_with_split.csv', index=False)
    shutil.copy2(args.config, out / 'config.json')
    imaging = labels.intravascular_imaging.isin(['oct', 'ivus', 'imaging_uncertain'])
    labels['imaging_catheter'] = labels.intravascular_imaging.map(
        {'oct': 'yes', 'ivus': 'yes', 'none': 'no', 'other_device': 'no'}).fillna('')
    tasks = cfg.get('tasks', ['imaging_catheter', 'intravascular_imaging', 'vessel',
                             'left_main_traversed', 'acquisition_phase'])
    import wandb
    run = wandb.init(entity=cfg['wandb_entity'], project=cfg['wandb_project'], name=cfg['name'],
                     config=cfg, mode='offline' if args.offline else 'online',
                     tags=['frozen-encoder', 'human-labels', 'patient-disjoint', 'review-required'])
    (out / 'wandb.json').write_text(json.dumps({'id': run.id, 'url': run.url}, indent=2))
    all_metrics, models = {}, {}
    predictions = manifest[['global_key', 'StudyInstanceUID', 'FileName']].copy()
    predictions['embedding_valid'] = valid
    for task in tasks:
        eligible = labels[task].ne('')
        if task in ['vessel', 'left_main_traversed', 'acquisition_phase']:
            eligible &= imaging
        counts = labels.loc[eligible & labels.split.eq('train'), task].value_counts()
        supported = counts[counts >= cfg['min_train_per_class']].index.tolist()
        train = eligible & labels.split.eq('train') & labels[task].isin(supported)
        val = eligible & labels.split.eq('val')
        test = eligible & labels.split.eq('test')
        info = {'counts': {s: labels.loc[eligible & labels.split.eq(s), task].value_counts().to_dict()
                           for s in ['train', 'val', 'test']},
                'unsupported_classes': sorted(set(labels.loc[eligible, task]) - set(supported))}
        if len(supported) < 2 or not val.any():
            info['status'] = 'insufficient_support'
            all_metrics[task] = info
            continue
        best, best_score = None, -1
        sweep = []
        for c in cfg['C_grid']:
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=c, class_weight='balanced', max_iter=3000, random_state=cfg['seed']))
            model.fit(L[train], labels.loc[train, task])
            score = f1_score(labels.loc[val, task], model.predict(L[val]),
                             average='macro', zero_division=0)
            sweep.append({'C': c, 'validation_macro_f1': score})
            if score > best_score:
                best, best_score = model, score
        classes = best.classes_
        info.update({'status': 'trained', 'sweep': sweep,
                     'selected_C': best.named_steps['logisticregression'].C,
                     'classes': classes.tolist()})
        for split, mask in [('val', val), ('test', test)]:
            if not mask.any():
                continue
            probs = best.predict_proba(L[mask])
            info[split] = metrics(labels.loc[mask, task].to_numpy(), probs, classes)
            frame = labels.loc[mask, ['global_key', task, 'split']].copy()
            frame['prediction'] = classes[probs.argmax(axis=1)]
            for i, cls in enumerate(classes):
                frame[f'p_{cls}'] = probs[:, i]
            frame.to_csv(out / f'{task}_{split}_predictions.csv', index=False)
            run.log({f'{task}/{split}/{k}': v for k, v in info[split].items()
                     if isinstance(v, (int, float))})
        probs = best.predict_proba(X[valid])
        predictions.loc[valid, f'{task}_suggestion'] = classes[probs.argmax(axis=1)]
        for i, cls in enumerate(classes):
            predictions.loc[valid, f'{task}_p_{cls}'] = probs[:, i]
        models[task] = best
        all_metrics[task] = info
        print(task, json.dumps(info), flush=True)
    joblib.dump({'models': models, 'config': cfg, 'feature_dim': 1024}, out / 'classifier.joblib')
    predictions['human_label_exists'] = predictions.global_key.isin(labels.global_key)
    predictions['requires_human_review'] = True
    predictions.to_csv(out / 'all_predictions.csv', index=False)
    predictions.loc[~predictions.human_label_exists].to_csv(out / 'remaining_predictions.csv', index=False)
    audit = {'source_export_sha256': sha256(cfg['export']), 'source_labels_sha256': sha256(cfg['labels']),
             'manifest_sha256': sha256(cfg['manifest']), 'embedding_sources': sources,
             'n_labels': len(labels), 'n_manifest': len(manifest), 'n_valid_embeddings': int(valid.sum()),
             'n_remaining': int((~predictions.human_label_exists).sum()),
             'encoder': cfg['encoder_checkpoint'], 'source_export_date': cfg.get('source_export_date', '2026-10-19'),
             'limitations': ['Labels have not been marked reviewed in Labelbox.',
                             'Only one IVUS label; IVUS subtype unsupported.',
                             'Candidate-enriched cohort; performance does not estimate screening population performance.',
                             'Model predictions require human review; inspect labelbox_sync.json for upload status.']}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2))
    (out / 'metrics.json').write_text(json.dumps(all_metrics, indent=2))
    artifact = wandb.Artifact(cfg['name'], type='model', metadata={'encoder': cfg['encoder_checkpoint']})
    for filename in ['classifier.joblib', 'config.json', 'metrics.json', 'audit.json']:
        artifact.add_file(str(out / filename))
    run.log_artifact(artifact)
    run.finish()
    print('Completed:', out, flush=True)


if __name__ == '__main__':
    main()
