"""Prepare/upload model-assisted suggestions and export current Labelbox labels.

Without --upload this only writes a review queue and MAL payload on NAS.
With --upload, LABELBOX_API_KEY is required. A fresh export filters out any
new human answers before submission. Suggestions never become human labels.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def answered_keys(rows, project_id):
    return {row['data_row']['global_key'] for row in rows
            if any(c.get('name') == 'intravascular_imaging' and c.get('radio_answer')
                   for lab in row.get('projects', {}).get(project_id, {}).get('labels', [])
                   for c in lab.get('annotations', {}).get('classifications', []))}


def payload_for(row):
    output = []
    primary = row['intravascular_imaging_suggestion']
    tasks = ['intravascular_imaging']
    if primary in ['oct', 'ivus', 'imaging_uncertain']:
        tasks += ['vessel', 'acquisition_phase']
    for task in tasks:
        value = row.get(f'{task}_suggestion')
        if not value:
            continue
        output.append({'dataRow': {'globalKey': row['global_key']}, 'name': task,
                       'answer': {'name': value},
                       'confidence': float(row[f'{task}_p_{value}'])})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/ivus_oct_labelbox.json')
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg['output'])
    with open(cfg['export']) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    client = None
    if args.upload:
        import labelbox
        client = labelbox.Client()
        task = client.get_project(cfg['project_id']).export(params={'label_details': True})
        task.wait_till_done()
        rows = task.result
        if not isinstance(rows, list) or not rows:
            raise ValueError('Fresh project export did not return any rows')
        with (out / 'current_labelbox_export.ndjson').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
    answered = answered_keys(rows, cfg['project_id'])
    predictions = pd.read_csv(out / 'all_predictions.csv', keep_default_na=False)
    queue = predictions.loc[~predictions.global_key.isin(answered)].copy()
    if not queue.embedding_valid.all():
        raise ValueError('Review queue contains missing embeddings')
    queue['review_priority'] = 1 - queue.filter(regex='^intravascular_imaging_p_').max(axis=1)
    queue = queue.sort_values('review_priority', ascending=False)
    queue.to_csv(out / 'reannotation_queue.csv', index=False)
    payload = [annotation for row in queue.to_dict('records') for annotation in payload_for(row)]
    with (out / 'mal_predictions.ndjson').open('w') as stream:
        for annotation in payload:
            stream.write(json.dumps(annotation) + '\n')
    status = {'n_answered': len(answered), 'n_review_cines': len(queue),
              'n_annotations': len(payload), 'uploaded': False,
              'source': 'fresh export' if args.upload else cfg['export']}
    if args.upload:
        from labelbox.schema.annotation_import import MALPredictionImport
        job = MALPredictionImport.create_from_objects(
            client, project_id=cfg['project_id'], name=cfg['name'] + '-suggestions', predictions=payload)
        job.wait_until_done()
        status.update({'job_id': job.uid, 'errors': job.errors, 'uploaded': not bool(job.errors)})
    (out / 'labelbox_sync.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
