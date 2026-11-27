"""Prepare/upload model-assisted suggestions and export current Labelbox labels.

Without --upload this only writes a review queue and MAL payload on NAS.
With --upload, LABELBOX_API_KEY is required. A fresh export filters out any
new human answers before submission. Suggestions never become human labels.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def export_rows(task):
    """SDK releases return either in-memory rows or a signed NDJSON URL."""
    result = task.result
    if isinstance(result, str):
        if result.lstrip().startswith('{'):
            rows = []
            task.get_buffered_stream().start(stream_handler=lambda output: rows.append(output.json))
            return rows
        import requests
        response = requests.get(result, timeout=120)
        response.raise_for_status()
        return [json.loads(line) for line in response.text.splitlines() if line.strip()]
    return result


def answered_keys(rows, project_id):
    return {row['data_row']['global_key'] for row in rows
            if any(c.get('name') == 'intravascular_imaging' and c.get('radio_answer')
                   for lab in row.get('projects', {}).get(project_id, {}).get('labels', [])
                   for c in lab.get('annotations', {}).get('classifications', []))}


def payload_for(row):
    output = []
    primary = row['intravascular_imaging_suggestion']
    tasks = ['intravascular_imaging']
    if row.get('coronary_system_suggestion'):
        tasks.append('coronary_system')
    if primary in ['oct', 'ivus', 'imaging_uncertain']:
        tasks += ['vessel', 'acquisition_phase']
    for task in tasks:
        value = row.get(f'{task}_suggestion')
        if not value:
            continue
        output.append({'dataRow': {'globalKey': row['global_key']}, 'name': task,
                       'answer': {'name': value}})
    return output


def resolve_schema_ids(payload, ontology):
    """Names/values can differ from display titles; import by immutable IDs."""
    questions = {q['name']: q for q in ontology['classifications']}
    result = []
    for annotation in payload:
        question = questions[annotation['name']]
        options = {o['value']: o for o in question['options']}
        option = options[annotation['answer']['name']]
        result.append({'dataRow': annotation['dataRow'],
                       'schemaId': question['featureSchemaId'],
                       'answer': {'schemaId': option['featureSchemaId']}})
    return result


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
        rows = export_rows(task)
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
    if args.upload:
        ontology = client.get_project(cfg['project_id']).ontology().normalized
        (out / 'upload_ontology.json').write_text(json.dumps(ontology, indent=2))
        payload = resolve_schema_ids(payload, ontology)
    with (out / 'mal_predictions.ndjson').open('w') as stream:
        for annotation in payload:
            stream.write(json.dumps(annotation) + '\n')
    status = {'n_answered': len(answered), 'n_review_cines': len(queue),
              'n_annotations': len(payload), 'uploaded': False,
              'source': 'fresh export' if args.upload else cfg['export']}
    if args.upload:
        from labelbox.schema.annotation_import import MALPredictionImport
        job = MALPredictionImport.create_from_objects(
            client, project_id=cfg['project_id'],
            name=cfg['name'] + '-suggestions-' + datetime.now(timezone.utc).strftime('%H%M%S'),
            predictions=payload)
        status.update({'job_id': job.uid, 'state': str(job.state)})
        (out / 'labelbox_sync.json').write_text(json.dumps(status, indent=2))
        job.wait_till_done()
        status.update({'job_id': job.uid, 'state': str(job.state),
                       'errors': job.errors,
                       'uploaded': job.state.value == 'FINISHED' and not bool(job.errors)})
    (out / 'labelbox_sync.json').write_text(json.dumps(status, indent=2))
    summary = {key: value for key, value in status.items() if key != 'errors'}
    summary['error_count'] = len(status.get('errors') or [])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
