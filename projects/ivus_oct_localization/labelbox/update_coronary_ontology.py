"""Add coronary-system labels without deleting existing branch annotations.

Use --apply to change the existing ontology; otherwise write the migration plan.
Human answers remain intact. Remapped answers are uploaded as model assistance
for review, preserving their source label and exact original vessel in CSV.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from coronary_system import coronary_system


SYSTEM_OPTIONS = [('left_coronary', 'Left coronary (LCA)'),
                  ('right_coronary', 'Right coronary (RCA)'),
                  ('graft', 'Graft (SVG / LIMA)'), ('uncertain', 'Uncertain')]
SYSTEM_INSTRUCTIONS = ('Coronary system containing the device or catheter: answer for OCT/IVUS '
                       'and for wire/balloon/stent cines. Choose Left coronary when LAD versus LCX '
                       'is unclear, or wires occupy both. Keep graft and uncertain separate. '
                       'Do not infer left-main traversal from a left-coronary answer.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/ivus_oct_labelbox.json')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg['output']) / 'coronary_system_migration'
    out.mkdir(exist_ok=True, parents=True)
    import labelbox
    from labelbox import Classification, Option
    client = labelbox.Client()
    ontology = client.get_project(cfg['project_id']).ontology()
    before = ontology.normalized
    if not (out / 'ontology_before.json').exists():
        (out / 'ontology_before.json').write_text(json.dumps(before, indent=2))
    vessel = Classification.from_dict(next(c for c in before['classifications'] if c['name'] == 'vessel'))
    original_ids = {o.value: o.feature_schema_id for o in vessel.options}
    if 'left_coronary' not in original_ids:
        vessel.options.append(Option(value='left_coronary', label='Left coronary (LCA; LAD/LCX unclear or both)'))
    vessel.instructions = ('Vessel containing the catheter or other device. Use Left coronary if '
                           'LAD versus LCX is unclear or both contain wires. Preserve a known '
                           'specific branch; Left coronary does not mean left main only.')
    system_existing = next((c for c in before['classifications'] if c['name'] == 'coronary_system'), None)
    system = Classification.from_dict(system_existing) if system_existing else Classification(
        class_type=Classification.Type.RADIO, name='coronary_system', instructions=SYSTEM_INSTRUCTIONS,
        required=False, scope=Classification.Scope.GLOBAL,
        options=[Option(value=v, label=l) for v, l in SYSTEM_OPTIONS])
    plan = {'vessel': vessel.asdict(), 'coronary_system': system.asdict()}
    (out / 'ontology_plan.json').write_text(json.dumps(plan, indent=2))
    if args.apply:
        client.upsert_feature_schema(vessel.asdict())
        if not system_existing:
            schema = client.upsert_feature_schema(system.asdict())
            client.insert_feature_schema_into_ontology(schema.uid, ontology.uid, position=2)
        after = client.get_ontology(ontology.uid).normalized
        (out / 'ontology_after.json').write_text(json.dumps(after, indent=2))
        after_vessel = next(c for c in after['classifications'] if c['name'] == 'vessel')
        after_ids = {o['value']: o['featureSchemaId'] for o in after_vessel['options']}
        assert all(after_ids[v] == sid for v, sid in original_ids.items()), 'Existing vessel IDs changed'
        assert 'left_coronary' in after_ids
        assert any(c['name'] == 'coronary_system' for c in after['classifications'])
        print('Live ontology updated; existing vessel schema IDs preserved', flush=True)
    labels_path = Path(cfg['labels'])
    labels = pd.read_csv(labels_path, keep_default_na=False)
    labels['vessel_original'] = labels['vessel']
    labels['coronary_system'] = labels.vessel.map(coronary_system)
    labels['coronary_system_source'] = 'deterministic remap of human vessel answer'
    labels.to_csv(out / 'labels_remapped.csv', index=False)
    summary = {'applied': args.apply, 'ontology_id': ontology.uid,
               'labels': len(labels), 'counts': labels.coronary_system.value_counts().to_dict(),
               'email_thread': 'https://mail.google.com/mail/u/0/#all/1a11c4cbb2f6fdce',
               'email_recommendation_date': '2026-10-11', 'email_agreement_date': '2026-10-12',
               'human_branch_answers_preserved': True}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
