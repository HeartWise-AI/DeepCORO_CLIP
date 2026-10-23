"""Regression checks for evaluation and human-label-preserving suggestions."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load_module('train_labelbox', 'train_labelbox.py')
sync = load_module('sync_classifier', 'labelbox/sync_classifier.py')
system = load_module('coronary_system', 'labelbox/coronary_system.py')


class ClassifierChecks(unittest.TestCase):
    def test_macro_metric_and_unsupported_class(self):
        result = train.metrics(np.array(['yes', 'no', 'ivus']),
                               np.array([[.1, .9], [.8, .2], [.4, .6]]), np.array(['no', 'yes']))
        self.assertAlmostEqual(result['macro_f1'], (1 + 2 / 3) / 3)
        self.assertEqual(result['report']['ivus']['recall'], 0)

    def test_failed_decodes_do_not_become_training_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'emb.npz'
            np.savez(path, paths=np.array(['valid', 'invalid']), mean=np.ones((2, 512)),
                     max=np.ones((2, 512)), ok=np.array([True, False]))
            features, _ = train.load_features(pd.DataFrame({'FileName': ['valid', 'invalid']}), [str(path)])
            self.assertTrue(np.isfinite(features[0]).all())
            self.assertTrue(np.isnan(features[1]).all())

    def test_only_answered_rows_are_protected(self):
        rows = [{'data_row': {'global_key': key}, 'projects': {'project': {'labels': [
            {'annotations': {'classifications': annotations}}]}}} for key, annotations in [
                ('answered', [{'name': 'intravascular_imaging', 'radio_answer': {'value': 'oct'}}]),
                ('skipped', [])]]
        self.assertEqual(sync.answered_keys(rows, 'project'), {'answered'})

    def test_secondary_questions_are_conditional(self):
        row = {'global_key': 'example', 'intravascular_imaging_suggestion': 'none',
               'intravascular_imaging_p_none': .7, 'vessel_suggestion': 'lad'}
        payload = sync.payload_for(row)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]['name'], 'intravascular_imaging')

    def test_coarse_remapping_preserves_ambiguity(self):
        for vessel in ['lm', 'lad', 'diagonal', 'lcx', 'om_ramus', 'left_coronary']:
            self.assertEqual(system.coronary_system(vessel), 'left_coronary')
        self.assertEqual(system.coronary_system('rca'), 'right_coronary')
        self.assertEqual(system.coronary_system('pda_plv'), 'uncertain')
        self.assertEqual(system.coronary_system('graft'), 'graft')
        self.assertEqual(system.coronary_system(''), '')

    def test_system_suggestion_also_applies_to_other_devices(self):
        row = {'global_key': 'example', 'intravascular_imaging_suggestion': 'other_device',
               'coronary_system_suggestion': 'left_coronary'}
        self.assertEqual([a['name'] for a in sync.payload_for(row)],
                         ['intravascular_imaging', 'coronary_system'])

    def test_stream_metadata_export(self):
        class Stream:
            def start(self, stream_handler):
                class Output:
                    json = {'data_row': {'global_key': 'example'}}
                stream_handler(Output())
        class Task:
            result = '{"result":{"totalLines":"1"}}'
            def get_buffered_stream(self):
                return Stream()
        self.assertEqual(sync.export_rows(Task()), [{'data_row': {'global_key': 'example'}}])

    def test_import_resolves_ids_independent_of_titles(self):
        ontology = {'classifications': [{'name': 'vessel', 'featureSchemaId': 'question',
                     'instructions': 'Vessel containing the device',
                     'options': [{'value': 'left_coronary', 'label': 'Left coronary (LCA)',
                                  'featureSchemaId': 'left-option'}]}]}
        payload = [{'dataRow': {'globalKey': 'example'}, 'name': 'vessel',
                    'answer': {'name': 'left_coronary'}}]
        resolved = sync.resolve_schema_ids(payload, ontology)
        self.assertEqual(resolved[0]['schemaId'], 'question')
        self.assertEqual(resolved[0]['answer'], {'schemaId': 'left-option'})


if __name__ == '__main__':
    unittest.main()
