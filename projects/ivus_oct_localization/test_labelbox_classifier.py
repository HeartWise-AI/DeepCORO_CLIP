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


if __name__ == '__main__':
    unittest.main()
