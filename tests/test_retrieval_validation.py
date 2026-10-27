from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from utils.retrieval_inference import (
    build_inference_rows, extract_video_embeddings, load_text_embeddings,
    run_retrieval_metadata_inference,
)


def test_masked_clip_pooling_ignores_padding_and_checks_shape():
    embeds = torch.tensor([[[2., 4.], [0., 0.]]])
    assert torch.equal(extract_video_embeddings(embeds, torch.tensor([[True, False]])), embeds[:, 0])
    with pytest.raises(ValueError, match='mask shape'):
        extract_video_embeddings(embeds, torch.ones(1, 3))


def test_nonfinite_text_embeddings_rejected(tmp_path):
    path = tmp_path / 'embeds.pt'
    torch.save(torch.tensor([[float('nan')]]), path)
    with pytest.raises(ValueError, match='non-finite'):
        load_text_embeddings(str(path), torch.device('cpu'))


def test_training_metadata_cannot_overwrite_input_identity():
    rows = build_inference_rows(
        topk_indices=torch.tensor([[0]]), identifiers=['input.mp4'],
        metadata=pd.DataFrame({'video_name': ['training.mp4'], 'stenosis': [50.]}),
        dataset=SimpleNamespace(multi_video_mode=False), groupby_col_name='study_id',
    )
    assert rows == [{'video_name': 'input.mp4', 'stenosis': 50.}]


class Loader:
    dataset = SimpleNamespace(multi_video_mode=False)

    def __init__(self, with_ids=True):
        self.with_ids = with_ids

    def __iter__(self):
        batch = {'videos': torch.ones(1, 1, 3, 2, 2, 2)}
        if self.with_ids:
            batch['paths'] = ['input.mp4']
        yield batch


class Encoder:
    def eval(self):
        return self

    def __call__(self, videos):
        assert torch.is_inference_mode_enabled()
        return torch.tensor([[1., 0.]])


def run(tmp_path, metadata, with_ids=True):
    path = tmp_path / 'embeds.pt'
    torch.save(torch.tensor([[1., 0.], [0., 1.]]), path)
    config = SimpleNamespace(text_embeddings_path=str(path), metadata_path='metadata.parquet', topk=1, is_ref_device=True)
    with patch('utils.retrieval_inference.pd.read_parquet', return_value=metadata):
        return run_retrieval_metadata_inference(
            video_encoder=Encoder(), val_loader=Loader(with_ids), config=config,
            device_id='cpu', world_size=1, output_dir=str(tmp_path),
        )


def test_embeddings_metadata_alignment_is_required(tmp_path):
    with pytest.raises(ValueError, match='same number'):
        run(tmp_path, pd.DataFrame({'value': [1]}))


def test_missing_identifiers_cannot_silently_repeat_across_batches(tmp_path):
    with pytest.raises(ValueError, match='identifier'):
        run(tmp_path, pd.DataFrame({'value': [1, 2]}), with_ids=False)


def test_real_tensor_retrieval_writes_predictions(tmp_path):
    assert run(tmp_path, pd.DataFrame({'value': [1, 2]})) == [{'video_name': 'input.mp4', 'value': 1.}]
    assert pd.read_csv(tmp_path / 'averaged_metadata.csv').iloc[0]['video_name'] == 'input.mp4'
