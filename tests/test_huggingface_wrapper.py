from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from utils.huggingface_wrapper import HuggingFaceWrapper


def test_existing_partial_directory_is_resumed(tmp_path):
    (tmp_path / 'model.pt').write_bytes(b'weights')
    wrapper = HuggingFaceWrapper()
    wrapper.api = MagicMock()
    wrapper.api.model_info.return_value = SimpleNamespace(siblings=[SimpleNamespace(rfilename='model.pt')])
    with patch('utils.huggingface_wrapper.snapshot_download', return_value=str(tmp_path)) as download:
        wrapper.get_model('heartwise/model', tmp_path, revision='commit', allow_patterns=['*.pt'], cache_dir='/cache')
        download.reset_mock()
        result = wrapper.get_model(
            'heartwise/model', tmp_path, revision='commit', local_files_only=True,
            allow_patterns=['*.pt'], cache_dir='/cache',
        )
    assert result == str(tmp_path)
    assert download.call_args.kwargs['revision'] == 'commit'
    assert download.call_args.kwargs['local_files_only'] is True
    assert download.call_args.kwargs['token'] is None
    (tmp_path / 'model.pt').unlink()
    with patch('utils.huggingface_wrapper.snapshot_download', return_value=str(tmp_path)):
        with pytest.raises(FileNotFoundError, match='Incomplete'):
            wrapper.get_model('heartwise/model', tmp_path, revision='commit', local_files_only=True, allow_patterns=['*.pt'])


def test_offline_partial_folder_without_manifest_fails(tmp_path):
    with patch('utils.huggingface_wrapper.snapshot_download', return_value=str(tmp_path)):
        with pytest.raises(ValueError, match='completion manifest'):
            HuggingFaceWrapper().get_model('owner/model', tmp_path, local_files_only=True)


def test_online_fallback_to_incomplete_local_folder_fails(tmp_path):
    wrapper = HuggingFaceWrapper()
    wrapper.api = MagicMock()
    wrapper.api.model_info.return_value = SimpleNamespace(siblings=[SimpleNamespace(rfilename='model.pt')])
    with patch('utils.huggingface_wrapper.snapshot_download', return_value=str(tmp_path)):
        with pytest.raises(FileNotFoundError, match='Incomplete snapshot'):
            wrapper.get_model('owner/model', tmp_path)


def test_upload_is_one_folder_commit_and_errors_propagate(tmp_path):
    (tmp_path / 'model.pt').write_bytes(b'checkpoint')
    with patch('utils.huggingface_wrapper.HfApi') as api:
        api.return_value.upload_folder.return_value = 'commit'
        wrapper = HuggingFaceWrapper('test-token')
        assert wrapper.upload_model('owner/model', tmp_path, 'Weights') == 'commit'
        api.return_value.create_repo.assert_called_once_with(
            repo_id='owner/model', repo_type='model', private=True, exist_ok=True,
        )
        assert api.return_value.upload_folder.call_args.kwargs['commit_message'] == 'Weights'
        api.return_value.upload_folder.side_effect = PermissionError('denied')
        with pytest.raises(PermissionError):
            wrapper.upload_model('owner/model', tmp_path)


def test_upload_validates_folder_before_creating_repo(tmp_path):
    wrapper = HuggingFaceWrapper()
    wrapper.api = MagicMock()
    with pytest.raises(ValueError, match='empty'):
        wrapper.upload_model('owner/model', tmp_path)
    with pytest.raises(FileNotFoundError):
        wrapper.upload_model('owner/model', tmp_path / 'missing')
    wrapper.api.create_repo.assert_not_called()


def test_official_downloads_default_to_pinned_revision(tmp_path):
    from utils.huggingface_wrapper import DEFAULT_REVISIONS
    (tmp_path / 'model.pt').write_bytes(b'weights')
    with patch('utils.huggingface_wrapper.snapshot_download', return_value=str(tmp_path)) as download:
        wrapper = HuggingFaceWrapper()
        wrapper.api = MagicMock()
        wrapper.api.model_info.return_value = SimpleNamespace(siblings=[SimpleNamespace(rfilename='model.pt')])
        wrapper.get_model('heartwise/deepcoro_clip_stenosis')
        assert download.call_args.kwargs['revision'] == DEFAULT_REVISIONS['heartwise/deepcoro_clip_stenosis']
        wrapper.get_model('heartwise/deepcoro_clip_stenosis', revision='custom')
        assert download.call_args.kwargs['revision'] == 'custom'


def test_downloader_can_use_sdk_login_without_legacy_key_file():
    from utils import download_pretrained_weights as downloader
    with patch('utils.files_handler.read_api_key', side_effect=FileNotFoundError), patch.object(
        downloader, 'download_selected_models',
    ) as download, patch.dict('os.environ', {}, clear=True):
        downloader.main(['--models', 'stenosis'])
    assert download.call_args.args[1] is None
