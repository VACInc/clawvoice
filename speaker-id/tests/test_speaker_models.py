"""Embedding adapter contracts using tiny tensors and fake dependency modules."""
from contextlib import nullcontext
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from speaker_embed.models import ReDimNetBackend, TitaNetBackend
from speaker_embed.store import FlatFileEmbeddingStore
from speaker_embed.config import SpeakerConfig


def test_negative_similarity_threshold_and_port_lower_boundary():
    config = SpeakerConfig.from_env({'LOW_THRESHOLD': '-0.5', 'HIGH_THRESHOLD': '0.5'})
    assert config.low_threshold == -0.5
    with pytest.raises(ValueError, match='SPEAKER_PORT'):
        SpeakerConfig.from_env({'SPEAKER_PORT': '0'})
    with pytest.raises(ValueError, match='MAX_AUDIO_SECONDS'):
        SpeakerConfig.from_env({'MAX_AUDIO_SECONDS': '0'})


def test_centroid_averages_samples_not_embedding_dimensions(tmp_path):
    store = FlatFileEmbeddingStore(tmp_path)
    store.add('model', 'alice', np.array([1.0, 0.0, 0.0]))
    store.add('model', 'alice', np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(store.centroids('model')['alice'],
                               [2**-0.5, 2**-0.5, 0.0], rtol=1e-6)


def test_backend_device_properties_report_loaded_device(tmp_path):
    for backend in (ReDimNetBackend(tmp_path, tmp_path / 'fake', ''),
                    TitaNetBackend(tmp_path / 'fake', '')):
        assert backend.device == 'unloaded'
        backend._device = 'cpu'
        assert backend.device == 'cpu'


def fake_imports(monkeypatch, *, checkpoint):
    torch = types.ModuleType('torch')
    torch.load = lambda *a, **kw: checkpoint
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torchaudio', types.ModuleType('torchaudio'))


def test_redimnet_rejects_checkpoint_without_required_mapping(tmp_path, monkeypatch):
    (tmp_path / 'redimnet').mkdir()
    (tmp_path / 'redimnet/model.py').write_text('# fixture import is replaced below\n')
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.write_bytes(b'fixture')
    fake_imports(monkeypatch, checkpoint={})
    package = types.ModuleType('redimnet')
    model = types.ModuleType('redimnet.model')
    model.ReDimNetWrap = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, 'redimnet', package)
    monkeypatch.setitem(sys.modules, 'redimnet.model', model)
    backend = ReDimNetBackend(tmp_path, checkpoint, '', verify_hash=False)
    with pytest.raises(RuntimeError, match='missing model_config/state_dict'):
        backend._load()
    assert backend.loaded is False


def test_titanet_missing_dependency_has_actionable_error(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.write_bytes(b'fixture')
    fake_imports(monkeypatch, checkpoint={})
    monkeypatch.setitem(sys.modules, 'nemo', None)
    backend = TitaNetBackend(checkpoint, '', verify_hash=False)
    with pytest.raises(RuntimeError, match='requires requirements-titanet.txt'):
        backend._load()
    assert backend.loaded is False


def test_redimnet_tuple_output_uses_final_embedding(tmp_path):
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)
        def unsqueeze(self, axis):
            return self
        def to(self, device):
            return self
        def detach(self):
            return self
        def float(self):
            return self
        def reshape(self, *shape):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.values

    backend = ReDimNetBackend(tmp_path, tmp_path / 'unused', '')
    backend._load = lambda: None
    backend._device = 'cpu'
    backend._torch = types.SimpleNamespace(from_numpy=Tensor, inference_mode=nullcontext)
    backend._model = lambda waveform: (Tensor([1, 0]), Tensor([0, 1]))
    np.testing.assert_array_equal(backend.embed(np.zeros(16), 16000), [0, 1])
