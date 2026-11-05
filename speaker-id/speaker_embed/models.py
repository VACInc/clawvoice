"""Lazy ReDimNet primary and optional TitaNet-L embedding adapters."""

from __future__ import annotations

import hashlib
import logging
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import REDIMNET_REVISION, TITANET_REVISION
from .store import normalize

_LOGGER = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    name: str
    store_key: str

    @property
    def loaded(self) -> bool:
        """Whether model weights have been loaded."""

    @property
    def ready(self) -> bool:
        """Whether local model artifacts are present."""

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a one-dimensional normalized speaker embedding."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(torch_module: Any, requested: str) -> str:
    """Resolve auto/cpu/cuda with an explicit CPU fallback for auto."""
    value = requested.strip().lower()
    if value == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if value == "cpu":
        return value
    if value == "cuda" or value.startswith("cuda:"):
        if not torch_module.cuda.is_available():
            raise RuntimeError(f"DEVICE={requested} requested but CUDA is unavailable")
        return value
    raise ValueError("DEVICE must be auto, cpu, cuda, or cuda:N")


class ReDimNetBackend:
    name = f"redimnet-M-ft_mix@{REDIMNET_REVISION[:8]}"
    store_key = f"redimnet_M_ft_mix_{REDIMNET_REVISION[:8]}"

    def __init__(
        self,
        source_dir: Path,
        checkpoint: Path,
        checkpoint_sha256: str,
        device: str = "auto",
        verify_hash: bool = True,
    ):
        self.source_dir = Path(source_dir)
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = checkpoint_sha256.lower()
        self.requested_device = device
        self.verify_hash = verify_hash
        self._model: Any = None
        self._torch: Any = None
        self._torchaudio: Any = None
        self._device = "unloaded"
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def ready(self) -> bool:
        return (
            self.source_dir / "redimnet" / "model.py"
        ).is_file() and self.checkpoint.is_file()

    @property
    def device(self) -> str:
        return self._device

    def _load(self) -> None:
        if self._model is not None:
            return
        if not (self.source_dir / "redimnet" / "model.py").is_file():
            raise FileNotFoundError(
                f"ReDimNet source missing at {self.source_dir}; run fetch_models.py"
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"ReDimNet checkpoint missing at {self.checkpoint}; run fetch_models.py"
            )
        if self.verify_hash:
            actual_hash = file_sha256(self.checkpoint)
            if actual_hash != self.checkpoint_sha256:
                raise RuntimeError(
                    "ReDimNet checkpoint SHA-256 mismatch: "
                    f"expected {self.checkpoint_sha256}, got {actual_hash}"
                )

        import torch
        import torchaudio

        source_text = str(self.source_dir)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        from redimnet.model import ReDimNetWrap  # type: ignore[import-not-found]

        device = resolve_device(torch, self.requested_device)
        try:
            checkpoint_data = torch.load(
                self.checkpoint, map_location="cpu", weights_only=True
            )
        except TypeError:  # PyTorch 2.0 compatibility
            checkpoint_data = torch.load(self.checkpoint, map_location="cpu")
        if not isinstance(checkpoint_data, dict):
            raise RuntimeError("ReDimNet checkpoint is not a mapping")
        if not {"model_config", "state_dict"}.issubset(checkpoint_data):
            raise RuntimeError("ReDimNet checkpoint is missing model_config/state_dict")
        model = ReDimNetWrap(**checkpoint_data["model_config"])
        model.load_state_dict(checkpoint_data["state_dict"], strict=True)
        model.eval()
        model.to(device)
        self._torch = torch
        self._torchaudio = torchaudio
        self._device = device
        self._model = model
        _LOGGER.info("Loaded %s on %s", self.name, device)

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        with self._lock:
            self._load()
            torch = self._torch
            waveform = torch.from_numpy(
                np.ascontiguousarray(samples, dtype=np.float32)
            ).unsqueeze(0)
            if sample_rate != 16000:
                waveform = self._torchaudio.functional.resample(
                    waveform, sample_rate, 16000
                )
            waveform = waveform.to(self._device)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self._device.startswith("cuda")
                else nullcontext()
            )
            with torch.inference_mode(), amp_context:
                embedding = self._model(waveform)
            if isinstance(embedding, (tuple, list)):
                embedding = embedding[-1]
            value = embedding.detach().float().reshape(-1).cpu().numpy()
            return normalize(value)


class TitaNetBackend:
    name = f"titanet-large@{TITANET_REVISION[:8]}"
    store_key = f"titanet_large_{TITANET_REVISION[:8]}"

    def __init__(
        self,
        checkpoint: Path,
        checkpoint_sha256: str,
        device: str = "auto",
        verify_hash: bool = True,
    ):
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = checkpoint_sha256.lower()
        self.requested_device = device
        self.verify_hash = verify_hash
        self._model: Any = None
        self._torch: Any = None
        self._torchaudio: Any = None
        self._device = "unloaded"
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def ready(self) -> bool:
        return self.checkpoint.is_file()

    @property
    def device(self) -> str:
        return self._device

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"TitaNet checkpoint missing at {self.checkpoint}; "
                "run fetch_models.py --titanet"
            )
        if self.verify_hash:
            actual_hash = file_sha256(self.checkpoint)
            if actual_hash != self.checkpoint_sha256:
                raise RuntimeError(
                    "TitaNet checkpoint SHA-256 mismatch: "
                    f"expected {self.checkpoint_sha256}, got {actual_hash}"
                )

        import torch
        import torchaudio

        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
        except ImportError as exc:
            raise RuntimeError(
                "TITANET_ENABLED=1 requires requirements-titanet.txt"
            ) from exc

        device = resolve_device(torch, self.requested_device)
        model = EncDecSpeakerLabelModel.restore_from(
            restore_path=str(self.checkpoint), map_location=device
        )
        model.eval()
        model.to(device)
        self._torch = torch
        self._torchaudio = torchaudio
        self._device = device
        self._model = model
        _LOGGER.info("Loaded %s on %s", self.name, device)

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        with self._lock:
            self._load()
            torch = self._torch
            waveform = torch.from_numpy(
                np.ascontiguousarray(samples, dtype=np.float32)
            ).unsqueeze(0)
            if sample_rate != 16000:
                waveform = self._torchaudio.functional.resample(
                    waveform, sample_rate, 16000
                )
            waveform = waveform.to(self._device)
            lengths = torch.tensor(
                [waveform.shape[-1]], dtype=torch.long, device=self._device
            )
            with torch.inference_mode():
                output = self._model.forward(
                    input_signal=waveform, input_signal_length=lengths
                )
            embedding = output[-1] if isinstance(output, (tuple, list)) else output
            value = embedding.detach().float().reshape(-1).cpu().numpy()
            return normalize(value)
