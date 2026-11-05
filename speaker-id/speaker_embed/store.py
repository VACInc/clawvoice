"""Flat-file enrollment embedding store."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Mapping, Protocol

import numpy as np

_LOGGER = logging.getLogger(__name__)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("embedding has zero or non-finite norm")
    return np.ascontiguousarray(value / norm, dtype=np.float32)


class EmbeddingStore(Protocol):
    def add(self, model_key: str, user: str, embedding: np.ndarray) -> int:
        """Append an embedding and return the user's sample count."""

    def centroids(self, model_key: str) -> Mapping[str, np.ndarray]:
        """Return normalized per-user average embeddings."""

    def summary(self) -> Mapping[str, Mapping[str, int]]:
        """Return model -> user -> sample count."""


class FlatFileEmbeddingStore:
    """One NPY plus one metadata JSON file per model/user pair."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _validate_component(value: str, label: str) -> None:
        if not _SAFE_COMPONENT.fullmatch(value):
            raise ValueError(f"invalid {label}: {value!r}")

    def _path(self, model_key: str, user: str) -> Path:
        self._validate_component(model_key, "model key")
        self._validate_component(user, "user")
        return self.root / f"{model_key}--{user}.npy"

    @staticmethod
    def _load_matrix(path: Path) -> np.ndarray:
        matrix = np.load(path, allow_pickle=False)
        if (matrix.ndim != 2) or (matrix.shape[0] < 1) or (matrix.shape[1] < 1):
            raise ValueError(f"invalid embedding matrix in {path.name}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"non-finite embedding in {path.name}")
        return np.asarray(matrix, dtype=np.float32)

    def add(self, model_key: str, user: str, embedding: np.ndarray) -> int:
        value = normalize(embedding)
        path = self._path(model_key, user)
        with self._lock:
            if path.exists():
                current = self._load_matrix(path)
                if current.shape[1] != value.size:
                    raise ValueError(
                        f"embedding dimension changed from {current.shape[1]} "
                        f"to {value.size} for {model_key}/{user}"
                    )
                matrix = np.concatenate((current, value[None, :]), axis=0)
            else:
                matrix = value[None, :]

            token = uuid.uuid4().hex
            temp_path = path.with_name(f".{path.name}.{token}.tmp")
            with temp_path.open("wb") as temp_file:
                np.save(temp_file, matrix, allow_pickle=False)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, path)

            metadata = {
                "model": model_key,
                "user": user,
                "samples": int(matrix.shape[0]),
                "dimension": int(matrix.shape[1]),
                "updated_at": time.time(),
            }
            metadata_path = path.with_suffix(".json")
            metadata_temp = metadata_path.with_name(
                f".{metadata_path.name}.{token}.tmp"
            )
            with metadata_temp.open("w", encoding="utf-8") as temp_file:
                json.dump(metadata, temp_file, sort_keys=True)
                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(metadata_temp, metadata_path)
            return int(matrix.shape[0])

    def centroids(self, model_key: str) -> Mapping[str, np.ndarray]:
        self._validate_component(model_key, "model key")
        result: Dict[str, np.ndarray] = {}
        prefix = f"{model_key}--"
        with self._lock:
            for path in sorted(self.root.glob(f"{prefix}*.npy")):
                user = path.stem[len(prefix) :]
                try:
                    self._validate_component(user, "user")
                    matrix = self._load_matrix(path)
                    result[user] = normalize(matrix.mean(axis=0))
                except Exception:  # fail closed if one enrollment file is corrupt
                    _LOGGER.exception("Ignoring invalid enrollment file %s", path.name)
        return result

    def summary(self) -> Mapping[str, Mapping[str, int]]:
        result: Dict[str, Dict[str, int]] = {}
        with self._lock:
            for path in sorted(self.root.glob("*.npy")):
                try:
                    model_key, user = path.stem.split("--", 1)
                    self._validate_component(model_key, "model key")
                    self._validate_component(user, "user")
                    matrix = self._load_matrix(path)
                except Exception:
                    _LOGGER.exception("Ignoring invalid enrollment file %s", path.name)
                    continue
                result.setdefault(model_key, {})[user] = int(matrix.shape[0])
        return result
