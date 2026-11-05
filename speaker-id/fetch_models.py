#!/usr/bin/env python3
"""Fetch immutable ReDimNet source/weights and optional TitaNet-L weights."""

from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

from speaker_embed.config import (
    REDIMNET_CHECKPOINT_SHA256,
    REDIMNET_REVISION,
    TITANET_CHECKPOINT_SHA256,
    TITANET_REVISION,
)

REDIMNET_SOURCE_URL = (
    f"https://github.com/IDRnD/ReDimNet/archive/{REDIMNET_REVISION}.tar.gz"
)
REDIMNET_WEIGHT_URL = (
    "https://github.com/IDRnD/ReDimNet/releases/download/latest/"
    "M-vb2+vox2+cnc-ft_mix.pt"
)
TITANET_WEIGHT_URL = (
    "https://huggingface.co/nvidia/speakerverification_en_titanet_large/"
    f"resolve/{TITANET_REVISION}/speakerverification_en_titanet_large.nemo"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str | None) -> None:
    if destination.exists():
        if (expected_sha256 is None) or (_sha256(destination) == expected_sha256):
            print(f"ok: {destination}")
            return
        raise RuntimeError(
            f"existing file has the wrong SHA-256: {destination}; move it aside first"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    request = urllib.request.Request(
        url, headers={"User-Agent": "speaker-id-fetch-models/1.0"}
    )
    print(f"fetch: {url}")
    with (
        urllib.request.urlopen(request, timeout=60) as response,
        partial.open("wb") as output_file,
    ):
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output_file.write(chunk)
        output_file.flush()
        os.fsync(output_file.fileno())
    if expected_sha256 is not None:
        actual = _sha256(partial)
        if actual != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}"
            )
    os.replace(partial, destination)
    print(f"saved: {destination}")


def _extract_source(archive: Path, destination: Path) -> None:
    marker = destination / ".revision"
    if (
        marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == REDIMNET_REVISION
    ):
        print(f"ok: {destination}")
        return
    if destination.exists():
        raise RuntimeError(
            f"source destination already exists without the expected marker: {destination}"
        )
    destination.mkdir(parents=True)
    allowed_roots = {
        f"ReDimNet-{REDIMNET_REVISION}",
        f"redimnet-{REDIMNET_REVISION}",
    }
    with tarfile.open(archive, "r:gz") as source_tar:
        for member in source_tar.getmembers():
            member_parts = Path(member.name).parts
            if not member_parts or member_parts[0] not in allowed_roots:
                continue
            relative_path = Path(*member_parts[1:])
            if not relative_path.parts:
                continue
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RuntimeError(f"unsafe archive member: {member.name}")
            target = destination / relative_path
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(f"unsupported archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source_file = source_tar.extractfile(member)
            if source_file is None:
                raise RuntimeError(f"could not read archive member: {member.name}")
            with source_file, target.open("xb") as output_file:
                while True:
                    chunk = source_file.read(1024 * 1024)
                    if not chunk:
                        break
                    output_file.write(chunk)
    if not (destination / "redimnet" / "model.py").is_file():
        raise RuntimeError("downloaded archive did not contain redimnet/model.py")
    marker.write_text(f"{REDIMNET_REVISION}\n", encoding="utf-8")
    print(f"extracted: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "models"))
    )
    parser.add_argument(
        "--titanet", action="store_true", help="also download TitaNet-L"
    )
    args = parser.parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    source_archive = model_dir / f"redimnet-{REDIMNET_REVISION}.tar.gz"
    source_dir = model_dir / f"redimnet-{REDIMNET_REVISION}"
    _download(REDIMNET_SOURCE_URL, source_archive, expected_sha256=None)
    _extract_source(source_archive, source_dir)
    _download(
        REDIMNET_WEIGHT_URL,
        model_dir / "M-vb2+vox2+cnc-ft_mix.pt",
        REDIMNET_CHECKPOINT_SHA256,
    )
    if args.titanet:
        _download(
            TITANET_WEIGHT_URL,
            model_dir / "speakerverification_en_titanet_large.nemo",
            TITANET_CHECKPOINT_SHA256,
        )


if __name__ == "__main__":
    main()
