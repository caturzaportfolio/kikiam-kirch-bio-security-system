"""Verified local acquisition of OpenCV Zoo models used by BIO-003."""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class ModelAssetError(RuntimeError):
    """Raised when a required model cannot be acquired or verified."""


@dataclass(frozen=True, slots=True)
class ModelAsset:
    filename: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class ModelBundle:
    yunet: Path
    sface: Path


YUNET = ModelAsset(
    filename="face_detection_yunet_2023mar.onnx",
    url=(
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    size=232589,
)

SFACE = ModelAsset(
    filename="face_recognition_sface_2021dec.onnx",
    url=(
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    size=38696353,
)

ProgressCallback = Callable[[str], None]


def default_model_dir() -> Path:
    """Return the per-user model directory outside the Git repository."""

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    else:
        base = Path.home() / ".local" / "share"
    return base / "KikiamBioSecurity" / "models"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_models(
    directory: Path | None = None,
    progress: ProgressCallback | None = None,
) -> ModelBundle:
    """Ensure YuNet and SFace exist locally and match official upstream hashes."""

    model_dir = directory or default_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    yunet = _ensure_model(YUNET, model_dir, progress)
    sface = _ensure_model(SFACE, model_dir, progress)
    return ModelBundle(yunet=yunet, sface=sface)


def _ensure_model(
    asset: ModelAsset,
    directory: Path,
    progress: ProgressCallback | None,
) -> Path:
    target = directory / asset.filename
    if target.exists():
        if target.stat().st_size == asset.size and sha256_file(target) == asset.sha256:
            return target
        target.unlink()

    temp = target.with_suffix(target.suffix + ".part")
    temp.unlink(missing_ok=True)
    if progress is not None:
        progress(f"Downloading {asset.filename}...")

    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": "Kikiam-Bio-Security/0.3"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temp.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    except (OSError, urllib.error.URLError) as exc:
        temp.unlink(missing_ok=True)
        raise ModelAssetError(f"failed to download {asset.filename}: {exc}") from exc

    if temp.stat().st_size != asset.size:
        actual = temp.stat().st_size
        temp.unlink(missing_ok=True)
        raise ModelAssetError(
            f"model size verification failed for {asset.filename}: {actual} != {asset.size}"
        )
    actual_hash = sha256_file(temp)
    if actual_hash != asset.sha256:
        temp.unlink(missing_ok=True)
        raise ModelAssetError(
            f"model SHA-256 verification failed for {asset.filename}: {actual_hash}"
        )

    temp.replace(target)
    if progress is not None:
        progress(f"Verified {asset.filename}.")
    return target
