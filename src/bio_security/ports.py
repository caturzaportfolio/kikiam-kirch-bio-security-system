"""Stable boundaries between the application core and runtime adapters."""

from __future__ import annotations

from typing import Any, Protocol

from bio_security.domain import BoundingBox, TimingSample

Frame = Any


class FrameSource(Protocol):
    """Source capable of opening, reading, and releasing video frames."""

    def open(self) -> None: ...

    def read(self) -> Frame: ...

    def close(self) -> None: ...

    @property
    def description(self) -> str: ...


class FaceDetector(Protocol):
    """Detection adapter. It reports measurements, not identity or access decisions."""

    @property
    def description(self) -> str: ...

    def detect(self, frame: Frame) -> tuple[BoundingBox, ...]: ...


class LocalDisplay(Protocol):
    """Ephemeral local visualization boundary; frames are not persisted."""

    def show(
        self,
        frame: Frame,
        boxes: tuple[BoundingBox, ...],
        timing: TimingSample,
    ) -> None: ...

    def exit_requested(self) -> bool: ...

    def close(self) -> None: ...
