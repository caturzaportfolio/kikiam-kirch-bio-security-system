"""Cross-platform Tk preview runtime for BIO-002 using Pillow/ImageTk."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import cast

import cv2
from PIL import Image, ImageTk

from bio_security.gui import KikiamDesktopApp
from bio_security.infrastructure.opencv_runtime import (
    CameraOpenError,
    DetectorLoadError,
    OpenCVCamera,
    OpenCVHaarFaceDetector,
)
from bio_security.infrastructure.sqlite_identity import (
    SQLiteIdentityRepository,
    default_registry_path,
)
from bio_security.ports import Frame


class PillowKikiamDesktopApp(KikiamDesktopApp):
    """BIO-002 GUI with a Pillow-backed camera preview.

    Pillow/ImageTk is used instead of Tk's direct PPM parser because Windows
    Tcl/Tk builds can reject otherwise-valid in-memory PPM payloads.
    """

    def _display_frame(self, frame: Frame) -> None:
        height, width = frame.shape[:2]
        max_width = 780
        max_height = 620
        scale = min(max_width / width, max_height / height, 1.0)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=image, master=self.root)
        self._photo = cast(tk.PhotoImage, photo)
        self.camera_label.configure(image=photo, text="")


def run_gui(
    *,
    camera_index: int = 0,
    registry_path: Path | None = None,
    threshold: float = 0.86,
    enrollment_samples: int = 8,
) -> int:
    """Launch BIO-002 with the Pillow-backed Tk camera preview."""

    camera = OpenCVCamera(index=camera_index)
    detector = OpenCVHaarFaceDetector()
    repository = SQLiteIdentityRepository(registry_path or default_registry_path())
    root: tk.Tk | None = None

    try:
        camera.open()
        root = tk.Tk()
        PillowKikiamDesktopApp(
            root,
            camera=camera,
            detector=detector,
            repository=repository,
            threshold=threshold,
            enrollment_samples=enrollment_samples,
        )
        root.mainloop()
    except (CameraOpenError, DetectorLoadError, tk.TclError) as exc:
        if root is not None:
            from tkinter import messagebox

            messagebox.showerror("BIO-002 startup error", str(exc), parent=root)
        return 2
    finally:
        camera.close()
    return 0
