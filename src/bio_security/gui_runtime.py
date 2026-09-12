"""BIO-003 Windows GUI runtime using YuNet + SFace + Pillow/ImageTk."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from time import monotonic
from tkinter import messagebox
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image, ImageTk

from bio_security.biometrics import TemplateMatcher
from bio_security.dnn_face import (
    DNNModelLoadError,
    SFaceFeatureExtractor,
    TemporalMatchStabilizer,
    YuNetFaceDetector,
)
from bio_security.domain import BoundingBox
from bio_security.gui import KikiamDesktopApp
from bio_security.identity import FeatureVector
from bio_security.infrastructure.opencv_runtime import (
    CameraOpenError,
    CameraReadError,
    OpenCVCamera,
)
from bio_security.infrastructure.sqlite_identity import (
    DuplicatePersonError,
    SQLiteIdentityRepository,
    default_registry_path,
)
from bio_security.model_assets import ModelAssetError, ensure_models
from bio_security.ports import Frame


class MirroredOpenCVCamera(OpenCVCamera):
    """Mirror GUI frames horizontally for natural webcam interaction."""

    def read(self) -> Frame:
        return cv2.flip(super().read(), 1)


def default_sface_registry_path() -> Path:
    """Keep BIO-003 SFace templates isolated from incompatible BIO-002 LBP data."""

    return default_registry_path().with_name("registry_sface.sqlite3")


class PillowKikiamDesktopApp(KikiamDesktopApp):
    """BIO-003 GUI with DNN recognition, duplicate checks, and temporal stability."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        camera: OpenCVCamera,
        detector: YuNetFaceDetector,
        extractor: SFaceFeatureExtractor,
        repository: SQLiteIdentityRepository,
        threshold: float,
        enrollment_samples: int,
    ) -> None:
        super().__init__(
            root,
            camera=camera,
            detector=cast(Any, detector),
            repository=repository,
            threshold=threshold,
            enrollment_samples=enrollment_samples,
        )
        self.dnn_detector = detector
        self.dnn_extractor = extractor
        self.matcher = TemplateMatcher(threshold=threshold, top_templates=3)
        self.duplicate_matcher = TemplateMatcher(threshold=0.55, top_templates=3)
        self.stabilizer = TemporalMatchStabilizer(window_size=5, required_matches=3)
        self.root.title("Kikiam Bio Security — BIO-003")
        self.status_var.set("BIO-003 ready: YuNet detection + SFace recognition.")

    def _process_enrollment(self, frame: Frame, boxes: tuple[BoundingBox, ...]) -> None:
        state = self.enrollment
        if state is None:
            return
        if len(boxes) != 1:
            self.status_var.set(
                "Enrollment paused: exactly one face must be visible."
                if boxes
                else "Enrollment paused: no face detected."
            )
            return

        box = boxes[0]
        if box.width < 100 or box.height < 100:
            self.status_var.set("Enrollment paused: move closer to the camera.")
            return

        now = monotonic()
        if now - state.last_capture_at < 0.45:
            return

        feature = self.dnn_extractor.extract(frame, box)
        state.features.append(feature)
        state.last_capture_at = now
        count = len(state.features)
        self.enrollment_progress["value"] = count
        self.enrollment_progress_var.set(f"{count} / {state.target_samples} samples")
        self.status_var.set(
            f"Captured SFace sample {count}/{state.target_samples}. Move slightly."
        )

        if count < state.target_samples:
            return

        duplicate = self.duplicate_matcher.match(self._centroid(state.features), self.templates)
        if duplicate.status == "MATCH" and duplicate.person is not None:
            existing = duplicate.person
            self.enrollment = None
            self.enrollment_progress["value"] = 0
            self.enrollment_progress_var.set(f"0 / {self.enrollment_samples} samples")
            self.status_var.set(
                f"Enrollment blocked: face already matches {existing.display_name}."
            )
            messagebox.showerror(
                "Duplicate biometric enrollment",
                (
                    "This face already matches an enrolled person.\n\n"
                    f"Name: {existing.display_name}\n"
                    f"ID: {existing.employee_id}\n"
                    f"Similarity: {duplicate.similarity:.3f}\n\n"
                    "Use the existing person record instead of creating another identity."
                ),
                parent=self.root,
            )
            return

        try:
            person = self.repository.enroll(state.registration, state.features)
        except DuplicatePersonError:
            self.enrollment = None
            raise

        self.templates = self.repository.load_templates()
        self.enrollment = None
        self._clear_enrollment_form()
        self._refresh_people()
        self.stabilizer.reset()
        self.status_var.set(
            f"SFace enrollment complete: {person.display_name} ({person.employee_id})."
        )
        messagebox.showinfo(
            "Enrollment complete",
            (
                f"{person.display_name}\n"
                f"ID: {person.employee_id}\n"
                f"SFace templates: {count}\n\n"
                "Raw camera frames were not saved."
            ),
            parent=self.root,
        )

    def _process_recognition(
        self,
        frame: Frame,
        display: Frame,
        boxes: tuple[BoundingBox, ...],
    ) -> None:
        if not boxes:
            self.stabilizer.reset()
            self._show_result(None)
            self.status_var.set("No face detected.")
            return

        if len(boxes) != 1:
            self.stabilizer.reset()
            self._show_result(None)
            for box in boxes:
                result = self.matcher.match(self.dnn_extractor.extract(frame, box), self.templates)
                if result.status == "MATCH" and result.person is not None:
                    label = f"{result.person.display_name} {result.similarity:.3f}"
                else:
                    label = f"UNKNOWN {result.similarity:.3f}"
                self._put_label(display, box, label)
            self.status_var.set(
                f"{len(boxes)} faces detected. Identity panel suppressed until one face remains."
            )
            return

        box = boxes[0]
        instant = self.matcher.match(self.dnn_extractor.extract(frame, box), self.templates)
        stable = self.stabilizer.observe(instant)
        self._show_result(stable)

        if stable.status == "MATCH" and stable.person is not None:
            label = f"{stable.person.display_name} {stable.similarity:.3f}"
            self.status_var.set("Stable SFace match confirmed across recent frames.")
        elif instant.status == "MATCH" and instant.person is not None:
            label = f"VERIFYING {instant.similarity:.3f}"
            self.status_var.set("Candidate match detected; waiting for temporal confirmation.")
        else:
            label = f"UNKNOWN {instant.similarity:.3f}"
            self.status_var.set("Face detected, but no enrolled SFace identity met the threshold.")
        self._put_label(display, box, label)

    @staticmethod
    def _centroid(features: list[FeatureVector]) -> FeatureVector:
        matrix = np.stack(features).astype(np.float32, copy=False)
        centroid = matrix.mean(axis=0).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(centroid))
        if norm <= 0.0:
            raise ValueError("enrollment centroid has zero norm")
        centroid /= norm
        return cast(FeatureVector, centroid)

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
    threshold: float = 0.45,
    enrollment_samples: int = 8,
) -> int:
    """Launch BIO-003 with verified OpenCV Zoo models."""

    try:
        models = ensure_models(progress=print)
        detector = YuNetFaceDetector(models.yunet)
        extractor = SFaceFeatureExtractor(models.sface, detector)
    except (ModelAssetError, DNNModelLoadError, ValueError) as exc:
        print(f"BIO-003 model error: {exc}", file=sys.stderr)
        return 2

    camera = MirroredOpenCVCamera(index=camera_index)
    repository = SQLiteIdentityRepository(registry_path or default_sface_registry_path())
    root: tk.Tk | None = None

    try:
        camera.open()
        root = tk.Tk()
        PillowKikiamDesktopApp(
            root,
            camera=camera,
            detector=detector,
            extractor=extractor,
            repository=repository,
            threshold=threshold,
            enrollment_samples=enrollment_samples,
        )
        root.mainloop()
    except CameraReadError as exc:
        print(f"BIO-003 camera error: {exc}", file=sys.stderr)
        return 2
    except (CameraOpenError, tk.TclError) as exc:
        if root is not None:
            messagebox.showerror("BIO-003 startup error", str(exc), parent=root)
        return 2
    finally:
        camera.close()
    return 0
