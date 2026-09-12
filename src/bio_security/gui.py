"""Tkinter desktop GUI for local enrollment and recognition research."""

from __future__ import annotations

import base64
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from tkinter import messagebox, ttk

import cv2

from bio_security.biometrics import LBPFeatureExtractor, TemplateMatcher
from bio_security.domain import BoundingBox
from bio_security.identity import FeatureVector, PersonRegistration, RecognitionResult
from bio_security.infrastructure.opencv_runtime import (
    CameraOpenError,
    CameraReadError,
    DetectorLoadError,
    OpenCVCamera,
    OpenCVHaarFaceDetector,
)
from bio_security.infrastructure.sqlite_identity import (
    DuplicatePersonError,
    SQLiteIdentityRepository,
    default_registry_path,
)
from bio_security.ports import Frame


@dataclass(slots=True)
class EnrollmentState:
    registration: PersonRegistration
    target_samples: int
    features: list[FeatureVector] = field(default_factory=list)
    last_capture_at: float = 0.0


class KikiamDesktopApp:
    """Local GUI that keeps recognition output separate from access decisions."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        camera: OpenCVCamera,
        detector: OpenCVHaarFaceDetector,
        repository: SQLiteIdentityRepository,
        threshold: float,
        enrollment_samples: int,
    ) -> None:
        if enrollment_samples < 3:
            raise ValueError("enrollment_samples must be at least 3")

        self.root = root
        self.camera = camera
        self.detector = detector
        self.repository = repository
        self.extractor = LBPFeatureExtractor()
        self.matcher = TemplateMatcher(threshold=threshold)
        self.enrollment_samples = enrollment_samples
        self.templates = self.repository.load_templates()
        self.enrollment: EnrollmentState | None = None
        self._photo: tk.PhotoImage | None = None
        self._running = True

        self._build_ui()
        self._refresh_people()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(20, self._tick)

    def _build_ui(self) -> None:
        self.root.title("Kikiam Bio Security — BIO-002")
        self.root.geometry("1220x760")
        self.root.minsize(980, 680)

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="Kikiam Bio Security",
            font=("Segoe UI", 18, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Label(
            header,
            text="BIO-002 local enrollment + recognition research",
        ).pack(side=tk.LEFT, padx=(12, 0))

        body = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, pady=(12, 8))

        camera_frame = ttk.Labelframe(body, text="Live camera")
        controls_frame = ttk.Frame(body)
        body.add(camera_frame, weight=3)
        body.add(controls_frame, weight=2)

        self.camera_label = ttk.Label(
            camera_frame,
            text="Starting camera...",
            anchor=tk.CENTER,
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.notebook = ttk.Notebook(controls_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._build_recognition_tab()
        self._build_enrollment_tab()
        self._build_people_tab()

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            outer,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(8, 5),
        ).pack(fill=tk.X)

    def _build_recognition_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(tab, text="Recognition")

        ttk.Label(
            tab,
            text="Recognition result",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 12))

        self.result_vars = {
            "status": tk.StringVar(value="UNKNOWN"),
            "name": tk.StringVar(value="—"),
            "employee_id": tk.StringVar(value="—"),
            "department": tk.StringVar(value="—"),
            "position": tk.StringVar(value="—"),
            "similarity": tk.StringVar(value="—"),
        }
        rows = (
            ("Result", "status"),
            ("Name", "name"),
            ("Employee / Person ID", "employee_id"),
            ("Department", "department"),
            ("Position", "position"),
            ("Similarity", "similarity"),
        )
        for index, (label, key) in enumerate(rows, start=1):
            ttk.Label(tab, text=f"{label}:").grid(
                row=index,
                column=0,
                sticky=tk.W,
                padx=(0, 12),
                pady=5,
            )
            ttk.Label(
                tab,
                textvariable=self.result_vars[key],
                font=("Segoe UI", 10, "bold") if key == "status" else None,
            ).grid(row=index, column=1, sticky=tk.W, pady=5)

        ttk.Separator(tab).grid(row=7, column=0, columnspan=2, sticky=tk.EW, pady=14)

        ttk.Label(
            tab,
            text=(
                "Recognition is a measurement only. This prototype does not grant or deny "
                "physical access."
            ),
            wraplength=360,
        ).grid(row=8, column=0, columnspan=2, sticky=tk.W)

        tab.columnconfigure(1, weight=1)

    def _build_enrollment_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(tab, text="Enroll person")

        self.employee_id_var = tk.StringVar()
        self.first_name_var = tk.StringVar()
        self.last_name_var = tk.StringVar()
        self.department_var = tk.StringVar()
        self.position_var = tk.StringVar()

        fields = (
            ("Employee / Person ID", self.employee_id_var),
            ("First name", self.first_name_var),
            ("Last name", self.last_name_var),
            ("Department", self.department_var),
            ("Position", self.position_var),
        )
        for row, (label, variable) in enumerate(fields):
            ttk.Label(tab, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
            ttk.Entry(tab, textvariable=variable).grid(
                row=row,
                column=1,
                sticky=tk.EW,
                padx=(12, 0),
                pady=5,
            )

        self.enrollment_progress = ttk.Progressbar(
            tab,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=self.enrollment_samples,
        )
        self.enrollment_progress.grid(
            row=len(fields),
            column=0,
            columnspan=2,
            sticky=tk.EW,
            pady=(18, 6),
        )
        self.enrollment_progress_var = tk.StringVar(
            value=f"0 / {self.enrollment_samples} samples"
        )
        ttk.Label(tab, textvariable=self.enrollment_progress_var).grid(
            row=len(fields) + 1,
            column=0,
            columnspan=2,
            sticky=tk.W,
        )

        buttons = ttk.Frame(tab)
        buttons.grid(
            row=len(fields) + 2,
            column=0,
            columnspan=2,
            sticky=tk.EW,
            pady=(14, 0),
        )
        ttk.Button(
            buttons,
            text="Start enrollment",
            command=self._start_enrollment,
        ).pack(side=tk.LEFT)
        ttk.Button(
            buttons,
            text="Cancel",
            command=self._cancel_enrollment,
        ).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(
            tab,
            text=(
                "Enrollment requires exactly one visible face. Move slightly between samples. "
                "Only derived biometric templates are stored; raw camera frames are not saved."
            ),
            wraplength=360,
        ).grid(
            row=len(fields) + 3,
            column=0,
            columnspan=2,
            sticky=tk.W,
            pady=(18, 0),
        )

        tab.columnconfigure(1, weight=1)

    def _build_people_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="People")

        columns = ("employee_id", "name", "department", "position", "templates")
        self.people_tree = ttk.Treeview(tab, columns=columns, show="headings", height=18)
        headings = {
            "employee_id": "ID",
            "name": "Name",
            "department": "Department",
            "position": "Position",
            "templates": "Templates",
        }
        widths = {
            "employee_id": 110,
            "name": 150,
            "department": 130,
            "position": 130,
            "templates": 80,
        }
        for column in columns:
            self.people_tree.heading(column, text=headings[column])
            self.people_tree.column(column, width=widths[column], anchor=tk.W)

        self.people_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Button(tab, text="Refresh", command=self._refresh_people).pack(
            anchor=tk.E,
            pady=(8, 0),
        )

    def _start_enrollment(self) -> None:
        try:
            registration = PersonRegistration(
                employee_id=self.employee_id_var.get(),
                first_name=self.first_name_var.get(),
                last_name=self.last_name_var.get(),
                department=self.department_var.get(),
                position=self.position_var.get(),
            )
        except ValueError as exc:
            messagebox.showerror("Enrollment", str(exc), parent=self.root)
            return

        self.enrollment = EnrollmentState(
            registration=registration,
            target_samples=self.enrollment_samples,
        )
        self.enrollment_progress["value"] = 0
        self.enrollment_progress_var.set(f"0 / {self.enrollment_samples} samples")
        self.status_var.set("Enrollment started. Keep exactly one face in view.")
        self.notebook.select(1)

    def _cancel_enrollment(self) -> None:
        self.enrollment = None
        self.enrollment_progress["value"] = 0
        self.enrollment_progress_var.set(f"0 / {self.enrollment_samples} samples")
        self.status_var.set("Enrollment cancelled.")

    def _tick(self) -> None:
        if not self._running:
            return

        try:
            frame = self.camera.read()
            boxes = tuple(self.detector.detect(frame))
            display = frame.copy()

            if self.enrollment is not None:
                self._process_enrollment(frame, boxes)
            else:
                self._process_recognition(frame, display, boxes)

            self._draw_boxes(display, boxes)
            self._display_frame(display)
        except CameraReadError as exc:
            self.status_var.set(str(exc))
            messagebox.showerror("Camera error", str(exc), parent=self.root)
            self.close()
            return
        except (ValueError, DuplicatePersonError) as exc:
            self.status_var.set(str(exc))
            self.enrollment = None
            messagebox.showerror("BIO-002", str(exc), parent=self.root)

        self.root.after(30, self._tick)

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
        if now - state.last_capture_at < 0.35:
            return

        feature = self.extractor.extract(frame, box)
        state.features.append(feature)
        state.last_capture_at = now
        count = len(state.features)
        self.enrollment_progress["value"] = count
        self.enrollment_progress_var.set(f"{count} / {state.target_samples} samples")
        self.status_var.set(
            f"Captured enrollment sample {count}/{state.target_samples}. Move slightly."
        )

        if count >= state.target_samples:
            person = self.repository.enroll(state.registration, state.features)
            self.templates = self.repository.load_templates()
            self.enrollment = None
            self._clear_enrollment_form()
            self._refresh_people()
            self.status_var.set(
                f"Enrollment complete: {person.display_name} ({person.employee_id})."
            )
            messagebox.showinfo(
                "Enrollment complete",
                (
                    f"{person.display_name}\n"
                    f"ID: {person.employee_id}\n"
                    f"Templates: {count}\n\n"
                    "Derived templates were stored locally. Raw frames were not saved."
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
            self._show_result(None)
            self.status_var.set("No face detected.")
            return

        strongest: RecognitionResult | None = None
        for box in boxes:
            query = self.extractor.extract(frame, box)
            result = self.matcher.match(query, self.templates)
            if strongest is None or result.similarity > strongest.similarity:
                strongest = result

            if result.status == "MATCH" and result.person is not None:
                label = f"{result.person.display_name} {result.similarity:.3f}"
            else:
                label = f"UNKNOWN {result.similarity:.3f}"
            self._put_label(display, box, label)

        self._show_result(strongest)
        if len(boxes) > 1:
            self.status_var.set(
                f"{len(boxes)} faces detected. Recognition only; no access decision."
            )
        elif strongest is not None and strongest.status == "MATCH":
            self.status_var.set("Known person matched against local enrolled templates.")
        else:
            self.status_var.set("Face detected, but no enrolled identity met the threshold.")

    def _show_result(self, result: RecognitionResult | None) -> None:
        if result is None or result.status != "MATCH" or result.person is None:
            self.result_vars["status"].set("UNKNOWN")
            self.result_vars["name"].set("—")
            self.result_vars["employee_id"].set("—")
            self.result_vars["department"].set("—")
            self.result_vars["position"].set("—")
            if result is None:
                self.result_vars["similarity"].set("—")
            else:
                self.result_vars["similarity"].set(f"{result.similarity:.3f}")
            return

        person = result.person
        self.result_vars["status"].set("MATCH")
        self.result_vars["name"].set(person.display_name)
        self.result_vars["employee_id"].set(person.employee_id)
        self.result_vars["department"].set(person.department or "—")
        self.result_vars["position"].set(person.position or "—")
        self.result_vars["similarity"].set(f"{result.similarity:.3f}")

    @staticmethod
    def _put_label(display: Frame, box: BoundingBox, label: str) -> None:
        cv2.putText(
            display,
            label,
            (box.x, max(24, box.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_boxes(display: Frame, boxes: tuple[BoundingBox, ...]) -> None:
        for box in boxes:
            cv2.rectangle(
                display,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                (0, 255, 0),
                2,
            )

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
        image_height, image_width = rgb.shape[:2]
        ppm = f"P6\n{image_width} {image_height}\n255\n".encode("ascii") + rgb.tobytes()
        encoded = base64.b64encode(ppm)
        self._photo = tk.PhotoImage(data=encoded, format="PPM")
        self.camera_label.configure(image=self._photo, text="")

    def _refresh_people(self) -> None:
        for item in self.people_tree.get_children():
            self.people_tree.delete(item)

        template_counts: dict[int, int] = {}
        for template in self.repository.load_templates():
            person_id = template.person.person_id
            template_counts[person_id] = template_counts.get(person_id, 0) + 1

        for person in self.repository.list_people():
            self.people_tree.insert(
                "",
                tk.END,
                values=(
                    person.employee_id,
                    person.display_name,
                    person.department,
                    person.position,
                    template_counts.get(person.person_id, 0),
                ),
            )

    def _clear_enrollment_form(self) -> None:
        self.employee_id_var.set("")
        self.first_name_var.set("")
        self.last_name_var.set("")
        self.department_var.set("")
        self.position_var.set("")
        self.enrollment_progress["value"] = 0
        self.enrollment_progress_var.set(f"0 / {self.enrollment_samples} samples")

    def close(self) -> None:
        if not self._running:
            return
        self._running = False
        self.camera.close()
        self.root.destroy()


def run_gui(
    *,
    camera_index: int = 0,
    registry_path: Path | None = None,
    threshold: float = 0.86,
    enrollment_samples: int = 8,
) -> int:
    """Launch the BIO-002 desktop GUI."""

    camera = OpenCVCamera(index=camera_index)
    detector = OpenCVHaarFaceDetector()
    repository = SQLiteIdentityRepository(registry_path or default_registry_path())
    root: tk.Tk | None = None

    try:
        camera.open()
        root = tk.Tk()
        KikiamDesktopApp(
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
            messagebox.showerror("BIO-002 startup error", str(exc), parent=root)
        return 2
    finally:
        camera.close()
    return 0
