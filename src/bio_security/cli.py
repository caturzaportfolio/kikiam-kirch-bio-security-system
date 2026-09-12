"""CLI entry point for the Kikiam biometric research baseline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from contextlib import suppress

from bio_security.application import DetectionSession
from bio_security.infrastructure.opencv_runtime import (
    CameraOpenError,
    CameraReadError,
    DetectorLoadError,
    OpenCVCamera,
    OpenCVHaarFaceDetector,
    OpenCVWindow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bio-security",
        description="Kikiam local biometric research and benchmarking CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="run the BIO-001 camera/detection baseline")
    detect.add_argument("--camera", type=int, default=0, help="zero-based camera index")
    detect.add_argument("--width", type=int, default=None, help="requested capture width")
    detect.add_argument("--height", type=int, default=None, help="requested capture height")
    detect.add_argument("--scale-factor", type=float, default=1.1)
    detect.add_argument("--min-neighbors", type=int, default=5)
    detect.add_argument("--min-face-size", type=int, default=40)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "detect":
        return _run_detection(args)
    return 2


def _run_detection(args: argparse.Namespace) -> int:
    camera = OpenCVCamera(index=args.camera, width=args.width, height=args.height)
    window = OpenCVWindow()
    session: DetectionSession | None = None

    try:
        detector = OpenCVHaarFaceDetector(
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_face_size=args.min_face_size,
        )
        camera.open()
        session = DetectionSession(frame_source=camera, detector=detector)

        print("Kikiam BIO-001 camera baseline")
        print(f"Camera:   {camera.description}")
        print(f"Detector: {detector.description}")
        print("Press Q or ESC to stop. Frames are displayed locally and are not saved.")

        while True:
            observation = session.process_next_frame()
            window.show(observation.frame, observation.boxes, observation.timing)
            if window.exit_requested():
                break

    except KeyboardInterrupt:
        print("\nInterrupted by operator.")
    except (CameraOpenError, CameraReadError, DetectorLoadError, ValueError) as exc:
        print(f"BIO-001 runtime error: {exc}", file=sys.stderr)
        return 2
    finally:
        camera.close()
        # Window creation may never have succeeded (for example on a headless machine).
        with suppress(Exception):
            window.close()

    summary = session.metrics.summary() if session is not None else None
    if summary is not None:
        print(
            "Summary: "
            f"frames={summary.frames} "
            f"detection_median={summary.median_detection_ms:.2f}ms "
            f"detection_p95={summary.p95_detection_ms:.2f}ms "
            f"total_median={summary.median_total_ms:.2f}ms "
            f"total_p95={summary.p95_total_ms:.2f}ms "
            f"mean_fps={summary.mean_fps:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
