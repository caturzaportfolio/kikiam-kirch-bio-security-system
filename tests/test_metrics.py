import pytest

from bio_security.domain import BoundingBox, TimingAccumulator, TimingSample


def test_bounding_box_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        BoundingBox(x=0, y=0, width=0, height=10)


def test_timing_summary_reports_expected_aggregates() -> None:
    metrics = TimingAccumulator()
    metrics.add(TimingSample(capture_ms=2.0, detection_ms=10.0, total_ms=20.0, face_count=1))
    metrics.add(TimingSample(capture_ms=4.0, detection_ms=20.0, total_ms=40.0, face_count=0))

    summary = metrics.summary()

    assert summary is not None
    assert summary.frames == 2
    assert summary.mean_capture_ms == pytest.approx(3.0)
    assert summary.median_detection_ms == pytest.approx(15.0)
    assert summary.p95_detection_ms == pytest.approx(19.5)
    assert summary.median_total_ms == pytest.approx(30.0)
    assert summary.p95_total_ms == pytest.approx(39.0)
    assert summary.mean_fps == pytest.approx(37.5)
