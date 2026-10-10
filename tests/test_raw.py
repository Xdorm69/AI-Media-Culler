"""RAW decoding (rawpy faked: no real RAW sample ships with the tests) and RAW+JPEG pairs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tests.conftest import save, save_timed, textured

from media_culler import imaging
from media_culler.analyzers.technical import TechnicalAnalyzer
from media_culler.exceptions import ImageLoadError
from media_culler.models import GroupRole, ImageRecord, Recommendation
from media_culler.pairing import inherit_companion_decisions, link_raw_companions
from media_culler.pipeline import Pipeline
from media_culler.reporting import build_report

WHEN = "2026:06:12 07:07:26"


class FakeRawpy:
    """Just enough of rawpy's surface: imread() context manager, sizes, postprocess, thumb."""

    ThumbFormat = SimpleNamespace(JPEG="jpeg", BITMAP="bitmap")

    def __init__(self, image: np.ndarray, *, flip: int = 0, thumb: bytes | None = None) -> None:
        self.image, self.flip, self.thumb = image, flip, thumb
        self.postprocess_kwargs: dict[str, object] = {}
        self.opened: list[str] = []

    def imread(self, path: str) -> FakeRawpy:
        if path.endswith("broken.cr2"):
            raise OSError("LibRaw: unsupported file format")
        self.opened.append(path)
        return self

    def __enter__(self) -> FakeRawpy:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    @property
    def sizes(self) -> SimpleNamespace:
        height, width = self.image.shape[:2]
        return SimpleNamespace(width=width, height=height, flip=self.flip)

    def postprocess(self, **kwargs: object) -> np.ndarray:
        self.postprocess_kwargs = kwargs
        return self.image

    def extract_thumb(self) -> SimpleNamespace:
        if self.thumb is None:
            raise OSError("no thumbnail")
        return SimpleNamespace(format="jpeg", data=self.thumb)


@pytest.fixture
def fake_raw(monkeypatch: pytest.MonkeyPatch):
    def install(image: np.ndarray, **kwargs: object) -> FakeRawpy:
        fake = FakeRawpy(image, **kwargs)  # type: ignore[arg-type]
        monkeypatch.setattr(imaging, "rawpy", fake, raising=False)
        monkeypatch.setattr(imaging, "RAW_AVAILABLE", True)
        monkeypatch.setattr(imaging, "_raw_half_size", False)
        return fake

    return install


def raw_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really a raw file")
    return path


def test_raw_is_decoded_with_rawpy_not_pillow(tmp_path, fake_raw):
    image = textured(1, (320, 240))
    fake = fake_raw(image)
    rgb = imaging.load_rgb(raw_file(tmp_path / "a.CR2"), 10_000_000)
    assert rgb.shape == (240, 320, 3) and rgb.dtype == np.uint8
    assert rgb.flags["C_CONTIGUOUS"] and np.array_equal(rgb, image)
    assert fake.postprocess_kwargs == {"use_camera_wb": True, "half_size": False, "output_bps": 8}


def test_half_size_is_configurable(tmp_path, fake_raw):
    fake = fake_raw(textured(1, (320, 240)))
    imaging.configure_raw(True)
    try:
        imaging.load_rgb(raw_file(tmp_path / "a.nef"), 10_000_000)
    finally:
        imaging.configure_raw(False)
    assert fake.postprocess_kwargs["half_size"] is True


def test_header_reports_dimensions_orientation_and_capture_time(tmp_path, fake_raw):
    preview = tmp_path / "preview.png"
    save_timed(textured(2, (64, 48)), preview, WHEN, "40")
    fake_raw(textured(1, (320, 240)), flip=6, thumb=preview.read_bytes())
    header = imaging.read_header(raw_file(tmp_path / "a.dng"), 10_000_000)
    assert (header.width, header.height, header.format) == (240, 320, "RAW")  # 90-degree flip
    assert header.captured_at == "2026-06-12T07:07:26.400000"


def test_header_without_any_exif_has_no_time_rather_than_a_guess(tmp_path, fake_raw):
    fake_raw(textured(1, (320, 240)))
    header = imaging.read_header(raw_file(tmp_path / "a.arw"), 10_000_000)
    assert header.captured_at is None and (header.width, header.height) == (320, 240)


def test_size_limit_applies_to_raw(tmp_path, fake_raw):
    fake_raw(textured(1, (320, 240)))
    path = raw_file(tmp_path / "a.raf")
    with pytest.raises(ImageLoadError, match="too large"):
        imaging.read_header(path, 1000)
    with pytest.raises(ImageLoadError, match="too large"):
        imaging.load_rgb(path, 1000)


def test_corrupt_raw_is_a_per_file_error(tmp_path, fake_raw):
    fake_raw(textured(1, (320, 240)))
    path = raw_file(tmp_path / "broken.cr2")
    with pytest.raises(ImageLoadError, match="Cannot open RAW"):
        imaging.read_header(path, 10_000_000)
    with pytest.raises(ImageLoadError, match="Cannot decode RAW"):
        imaging.load_rgb(path, 10_000_000)


def test_missing_rawpy_gives_an_actionable_error_instead_of_a_misleading_preview(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(imaging, "RAW_AVAILABLE", False)
    path = raw_file(tmp_path / "a.cr3")
    for call in (imaging.read_header, imaging.load_rgb):
        with pytest.raises(ImageLoadError, match=r"pip install 'ai-media-culler\[formats\]'"):
            call(path, 10_000_000)


def test_non_raw_files_never_touch_rawpy(tmp_path, fake_raw):
    fake = fake_raw(textured(1, (320, 240)))
    path = save(textured(3, (64, 48)), tmp_path / "a.png")
    imaging.load_rgb(path, 10_000_000)
    imaging.read_header(path, 10_000_000)
    assert fake.opened == []


# ------------------------------------------------------------------------- pairing


def rec(relative: str, *, status: str = "valid", at: str | None = None) -> ImageRecord:
    return ImageRecord(
        path=f"/photos/{relative}",
        relative_path=relative,
        filename=Path(relative).name,
        extension=Path(relative).suffix,
        status=status,  # type: ignore[arg-type]
        captured_at=at,
    )


def test_pairs_match_on_folder_and_stem_case_insensitively():
    records = [
        rec("trip/IMG_1.CR2"),
        rec("trip/img_1.jpg"),
        rec("trip/IMG_2.jpg"),  # no RAW
        rec("other/IMG_1.jpg"),  # same stem, different folder
    ]
    assert link_raw_companions(records) == 1
    assert [r.paired_with for r in records] == [None, "trip/IMG_1.CR2", None, None]


def test_a_failed_raw_does_not_swallow_its_jpeg():
    records = [rec("a.cr2", status="error"), rec("a.jpg")]
    assert link_raw_companions(records) == 0 and records[1].paired_with is None


def test_same_name_but_different_moment_is_not_a_pair():
    records = [
        rec("a.nef", at="2026-06-12T07:07:26"),
        rec("a.jpg", at="2026-06-12T09:30:00"),  # counter rolled over
    ]
    assert link_raw_companions(records) == 0
    near = [rec("b.nef", at="2026-06-12T07:07:26"), rec("b.jpg", at="2026-06-12T07:07:27")]
    assert link_raw_companions(near) == 1


def test_relinking_clears_stale_links():
    records = [rec("a.cr2"), rec("a.jpg")]
    link_raw_companions(records)
    records[0].status = "error"
    link_raw_companions(records)
    assert records[1].paired_with is None


def test_companion_inherits_decisions():
    raw, jpg = rec("a.cr2"), rec("a.jpg")
    raw.recommendation, raw.recommendation_reason = Recommendation.REVIEW, "why"
    raw.group_role, raw.is_group_representative = GroupRole.SIMILAR, False
    raw.selection_reason = "Closed eyes on 1 face(s)"
    link_raw_companions([raw, jpg])
    inherit_companion_decisions([raw, jpg])
    assert jpg.recommendation is Recommendation.REVIEW and jpg.group_role is GroupRole.SIMILAR
    assert jpg.is_group_representative is False
    assert (
        jpg.selection_reason
        == "Companion of a.cr2; follows its decision (Closed eyes on 1 face(s))"
    )


def test_raw_plus_jpeg_is_one_shot_not_a_burst(settings, tmp_path, fake_raw):
    frame = textured(7, (640, 480))
    root = tmp_path / "in"
    save_timed(frame, root / "IMG_0001.JPG", WHEN, "10")
    raw_file(root / "IMG_0001.CR2")
    preview = tmp_path / "p.png"
    save_timed(textured(2, (64, 48)), preview, WHEN, "10")
    fake_raw(frame, thumb=preview.read_bytes())

    off = settings.model_copy(deep=True)
    off.scan.pair_raw_jpeg = False
    unpaired = Pipeline(off, analyzers=[TechnicalAnalyzer()]).run(root)
    assert {r.filename for r in unpaired.records if r.duplicate_group} == {
        "IMG_0001.JPG",
        "IMG_0001.CR2",
    }, "premise: without pairing the two files look like a duplicate group"

    result = Pipeline(settings, analyzers=[TechnicalAnalyzer()]).run(root)
    by = {r.filename: r for r in result.records}
    assert all(r.status == "valid" for r in by.values())
    assert by["IMG_0001.JPG"].paired_with == "IMG_0001.CR2"
    assert result.duplicate_groups == 0
    assert all(r.duplicate_group is None for r in by.values())
    assert by["IMG_0001.JPG"].recommendation == by["IMG_0001.CR2"].recommendation

    report = build_report(result)
    assert report.summary.redundant_images == 0 and report.summary.duplicate_groups == 0
