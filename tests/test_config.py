from pathlib import Path

import pytest
import yaml

from media_culler.config import DEFAULT_CONFIG_YAML, Settings, load_settings
from media_culler.exceptions import ConfigError


def test_defaults_are_valid():
    s = Settings()
    assert s.thresholds.select > s.thresholds.usable > s.thresholds.review


def test_shipped_template_matches_defaults():
    """`culler config init` must never drift from the real defaults."""
    assert Settings.model_validate(yaml.safe_load(DEFAULT_CONFIG_YAML)) == Settings()


def test_unknown_key_is_rejected(tmp_path: Path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("thresholds:\n  selct: 90\n")
    with pytest.raises(ConfigError, match=r"thresholds\.selct"):
        load_settings(cfg)


def test_threshold_order_enforced():
    with pytest.raises(ConfigError, match="select > usable > review"):
        load_settings(None, {"thresholds": {"select": 50, "usable": 60}})


def test_extensions_are_normalised():
    s = load_settings(None, {"scan": {"extensions": ["JPG", ".Png"]}})
    assert s.scan.extensions == [".jpg", ".png"]


def test_overrides_beat_file(tmp_path: Path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("duplicates:\n  hamming_threshold: 3\n")
    assert (
        load_settings(cfg, {"duplicates": {"hamming_threshold": 5}}).duplicates.hamming_threshold
        == 5
    )


def test_legacy_analysis_section_is_ignored(tmp_path: Path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("analysis:\n  blur_detection: true\n")
    assert load_settings(cfg) == Settings()


def test_missing_and_non_mapping_files(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n- list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_settings(bad)


def test_technical_weight_must_be_positive():
    with pytest.raises(ConfigError):
        load_settings(None, {"weights": {"technical": 0}})
