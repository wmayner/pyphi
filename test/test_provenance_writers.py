"""Tests for the script-facing provenance writers."""

import json

import numpy as np
import pytest

from pyphi import provenance


class TestFormatStem:
    def test_params_in_insertion_order(self):
        assert (
            provenance.format_stem("study", {"seed": 42, "trials": 60})
            == "study_seed42_trials60"
        )

    def test_float_dot_becomes_p(self):
        assert provenance.format_stem("study", {"noise": 0.7}) == "study_noise0p7"

    def test_unsafe_characters_sanitized(self):
        assert provenance.format_stem("study", {"tag": "a/b c"}) == "study_taga-b-c"

    def test_run_label_appended(self):
        assert (
            provenance.format_stem("study", {"seed": 1}, "post_reduction")
            == "study_seed1_post_reduction"
        )

    def test_no_params(self):
        assert provenance.format_stem("study") == "study"


class TestUniquePath:
    def test_fresh_path(self, tmp_path):
        assert provenance.unique_path(tmp_path, "run", ".json") == tmp_path / "run.json"

    def test_versions_on_collision(self, tmp_path):
        (tmp_path / "run.json").touch()
        assert (
            provenance.unique_path(tmp_path, "run", ".json") == tmp_path / "run_v2.json"
        )
        (tmp_path / "run_v2.json").touch()
        assert (
            provenance.unique_path(tmp_path, "run", ".json") == tmp_path / "run_v3.json"
        )

    def test_creates_directory(self, tmp_path):
        target = tmp_path / "a" / "b"
        provenance.unique_path(target, "run", ".json")
        assert target.is_dir()


class TestSaveJson:
    def test_envelope_and_numpy_values(self, tmp_path):
        path = provenance.save_json(
            {"phi": np.float64(0.5), "counts": np.arange(3)},
            tmp_path,
            "study",
            params={"seed": 42},
        )
        assert path == tmp_path / "study_seed42.json"
        document = json.loads(path.read_text())
        assert document["data"] == {"phi": 0.5, "counts": [0, 1, 2]}
        assert document["params"] == {"seed": 42}
        assert document["provenance"]["seed"] == 42
        assert document["provenance"]["pyphi_version"]
        assert document["provenance"]["timestamp"]

    def test_explicit_seed_overrides_params(self, tmp_path):
        path = provenance.save_json({}, tmp_path, "study", params={"seed": 42}, seed=7)
        assert json.loads(path.read_text())["provenance"]["seed"] == 7

    def test_note_stored(self, tmp_path):
        path = provenance.save_json({}, tmp_path, "study", note="pilot run")
        assert json.loads(path.read_text())["provenance"]["note"] == "pilot run"

    def test_unserializable_payload_raises(self, tmp_path):
        with pytest.raises(TypeError, match="not JSON serializable"):
            provenance.save_json({"bad": object()}, tmp_path, "study")

    def test_no_clobber(self, tmp_path):
        first = provenance.save_json({"x": 1}, tmp_path, "study", params={"seed": 1})
        second = provenance.save_json({"x": 2}, tmp_path, "study", params={"seed": 1})
        assert second == tmp_path / "study_seed1_v2.json"
        assert json.loads(first.read_text())["data"] == {"x": 1}
