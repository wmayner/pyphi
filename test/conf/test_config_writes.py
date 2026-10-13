"""Every way of writing to ``pyphi.config`` runs the cross-field check, and
every documented way of switching versions of IIT still works."""

from __future__ import annotations

import dataclasses
import textwrap
from pathlib import Path

import pytest

import pyphi
from pyphi.conf import ConfigurationError
from pyphi.conf import config
from pyphi.conf import presets


@pytest.fixture(autouse=True)
def _restore():
    with pyphi.config.override():
        yield


def test_version_alone_raises_on_item_assignment():
    with pytest.raises(ConfigurationError):
        pyphi.config["iit.version"] = "IIT_4_0_2023"


def test_unknown_version_raises_at_assignment():
    with pytest.raises(ConfigurationError):
        pyphi.config["iit.version"] = "IIT4"


def test_ambiguity_message_recommends_a_preset():
    with pytest.raises(ConfigurationError, match=r"iit4_2023|formalism="):
        pyphi.config.version = "IIT_4_0_2023"


def test_writing_through_a_layer_says_how_to_set_it():
    with pytest.raises(ConfigurationError, match=r"config\.precision = "):
        pyphi.config.numerics.precision = 6


def test_typo_suggests_the_option():
    with pytest.raises(ConfigurationError, match="progress_bars"):
        pyphi.config.progres_bars = False


def test_presets_and_overrides_still_apply():
    with pyphi.config.override(**pyphi.iit3):
        assert pyphi.config.formalism.iit.version == "IIT_3_0"
    pyphi.config.precision = 6
    assert pyphi.config.precision == 6


class TestEveryWritePathIsChecked:
    def test_flat_write(self):
        with pytest.raises(ConfigurationError, match="mechanism_phi_measure"):
            config.mechanism_phi_measure = "EMD"

    def test_item_write_of_a_measure(self):
        with pytest.raises(ConfigurationError, match="system_phi_measure"):
            config["iit.system_phi_measure"] = "GENERALIZED_INTRINSIC_DIFFERENCE"

    def test_subnamespace_replacement(self):
        with pytest.raises(ConfigurationError):
            config.iit = dataclasses.replace(config.formalism.iit, version="IIT_3_0")

    def test_layer_replacement(self):
        iit = dataclasses.replace(config.formalism.iit, version="IIT_4_0_2023")
        with pytest.raises(ConfigurationError):
            config.formalism = dataclasses.replace(config.formalism, iit=iit)

    def test_rejected_write_leaves_the_config_unchanged(self):
        before = config.snapshot()
        with pytest.raises(ConfigurationError):
            config["iit.version"] = "IIT_3_0"
        assert config.snapshot() == before

    def test_message_says_how_to_change_options_together(self):
        with pytest.raises(ConfigurationError, match=r"override\(\*\*pyphi\.iit3\)"):
            config["iit.version"] = "IIT_3_0"

    def test_validate_config_false_skips_the_check(self):
        with config.override(validate_config=False):
            config["iit.version"] = "IIT_3_0"
            assert config.formalism.iit.version == "IIT_3_0"

    def test_infrastructure_writes_still_work(self):
        config.parallel = True
        config["infrastructure.progress_bars"] = False
        assert config.parallel is True
        assert config.progress_bars is False

    def test_install_snapshot_is_not_checked(self):
        """Workers install the parent's snapshot wholesale, without the check."""
        snap = config.snapshot()
        iit = dataclasses.replace(snap.formalism.iit, version="IIT_3_0")
        mixed = dataclasses.replace(
            snap, formalism=dataclasses.replace(snap.formalism, iit=iit)
        )
        config.install_snapshot(mixed)
        assert config.formalism.iit.version == "IIT_3_0"


class TestDocumentedVersionSwitches:
    """Each way earlier-versions.md, the migration guide and the tutorials
    switch versions of IIT works under the check."""

    @pytest.mark.parametrize("name", ["iit3", "iit4_2023", "iit4_2026"])
    def test_override_with_a_preset(self, name):
        preset = getattr(pyphi, name)
        with config.override(**preset):
            assert config.formalism.iit == preset["iit"]

    @pytest.mark.parametrize("name", ["IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026"])
    def test_formalism_argument(self, name):
        with presets.use(name):
            assert config.formalism.iit.version == name

    @pytest.mark.parametrize("name", ["iit3", "iit4_2023", "iit4_2026"])
    def test_assigning_the_iit_subnamespace(self, name):
        config.iit = getattr(presets, name)["iit"]
        assert config.formalism.iit == getattr(presets, name)["iit"]

    @pytest.mark.parametrize("name", ["iit4_2023", "iit4_2026"])
    def test_replacing_the_formalism_layer(self, name):
        preset = getattr(pyphi, name)
        config.formalism = dataclasses.replace(config.formalism, **preset)
        assert config.formalism.iit == preset["iit"]

    def test_assigning_each_preset_entry(self):
        for key, value in pyphi.iit3.items():
            setattr(config, key, value)
        assert config.formalism.iit == pyphi.iit3["iit"]
        assert config.precision == 6

    def test_loading_the_shipped_iit3_file(self):
        path = Path(__file__).parents[2] / "pyphi_config_3.0.yml"
        config.load_yaml(path)
        assert config.formalism.iit == pyphi.iit3["iit"]

    def test_loading_a_file_that_sets_the_version_alone_raises(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text("formalism:\n  iit:\n    version: IIT_4_0_2023\n")
        with pytest.raises(ConfigurationError):
            config.load_yaml(path)

    def test_snapshot_overrides_reproduce_another_version(self):
        with config.override(**pyphi.iit3):
            snap = config.snapshot()
        with config.override(**snap.as_overrides()):
            assert config.snapshot() == snap


class TestLayerObjectsSayHowToWrite:
    def test_flat_option(self):
        with pytest.raises(ConfigurationError, match=r"pyphi\.config\.parallel = "):
            config.infrastructure.parallel = False

    def test_colliding_option_names_the_qualified_path(self):
        with pytest.raises(
            ConfigurationError, match=r'config\["iit\.mechanism_partition_scheme"\]'
        ):
            config.formalism.iit.mechanism_partition_scheme = "JOINT_BIPARTITION"

    def test_version_recommends_a_preset(self):
        with pytest.raises(ConfigurationError, match="iit4_2023"):
            config.formalism.iit.version = "IIT_4_0_2023"

    def test_subnamespace(self):
        with pytest.raises(ConfigurationError, match=r"pyphi\.config\.iit = "):
            config.formalism.iit = presets.iit3["iit"]

    def test_standalone_layer(self):
        with pytest.raises(ConfigurationError, match=r"dataclasses\.replace"):
            presets.iit3["iit"].mechanism_phi_measure = "KLD"


class TestFormerNames:
    def test_renamed_option_write_names_its_replacement(self):
        with pytest.raises(ConfigurationError, match=r"iit\.version"):
            config.IIT_VERSION = 3.0

    def test_renamed_option_read_names_its_replacement(self):
        with pytest.raises(AttributeError, match="parallel_partition_evaluation"):
            _ = config.PARALLEL_CUT_EVALUATION

    def test_removed_option_says_so(self):
        with pytest.raises(ConfigurationError, match="enable_logging"):
            config.LOG_STDOUT_LEVEL = "INFO"

    def test_typo_read_suggests_the_option(self):
        with pytest.raises(AttributeError, match="progress_bars"):
            _ = config.progres_bars

    def test_flat_file_names_each_key_and_its_replacement(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text(
            textwrap.dedent("""\
            PRECISION: 6
            PARALLEL_CUT_EVALUATION: false
            SYSTEM_CUTS: 3.0_STYLE
            """)
        )
        with pytest.raises(ConfigurationError) as exc:
            config.load_yaml(path)
        message = str(exc.value)
        assert "numerics.precision" in message
        assert "infrastructure.parallel_partition_evaluation" in message
        assert "SYSTEM_CUTS" in message


class TestAWriteOfTheWrongTypeIsUndone:
    """A value of the wrong type fails the check with a ConfigurationError and
    is rolled back, so later writes are not blocked by it."""

    def test_wrong_type_raises_configuration_error_and_rolls_back(self):
        before = config.snapshot()
        with pytest.raises(ConfigurationError, match="sia_tie_resolution"):
            config.sia_tie_resolution = 5
        assert config.snapshot() == before
        config.precision = 6
        assert config.precision == 6

    def test_wrong_type_under_iit3_raises_configuration_error(self):
        with config.override(**pyphi.iit3):
            with pytest.raises(ConfigurationError, match="sia_tie_resolution"):
                config.sia_tie_resolution = 5
            assert config.formalism.iit.sia_tie_resolution == ("PHI", "PARTITION_LEX")

    def test_any_failure_of_the_check_rolls_back(self, monkeypatch):
        from pyphi.conf import constraints

        def explode(_config):
            raise RuntimeError("check failed")

        before = config.snapshot()
        monkeypatch.setattr(constraints, "check_config_constraints", explode)
        with pytest.raises(RuntimeError, match="check failed"):
            config.precision = 3
        assert config.snapshot() == before


class TestValuesThatWouldFailLaterAreRejected:
    """A value that is accepted but fails, or is ignored, at compute time is
    rejected when it is written."""

    @pytest.mark.parametrize(
        ("path", "value", "match"),
        [
            ("actual_causation.version", "NOPE", "actual-causation formalism"),
            ("actual_causation.alpha_measure", "NOPE", "alpha measure"),
            ("actual_causation.alpha_measure", "KLD", "alpha measure"),
            ("infrastructure.parallel_workers", 0, "must not be 0"),
            ("infrastructure.parallel_backend", "nope", "parallel_backend"),
            ("infrastructure.memory_ceiling_percentage", 200, "between 0 and 100"),
            ("infrastructure.memory_ceiling_percentage", -1, "between 0 and 100"),
            (
                "infrastructure.parallel_partition_evaluation",
                {"parallel": "no"},
                r"\['parallel'\] must be bool",
            ),
            (
                "infrastructure.parallel_partition_evaluation",
                {"chunksize": 0},
                "at least 1",
            ),
            ("iit.mip_tie_resolution", [], "non-empty"),
        ],
    )
    def test_rejected_at_write(self, path, value, match):
        before = config.snapshot()
        with pytest.raises(ValueError, match=match):
            config[path] = value
        assert config.snapshot() == before

    def test_numpy_values_are_accepted_in_level_settings(self):
        import numpy as np

        path = "infrastructure.parallel_partition_evaluation"
        config[path] = {"parallel": np.True_, "chunksize": np.int64(4)}
        level = config.infrastructure.parallel_partition_evaluation
        assert level["parallel"] is True
        assert level["chunksize"] == 4
        assert type(level["chunksize"]) is int

    def test_an_actual_causation_measure_error_does_not_recommend_iit_presets(self):
        from pyphi.formalism.actual_causation.formalism import _resolve_ac_measures
        from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY
        from pyphi.formalism.base import MeasureNotCompatibleError

        with pytest.raises(MeasureNotCompatibleError) as raised:
            _resolve_ac_measures(
                ACTUAL_CAUSATION_FORMALISM_REGISTRY["AC_2019"], alpha_measure_name="KLD"
            )
        assert "iit3" not in str(raised.value)
        assert "actual_causation.alpha_measure" in str(raised.value)


class TestEveryWritePathReportsATypoAlike:
    """A misspelt option raises ConfigurationError with a suggestion, whichever
    way it is written."""

    def test_item_write(self):
        with pytest.raises(
            ConfigurationError, match="Did you mean 'mechanism_phi_measure'"
        ):
            config["iit.mechanism_phi_measur"] = "X"

    def test_override(self):
        with (
            pytest.raises(
                ConfigurationError, match="Did you mean 'mechanism_phi_measure'"
            ),
            config.override({"iit.mechanism_phi_measur": "X"}),
        ):
            pass

    def test_a_path_below_an_option(self):
        with pytest.raises(ConfigurationError, match="is an option, not a section"):
            config["numerics.precision.real"] = 3

    def test_an_unregistered_measure_is_called_unregistered(self):
        with pytest.raises(ConfigurationError) as raised:
            config.mechanism_phi_measure = "INTRINSIC_INFO"
        message = str(raised.value)
        assert "not a registered measure" in message
        assert "Did you mean 'INTRINSIC_INFORMATION'" in message
        assert "must agree" not in message

    def test_override_of_the_version_alone_recommends_its_preset(self):
        with (
            pytest.raises(ConfigurationError, match=r"override\(\*\*pyphi\.iit3\)"),
            config.override({"iit.version": "IIT_3_0"}),
        ):
            pass
