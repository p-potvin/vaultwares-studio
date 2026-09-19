"""The mapper's bundle-adjustment flags, and the 3.11/3.12 rename.

COLMAP exits on an unrecognised option rather than ignoring it, and 3.12
renamed the global-BA trigger from ``images`` to ``frames`` when rigs arrived.
The deployed worker image was built in July and the local build is 3.12, so the
help text has to decide which name to send.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ENTRYPOINT = Path(__file__).resolve().parents[1] / "docker" / "worker" / "recon_entrypoint.py"

# Trimmed to the option names; COLMAP prints one "--Mapper.<name> arg (=<default>)"
# line per option and the probe only ever asks whether a name is present.
HELP_312 = """
  --Mapper.ba_global_frames_ratio arg (=1.1)
  --Mapper.ba_global_frames_freq arg (=500)
  --Mapper.ba_global_max_refinements arg (=5)
  --Mapper.ba_global_max_num_iterations arg (=50)
  --Mapper.ba_global_function_tolerance arg (=0)
  --Mapper.ba_use_gpu arg (=0)
"""

HELP_311 = """
  --Mapper.ba_global_images_ratio arg (=1.1)
  --Mapper.ba_global_images_freq arg (=500)
  --Mapper.ba_global_max_refinements arg (=5)
  --Mapper.ba_global_max_num_iterations arg (=50)
  --Mapper.ba_global_function_tolerance arg (=0)
"""


def _load():
    spec = importlib.util.spec_from_file_location("recon_entrypoint_ba", ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:  # pragma: no cover - the script guards its own main
        pass
    return module


def _pairs(options: list[str]) -> dict[str, str]:
    return dict(zip(options[::2], options[1::2]))


def test_312_gets_the_frames_names():
    options = _pairs(_load().mapper_ba_options(HELP_312))
    assert "--Mapper.ba_global_frames_ratio" in options
    assert "--Mapper.ba_global_images_ratio" not in options


def test_311_gets_the_images_names():
    options = _pairs(_load().mapper_ba_options(HELP_311))
    assert "--Mapper.ba_global_images_ratio" in options
    assert "--Mapper.ba_global_frames_ratio" not in options


def test_the_defaults_are_the_ones_that_cut_repeated_work():
    """Refinements and the function tolerance are the two that matter.

    5 refinements is a 5x multiplier on every trigger, and a tolerance of 0
    means each solve runs its whole iteration budget regardless.
    """
    options = _pairs(_load().mapper_ba_options(HELP_312))
    assert options["--Mapper.ba_global_max_refinements"] == "2"
    assert float(options["--Mapper.ba_global_function_tolerance"]) > 0
    assert float(options["--Mapper.ba_global_frames_ratio"]) > 1.1


def test_a_flag_this_colmap_does_not_know_is_dropped_not_guessed():
    """3.11 has no ba_use_gpu. Sending it would abort the mapper outright."""
    options = _pairs(_load().mapper_ba_options(HELP_311, use_gpu=True))
    assert "--Mapper.ba_use_gpu" not in options
    assert "--Mapper.ba_global_max_refinements" in options


def test_gpu_is_opt_in_and_off_by_default():
    module = _load()
    assert "--Mapper.ba_use_gpu" not in _pairs(module.mapper_ba_options(HELP_312))
    assert _pairs(module.mapper_ba_options(HELP_312, use_gpu=True))["--Mapper.ba_use_gpu"] == "1"


def test_an_unprobeable_colmap_still_yields_flags():
    """Empty help means the probe failed, not that COLMAP knows nothing.

    Dropping every flag there would silently restore the defaults this exists
    to replace, so the stable names are sent and the renamed pair is skipped.
    """
    options = _pairs(_load().mapper_ba_options(""))
    assert options["--Mapper.ba_global_max_refinements"] == "2"
