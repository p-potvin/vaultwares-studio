from vaultwares_studio.presets import DEFAULT_PRESET_KEY, PRESETS, get_preset


def test_default_preset_is_standard():
    assert DEFAULT_PRESET_KEY == "standard"
    assert get_preset(None).key == "standard"
    assert get_preset("nonsense").key == "standard"
    assert get_preset("HIGH").key == "high"


def test_preset_ladder_is_ordered():
    assert PRESETS["draft"].iterations < PRESETS["standard"].iterations < PRESETS["high"].iterations
    assert PRESETS["local-debug"].iterations == 250


def test_train_args_carry_vram_savers():
    for key in ("draft", "standard", "high"):
        args = PRESETS[key].train_args()
        assert "--max-num-iterations" in args
        assert "cpu" in args  # --pipeline.datamanager.cache-images cpu
        assert "--vis" in args


def test_preset_costs_are_plausible():
    draft = PRESETS["draft"].cost()
    standard = PRESETS["standard"].cost()
    high = PRESETS["high"].cost()
    assert draft.est_usd < 1.0
    assert standard.est_usd < 1.0
    assert high.est_usd < 5.0
    assert draft.est_usd <= standard.est_usd <= high.est_usd
