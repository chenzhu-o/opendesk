from opendesk.learning.action_space import (
    POINTER_FALLBACK_WEIGHT,
    map_legacy_completion,
)


def test_unique_click_maps_to_ui():
    tree = "push-button\tSave\t(100, 200)\t(40, 20)"
    legacy = "Agent.click(coordinates=[100, 200])"
    m = map_legacy_completion(legacy, tree)
    assert m.status == "mapped_ui"
    assert m.tier == "ui"
    assert "click(name=" in m.primary
    assert m.sample_weight == 1.0
    assert m.legacy == legacy


def test_tool_passthrough():
    gold = "ImpressTools.duplicate_slide()"
    m = map_legacy_completion(gold, "")
    assert m.status == "passthrough_tool"
    assert m.primary == gold


def test_missing_target_retains_pointer_with_downweight():
    tree = "push-button\tOK\t(10, 10)\t(5, 5)"
    legacy = "Agent.click(coordinates=[999, 999])"
    m = map_legacy_completion(legacy, tree)
    assert m.status == "retained_pointer"
    assert m.primary == legacy
    assert m.sample_weight == POINTER_FALLBACK_WEIGHT


def test_ambiguous_click_not_faked():
    tree = (
        "push-button\tA\t(100, 200)\t(10, 10)\n"
        "push-button\tB\t(102, 198)\t(10, 10)"
    )
    legacy = "Agent.click(coordinates=[100, 200])"
    m = map_legacy_completion(legacy, tree)
    assert m.status == "retained_pointer"
    assert m.primary == legacy
