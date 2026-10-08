"""Tests for verifiable GUI action rewards."""

from opendesk.learning import action_rewards as ar


def test_tool_call_requires_exact_match():
    gold = 'ImpressTools.duplicate_slide()'
    pred = 'Agent.click(coordinates=[100, 100])'
    rep = ar.score_step_action(pred, gold, tree="menu\tImpress\t(100, 100)\t(10, 10)")
    assert rep.components["tool_identity"] == 0.0
    assert rep.reward < 0.5


def test_click_coords_when_visible():
    gold = "Agent.click(coordinates=[188, 270])"
    tree = "push-button\tOK\t(188, 270)\t(40, 20)"
    good = ar.score_step_action(gold, gold, tree)
    assert good.reward > 0.95
    bad = ar.score_step_action("Agent.click(coordinates=[0, 0])", gold, tree)
    assert bad.components["pointer"] == 0.0
    assert bad.reward < good.reward


def test_hard_negative_prefers_wrong_tool_over_random_click():
    gold = "ImpressTools.save()"
    rej, g_rep, r_rep = ar.hard_negative(gold, tree="")
    assert g_rep.reward > r_rep.reward
    assert rej != gold
    assert ar.dpo_margin_from_reports(g_rep, r_rep) >= 0.05


def test_coordinate_not_in_tree_does_not_require_pixel_match():
    gold = "Agent.click(coordinates=[9999, 9999])"
    tree = "push-button\tOK\t(10, 10)\t(5, 5)"
    rep = ar.score_step_action("Agent.click(coordinates=[10, 10])", gold, tree)
    assert rep.gold_visible is False
