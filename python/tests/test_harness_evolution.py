from opendesk.learning.harness_evolution import (
    HarnessProfile,
    analyze_episodes,
    export_report,
)


def test_analyze_suggests_tree_budget_on_many_pointers(tmp_path):
    eps = []
    for i in range(10):
        eps.append({
            "task": "t%d" % i,
            "_tree": "",
            "_response": "Agent.click(coordinates=[1, 2])",
            "process": {"metrics": {"no_effect_steps": 0}},
        })
    report = analyze_episodes(
        eps,
        profile=HarnessProfile(max_tree_chars=8000),
        legacy_resolver=lambda e: e.get("_response"),
    )
    ids = {s.suggestion_id for s in report.suggestions}
    assert "raise_tree_budget" in ids


def test_export_report_writes_json_and_md(tmp_path):
    report = analyze_episodes([], profile=HarnessProfile())
    path = tmp_path / "harness_report.json"
    export_report(report, str(path))
    assert path.is_file()
    assert (tmp_path / "harness_report.md").is_file()
