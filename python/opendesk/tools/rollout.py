"""RolloutTool — export trajectories and build preference data.

Turns recorded episodes into artefacts a training pipeline can consume:
JSONL trajectories and chosen/rejected preference pairs.

Example::

    rollout(action="export", episode_id="a1b2c3", path="runs/ep1.jsonl")
    rollout(action="pairs", task="export the invoice", path="runs/prefs.jsonl")
"""

from __future__ import annotations

import os
from typing import Literal, Optional

from pydantic import Field

from opendesk.tools.base import Tool, ToolContext, ToolResult


class RolloutTool(Tool):
    """Export RL-ready trajectories and preference pairs."""

    name = "rollout"
    description = (
        "Turn recorded episodes into training data.\n\n"
        "  action='export' — write an episode as JSONL (one record per step),\n"
        "                    with per-step reward and discounted return\n"
        "  action='rank'   — rank episodes best-first (best-of-N view)\n"
        "  action='pairs'  — build chosen/rejected preference pairs from episodes\n"
        "  action='dataset'— write trainer-ready rows (sft / dpo / grpo)\n"
        "  action='list'   — list episodes available for export\n\n"
        "Screenshots are written as PNG files next to the JSONL by default; pass "
        "embed_images=true for one self-contained file.\n\n"
        "Typical flow: run a task as an episode with the 'reward' tool, repeat it "
        "a few times, then build pairs — attempts that satisfied the checklist "
        "become the chosen examples, and shorter successful runs win ties. "
        "'dataset' goes one step further and renders those into the row shape a "
        "trainer reads; a completion holds the agent's actions only, never the "
        "rewards, because a model trained on the reward tokens learns to emit "
        "them instead of solving the task."
    )

    class Params(Tool.Params):
        action: Literal["export", "rank", "pairs", "dataset", "list"] = Field(
            default="export", description="What to do."
        )
        episode_id: Optional[str] = Field(
            default=None, description="Episode to export. Defaults to the latest closed one."
        )
        task: Optional[str] = Field(
            default=None,
            description="For 'rank'/'pairs': only consider episodes whose task contains this text.",
        )
        path: Optional[str] = Field(
            default=None,
            description=(
                "Destination file. Defaults to './opendesk-rollouts/<episode>.jsonl' "
                "for export, or './opendesk-rollouts/preferences.jsonl' for pairs."
            ),
        )
        image_dir: Optional[str] = Field(
            default=None, description="Directory for step screenshots (export only)."
        )
        embed_images: bool = Field(
            default=False,
            description="Inline screenshots as base64 instead of writing PNG files.",
        )
        min_margin: float = Field(
            default=0.0,
            description="For 'pairs': minimum reward difference to form an outcome pair.",
        )
        max_pairs: Optional[int] = Field(
            default=None, description="For 'pairs': cap on the number of pairs."
        )
        strategy: Literal["best_vs_worst", "all_vs_best", "adjacent"] = Field(
            default="best_vs_worst", description="For 'pairs': how attempts are paired."
        )
        include_open: bool = Field(
            default=False,
            description="Include episodes that were never closed (reward unknown).",
        )
        format: Literal["sft", "dpo", "grpo"] = Field(
            default="sft",
            description=(
                "For 'dataset': 'sft' (prompt→completion, successful attempts "
                "only), 'dpo' (prompt→chosen/rejected from the pairs), or 'grpo' "
                "(one row per task holding every attempt and its reward)."
            ),
        )
        min_trust: Optional[Literal["ui", "screen", "action"]] = Field(
            default=None,
            description=(
                "For 'dataset': require step effects to have been read from at "
                "least this signal. Omit to keep everything and report the "
                "split — rows always carry a 'trust' field. Worth setting to "
                "'screen' for grpo, where the per-step returns are the signal."
            ),
        )
        min_reward: float = Field(
            default=1.0, description="For 'dataset' sft: minimum outcome reward."
        )
        val_ratio: float = Field(
            default=0.0,
            description=(
                "For 'dataset': hold out this share for validation. Splits by "
                "task, not by row — attempts at one task are near-duplicates, so "
                "splitting them leaks the eval set into training."
            ),
        )
        test_ratio: float = Field(
            default=0.0, description="For 'dataset': hold out this share for test."
        )

    async def execute(self, ctx: ToolContext, params: "RolloutTool.Params") -> ToolResult:
        handler = {
            "export": self._export,
            "rank": self._rank,
            "pairs": self._pairs,
            "dataset": self._dataset,
            "list": self._list,
        }[params.action]
        return await handler(ctx, params)

    # ------------------------------------------------------------------

    def _episodes(self, ctx, params):
        from opendesk.learning.trajectories import list_episodes

        eps = list_episodes(ctx.session_id)
        if not params.include_open:
            eps = [e for e in eps if e.finished]
        if params.task:
            needle = params.task.lower()
            eps = [e for e in eps if needle in e.task.lower()]
        return eps

    async def _export(self, ctx, params) -> ToolResult:
        from opendesk.computer.observations import get_store
        from opendesk.computer.sandbox import get_sandbox
        from opendesk.learning.assertions import for_episode
        from opendesk.learning.trajectories import export, get_episode, list_episodes

        ep = None
        if params.episode_id:
            ep = get_episode(params.episode_id, ctx.session_id)
            if ep is None:
                return ToolResult(
                    title="Rollout export",
                    output=f"No episode {params.episode_id!r} in this session.",
                    error=True,
                )
        else:
            closed = [e for e in list_episodes(ctx.session_id) if e.finished]
            ep = closed[-1] if closed else None
        if ep is None:
            return ToolResult(
                title="Rollout export",
                output=(
                    "No closed episode to export. Run a task with the 'reward' tool "
                    "(begin → work → end), then export it."
                ),
                error=True,
            )

        sandbox = get_sandbox(ctx.session_id)
        store = get_store(ctx.session_id)
        path = params.path or f"./opendesk-rollouts/{ep.id}.jsonl"

        try:
            manifest = export(
                ep,
                entries=sandbox.export_audit_log(),
                store=store,
                process=ep.process,
                assertions=for_episode(ep.id, ctx.session_id),
                path=path,
                image_dir=params.image_dir,
                embed_images=params.embed_images,
            )
        except Exception as exc:
            return ToolResult(
                title="Rollout export", output=f"Export failed: {exc}", error=True
            )

        await self._audit(
            ctx, "rollout_export", {"episode": ep.id}, manifest["path"]
        )
        lines = [
            f"Exported episode {ep.id} — {manifest['steps']} step(s)",
            f"  trajectory: {manifest['path']}",
        ]
        if manifest.get("image_dir"):
            lines.append(f"  screenshots: {manifest['image_dir']} ({manifest['images']} PNG)")
        elif manifest.get("images"):
            lines.append(f"  screenshots: {manifest['images']} embedded (base64)")
        if ep.reward is not None:
            lines.append(f"  outcome reward: {ep.reward:.0f}")
        if ep.assertions:
            m = ep.assertions.get("metrics") or {}
            lines.append(
                f"  assertions: {m.get('declared', 0)} declared, "
                f"{m.get('held', 0)} held, {m.get('regressed', 0)} regressed "
                f"(precision {m.get('precision', 0):.0%})"
            )
        lines.append(f"  schema: {manifest['schema']}")
        return ToolResult(
            title=f"Exported {manifest['steps']} step(s)",
            output="\n".join(lines),
            metadata=manifest,
        )

    async def _rank(self, ctx, params) -> ToolResult:
        from opendesk.learning.preference import Rollout, build_pairs

        eps = self._episodes(ctx, params)
        if not eps:
            return ToolResult(
                title="Rollout rank",
                output=(
                    "No closed episodes to rank. Run the task a few times with the "
                    "'reward' tool, then rank them."
                ),
            )

        dataset = build_pairs(
            [Rollout.from_episode(e) for e in eps],
            min_margin=params.min_margin, max_pairs=params.max_pairs,
            strategy=params.strategy,
        )
        return ToolResult(
            title=f"Ranked {len(eps)} attempt(s)",
            output=dataset.summary_text(),
            metadata=dataset.to_dict()["stats"],
        )

    async def _pairs(self, ctx, params) -> ToolResult:
        from opendesk.learning.preference import Rollout, build_pairs, export_pairs

        eps = self._episodes(ctx, params)
        if len(eps) < 2:
            return ToolResult(
                title="Preference pairs",
                output=(
                    f"Preference pairs need at least 2 comparable attempts; found "
                    f"{len(eps)}. Repeat the task as separate episodes, then retry."
                ),
                error=True,
            )

        dataset = build_pairs(
            [Rollout.from_episode(e) for e in eps],
            min_margin=params.min_margin, max_pairs=params.max_pairs,
            strategy=params.strategy,
        )

        lines = [dataset.summary_text()]
        if not dataset.pairs:
            return ToolResult(
                title="Preference pairs",
                output="\n".join(lines) + "\n\nNo pairs could be formed.",
                metadata=dataset.to_dict()["stats"],
            )

        path = params.path or "./opendesk-rollouts/preferences.jsonl"
        try:
            manifest = export_pairs(dataset, path=path)
        except Exception as exc:
            return ToolResult(
                title="Preference pairs", output=f"Write failed: {exc}", error=True
            )

        await self._audit(
            ctx, "preference_build", {"task": params.task},
            f"{manifest['pairs']} pairs",
        )
        lines.append("")
        lines.append(
            f"Wrote {manifest['pairs']} pair(s) to {manifest['path']} "
            f"({manifest['outcome_pairs']} from outcome, "
            f"{manifest['efficiency_pairs']} from efficiency)"
        )
        return ToolResult(
            title=f"{manifest['pairs']} preference pair(s)",
            output="\n".join(lines),
            metadata=manifest,
        )

    async def _dataset(self, ctx, params) -> ToolResult:
        from opendesk.computer.observations import get_store
        from opendesk.computer.sandbox import get_sandbox
        from opendesk.learning import training
        from opendesk.learning import trajectories as T
        from opendesk.learning.assertions import for_episode
        from opendesk.learning.preference import Rollout, build_pairs

        eps = self._episodes(ctx, params)
        if not eps:
            return ToolResult(
                title="Dataset",
                output=(
                    "No closed episodes to build from. Run the task with the "
                    "'reward' tool first (begin → work → end)."
                ),
                error=True,
            )

        sandbox = get_sandbox(ctx.session_id)
        store = get_store(ctx.session_id)
        entries = sandbox.export_audit_log()
        records = [
            T.build_trajectory(
                ep, entries=entries, store=store, process=ep.process,
                assertions=for_episode(ep.id, ctx.session_id),
            )
            for ep in eps
        ]

        pairs = None
        if params.format == "dpo":
            dataset = build_pairs(
                [Rollout.from_episode(e) for e in eps],
                min_margin=params.min_margin, max_pairs=params.max_pairs,
                strategy=params.strategy,
            )
            if not dataset.pairs:
                return ToolResult(
                    title="Dataset",
                    output=(
                        f"dpo needs at least two attempts that differ; "
                        f"{len(eps)} episode(s) produced no pairs. Run the task "
                        f"again, or add checks that discriminate between runs."
                    ),
                    error=True,
                )
            pairs = [p.to_dict() for p in dataset.pairs]

        try:
            job = training.build_dataset(
                records, pairs=pairs, format=params.format,
                min_reward=params.min_reward, min_trust=params.min_trust,
            )
        except ValueError as exc:
            return ToolResult(title="Dataset", output=str(exc), error=True)

        if not job.rows:
            return ToolResult(
                title="Dataset",
                output=job.summary_text(),
                metadata=job.stats,
            )

        stem = params.path or f"./opendesk-rollouts/{params.format}.jsonl"
        await self._audit(
            ctx, "dataset_build", {"format": params.format},
            f"{len(job.rows)} rows",
        )

        lines = [job.summary_text(), ""]
        if params.val_ratio or params.test_ratio:
            parts = training.split_rows(
                job.rows, val_ratio=params.val_ratio,
                test_ratio=params.test_ratio,
            )
            base, ext = os.path.splitext(stem)
            written = {}
            for where, rows in parts.items():
                m = training.export_jsonl(
                    rows, f"{base}.{where}{ext}", manifest=job.stats
                )
                written[where] = m
                lines.append(f"  {where:<5} {len(rows):>4} row(s)  {m['path']}")
            metadata = {"splits": {k: len(v) for k, v in parts.items()},
                        **job.stats}
        else:
            m = training.export_jsonl(job.rows, stem, manifest=job.stats)
            lines.append(f"Wrote {m['rows']} row(s) to {m['path']}")
            lines.append(f"  manifest: {m['path']}.manifest.json")
            metadata = m

        return ToolResult(
            title=f"{len(job.rows)} {params.format} row(s)",
            output="\n".join(lines),
            metadata=metadata,
        )

    async def _list(self, ctx, params) -> ToolResult:
        eps = self._episodes(ctx, params)
        if not eps:
            return ToolResult(
                title="Rollouts",
                output="No closed episodes available for export.",
            )
        lines = [f"{len(eps)} exportable episode(s):\n"]
        for ep in eps:
            rew = f"reward={ep.reward:.0f}" if ep.reward is not None else "reward=—"
            metrics = (ep.process or {}).get("metrics") or {}
            steps = metrics.get("total_steps", "?")
            lines.append(f"  {ep.id}  {rew}  {steps} step(s)  {ep.task[:50]}")
        return ToolResult(title=f"Rollouts ({len(eps)})", output="\n".join(lines))

    # ------------------------------------------------------------------

    @staticmethod
    async def _audit(ctx, action: str, params: dict, result: str) -> None:
        try:
            from opendesk.computer.sandbox import ActionType, get_sandbox

            await get_sandbox(ctx.session_id).record_action(
                ActionType(action), params=params, result=result
            )
        except Exception:
            pass
