# `rollout` — Trajectories & Preference Data

`rollout` turns recorded episodes into artefacts a training pipeline can
consume: JSONL trajectories with per-step rewards, and chosen/rejected
preference pairs.

```
episodes ──► rollout(export) ──► run.jsonl + run.jsonl_images/*.png
         └─► rollout(pairs)  ──► preferences.jsonl
```

## Why it exists

A session log is a record. A *trajectory* is training data. The difference is
structure: an episode has a task, a start, an end, an outcome reward, and a step
sequence in which each step ties an observation to the action taken against it
and the reward that followed.

Run the same task several times and you also get something an agent that cannot
update its weights can still use — **best-of-N selection** — and, if it can,
preference data.

## Trajectory format

`export` writes JSONL, one record per line.

**Line 1** — the episode header:

```json
{"type": "episode", "schema": "opendesk.trajectory/1", "episode_id": "a1b2c3",
 "task": "Export the invoice", "goal": "done", "reward_spec": {...},
 "outcome": {"reward": 1.0, "score": 1.0}, "process": {"gamma": 0.95, ...},
 "assertions": {"metrics": {...}, "assertions": [...]},
 "step_count": 12}
```

**Lines 2..N** — the steps:

```json
{"type": "step", "step": 1, "timestamp": 1770000000.1,
 "observation": {"screen": "3f2a…", "observation_index": 4,
                 "app": "Chrome", "window": "Invoices",
                 "screenshot": "run.jsonl_images/a1b2c3_step001.png"},
 "action": {"type": "mouse_click", "params": {"x": 412, "y": 288}},
 "error": null, "effect": "changed", "next_state": "S3",
 "reward": 0.9, "return": 8.41, "assertions": [], "done": false}
```

Screenshots are written as PNG files next to the JSONL by default. Pass
`embed_images=true` for one self-contained file — larger, but easier to ship.

Journal entries (the reward tool's own checks, episode markers, the agent's
assertions) are excluded, so every step corresponds to something the agent
actually did.

## Assertion milestones

A claim the agent declared mid-episode (see
[`reward`](reward.md#agent-assertions)) rides on the step it followed, so credit
assignment has a dense hook that is not a pixel diff:

```json
{"type": "step", "step": 4, …,
 "assertions": [{"name": "invoice downloaded", "held": true,
                 "claim": "the export finished and the PDF is on disk",
                 "held_late": true, "regressed": false}]}
```

The header carries the whole picture, including calibration:

```json
"assertions": {"metrics": {"declared": 3, "held": 3, "failed": 0,
                           "regressed": 1, "rechecked": 3,
                           "precision": 1.0, "stability": 0.667}}
```

`precision` is how many claims were true when made — the agent's
self-knowledge. `stability` is how many of the verified ones survived to the
end; a drop is a regression, where the agent declared success and later broke
it. Neither contributes to `reward`: the agent authors its own claims, so they
are evidence *about* the agent rather than about the task.

## Per-step rewards

`reward` computes step-level rewards and discounted returns, which `export`
folds into each step:

| Signal | Shaping |
|---|---|
| state changed | `+progress_reward` |
| no visible change | `-no_effect_penalty` |
| error | `-error_penalty` |
| repeated cycle | `-loop_penalty` |
| every step | `step_cost` (acting is not free) |
| last step | `+success_bonus × outcome_reward` |

`return` is the discounted sum from that step onward
(`G_t = r_t + γ·G_{t+1}`), so `returns[0]` values the whole trajectory and
`returns[-1]` is dominated by the outcome.

Observation actions — screenshots, OCR, reads — are recorded as steps but carry
no reward: they observe rather than act.

## Preference pairs

When a task runs more than once, `pairs` ranks the attempts and emits
chosen/rejected examples.

Ranking prefers, in order: **higher outcome reward**, then **higher dense
score**, then **fewer steps**, then **fewer errors**.

Two kinds of pair result:

- **outcome** — the attempts differ in reward. The usual case.
- **efficiency** — every attempt succeeded, so reward carries no signal and the
  shorter run wins. Without this, a task the agent has already mastered
  produces no usable pairs at all.

`strategy` controls the pairing:

| Strategy | Pairs |
|---|---|
| `best_vs_worst` (default) | strongest vs weakest — one clean contrast |
| `all_vs_best` | best against each weaker attempt |
| `adjacent` | neighbours after ranking — finest-grained contrast |

## Actions

| Action | Arguments | What it does |
|---|---|---|
| `export` | `episode_id?`, `path?`, `image_dir?`, `embed_images?` | Write a trajectory |
| `rank` | `task?` | Rank attempts best-first |
| `pairs` | `task?`, `path?`, `strategy?`, `min_margin?`, `max_pairs?` | Build preference pairs |
| `dataset` | `format?`, `path?`, `min_trust?`, `min_reward?`, `val_ratio?`, `test_ratio?`, `strategy?` | Write trainer-ready rows |
| `list` | `task?` | Show exportable episodes |

## The `dataset` action

`export` and `pairs` produce the environment half — trajectories and preference
pairs. `dataset` is the last mile: it renders those into the row shape a trainer
reads, in one of three formats.

```json
{"action": "dataset", "format": "dpo", "path": "runs/dpo.jsonl",
 "strategy": "all_vs_best", "val_ratio": 0.2}
```

| `format` | Rows | Trains on |
|---|---|---|
| `sft` | `prompt` → `completion`, successful attempts only | the outcome reward |
| `dpo` | `prompt` → `chosen` / `rejected`, joined from the pairs | which attempt passed |
| `grpo` | one row per task: every attempt, its reward, its returns | the per-step returns |

A **completion holds the agent's actions and nothing else**. Every other field in
a trajectory — per-step reward, return, effect, outcome — is a label, and a model
trained on the labels learns to emit them instead of solving the task. The labels
move to the row's metadata, where a loss can still use them.

Setting `val_ratio` or `test_ratio` writes one file per split and splits **by
task**, not by row: attempts at one task are near-duplicates, so splitting them
across train and eval leaks the evaluation set into training.

A `<path>.manifest.json` is written alongside the data with the render version,
the trust tally and what was dropped.

See [The Training Handoff](../architecture/training.md) for the full picture —
including why a canonical rendering is not the policy's own tokens, and what to
do about it.

## Examples

```python
from opendesk.registry import create_registry
from opendesk.tools.base import allow_all_context

tools = create_registry()
rollout = tools.get("rollout")
ctx = allow_all_context()

# One trajectory
await rollout.execute(ctx, rollout.parse_params({
    "action": "export", "path": "runs/ep1.jsonl",
}))

# After running the same task a few times as separate episodes
await rollout.execute(ctx, rollout.parse_params({
    "action": "pairs", "task": "Export the invoice",
    "path": "runs/prefs.jsonl", "strategy": "all_vs_best",
}))
```

## Consuming the output

```python
import json

with open("runs/ep1.jsonl", encoding="utf-8") as fh:
    header = json.loads(fh.readline())
    steps = [json.loads(line) for line in fh]

# (observation, action, reward, done) tuples for an offline RL loop
transitions = [
    (s["observation"], s["action"], s["reward"], s["done"]) for s in steps
]
```

The format is deliberately plain JSONL — no framework import, no schema
registry. `verl`, `TRL`, `OpenRLHF` and a hand-written loop can all read it.

For the step after this — rendering those trajectories into rows a trainer reads
directly — see [The Training Handoff](../architecture/training.md) and the
`dataset` action above.
