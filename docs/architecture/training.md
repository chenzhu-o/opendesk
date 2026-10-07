# The Training Handoff

The [learning layer](learning.md) ends at JSONL: reward-labeled trajectories and
chosen/rejected pairs. This page is the last mile — turning those into the row
shape a trainer actually reads.

```python
from opendesk.learning import training

episodes = training.load_trajectories("runs/")          # or a single .jsonl
pairs = training.load_pairs("runs/preferences.jsonl")

job = training.build_dataset(episodes, pairs=pairs, format="dpo")
training.export_jsonl(job.rows, "runs/dpo.jsonl")
print(job.summary_text())
```

Still no gradients, still no policy. This module reshapes data.

## The rule that matters

**A completion holds the agent's actions and nothing else.**

A trajectory is dense with labels: per-step reward, discounted return, effect,
outcome. Every one of them is the grader's answer key. Put them in the token
stream and you have not built a harder task — you have built a shortcut:

| Leak | What the model learns |
|---|---|
| `<outcome reward="1"/>` at the end | Emit `reward="1"`. Cheaper than working. |
| `reward=+1.00` per step | Emit the annotation tokens in the right pattern. |
| `efficiency=0.67` in the header | Predict the metric, not the task. |

This is worst for preference data, where it is also hardest to notice. DPO's
loss is a function of the *difference* between the two log-probabilities. If
`chosen` and `rejected` are identical except for a trailing reward annotation,
nearly every token contributes zero and the entire gradient focuses on that one
annotation. Training converges beautifully — on the wrong objective.

So:

```python
training.render_completion(episode)
# 'click(name="Export", role="button")\ntype(text="~/invoice.pdf")'
```

Actions only. `render_action` quotes strings and sorts params so the same
behaviour always renders byte-identically; without that, two rollouts that did
the same thing could produce different completions and every pair built from
them would carry noise instead of signal.

It also drops the params recorded as `null`, and that is not cosmetic. The audit
log stores the tool's *parsed* params — which is its whole signature — so an
action that set one field arrives with every other field present and empty:

```python
# what the log holds
file_write(action="write_file", argv=None, command=None, content="total: 42\n",
           cwd=None, dst=None, path="report.txt", timeout=None)
# what it renders
file_write(action="write_file", content="total: 42\n", path="report.txt")
```

Six of those eight params are `null` on every single step. Rendering them doubles
the transcript and, worse, teaches the model to recite
`argv=null, command=null, timeout=null` — the harness's plumbing, not the task.
A param whose value merely repeats the action's own name goes too; a *different*
one stays, because `system`'s `file_write(action="write_file")` is a sub-action
the agent actually chose.

The labels are not discarded — they move to the row's metadata, where a loss can
use them for credit assignment:

```python
row = job.rows[0]
row["chosen"]          # "click(name=…)\ntype(text=…)"
row["chosen_reward"]   # 1.0     ← a label, in metadata
row["chosen_returns"]  # [8.4, 7.6, 6.5, …]
```

`describe_trajectory()` renders the other view — labels included — for reading a
log or debugging a reward. Do not train on it.

## What a completion is not

opendesk records structured actions. There is no `completion` field in the audit
log to recover, so what `render_completion` produces is a *canonical
serialization* of what the agent did, not the tokens your policy emitted.

That distinction is not cosmetic. Training on the canonical form teaches the
rendering — the exact spelling of `click(name="Export", role="button")` — as
much as it teaches the task, and every row carries
`"completion_source": "opendesk.canonical/1"` to say so.

If your harness logged the model's own output alongside the episode, hand it
back and it wins:

```python
job = training.build_dataset(episodes, format="sft", resolver=my_log_lookup)
job.rows[0]["completion_source"]   # "caller"
```

The resolver returns `None` for episodes you have no log for, and those fall
back to the canonical rendering. Mixing is fine and is reported per row.

### What must not be memorised

The audit log records **absolute paths**, so a run in a temporary directory
bakes that directory's random name into every completion:

```
file_write(content="total: 42\n", path="opendesk-eval-jwcw5kxa/report.txt")
```

The same task then renders differently on every run — the model is taught a path
that will never exist again, and the identical-completion check that drops
degenerate DPO pairs can never fire. Pass `sanitize` to rewrite the episode
before anything is rendered; it receives the whole record, so you can strip a
sandbox root, redact a home directory, or normalise separators.

There is usually a second prefix to remove, and it is easy to miss. A harness
that runs N attempts puts each in its own directory, so the paths carry the
*attempt number*:

```python
def sanitize(episode):
    needle = str(SANDBOX_ROOT)
    attempt_seg = re.compile(r"attempt\d+[\\/]")

    def scrub(value):
        if isinstance(value, str):
            if value.startswith(needle):
                value = value[len(needle):].lstrip("\\/")
            return attempt_seg.sub("", value)
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    return scrub(episode)   # the whole record: reward_spec holds paths too

job = training.build_dataset(episodes, format="sft", sanitize=sanitize)
```

The attempt number is an artefact of the harness, not a property of the task:
rendering it teaches the model which attempt it is, and it also means three tries
at the same task produce three *different* completions, so the degenerate-pair
check below never fires. Note that `scrub` runs over the whole record — for GRPO
the `reward_spec` holds the same absolute paths, and that spec is what a trainer
sends back to re-verify.

[`examples/learning-eval`](../../examples/learning-eval/) does exactly this, and
prints the difference.

## Three row shapes

### `sft` — imitate what worked

One row per successful attempt. `min_reward` (default `1.0`) is the verifiable
outcome reward, not a score threshold.

```json
{"prompt": "Task: Export the invoice", "completion": "click(…)\ntype(…)",
 "episode_id": "a1b2c3", "reward": 1.0, "steps": 3,
 "returns": [8.4, 7.6, 6.5], "trust": "ui",
 "completion_source": "opendesk.canonical/1"}
```

### `dpo` — prefer what worked

Pairs are joined to their trajectories by `episode_id`, so the prompt is
guaranteed identical on both sides.

```json
{"prompt": "Task: Export the invoice",
 "chosen": "click(name=\"Export\", role=\"button\")\n…",
 "rejected": "click(name=\"Settings\", role=\"button\")\n…",
 "criterion": "outcome", "margin": 1.0,
 "chosen_reward": 1.0, "rejected_reward": 0.0}
```

Pairs whose two completions render identically are **dropped** and counted. Such
a pair has no gradient to give, and a file full of them looks exactly like a
hung trainer. `drop_degenerate=false` keeps them if you are inspecting rather
than training.

### `grpo` — try it yourself

The on-policy shape. One row per task, holding every attempt with its verifiable
reward and per-step returns:

```json
{"prompt": "Task: Export the invoice", "group_id": "Export the invoice",
 "reward_spec": {"task": "…", "checks": [{"kind": "file_exists", …}]},
 "completions": [{"text": "…", "reward": 1.0, "returns": [8.4, …]}, …],
 "rewards": [1.0, 0.0], "reward_mean": 0.5, "reward_spread": 1.0}
```

`reward_spec` travels with the row deliberately. The trainer samples from the
prompt, then sends the spec back to opendesk's `reward` tool to re-verify — so
the reward still comes from files, shell output and the accessibility tree
rather than from a model. A task with only one attempt is dropped: every
group-relative advantage computed against a single sample is zero.

A group where every attempt scored the same is reported but **kept**:

```
4 grpo row(s) from 12 episode(s)
  note: 3 group(s) have no reward spread, so their group-relative
        advantages are all zero. Keep them only if your objective uses
        the reward directly.
```

Its advantage is zero under GRPO, so the row trains nothing — but a group that
all failed is still a negative example for an objective that uses the reward
directly, and only the caller knows which loss it is running. Same reasoning as
the trust floor: report, do not silently mutate. Filter on `reward_spread > 0`
if your loss is purely group-relative.

## Trust, reported rather than enforced

`process.signal_source` says how the step effects were read — and the three
sources differ in how much they can be trusted:

1. **`ui`** — the accessibility tree. Sees a single edited glyph.
2. **`screen`** — screenshot fingerprints. Groups by layout, so a content edit
   inside an unchanged layout reads as no effect.
3. **`action`** — the non-visual fallback, which *infers* effect from the action
   instead of observing it.
4. **`unknown`** — the field is absent, so nothing is claimed.

Every row carries its `trust`, and the manifest tallies them. `min_trust` can
set a floor, but the default is **no floor** — because trust only describes the
*step-level* signal, and that is not what every format trains on:

| Format | Training signal | Needs trust? |
|---|---|---|
| `sft` | outcome reward + the actions | no |
| `dpo` | which attempt passed the checks | no |
| `grpo` | per-step returns → advantages | **yes** |

Default-filtering on it would silently delete valid SFT and DPO rows — the
failure mode being a plausible-looking zero-row file. Silently *including*
inferred returns in a GRPO run is the more dangerous error, because the training
runs and just learns from a guess. So the module does neither: it keeps the data
and says so.

```
2 sft row(s) from 3 episode(s)
  tasks 2  groups 2  successes 1
  effect signal: action=1, ui=2
  note: 1 row(s) have an inferred or unrecorded effect signal;
        set min_trust='screen' to require observed effects.
```

Set `min_trust="screen"` for GRPO. Rows are always filterable afterwards either
way.

## Splitting without leaking

```python
parts = training.split_rows(job.rows, val_ratio=0.2, test_ratio=0.1)
```

Splits are taken **by task, not by row**. Attempts at one task are
near-duplicates — same prompt, overlapping actions — so splitting them across
train and eval leaks the evaluation set into training and reports a number that
means nothing. Grouping by `group_id` (or `task`) keeps every attempt of a task
on one side.

Assignment is a hash of the group key, so it is deterministic across runs and
machines and does not depend on row order. Changing `seed` reshuffles it.

## Through the tool

`rollout(action="dataset")` is the same thing over MCP, reading episodes from the
live session:

```json
{"action": "dataset", "format": "dpo", "path": "runs/dpo.jsonl",
 "strategy": "all_vs_best", "val_ratio": 0.2}
```

It writes one file per split when a ratio is set, and a
`<path>.manifest.json` beside the data recording the render version, the trust
tally and the row count. A directory of `.jsonl` with no record of which
renderer or floor produced it is not reproducible.

## Folding in the observation

`render_prompt` is the task and goal, because the per-step observation lives in
your harness. When you fold it back in — a `prompt_resolver` returning the
accessibility tree — the budget you give it decides what the row can teach, and
the failure is quiet: a truncated observation does not look broken, it looks
like a model that will not learn.

Measured on 2228 recorded steps, asking whether the gold action's coordinate is
present in the text the model actually receives:

| observation budget | targets visible |
|---|---|
| 3 000 chars | 30.4% |
| 6 000 chars | 43.6% |
| 12 000 chars | 53.1% |
| uncapped | 53.1% |

Two things follow. A 3 000-character tree hides two thirds of the targets, so
most rows ask for a coordinate that cannot be read off the input at all. And the
ceiling saturates: past roughly 12 000 characters no further target becomes
visible, because the remaining ones are absent from the tree itself — so about
47% of coordinate actions are unpredictable from this observation by
construction, whatever model is applied to it.

The tree is also larger than it looks in a rendered prompt, so the character
budget is not the only cut. A 3 000-character tree costs about 1 769 prompt
tokens; a token cap set below that truncates the same text a second time,
leaving less than half of an already-halved observation.

Two reporting habits catch this before it costs a training run:

* **Put a floor next to the score.** `func_match` compares call *names* — the
  regex captures the identifier, not the arguments — so on a small action
  vocabulary it largely measures the base rate. A predictor that emits one verb
  unconditionally scored 0.430 against a fine-tuned model's 0.415. The model sat
  below a constant, and neither number was wrong on its own.
* **Report per-verb recall, not only the pooled figure.** The same run scored
  0.814 on the most common verb, 0.038 on the next, and 0.000 on five more. It
  had learned which action was most common, not which action to take.

## Remote SFT on a T4 (Kaggle)

Reference run: `markov-ai/computer-use` → `training.build_dataset` → full
fine-tune of Qwen2.5-0.5B-Instruct on a 14.56 GiB Tesla T4.

**Observation budget vs VRAM.** With the tree folded in at 12 800 characters,
most rows need thousands of prompt tokens. On sm_75 the attention peak scales
with sequence length squared; a 7168-token row can request ~2.7 GiB for a single
attention materialisation, on top of ~8 GiB for fp32 weights, gradients, and
AdamW state. Practical mitigations that kept the run honest:

* memory-efficient SDPA (flash is unavailable below sm_80);
* 8-bit AdamW optimizer state;
* LM-head loss only on supervised completion tokens (not the full vocabulary
  projection over the prompt);
* **probe the worst real training row** at each candidate prompt cap before
  building the dataset — synthetic probes that shorten the completion lied (v9
  passed at 6144 then OOM'd on step 2).

**Honest baselines at full tree budget (before cap downshift).** An untrained
model at the expanded observation budget scored held-out NLL ≈ 4.22 and
`func_match` 0.0 — the earlier ~0.45 figure was a truncation artefact (marginal
click rate with no usable coordinates in the input).

**Ceiling still applies.** Even with cap chosen to fit VRAM, ~47% of coordinate
targets remain absent from accessibility trees; generation metrics must be read
with `floor`, per-verb recall, and `visible_target_rate` alongside pooled
`func_match`.

**First successful training pass (v10, cap 4608).** Worst-row probing picked
4608 prompt tokens (~3.7 GiB headroom on the probe). After one epoch: held-out
teacher-forced NLL dropped from 4.49 → 0.98 (token accuracy 0.40 → 0.73).
In-loop epoch eval was removed in v11 — it OOM'd from fragmentation after epoch 1
while the training steps themselves fit.

## What this will not do

- **No gradients, no tokenizer, no framework import.** The output is JSONL. If
  your trainer wants `datasets.Dataset`, one line does it — but that dependency
  is not opendesk's to impose.
- **No token accounting.** Chat templates and truncation are trainer
  configuration. A completion here is the action sequence; how your template
  frames it is yours.
- **No screen content.** Observations are stored as metadata (app, window, size,
  fingerprint, screenshot path), not text, so `render_prompt` is the task and
  goal. The per-step observation your policy conditioned on lives in your
  harness — fold it in there.
- **No model in the reward path.** `reward_spec` is returned *for* re-verification
  by opendesk, not replaced by a learned reward.
