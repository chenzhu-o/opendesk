# Learning-layer evaluation

A runnable harness for the reward / process / preference machinery — no GUI, no
display, no API keys. It runs four tasks, three times each, and reports what each
attempt scored and what the agent claimed about its own state.

```bash
python examples/learning-eval/run.py
python examples/learning-eval/run.py --verbose
python examples/learning-eval/run.py --export runs/        # one trajectory per attempt
python examples/learning-eval/run.py --dataset data/       # sft / dpo / grpo rows
python examples/learning-eval/run.py --keep --root /tmp/x  # inspect the files
```

```
opendesk learning-layer evaluation
  4 task(s) x 3 attempt(s)

  [PASS] write-report   attempt 0  steps=1 eff=100% claims=2/2 regressed=0
  ...
  [fail] regression     attempt 2  steps=2 eff=100% claims=1/1 regressed=1

task               pass   score  steps   eff   prec  regr
---------------------------------------------------------
write-report        3/3   100%    1.0  100%   100%     0
rename-log          3/3   100%    2.0  100%   100%     0
two-files           2/3    83%    1.7  100%   100%     0
regression          0/3     0%    2.0  100%   100%     3
---------------------------------------------------------
  8/12 attempts passed  |  20 step(s) total
  15 claim(s) declared, 15 held (100% precision), 3 regressed
```

## What it is measuring — and what it is not

**Not** a model benchmark. The "agent" is a scripted list of actions, so the
numbers say nothing about how well a model drives a desktop. opendesk does not
ship a policy, and a harness that pretended to measure one would be measuring the
script.

**Is** the environment half of the loop, exercised for real. Every task declares
machine-checkable success criteria; every attempt produces a real audit log, real
step-level process rewards, real agent declarations, and real best-of-N preference
pairs. The only stubbed component is the policy.

| Column | Comes from |
|---|---|
| `pass` | the task's own `checks`, evaluated by `reward(action="end")` |
| `score` | fraction of required checks passed — partial credit |
| `steps` | agent actions in the audit log (bookkeeping excluded) |
| `eff` | share of judgeable steps that changed the state — `score_process` |
| `prec` | claims that were true **when declared** |
| `regr` | claims that held when declared and were false at the end |

`prec` and `regr` are not part of the reward and never can be: claims are the
agent's own, and letting them score the task is the reward hack the assertion
track exists to avoid. They measure something else — how well the agent knows its
own state, which is what a self-verifying policy needs to be trained against.

## The tasks

Each task is a separate `checks` spec and a separate scripted `attempt`. The
attempt does not know what it is graded on.

| Task | What it tests |
|---|---|
| `write-report` | two required checks, both pass — the straightforward case |
| `rename-log` | a move: one file must be gone and another present |
| `two-files` | a deliberately unreliable policy, so attempts diverge |
| `regression` | the agent claims success, then breaks it |

`two-files` is the one that makes best-of-N meaningful. Without a spread of
outcomes every attempt ties and there is nothing to pair. `regression` is the
case the terminal reward cannot express: the run failed, and the interesting fact
is that the agent believed otherwise two steps earlier.

## Why file tasks

They run anywhere: any OS, a temp directory, no display. That is what makes this
something you can run in CI and diff over time rather than a demo that only works
on the author's laptop. It also means the actions go through the `system` tool's
filesystem operations rather than a shell string — `echo` and `>` are not
portable (on Windows PowerShell they write UTF-16, and `rm -f` is an ambiguous
parameter), and a harness whose *fixtures* are platform-dependent cannot report
on anything else.

The same machinery is what the GUI tools feed: the same check kinds, the same
audit log, the same export. With no screenshots, `score_process` falls back to
inferring effects from the actions themselves and reports
`signal_source: "action"` — the documented path for a CLI-only session.

## From attempts to training rows

`--dataset` runs the same four tasks and then does the last mile: it exports the
trajectories as JSONL, reads them back with `training.load_trajectories`, and
renders three datasets — one per row shape a trainer reads.

```
  Training datasets → data/

  8 sft row(s) from 12 episode(s)
    tasks 3  groups 3  successes 8
    dropped: 4 no verifiable success
    effect signal: action=8
    note: 8 row(s) have an inferred or unrecorded effect signal; set
          min_trust='screen' to require observed effects.

  A completion holds actions only — no reward, no return:
    file_write(action="write_file", content="old\n", path="rename-log/app.log")
    file_move(action="move", dst="rename-log/app.log.bak", path="rename-log/app.log")
    (labels live beside it: reward=1.0, returns=[10.305, 9.9])
```

Three things this run demonstrates that a description would not:

**The labels stay out of the text.** The completion above is the agent's
actions; the reward and the discounted returns sit beside it in the row. Put
them *in* the text and the cheapest way for DPO to raise its log-probability
difference is to print `reward="1"` — see
[the training handoff](../../docs/architecture/training.md).

**Trust is reported, not enforced.** Every step here is scored from the
non-visual fallback (`signal_source: "action"`), because a CLI-only harness has
no screenshots and `score_process` infers effects from the actions themselves.
The rows are still produced — SFT and DPO do not train on the step signal at
all — and the manifest says so. `min_trust='screen'` would keep **0** of the 4
GRPO rows, which is what that floor is for.

**Paths are scrubbed before rendering.** Each attempt lives in its own directory
under `--root`, and the audit log records those absolutely. The run passes a
`sanitize` callback that rewrites the whole record — so the completion above
reads `rename-log/app.log`, not
`<tmp>/opendesk-eval-jwcw5kxa/rename-log/attempt0/app.log`.

Two prefixes have to go, and the second is the subtle one:

- the **root**, because it is a temp directory whose name changes per run;
- the **`attemptN/` segment**, because the attempt number is an artefact of this
  harness, not a property of the task. Left in, three tries at one task render as
  three *different* completions — so identical behaviour never looks identical
  and the check that drops degenerate DPO pairs can never fire. On this run every
  GRPO group went from three distinct texts to one.

## Reading a failure

`regression` scores 0/3 by design. Its exported trajectory is the most
interesting one to open:

```json
{"type": "episode", "outcome": {"reward": 0.0, "passed": false, ...},
 "assertions": {"metrics": {"declared": 1, "held": 1, "regressed": 1,
                            "precision": 1.0, "stability": 0.0}}}
{"type": "step", "step": 1, "action": {"type": "file_write"}, "effect": "changed",
 "assertions": [{"name": "output written", "held": true, "regressed": true}]}
{"type": "step", "step": 2, "action": {"type": "file_delete"}, "effect": "none"}
```

`precision 1.0` — every claim was true when made. `stability 0.0` — none survived.
The milestone sits on step 1, the step that caused it, not on the step that
noticed. That is the annotation a step-level credit assignment needs and an
outcome-only reward cannot produce.
