# OpenDesk Examples

Practical examples showing OpenDesk in action against real applications.

## UI Testing

Use OpenDesk to test a web app's UI through natural language.

**[TaskFlow](https://github.com/vitalops/taskflow-demo)** is a task management app with the kind of UI you'd find in production -- sidebar nav, tabs, modals, tables, forms, toggles, and notifications. We use it here as a test target to walk through OpenDesk's tools.

See [ui-testing/](ui-testing/) for setup and examples.

## Learning-layer evaluation

A runnable harness for the reward, process-reward and preference machinery —
four tasks, three attempts each, no GUI required. Pass `--dataset` to also render
the run into SFT / DPO / GRPO rows.

```bash
python examples/learning-eval/run.py
python examples/learning-eval/run.py --dataset data/
```

See [learning-eval/](learning-eval/) for what it measures and, more importantly,
what it does not.

---

More examples coming soon.
