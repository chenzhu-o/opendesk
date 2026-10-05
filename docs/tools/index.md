# Tools Reference

All tools share the same interface: `await tool.execute(ctx, params) -> ToolResult`.

**Tool priority rule:** `system` (CLI/filesystem) → `ui` → `screenshot(marks=True)` → `mouse` with image dimensions.

| Tool | Description |
|---|---|
| [`system`](system.md) | Hybrid CLI + filesystem layer — run commands, read/write files |
| [`ui`](ui.md) | Accessibility-based UI interaction — click by name, type, read values |
| [`screenshot`](screenshot.md) | Screen capture with optional Set-of-Marks overlay |
| [`mouse`](mouse.md) | Pixel-level mouse control with HiDPI coordinate translation |
| [`keyboard`](keyboard.md) | Type text, press keys, send hotkeys |
| [`app`](app.md) | Open, close, focus, and list applications |
| [`clipboard`](clipboard.md) | Read and write clipboard text |
| [`ocr`](ocr.md) | Extract text from any screen region |
| [`audit`](audit.md) | Read the session audit log |
| [`skill`](skills.md) | Save, find, and run reusable parameterised skills |
| [`learn`](learn.md) | Record and replay desktop workflows |
| [`memory`](memory.md) | Recall past screen observations (lossless visual memory) |
| [`diagnose`](diagnose.md) | State-transition diagnosis — where a session stalled |
| [`reward`](reward.md) | Verifiable task rewards, goal states, agent assertions, episode bookkeeping |
| [`rollout`](rollout.md) | Export RL-ready trajectories and preference pairs |

---

## Learning & diagnosis

These four tools turn a session into something measurable and reusable. None of
them calls a model — every signal is derived from observable state.

```
screenshot ──► memory    ──► look back at what was on screen
audit log  ──► diagnose  ──► where did it stall?
reward     ──►            ──► did it satisfy the checklist?
rollout    ──►            ──► training data out
```

See [The Learning Layer](../architecture/learning.md) for how they fit together.

---

## ToolResult

All tools return:

```python
@dataclass
class ToolResult:
    title: str           # short human-readable label
    output: str          # text returned to the LLM
    error: bool          # True if the action failed
    attachments: list[Attachment]  # binary files (screenshots, etc.)
    metadata: dict       # extra data for programmatic consumers
```

`Attachment`:

```python
@dataclass
class Attachment:
    filename: str
    content: bytes
    media_type: str      # e.g. "image/png"

    def to_base64(self) -> str: ...
```

---

Start with the primary interaction tool: [ui →](ui.md)
