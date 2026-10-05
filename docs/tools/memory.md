# `memory` — Lossless Visual Memory

The `memory` tool keeps every screenshot a session has taken and lets the agent
look at them again, instead of only ever seeing the newest frame.

```
screenshot ──► observation store (lossless PNG) ──► memory(action="recall")
                     │
                     └─────────────────────────────► memory(action="diff")
```

## Why it exists

An agent that perceives the screen one frame at a time cannot answer "what did
this dialog look like before I dismissed it?" — the frame is gone. The usual
workaround is to navigate back and re-capture, which costs actions and may not
even reproduce the earlier state (a confirmation toast is gone for good).

Recent multimodal-agent work argues the opposite: give the model a
**lossless visual memory** and let it retrieve past observations as it reasons.
The retrieval is what matters — the model decides which moment to look at.

`memory` provides both halves: an observation store that keeps original bytes,
and a query surface the agent can drive.

## What is stored

Every `screenshot` call records an `Observation`:

| Field | Meaning |
|---|---|
| `index` | Stable absolute index (survives eviction) |
| `png` | The original PNG bytes — no re-encoding, no re-downscaling |
| `fingerprint` | Perceptual screen fingerprint (see `diagnose`) |
| `app`, `window` | Foreground context at capture time |
| `change_fraction`, `changed_region` | Pixel delta vs the previous observation |
| `marks_summary` | The Set-of-Marks listing, if marks were drawn |

Memory is a bounded ring buffer. When it fills, the oldest observations are
dropped and `memory(action="stats")` reports how many — eviction is always
visible, never silent.

## Actions

| Action | Arguments | What it does |
|---|---|---|
| `timeline` | `limit?` | List recent observations, oldest first |
| `recall` | `ref`, `include_image?` | Bring back one observation, optionally with its image |
| `find` | `app?`, `text?`, `since?`, `limit?` | Filter observations |
| `diff` | `ref`, `ref_b`, `ignore_regions?` | Compare two stored observations — **no re-capture** |
| `stats` | `cap?` | Held / evicted / bytes / distinct screens |
| `clear` | — | Drop the history |

### References

`ref` and `ref_b` accept:

- `"latest"` (default) — the most recent observation
- `"first"` — the oldest still held
- `"previous"` — one before the most recent
- an integer index — `7`, or `-1`, `-2` … counted from the end

### Capacity

Default capacity is **80 observations** (~100–300 MB at 1920×1080). Raise it for
long-horizon tasks:

```python
memory(action="stats", cap=300)
```

## Examples

```python
from opendesk.registry import create_registry
from opendesk.tools.base import allow_all_context

tools = create_registry()
mem = tools.get("memory")
ctx = allow_all_context()

# What have I seen recently?
await mem.execute(ctx, mem.parse_params({"action": "timeline", "limit": 5}))

# Show me the dialog from two frames ago
await mem.execute(ctx, mem.parse_params({
    "action": "recall", "ref": -2, "include_image": True,
}))

# Did the last three actions actually change anything?
await mem.execute(ctx, mem.parse_params({
    "action": "diff", "ref": 4, "ref_b": "latest",
}))
```

`diff` is the useful one when an agent is stuck: if the fingerprint distance
between two observations is a small number of bits, the two screens are the
same **functional state**. That is a structural statement — a typed glyph or a
tick of the clock also reads as a small distance, so pair it with
`change_fraction` (which counts changed pixels) before concluding the actions in
between accomplished nothing. When the churn lives in a known place — a clock, a
taskbar — pass `ignore_regions` so `change_fraction` measures the work rather
than the weather.

## Notes

- Recall attaches the **original** screenshot. Set `include_image=false` for a
  text-only description when you only need to know *which* observation it was.
- `changed_region` tells you *where* the screen changed since the previous
  capture — a cheap attention hint before a full visual re-read. The pixel diff
  reports changes down to a few dozen pixels, so a small edit is not filtered
  out; a live clock or moved cursor also registers, since that too is a genuine
  pixel change.
- `ignore_regions` excludes `[x, y, w, h]` boxes from a `diff`. A ticking clock
  is real pixel change, so no threshold can separate it from a real edit —
  excluding where it lives is the fix. Get the boxes from the accessibility tree
  rather than hardcoding them, so they survive a resolution or theme change:

  ```python
  from opendesk.computer.capture import regions_for_roles

  churn = regions_for_roles(ui_tree, ["menu bar", "static text"])
  await mem.execute(ctx, mem.parse_params({
      "action": "diff", "ref": 4, "ref_b": "latest", "ignore_regions": churn,
  }))
  ```

  The result reports `suppressed_pixels`, so an exclusion that quietly swallows
  a real edit shows up in the numbers instead of passing silently.
- When a capture recorded the accessibility tree (`tree=true` on
  [screenshot](screenshot.md)), `find(text=…)` also searches what those screens
  *said* — a value in a field, a row in a list — not just the app and window
  labels. Punctuation and digits are searchable, so "what did that invoice
  number look like five steps ago?" has an answer.
- Mark numbers in `marks_summary` are from capture time. `recall` warns about
  this: re-capture with `marks=true` before clicking by mark.
