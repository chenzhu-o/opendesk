# `ui` — Accessibility-based UI Interaction

The primary interaction tool. Clicks, types, and reads values using the platform's native accessibility API. No pixel coordinates needed.

```python
from opendesk.tools.ui import UITool
tool = UITool()
```

## Actions

| Action | Required params | Description |
|--------|----------------|-------------|
| `get_tree` | `app` | List all accessible elements in the app window |
| `click` | `app`, `title` or `role` | Click a button or element by its visible label |
| `click_menu` | `app`, `menu`, `menu_item` | Click a menu bar item: `File → Save` |
| `type` | `app`, `text` | Type text into the focused element (Unicode-safe). `verify=true` reports whether the content changed |
| `press_key` | `app`, `key` | Press a key or chord. `verify=true` as above |
| `get_value` | `app`, `title` or `role` | Read the current text value of an element |

## Ask Claude

> "Click the Save button in TextEdit"

> "Open the File menu in Safari and click New Window"

> "Type 'hello' into the search box in Finder"

> "What buttons are visible in this app?"

> "What's the current value of the address bar in Safari?"

This is the most reliable way to interact with apps — Claude uses the accessibility tree so it doesn't need to guess pixel coordinates.

---

## SDK examples

```python
params = UITool.Params

# List what's in the window
await tool.execute(ctx, params(action="get_tree", app="TextEdit"))

# Click a button
await tool.execute(ctx, params(action="click", app="TextEdit", title="Save"))

# Open File menu → New
await tool.execute(ctx, params(action="click_menu", app="TextEdit", menu="File", menu_item="New"))

# Type text (clipboard-paste, full Unicode)
await tool.execute(ctx, params(action="type", app="TextEdit", text="Hello, 世界 🌍"))

# Press Cmd+S to save
await tool.execute(ctx, params(action="press_key", app="TextEdit", key="s", modifiers=["command"]))

# Read a text field value
await tool.execute(ctx, params(action="get_value", app="Safari", title="Address and Search Bar"))
```

## Platform notes

- **macOS**: uses AppleScript / System Events. App name must match Activity Monitor (e.g. `"Google Chrome"` not `"chrome"`).
- **Linux**: uses AT-SPI2 (`pyatspi`) with `xdotool` fallback for type/press_key.
- **Windows**: uses UI Automation with Win32 fallback. App name matches window title or process name.

When an app uses custom rendering (canvas apps, games, Electron), `get_tree` may return an empty tree — fall back to `screenshot(marks=True)` and the `mouse` tool.

## Reading the tree keeps the content digest current

Every action that reads the tree — `get_tree`, `click`, `get_value` — records a
**content digest** of it on the session. Each action logged from then on carries
that digest, which lets [`diagnose`](diagnose.md) group states by what the screen
*said* rather than how it looked. That matters because a perceptual fingerprint
cannot see a single edited character: a glyph edit and a clock tick both move a
downscaled global hash by zero bits, while the digest separates them exactly.

It costs nothing extra there — the tree is read anyway to resolve the target. See
[The Accessibility Channel](../architecture/accessibility.md).

### Keyboard actions refresh it too

`type`, `press_key` and a native `click_menu` mutate the interface without needing
the tree to find anything, so they take a digest reading themselves. Without that
they would stamp whatever digest an *earlier* action happened to leave behind, and
an edit they made would be credited to whichever step next read a tree — or to no
step at all, if none did.

The read is best-effort. If the tree is unreadable the keystrokes still go
through: a keyboard action works on hosts where accessibility does not, and
refusing to type because the tree cannot be read would be a much worse trade.

### Checking that a keystroke landed

`type` and `press_key` accept `verify=true`, which reads the tree again afterwards
and compares content:

```python
await tool.execute(ctx, params(
    action="type", app="Numbers", text="150", verify=True,
))
# → Typed 3 chars into Numbers: '150'
#     verification: content changed (c23b50ef → 9f9aebb4).
```

It answers a question an agent otherwise cannot: *did that land*. Because the
comparison is content and not pixels, and because ambient furniture is pruned, a
ticking clock between the two reads does not make an inert keystroke look like a
success. Three outcomes, and they are deliberately distinct:

| Outcome | Means |
|---|---|
| `content changed` | something in the interface changed |
| `content UNCHANGED` | nothing did — check focus, or run `get_tree` |
| `verification: unavailable` | no tree to read; **unknown, not unchanged** |

The extra read is why this is opt-in.

---

Next: [screenshot →](screenshot.md)
