---
name: opendesk
description: Drive a real desktop with the OpenDesk MCP tools (screenshot, ui, mouse, keyboard, app, ocr, memory, opendesk_*). Use when the user asks to click, type, open apps, read the screen, automate a desktop app, recall what was on screen earlier, or control a paired remote machine.
---

# OpenDesk: computer use

OpenDesk gives you eyes and hands on a real macOS, Windows or Linux desktop through MCP tools. Everything runs locally; remote machines are reached over an end-to-end encrypted link only after a human pairs them.

## Before the first action

1. On macOS, the app that runs OpenDesk needs **Accessibility** and **Screen Recording**: Claude Desktop, the terminal running Claude Code, Cursor or VS Code. If a tool reports a permission error, ask the user to run:

   ```bash
   uvx opendesk check --open
   ```

   It opens the right System Settings pane. After granting, the user must restart that app.
2. Call `screenshot` once to see the current state.

## Workflow (cheapest and most reliable first)

1. **No GUI needed?** If the task can be done with files, a shell command or an app's API/CLI, do that instead of driving the screen.
2. **`ui`**: accessibility API. `ui(action="get_tree", app="Safari")`, then `click`, `type`, `click_menu` or `get_value` by element name. No coordinates.
3. **`screenshot(marks=true)`**: numbered boxes over interactive elements when the tree is missing or ambiguous.
4. **`mouse` / `keyboard`**: last resort. Pass `image_width` / `image_height` from the latest screenshot so coordinates scale correctly on Retina screens.
5. **Verify** every state-changing step with a fresh `screenshot` before continuing. Use `ocr` to read text the tree doesn't expose.

Use `app` to open, focus or close applications, and `clipboard` to move text in or out.

## Safety

- Some actions may require the user's approval. If a call is denied, don't retry it or look for a workaround; ask the user.
- The user can halt everything with the kill switch. If a call says it's engaged, stop.
- Don't enter passwords, payment details or other secrets unless the user typed them for this purpose. Confirm before sending messages, deleting data or buying anything.
- `audit` shows every action taken in the session.

## Remote machines

- `opendesk_peers` lists this machine (`local`) and the paired peers; `opendesk_describe` explains what a peer is for.
- `opendesk_use(peer="name")` makes a peer the default target, or pass `peer="name"` on a single call. `opendesk_use(peer="local")` switches back.
- Pairing is deliberately not a tool: the user runs `opendesk pair` on the target and `opendesk pair-with <host> <code>` on the controller.

## Screen memory

- `memory` (read-only) searches what was on screen earlier: `memory(action="search", query="invoice", since="last week")`, then `memory(action="show", id=...)`.
- `memory_manage` pauses capture, edits the per-app deny list, changes retention or deletes history. Deletion is a dry run until `confirm=true`; confirm with the user first.
- Capture only runs if the user started it (`opendesk memory start`). If `memory(action="status")` says the daemon isn't running, tell the user rather than starting it yourself.

Docs: https://vitalops.ai/opendesk/docs/
