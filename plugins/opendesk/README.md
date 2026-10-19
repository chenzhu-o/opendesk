# OpenDesk plugin for Claude Code

OpenDesk by VitalOps is an open-source computer-use MCP server. This plugin connects Claude Code to it, so Claude can see and control your macOS, Windows or Linux desktop: take screenshots, act on UI elements through the accessibility API, use the mouse and keyboard, read text with OCR, search a local screen history and drive paired remote machines. It is unrelated to the openDesk office suite.

## Install

```bash
claude plugin marketplace add vitalops/opendesk
claude plugin install opendesk@opendesk
```

Inside a session: `/plugin marketplace add vitalops/opendesk`, then `/plugin install opendesk@opendesk`.

The plugin starts the server with `uvx --from opendesk==0.3.0 opendesk-mcp`, so you need [uv](https://docs.astral.sh/uv/) installed. Nothing else is required.

## What's included

- **MCP server** `opendesk` (`.mcp.json`): the screenshot, ui, mouse, keyboard, app, clipboard, ocr, learn, schedule, audit, memory, memory_manage and `opendesk_*` peer tools.
- **Skill** `opendesk` (`skills/opendesk/SKILL.md`): the recommended workflow, permission setup, safety rules, peers and screen memory.

## macOS permissions

Grant **Accessibility** and **Screen Recording** to the terminal app that runs Claude Code, then restart it. `uvx opendesk check --open` opens the right System Settings pane.

## Privacy

Everything runs locally and nothing is uploaded. See [PRIVACY.md](https://github.com/vitalops/opendesk/blob/main/PRIVACY.md).

License: MIT.
