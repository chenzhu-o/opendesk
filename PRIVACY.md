# Privacy Policy

**OpenDesk** is a local-first desktop automation framework. Here's how it handles your data.

## Data Collection

OpenDesk does **not** collect, transmit, or store any user data on external servers. All processing happens locally on your machine.

## What stays on your machine

- **Screenshots** are captured in-memory and returned to the MCP client. They are never sent to any external service by OpenDesk itself.
- **Audit logs** are stored locally in `~/.opendesk/` and never leave your machine.
- **Recorded workflows** (learn/replay) are saved as local JSON files in `~/.opendesk/procedures/`.
- **Clipboard reads** are processed in-memory and not persisted.
- **Screen memory** (`opendesk memory start`, opt-in) stores OCR text and small thumbnails of your screen in `~/.opendesk/memory` (mode 0600). OCR runs locally. Nothing is uploaded. You control a per-app deny list (password managers are excluded by default), a pause hotkey, a storage cap with rolling deletion, and a retention window; `opendesk memory clear` deletes everything.

## Remote Computer Use

When using OpenDesk's remote control feature (`opendesk serve` / `opendesk pair`), data is transmitted between paired machines over an encrypted WebSocket connection:

- All frames are encrypted with ChaCha20-Poly1305 (AEAD).
- Pairing uses a 6-digit code-authenticated X25519 key exchange.
- Only explicitly paired machines can communicate — no data is sent to third parties.
- No relay servers are involved; connections are direct (LAN).

## MCP over HTTP

The MCP server talks to your client over stdio by default. The optional streamable HTTP transport (`opendesk mcp --transport http`) binds to `127.0.0.1` unless you choose another address, requires a bearer token (stored locally in `~/.opendesk/mcp-token`, mode 0600), and refuses non-loopback addresses without one.

## Third-Party Services

OpenDesk does **not** integrate with any analytics, telemetry, or tracking services. The MCP client you use (Claude Desktop, Claude Code, Cursor, etc.) has its own privacy policy — OpenDesk does not control what those clients do with tool results.

## Data Retention

- Audit logs are retained locally until you delete them.
- Screen memory frames are deleted automatically after the configured retention period (default 30 days) or when the storage cap (default 2 GB) is exceeded, oldest first.
- No data is retained on any remote server.

## Contact

For privacy questions, open an issue at [github.com/vitalops/opendesk/issues](https://github.com/vitalops/opendesk/issues).
