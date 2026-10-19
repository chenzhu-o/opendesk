# AGENTS.md

## Layout

- `python/` — Python SDK + MCP server (`opendesk`, PyPI). Tools in `python/opendesk/tools/`, MCP in `python/opendesk/integrations/mcp.py` (+ `mcp_http.py`).
- `js/` — JS/TS SDK + MCP server (`@vitalops/opendesk-sdk`, npm). Tools in `js/src/tools/`, MCP in `js/src/mcp.ts`.
- `docs/` — Sphinx/MyST docs (`docs/conf.py`, toctrees in `docs/index.md`); `mkdocs.yml` mirrors the nav.
- `plugins/opendesk/` — Claude Code plugin (MCP config + skill); `.claude-plugin/marketplace.json` — marketplace.
- `server.json` — MCP Registry entry. `manifest.json` + `scripts/build_mcpb.sh` — Claude Desktop `.mcpb` bundle.

## Python

```bash
cd python
pip install -e '.[all]'      # don't add [linux] off Linux: pyatspi isn't on PyPI
python -m pytest -q
```

## JavaScript

```bash
cd js
npm install
npx tsc --noEmit
npx vitest run
```

## Run the MCP server from source

```bash
uvx --from ./python opendesk mcp                         # stdio
uvx --from ./python opendesk mcp --transport http --port 8765
uvx --refresh --from ./python opendesk mcp               # after source changes
```

## Validate packaging

```bash
claude plugin validate --strict .
claude plugin validate --strict plugins/opendesk
mcp-publisher validate server.json
scripts/build_mcpb.sh                                    # → dist/opendesk-<version>.mcpb
```

## Conventions

- Every tool sets `title` and `annotations` (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`).
- Report failures as `ToolResult(error=True, ...)` (JS: `this.err(...)`) instead of raising.
- Keep Python and JS tools in sync (names, actions, parameters).
- Version bumps touch: `python/pyproject.toml`, `python/opendesk/__init__.py`, `js/package.json`, `js/src/mcp.ts` (`SERVER_VERSION`), `manifest.json`, `server.json`, `plugins/opendesk/.claude-plugin/plugin.json`, `plugins/opendesk/.mcp.json`.
- Commits: no Claude/AI co-author or attribution trailers.
