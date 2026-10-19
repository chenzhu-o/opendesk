/**
 * Base abstractions for all opendesk JS tools — mirrors Python's tools/base.py.
 */

export interface Attachment {
  filename: string;
  mediaType: string;
  content: Buffer;
}

export interface ToolResult {
  title: string;
  output: string;
  error: boolean;
  attachments: Attachment[];
  metadata: Record<string, unknown>;
}

export type PermissionHandler = (tool: string, argument: string, description: string) => Promise<void>;

export class PermissionDeniedError extends Error {}

export interface ToolContext {
  sessionId: string;
  permissionHandler?: PermissionHandler;
}

export function allowAllContext(sessionId = "default"): ToolContext {
  return { sessionId };
}

export async function checkPermission(
  ctx: ToolContext,
  tool: string,
  argument: string,
  description: string,
): Promise<void> {
  if (ctx.permissionHandler) {
    await ctx.permissionHandler(tool, argument, description);
  }
}

/** Appended to tool errors that look like a missing macOS permission. */
export const PERMISSION_FIX =
  "Fix: on macOS, grant Accessibility and Screen Recording to the app that runs OpenDesk " +
  "(Claude Desktop, your terminal for Claude Code, Cursor or VS Code) under System Settings → " +
  "Privacy & Security, then restart that app. With the Python package, `opendesk check --open` " +
  "jumps to the right pane.";

const PERMISSION_MARKERS = [
  "accessibility permission", "accessibility access", "screen recording", "assistive access",
  "not allowed assistive", "-1719", "-25211", "privacy & security", "not authorized to send apple events",
];

/** `PERMISSION_FIX` when `message` looks like a macOS permission failure, else "". */
export function permissionHint(message: string): string {
  const low = (message ?? "").toLowerCase();
  if (low.includes("opendesk check")) return "";
  return PERMISSION_MARKERS.some((m) => low.includes(m)) ? PERMISSION_FIX : "";
}

/** MCP `ToolAnnotations` behaviour hints. */
export interface ToolAnnotations {
  readOnlyHint?: boolean;
  destructiveHint?: boolean;
  idempotentHint?: boolean;
  openWorldHint?: boolean;
}

const ANNOTATION_KEYS = ["readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"] as const;

/** Common presets — copy with `{ ...READ_ONLY }`. */
export const READ_ONLY: ToolAnnotations = { readOnlyHint: true, openWorldHint: false };
export const DESKTOP_ACTION: ToolAnnotations = {
  readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: true,
};

/** `tool.title`, or a title-cased `tool.name` ("memory_manage" → "Memory Manage"). */
export function displayTitle(tool: { name: string; title?: string }): string {
  if (tool.title) return tool.title;
  return tool.name.split("_").map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w)).join(" ");
}

/**
 * Normalised annotations: only known MCP hint keys, `readOnlyHint` always
 * present, `destructiveHint` defaulting to true for non-read-only tools and
 * dropped for read-only ones (the spec ignores it there).
 */
export function toolAnnotations(tool: { annotations?: ToolAnnotations }): ToolAnnotations {
  const src = (tool.annotations ?? {}) as Record<string, unknown>;
  const out: Record<string, boolean> = {};
  for (const k of ANNOTATION_KEYS) if (typeof src[k] === "boolean") out[k] = src[k] as boolean;
  if (out.readOnlyHint === undefined) out.readOnlyHint = false;
  if (out.readOnlyHint) {
    delete out.destructiveHint;
    delete out.idempotentHint;
  } else if (out.destructiveHint === undefined) {
    out.destructiveHint = true;
  }
  return out as ToolAnnotations;
}

export abstract class Tool {
  abstract name: string;
  abstract description: string;
  abstract schema: Record<string, unknown>;
  /** Human-readable display name (MCP `title`). Defaults to a title-cased `name`. */
  title?: string;
  /** MCP behaviour hints (readOnlyHint / destructiveHint / idempotentHint / openWorldHint). */
  annotations?: ToolAnnotations;
  abstract execute(ctx: ToolContext, params: Record<string, unknown>): Promise<ToolResult>;

  ok(title: string, output: string, attachments: Attachment[] = [], metadata: Record<string, unknown> = {}): ToolResult {
    return { title, output, error: false, attachments, metadata };
  }

  err(title: string, output: string): ToolResult {
    return { title, output, error: true, attachments: [], metadata: {} };
  }
}
