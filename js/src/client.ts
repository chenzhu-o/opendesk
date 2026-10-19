/**
 * OpenDeskClient — programmatic interface to all native opendesk tools.
 * No Python required. All desktop automation runs in Node.js.
 *
 * Usage:
 *
 *   import { OpenDeskClient } from "@vitalops/opendesk-sdk";
 *
 *   const client = new OpenDeskClient();
 *   const result = await client.screenshot({ marks: true });
 *   console.log(result.output);
 */

import { createRegistry, ToolRegistry } from "./registry.js";
import { allowAllContext, ToolContext, ToolResult, PermissionHandler } from "./tools/base.js";

export interface OpenDeskClientOptions {
  sessionId?: string;
  permissionHandler?: PermissionHandler;
  registry?: ToolRegistry;
}

export class OpenDeskClient {
  private registry: ToolRegistry;
  private ctx: ToolContext;

  constructor(options: OpenDeskClientOptions = {}) {
    this.registry = options.registry ?? createRegistry();
    this.ctx = {
      sessionId: options.sessionId ?? "default",
      permissionHandler: options.permissionHandler,
    };
  }

  private call(name: string, params: Record<string, unknown> = {}): Promise<ToolResult> {
    return this.registry.get(name).execute(this.ctx, params);
  }

  screenshot(params: { marks?: boolean; savePath?: string; region?: number[] } = {}): Promise<ToolResult> {
    return this.call("screenshot", params as Record<string, unknown>);
  }

  mouse(params: { action: string; x?: number; y?: number; endX?: number; endY?: number; direction?: string; amount?: number }): Promise<ToolResult> {
    return this.call("mouse", params as Record<string, unknown>);
  }

  keyboard(params: { action: string; text?: string; key?: string; keys?: string[]; holdDuration?: number }): Promise<ToolResult> {
    return this.call("keyboard", params as Record<string, unknown>);
  }

  app(params: { action: string; name?: string }): Promise<ToolResult> {
    return this.call("app", params as Record<string, unknown>);
  }

  ui(params: { action: string; app: string; title?: string; role?: string; menu?: string; menuItem?: string; text?: string; key?: string; modifiers?: string[] }): Promise<ToolResult> {
    return this.call("ui", params as Record<string, unknown>);
  }

  clipboard(params: { action: string; text?: string }): Promise<ToolResult> {
    return this.call("clipboard", params as Record<string, unknown>);
  }

  ocr(params: { region?: number[] } = {}): Promise<ToolResult> {
    return this.call("ocr", params as Record<string, unknown>);
  }

  audit(params: { format?: "summary" | "full"; sessionId?: string } = {}): Promise<ToolResult> {
    return this.call("audit", params as Record<string, unknown>);
  }

  /** Screen memory (read-only) — search / show / timeline / status. */
  memory(params: {
    action: "search" | "show" | "timeline" | "status";
    query?: string; since?: string; until?: string; app?: string; limit?: number;
    id?: number; includeImage?: boolean;
  }): Promise<ToolResult> {
    return this.call("memory", params as Record<string, unknown>);
  }

  /** Run shell / AppleScript / JXA / PowerShell. Prefer this over GUI clicks for non-visual work. */
  script(params: {
    action: "shell" | "applescript" | "jxa" | "powershell";
    code: string; timeout?: number; cwd?: string;
  }): Promise<ToolResult> {
    return this.call("script", params as Record<string, unknown>);
  }

  /** Drive Chrome / Chromium / Edge / Brave over the DevTools Protocol (local machine only). */
  browser(params: {
    action: "launch" | "attach" | "tabs" | "open" | "navigate" | "read" | "click"
      | "type" | "wait_for" | "screenshot" | "eval" | "close_tab";
    url?: string; port?: number; browser?: string; headless?: boolean; profileDir?: string;
    tab?: number | string; ref?: string; selector?: string; text?: string; expression?: string;
    clear?: boolean; submit?: boolean; fullPage?: boolean; maxChars?: number; timeout?: number;
  }): Promise<ToolResult> {
    return this.call("browser", params as Record<string, unknown>);
  }

  /** Screen memory controls — pause / resume / deny / config / delete. */
  memoryManage(params: {
    action: "pause" | "resume" | "deny" | "config" | "delete";
    duration?: string; denyAdd?: string; denyRemove?: string;
    intervalSeconds?: number; storageCapMb?: number; retentionDays?: number;
    since?: string; until?: string; app?: string; confirm?: boolean;
  }): Promise<ToolResult> {
    return this.call("memory_manage", params as Record<string, unknown>);
  }

  listTools(): string[] {
    return this.registry.all().map((t) => t.name);
  }
}
