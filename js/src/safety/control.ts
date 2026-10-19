/**
 * Kill switch + take-over — mirrors python/opendesk/safety/control.py.
 * `<home>/control.json` = { mode: "running" | "stopped" | "takeover", since, reason }.
 */

import path from "path";
import { readJson, resolveHome, writeJson } from "./home.js";

export type ControlMode = "running" | "stopped" | "takeover";
export const MODES: ControlMode[] = ["running", "stopped", "takeover"];

export interface ControlState {
  mode: ControlMode;
  since: number | null;
  reason: string;
}

export function controlPath(home?: string): string {
  return path.join(resolveHome(home), "control.json");
}

export function readControl(home?: string): ControlState {
  const data = readJson(controlPath(home));
  if (!data) return { mode: "running", since: null, reason: "" };
  let mode = String(data["mode"] ?? "running").toLowerCase() as ControlMode;
  if (!MODES.includes(mode)) mode = "stopped"; // fail closed
  const since = typeof data["since"] === "number" ? (data["since"] as number) : null;
  return { mode, since, reason: String(data["reason"] ?? "") };
}

export function setMode(mode: string, home?: string, reason = ""): ControlState {
  const m = mode.toLowerCase() as ControlMode;
  if (!MODES.includes(m)) throw new Error(`mode must be one of ${MODES.join(", ")}`);
  const state: ControlState = { mode: m, since: Date.now() / 1000, reason };
  writeJson(controlPath(home), state);
  return state;
}

function sinceText(s: ControlState): string {
  if (!s.since) return "";
  return " since " + new Date(s.since * 1000).toISOString().replace("T", " ").slice(0, 19);
}

export function refusalMessage(state: ControlState): string | null {
  if (state.mode === "running") return null;
  const why = state.reason ? ` (${state.reason})` : "";
  if (state.mode === "takeover") {
    return `User has taken control of this machine${sinceText(state)}${why}. Do not retry. ` +
      "Ask them to run `opendesk handback` when they want you to continue.";
  }
  return `OpenDesk is stopped — the kill switch is engaged${sinceText(state)}${why}. ` +
    "All tool calls are refused. Ask the user to run `opendesk resume` when it is safe to continue.";
}

export function checkControl(home?: string): string | null {
  return refusalMessage(readControl(home));
}

/**
 * Arm the global kill hotkey (`kill_hotkey` in policy.json, pynput-style
 * combo) — pressing it engages the kill switch.  Uses the optional
 * `uiohook-napi` package; resolves to null when not configured / unavailable.
 */
export async function startKillHotkey(home?: string, hotkey?: string | null): Promise<{ stop(): void } | null> {
  let combo = hotkey;
  if (combo === undefined) {
    const { loadPolicy } = await import("./policy.js");
    combo = loadPolicy(home).kill_hotkey ?? null;
  }
  if (!combo) return null;
  const resolved = resolveHome(home);
  try {
    const { registerHotkey } = await import("../memory/hotkey.js");
    return await registerHotkey(combo, () => {
      try { setMode("stopped", resolved, `kill hotkey ${combo}`); } catch { /* best effort */ }
    });
  } catch {
    return null;
  }
}
