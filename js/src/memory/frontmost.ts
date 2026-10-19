/**
 * Frontmost application name, window title, and pid, per platform.  Used by
 * the screen-memory recorder for event triggering, deny-list matching, and
 * indexing.  Cheap enough to poll every second: on macOS it prefers the
 * native helper (when built) and otherwise makes a single `osascript` call.
 */

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { loadMacosNative } from "./native.js";

const exec = promisify(execFile);

export interface Frontmost {
  app: string;
  title: string;
  pid?: number | null;
}

async function run(cmd: string, args: string[], timeout = 5000): Promise<string> {
  try {
    const { stdout } = await exec(cmd, args, { timeout });
    return stdout.trim();
  } catch {
    return "";
  }
}

function intOrNull(v: unknown): number | null {
  if (v === null || v === undefined || String(v).trim() === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? Math.trunc(n) : null;
}

/** Accept whatever shape a probe returns: object, tuple, or null. */
export function coerceFrontmost(obj: unknown): Frontmost {
  if (!obj) return { app: "", title: "", pid: null };
  if (Array.isArray(obj)) {
    return { app: String(obj[0] ?? "").trim(), title: String(obj[1] ?? "").trim(), pid: intOrNull(obj[2]) };
  }
  const o = obj as Record<string, unknown>;
  const app = o["app"] ?? o["appName"] ?? o["app_name"] ?? o["name"] ?? o["application"] ?? "";
  const title = o["title"] ?? o["windowTitle"] ?? o["window_title"] ?? "";
  const pid = o["pid"] ?? o["processId"] ?? o["process_id"];
  return { app: String(app ?? "").trim(), title: String(title ?? "").trim(), pid: intOrNull(pid) };
}

const OSA_FRONT = `tell application "System Events"
set p to first process whose frontmost is true
set n to name of p
set i to unix id of p
set t to ""
try
set t to name of front window of p
end try
return n & linefeed & i & linefeed & t
end tell`;

export async function frontmost(): Promise<Frontmost> {
  if (process.platform === "darwin") {
    const native = await loadMacosNative();
    if (native?.frontmost) {
      try {
        const f = coerceFrontmost(await native.frontmost());
        if (f.app || f.title) return { ...f, title: f.title !== f.app ? f.title : "" };
      } catch { /* fall through */ }
    }
    const out = await run("osascript", ["-e", OSA_FRONT]);
    if (!out) return { app: "", title: "" };
    const [app = "", pid = "", ...rest] = out.split("\n");
    const title = rest.join(" ").trim();
    return { app: app.trim(), title: title && title !== app.trim() ? title : "", pid: intOrNull(pid) };
  }
  if (process.platform === "linux") {
    const title = await run("xdotool", ["getactivewindow", "getwindowname"]);
    const app = await run("xdotool", ["getactivewindow", "getwindowclassname"]);
    const pid = await run("xdotool", ["getactivewindow", "getwindowpid"]);
    return { app: app || title, title: app && title !== app ? title : "", pid: intOrNull(pid) };
  }
  if (process.platform === "win32") {
    const script = `
Add-Type @"
using System; using System.Runtime.InteropServices; using System.Text;
public class W { [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
[DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
[DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid); }
"@
$h = [W]::GetForegroundWindow(); $sb = New-Object System.Text.StringBuilder 512
[void][W]::GetWindowText($h, $sb, 512); $pid2 = 0; [void][W]::GetWindowThreadProcessId($h, [ref]$pid2)
$p = Get-Process -Id $pid2 -ErrorAction SilentlyContinue
Write-Output ($p.ProcessName); Write-Output ($pid2); Write-Output ($sb.ToString())`;
    const out = await run("powershell", ["-NonInteractive", "-Command", script]);
    const [app = "", pid = "", ...rest] = out.split(/\r?\n/);
    const title = rest.join(" ").trim();
    return { app: app.trim(), title: title !== app.trim() ? title : "", pid: intOrNull(pid) };
  }
  return { app: "", title: "" };
}
