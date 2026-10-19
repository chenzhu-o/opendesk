import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Tool, ToolContext, ToolResult, checkPermission, type ToolAnnotations } from "./base.js";
import { getSandbox } from "../computer/sandbox.js";
import * as native from "../computer/macosNative.js";
import { backgroundDefault } from "./mouse.js";

const exec = promisify(execFile);

interface MacParams {
  action: string; app: string; title?: string; role?: string; menu?: string; menuItem?: string;
  text?: string; key?: string; modifiers?: string[]; ref?: string;
}

export class UITool extends Tool {
  name = "ui";
  title = "Accessibility UI Actions";
  annotations: ToolAnnotations = { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: true };
  description =
    "Interact with UI elements by their accessible label — the primary interaction tool. " +
    "No pixel coordinates needed. Use this before the mouse tool. " +
    "get_tree shows each element as [ref] [role] label; pass ref to click / type / " +
    "get_value / press_key on exactly that element (fastest and most reliable).";

  schema = {
    type: "object",
    required: ["action", "app"],
    properties: {
      action: { type: "string", enum: ["get_tree", "click", "click_menu", "type", "press_key", "get_value"] },
      app: { type: "string", description: "Target application name" },
      title: { type: "string", description: "Element label — for click/get_value" },
      ref: { type: "string", description: "Element ref from get_tree output, e.g. 'k3x9q0' — targets exactly that element" },
      role: { type: "string", description: "Element role — for click/get_value" },
      menu: { type: "string", description: "Menu bar label — for click_menu" },
      menuItem: { type: "string", description: "Menu item label — for click_menu" },
      text: { type: "string", description: "Text to type — for type" },
      key: { type: "string", description: "Key name — for press_key" },
      modifiers: { type: "array", items: { type: "string" }, description: "Modifier keys — for press_key" },
    },
  };

  async execute(ctx: ToolContext, params: Record<string, unknown>): Promise<ToolResult> {
    const { action, app, title, role, menu, menuItem, text, key, modifiers = [], ref } = params as {
      action: string; app: string; title?: string; role?: string; menu?: string;
      menuItem?: string; text?: string; key?: string; modifiers?: string[]; ref?: string;
    };

    await checkPermission(ctx, "ui", `${action} in ${app}`, `UI ${action} in ${app}`);

    const platform = process.platform;

    try {
      if (platform === "darwin") {
        const p: MacParams = { action, app, title, role, menu, menuItem, text, key, modifiers, ref };
        const fast = await this.executeNative(p, ctx);
        if (fast) return fast;
        return await this.executeMac(p, ctx);
      } else if (platform === "win32") {
        return await this.executeWin({ action, app, title, text, key }, ctx);
      } else {
        return await this.executeLinux({ action, app, title, text, key }, ctx);
      }
    } catch (e) {
      return this.err("UI error", `${e}`);
    }
  }

  /**
   * Fast path through the persistent opendesk-ax helper.  Returns null to
   * fall back to AppleScript (helper unavailable, or no match by title).
   * Ref-based actions have no AppleScript equivalent, so they fail here.
   */
  private async executeNative(p: MacParams, ctx: ToolContext): Promise<ToolResult | null> {
    const background = backgroundDefault();
    const needsNative = Boolean(p.ref) || (background && (p.action === "type" || p.action === "press_key"));
    if (!(await native.available())) {
      if (!needsNative) return null;
      return this.err("UI error",
        "Acting by ref / in background mode needs the native macOS helper (opendesk-ax): " +
        "install Xcode Command Line Tools (swiftc) and grant Accessibility permission.");
    }
    const sandbox = getSandbox(ctx.sessionId);
    const replay = { action: p.action, app: p.app, title: p.title, role: p.role, ref: p.ref, text: p.text,
      key: p.key, modifiers: p.modifiers ?? [] };
    switch (p.action) {
      case "get_tree": {
        let t: native.AXTree;
        try {
          t = await native.tree({ app: p.app });
        } catch {
          return null;
        }
        const out = native.formatTree(t) ||
          `No accessible elements in '${p.app}'. Fall back to screenshot + mouse.`;
        sandbox.recordAction("ui_get_tree", { app: p.app }, "ok");
        return this.ok("UI tree", out);
      }
      case "click": {
        let node: native.AXNode | null = null;
        if (p.ref) {
          try {
            await native.perform(p.ref, "AXPress");
            sandbox.recordAction("ui_click", { app: p.app, ref: p.ref }, "ok", undefined, replay);
            return this.ok("UI click", `Clicked [${p.ref}] in ${p.app} (via a11y).`);
          } catch {
            node = await native.get(p.ref).catch(() => null);
            if (!node) return this.err("UI error", `No element with ref '${p.ref}' in ${p.app}. Run get_tree again.`);
          }
        } else {
          if (!p.title && !p.role) return null;
          try {
            node = native.findNode(await native.tree({ app: p.app }), { title: p.title, role: p.role });
          } catch {
            return null;
          }
          if (!node) return null;
          try {
            await native.perform(node.ref, "AXPress");
            sandbox.recordAction("ui_click", { app: p.app, title: p.title }, "ok", undefined, replay);
            return this.ok("UI click", `Clicked [${node.ref}] [${node.role}] '${native.nodeName(node)}' in ${p.app} (via a11y).`);
          } catch {
            /* no AXPress: click the element's centre below */
          }
        }
        if (!node.frame) return this.err("UI error", `Element '${native.nodeName(node)}' has no bounds and no AXPress action.`);
        const x = Math.round(node.frame.x + node.frame.w / 2);
        const y = Math.round(node.frame.y + node.frame.h / 2);
        let how = "bounds-center";
        if (background) {
          await native.clickPid({ x, y, pid: node.pid, app: node.pid ? undefined : p.app });
          how = "background click";
        } else {
          const { mouse, straightTo, Point, Button } = await import("@nut-tree-fork/nut-js");
          await mouse.move(straightTo(new Point(x, y)));
          await mouse.click(Button.LEFT);
        }
        sandbox.recordAction("ui_click", { app: p.app, title: p.title, ref: p.ref }, "ok", undefined, replay);
        return this.ok("UI click", `Clicked [${node.ref}] '${native.nodeName(node)}' in ${p.app} (via ${how}).`);
      }
      case "type": {
        if (!needsNative) return null;
        await native.keyPid({ ref: p.ref, app: p.ref ? undefined : p.app, text: p.text ?? "" });
        sandbox.recordAction("ui_type", { app: p.app, ref: p.ref }, "ok", undefined, replay);
        const where = p.ref ? `[${p.ref}] in ${p.app}` : p.app;
        return this.ok("UI type", `Typed ${(p.text ?? "").length} chars into ${where} (no activation).`);
      }
      case "press_key": {
        if (!needsNative) return null;
        await native.keyPid({ ref: p.ref, app: p.ref ? undefined : p.app, key: p.key, modifiers: p.modifiers });
        sandbox.recordAction("ui_press_key", { app: p.app, key: p.key, ref: p.ref }, "ok", undefined, replay);
        return this.ok("UI press key", `Pressed ${p.key} in ${p.app} (no activation).`);
      }
      case "get_value": {
        let node: native.AXNode | null = null;
        if (p.ref) {
          node = await native.get(p.ref).catch(() => null);
          if (!node) return this.err("UI error", `No element with ref '${p.ref}' in ${p.app}. Run get_tree again.`);
        } else {
          if (!p.title && !p.role) return null;
          try {
            node = native.findNode(await native.tree({ app: p.app }), { title: p.title, role: p.role });
          } catch {
            return null;
          }
          if (!node) return null;
        }
        const value = node.value ?? native.nodeName(node);
        sandbox.recordAction("ui_get_value", { app: p.app, title: p.title, ref: p.ref }, value);
        return this.ok("UI get value", value || "(no value)");
      }
      default:
        if (p.ref) return this.err("UI error", `Action '${p.action}' does not take a ref.`);
        return null;
    }
  }

  private async executeMac(p: MacParams, ctx: ToolContext): Promise<ToolResult> {
    const sandbox = getSandbox(ctx.sessionId);

    switch (p.action) {
      case "get_tree": {
        const script = `
          tell application "System Events"
            tell process "${p.app}"
              set allElements to every UI element of window 1
              set result to {}
              repeat with el in allElements
                try
                  set result to result & {name of el & " [" & role of el & "]"}
                end try
              end repeat
              return result
            end tell
          end tell`;
        const { stdout } = await exec("osascript", ["-e", script]);
        sandbox.recordAction("ui_get_tree", { app: p.app }, "ok");
        return this.ok("UI tree", stdout.trim());
      }
      case "click": {
        const by = p.title ? `whose name is "${p.title}"` : p.role ? `whose role is "${p.role}"` : "";
        const script = `
          tell application "System Events"
            tell process "${p.app}"
              click (first UI element of window 1 ${by})
            end tell
          end tell`;
        await exec("osascript", ["-e", script]);
        sandbox.recordAction("ui_click", { app: p.app, title: p.title }, "ok", undefined,
          { action: "click", app: p.app, title: p.title, role: p.role });
        return this.ok("UI click", `Clicked '${p.title ?? p.role}' in ${p.app}.`);
      }
      case "click_menu": {
        const script = `
          tell application "System Events"
            tell process "${p.app}"
              click menu item "${p.menuItem}" of menu "${p.menu}" of menu bar 1
            end tell
          end tell`;
        await exec("osascript", ["-e", script]);
        sandbox.recordAction("ui_click_menu", { app: p.app, menu: p.menu, menuItem: p.menuItem }, "ok", undefined,
          { action: "click_menu", app: p.app, menu: p.menu, menuItem: p.menuItem });
        return this.ok("UI click menu", `Clicked ${p.app} > ${p.menu} > ${p.menuItem}.`);
      }
      case "type": {
        const script = `
          tell application "System Events"
            tell process "${p.app}"
              keystroke "${p.text?.replace(/"/g, '\\"')}"
            end tell
          end tell`;
        await exec("osascript", ["-e", script]);
        sandbox.recordAction("ui_type", { app: p.app }, "ok", undefined,
          { action: "type", app: p.app, text: p.text });
        return this.ok("UI type", `Typed text in ${p.app}.`);
      }
      case "press_key": {
        const modMap: Record<string, string> = { command: "command down", shift: "shift down", option: "option down", control: "control down" };
        const using = (p.modifiers ?? []).map((m) => modMap[m] ?? m).join(", ");
        const usingClause = using ? ` using {${using}}` : "";
        const script = `
          tell application "System Events"
            tell process "${p.app}"
              keystroke "${p.key}"${usingClause}
            end tell
          end tell`;
        await exec("osascript", ["-e", script]);
        sandbox.recordAction("ui_press_key", { app: p.app, key: p.key }, "ok", undefined,
          { action: "press_key", app: p.app, key: p.key, modifiers: p.modifiers ?? [] });
        return this.ok("UI press key", `Pressed ${p.key} in ${p.app}.`);
      }
      case "get_value": {
        const by = p.title ? `whose name is "${p.title}"` : p.role ? `whose role is "${p.role}"` : "";
        const script = `
          tell application "System Events"
            tell process "${p.app}"
              get value of (first UI element of window 1 ${by})
            end tell
          end tell`;
        const { stdout } = await exec("osascript", ["-e", script]);
        sandbox.recordAction("ui_get_value", { app: p.app, title: p.title }, stdout.trim());
        return this.ok("UI get value", stdout.trim());
      }
      default:
        return this.err("UI error", `Unknown action: ${p.action}`);
    }
  }

  private async executeWin(
    p: { action: string; app: string; title?: string; text?: string; key?: string },
    ctx: ToolContext,
  ): Promise<ToolResult> {
    const script = (ps: string) => exec("powershell", ["-Command", ps]);
    const sandbox = getSandbox(ctx.sessionId);

    switch (p.action) {
      case "click": {
        const ps = `
          Add-Type -AssemblyName UIAutomationClient,UIAutomationTypes
          $desktop = [System.Windows.Automation.AutomationElement]::RootElement
          $cond = New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::NameProperty, "${p.title}")
          $el = $desktop.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cond)
          $invokePattern = $el.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
          $invokePattern.Invoke()`;
        await script(ps);
        sandbox.recordAction("ui_click", { app: p.app, title: p.title }, "ok", undefined,
          { action: "click", app: p.app, title: p.title });
        return this.ok("UI click", `Clicked '${p.title}' in ${p.app}.`);
      }
      case "type": {
        const ps = `Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('${p.text}')`;
        await script(ps);
        sandbox.recordAction("ui_type", { app: p.app }, "ok", undefined,
          { action: "type", app: p.app, text: p.text });
        return this.ok("UI type", `Typed text in ${p.app}.`);
      }
      default:
        return this.err("UI error", `Action '${p.action}' not yet supported on Windows.`);
    }
  }

  private async executeLinux(
    p: { action: string; app: string; title?: string; text?: string; key?: string },
    ctx: ToolContext,
  ): Promise<ToolResult> {
    const sandbox = getSandbox(ctx.sessionId);

    switch (p.action) {
      case "click": {
        await exec("xdotool", ["search", "--name", p.title ?? p.app, "click", "1"]);
        sandbox.recordAction("ui_click", { app: p.app, title: p.title }, "ok", undefined,
          { action: "click", app: p.app, title: p.title });
        return this.ok("UI click", `Clicked '${p.title}' in ${p.app}.`);
      }
      case "type": {
        await exec("xdotool", ["type", "--clearmodifiers", p.text ?? ""]);
        sandbox.recordAction("ui_type", { app: p.app }, "ok", undefined,
          { action: "type", app: p.app, text: p.text });
        return this.ok("UI type", `Typed text in ${p.app}.`);
      }
      default:
        return this.err("UI error", `Action '${p.action}' not yet supported on Linux.`);
    }
  }
}
