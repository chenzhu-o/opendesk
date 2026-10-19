import { Tool, ToolContext, ToolResult, checkPermission, type ToolAnnotations } from "./base.js";
import { getSandbox } from "../computer/sandbox.js";
import * as native from "../computer/macosNative.js";

/** True when OPENDESK_BACKGROUND asks for cursor-free operation. */
export function backgroundDefault(): boolean {
  return ["1", "true", "yes", "on"].includes((process.env["OPENDESK_BACKGROUND"] ?? "").trim().toLowerCase());
}

const CLICK_ACTIONS: Record<string, { button: "left" | "right"; count: number }> = {
  click: { button: "left", count: 1 },
  double_click: { button: "left", count: 2 },
  right_click: { button: "right", count: 1 },
};

/**
 * Click without moving the user's cursor (macOS): press the accessible element
 * under the point via AXPress, else post the click straight to its process.
 */
export async function backgroundClick(x: number, y: number, button: "left" | "right", count: number): Promise<string> {
  if (process.platform !== "darwin" || !(await native.available())) {
    throw new Error(
      "Background clicks need macOS with the native opendesk-ax helper (swiftc + Accessibility permission). " +
      "Use the ui tool with a ref, or unset OPENDESK_BACKGROUND / background.",
    );
  }
  const hit = await native.elementAt(x, y);
  if (hit && hit.pressable && button === "left" && count === 1) {
    try {
      await native.perform(hit.pressable, "AXPress");
      return `Pressed '${native.nodeName(hit) || hit.role}' at (${x}, ${y}) via accessibility (cursor not moved).`;
    } catch {
      /* fall through to a posted click */
    }
  }
  const res = await native.clickPid({ x, y, pid: hit?.pid, button, count });
  return `Posted ${button} click x${count} at (${x}, ${y}) to ${hit?.app ?? `pid ${res["pid"]}`} (cursor not moved).`;
}

export class MouseTool extends Tool {
  name = "mouse";
  title = "Mouse Control";
  annotations: ToolAnnotations = { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: true };
  description =
    "Control the mouse — click, double-click, right-click, scroll, drag, move. " +
    "Always prefer the ui tool first. Use mouse only for elements with no accessible label. " +
    "background=true clicks without moving the user's cursor (macOS).";

  schema = {
    type: "object",
    required: ["action"],
    properties: {
      action: {
        type: "string",
        enum: ["move", "click", "double_click", "right_click", "scroll", "drag", "cursor_position"],
      },
      x: { type: "integer" },
      y: { type: "integer" },
      endX: { type: "integer" },
      endY: { type: "integer" },
      direction: { type: "string", enum: ["up", "down", "left", "right"] },
      amount: { type: "integer", default: 3 },
      background: {
        type: "boolean",
        description: "Click without moving the cursor or activating the app (macOS). Default true when OPENDESK_BACKGROUND=1.",
      },
    },
  };

  async execute(ctx: ToolContext, params: Record<string, unknown>): Promise<ToolResult> {
    const { action, x, y, endX, endY, direction, amount = 3 } = params as {
      action: string; x?: number; y?: number; endX?: number; endY?: number;
      direction?: string; amount?: number;
    };
    const envBackground = backgroundDefault();
    const background = typeof params["background"] === "boolean" ? (params["background"] as boolean) : envBackground;

    if ((envBackground || background) && !(action in CLICK_ACTIONS) && action !== "cursor_position") {
      const why = envBackground
        ? "OPENDESK_BACKGROUND=1 is set, so actions that move the user's cursor are disabled"
        : "background=true only applies to click actions";
      return this.err("Mouse action refused",
        `mouse ${action}: ${why}. Use the ui tool instead — ui get_tree, then ui click/type/press_key ` +
        "with ref=<ref> — or mouse click with background=true.");
    }
    const useBackground = background || envBackground;

    await checkPermission(ctx, "mouse", `${action} at (${x},${y})${useBackground ? " (background)" : ""}`, `Mouse ${action}`);

    const sandbox = getSandbox(ctx.sessionId);
    if (x !== undefined && y !== undefined && !sandbox.isCoordinateAllowed(x, y)) {
      return this.err("Mouse denied", "Coordinates outside permitted screen region.");
    }

    if (useBackground && action in CLICK_ACTIONS) {
      if (x === undefined || y === undefined) return this.err("Mouse error", "x and y are required.");
      try {
        const { button, count } = CLICK_ACTIONS[action]!;
        const msg = await backgroundClick(x, y, button, count);
        sandbox.recordAction(`mouse_${action}`, { ...params, background: true }, "ok");
        return this.ok(`Mouse ${action}`, msg);
      } catch (e) {
        return this.err("Mouse error", `${e instanceof Error ? e.message : e}`);
      }
    }

    try {
      const { mouse, straightTo, Point, Button } = await import("@nut-tree-fork/nut-js");

      switch (action) {
        case "move":
          await mouse.move(straightTo(new Point(x!, y!)));
          break;
        case "click":
          await mouse.move(straightTo(new Point(x!, y!)));
          await mouse.click(Button.LEFT);
          break;
        case "double_click":
          await mouse.move(straightTo(new Point(x!, y!)));
          await mouse.doubleClick(Button.LEFT);
          break;
        case "right_click":
          await mouse.move(straightTo(new Point(x!, y!)));
          await mouse.click(Button.RIGHT);
          break;
        case "scroll":
          if (direction === "down") await mouse.scrollDown(amount as number);
          else if (direction === "up") await mouse.scrollUp(amount as number);
          else if (direction === "left") await mouse.scrollLeft(amount as number);
          else await mouse.scrollRight(amount as number);
          break;
        case "drag":
          await mouse.drag(straightTo(new Point(endX!, endY!)));
          break;
        case "cursor_position": {
          const pos = await mouse.getPosition();
          sandbox.recordAction("cursor_position", params, `(${pos.x},${pos.y})`);
          return this.ok("Cursor position", `x=${pos.x}, y=${pos.y}`);
        }
        default:
          return this.err("Mouse error", `Unknown action: ${action}`);
      }

      sandbox.recordAction(`mouse_${action}`, params, "ok");
      return this.ok(`Mouse ${action}`, `${action} at (${x ?? ""},${y ?? ""}) done.`);
    } catch (e) {
      return this.err("Mouse error", `${e}`);
    }
  }
}
