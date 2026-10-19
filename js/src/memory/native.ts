/**
 * Guarded access to the optional macOS native helpers in
 * `src/computer/macosNative.ts` (accessibility text, fast frontmost probe).
 *
 * The module may not exist (other platforms, older builds), so it is loaded
 * with a computed specifier — TypeScript does not resolve it at compile time
 * and a failed import simply yields null.
 *
 * Contract used here:
 *   windowText(pid?: number, maxChars?: number): string | null | Promise<string | null>
 *   frontmost(): { app, title, pid } | Promise<…>   (shape coerced defensively)
 */

export interface MacosNative {
  windowText?: (pid?: number, maxChars?: number) => unknown;
  frontmost?: () => unknown;
}

let cached: Promise<MacosNative | null> | null = null;

export function loadMacosNative(): Promise<MacosNative | null> {
  if (process.platform !== "darwin") return Promise.resolve(null);
  if (!cached) {
    const spec = ["..", "computer", "macosNative.js"].join("/");
    cached = import(/* @vite-ignore */ spec)
      .then((m: unknown) => {
        const usable = (x: unknown): x is MacosNative =>
          !!x && (typeof (x as MacosNative).windowText === "function" || typeof (x as MacosNative).frontmost === "function");
        if (usable(m)) return m;
        const dflt = (m as { default?: unknown }).default;
        return usable(dflt) ? dflt : null;
      })
      .catch(() => null);
  }
  return cached;
}

/** Reset the cached import (tests). */
export function resetMacosNativeCache(): void {
  cached = null;
}
