/**
 * opendesk screen memory — a local, searchable desktop history.
 *
 * A low-frequency background loop captures the screen, runs OCR locally,
 * and stores the text plus a small thumbnail in `~/.opendesk/memory`.  The
 * `memory` tool lets an agent search that history.  Nothing leaves the machine.
 */

export {
  MemoryConfig,
  type MemoryConfigData,
  type PauseState,
  type DaemonState,
  memoryDir,
  loadConfig,
  saveConfig,
  getPause,
  setPause,
  clearPause,
  isPaused,
  describePause,
  parseDuration,
  daemonAlive,
  readDaemonState,
} from "./config.js";
export {
  MemoryStore, TEXT_SOURCES, type Frame, type StoreStats, type QueryOpts, type ReindexCounts, type StoreOptions,
  type TextSource, queryTerms, makeSnippet, frameWhen,
} from "./store.js";
export {
  exportMemory, defaultExportDir, EXPORT_FORMATS, type ExportFormat, type ExportOptions, type ExportResult,
} from "./export.js";
export { parseWhen, parseRange, fmtTs, fmtDay, fmtRange, type TimeRange } from "./timeparse.js";
export {
  ScreenMemoryRecorder, startDaemon, titleKey, type RecorderOptions, type CaptureFn, type OcrFn, type FrontmostFn, type WindowTextFn,
} from "./recorder.js";
export { redact, redactText, luhnOk } from "./redact.js";
export {
  MemoryCipher, loadOrCreateKey, readKeyInfo, resolveKeyProvider, setKeyProviderResolver,
  KeychainProvider, SecretToolProvider, FileKeyProvider, StaticKeyProvider, type KeyProvider, type KeyInfo,
} from "./crypto.js";
export { makeThumbnail, frameSignature, signatureDelta, type Thumbnail } from "./image.js";
export { createOcrEngine, availableBackend, ensureVisionHelper, type OcrEngine, type OcrPreference } from "./ocr.js";
export { frontmost, type Frontmost } from "./frontmost.js";
export { installMemoryService, uninstallMemoryService, renderLaunchdPlist, renderSystemdUnit, renderSchtasksCommand } from "./service.js";
export { parseCombo, registerHotkey, type HotkeyHandle } from "./hotkey.js";
