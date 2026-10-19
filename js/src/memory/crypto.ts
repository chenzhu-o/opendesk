/**
 * Encryption at rest for screen memory — mirrors Python's
 * opendesk/memory/crypto.py and shares its key.
 *
 * One random 32-byte memory key protects everything.  Where it lives:
 *   - macOS: login Keychain, generic password service=opendesk,
 *     account=memory-key (hex), via /usr/bin/security — the same item the
 *     Python SDK uses.
 *   - Linux: the Secret Service via `secret-tool` (attributes
 *     service/username, compatible with Python's `keyring`).
 *   - otherwise: a 0600 file `<home>/memory/key` (weaker: readable by
 *     anything that can read your home directory, backups included).
 *
 * `<home>/memory/keyinfo.json` records which store holds the key (no secret
 * material) so both SDKs agree.  OPENDESK_MEMORY_KEYSTORE=keychain|keyring|file
 * forces a store.  In auto mode a custom home keeps its key in the file
 * store; only the default ~/.opendesk uses the OS store.
 *
 * Sub-keys: enc = HMAC(key, "opendesk-memory:enc") for AES-256-GCM,
 * index = HMAC(key, "opendesk-memory:index") for blind-index tokens.
 * Blob: "ODE1" | nonce(12) | ciphertext | tag(16).   Text: "v1:" + base64(nonce | ct | tag).
 */

import { execFileSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { DEFAULT_HOME, atomicWrite, memoryDir, resolveHomeDir } from "./config.js";

export const KEY_SERVICE = "opendesk";
export const KEY_ACCOUNT = "memory-key";
export const ENV_KEYSTORE = "OPENDESK_MEMORY_KEYSTORE";
export const BLOB_MAGIC = Buffer.from("ODE1");
export const TEXT_PREFIX = "v1:";
const NONCE_LEN = 12;
const TAG_LEN = 16;
const THUMB_AAD = Buffer.from("opendesk-memory-thumb");
const TEXT_AAD = Buffer.from("opendesk-memory-text");
const TOKEN_HEX_LEN = 20;

export interface KeyProvider {
  readonly name: string;
  load(): Buffer | null;
  save(key: Buffer): void;
}

function decodeKey(text: string | null | undefined): Buffer | null {
  const t = (text ?? "").trim();
  if (!/^[0-9a-fA-F]{64}$/.test(t)) return null;
  return Buffer.from(t, "hex");
}

function which(cmd: string): boolean {
  const dirs = (process.env["PATH"] ?? "").split(path.delimiter);
  return dirs.some((d) => d && fs.existsSync(path.join(d, cmd)));
}

/** macOS login Keychain via /usr/bin/security. */
export class KeychainProvider implements KeyProvider {
  readonly name = "keychain";
  constructor(readonly service = KEY_SERVICE, readonly account = KEY_ACCOUNT) {}

  static available(): boolean {
    return process.platform === "darwin" && (fs.existsSync("/usr/bin/security") || which("security"));
  }

  load(): Buffer | null {
    try {
      const out = execFileSync("security", ["find-generic-password", "-s", this.service, "-a", this.account, "-w"], {
        encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 15000,
      });
      return decodeKey(out);
    } catch {
      return null;
    }
  }

  save(key: Buffer): void {
    // `security -i` reads the command from stdin: the secret never shows up in `ps`.
    const cmd = `add-generic-password -U -s "${this.service}" -a "${this.account}" -l "opendesk screen memory" -w "${key.toString("hex")}"\n`;
    try {
      execFileSync("security", ["-i"], { input: cmd, stdio: ["pipe", "ignore", "pipe"], timeout: 15000 });
    } catch { /* verified below */ }
    const back = this.load();
    if (!back || !back.equals(key)) throw new Error("could not store the memory key in the Keychain");
  }
}

/** Linux Secret Service via `secret-tool` (same attributes Python's keyring uses). */
export class SecretToolProvider implements KeyProvider {
  readonly name = "keyring";
  constructor(readonly service = KEY_SERVICE, readonly account = KEY_ACCOUNT) {}

  static available(): boolean {
    return process.platform === "linux" && which("secret-tool");
  }

  load(): Buffer | null {
    try {
      const out = execFileSync("secret-tool", ["lookup", "service", this.service, "username", this.account], {
        encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 15000,
      });
      return decodeKey(out);
    } catch {
      return null;
    }
  }

  save(key: Buffer): void {
    execFileSync("secret-tool", [
      "store", "--label=opendesk screen memory", "service", this.service, "username", this.account,
    ], { input: key.toString("hex"), stdio: ["pipe", "ignore", "pipe"], timeout: 15000 });
  }
}

/** `<home>/memory/key`, mode 0600. */
export class FileKeyProvider implements KeyProvider {
  readonly name = "file";
  constructor(readonly file: string) {}

  load(): Buffer | null {
    try {
      return decodeKey(fs.readFileSync(this.file, "utf8"));
    } catch {
      return null;
    }
  }

  save(key: Buffer): void {
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    const tmp = this.file + ".tmp";
    fs.writeFileSync(tmp, key.toString("hex"), { mode: 0o600 });
    fs.renameSync(tmp, this.file);
    try { fs.chmodSync(this.file, 0o600); } catch { /* windows */ }
  }
}

/** In-memory key — tests and embedding. */
export class StaticKeyProvider implements KeyProvider {
  readonly name = "static";
  constructor(public key: Buffer | null = null) {}
  load(): Buffer | null { return this.key; }
  save(key: Buffer): void { this.key = key; }
}

export interface KeyInfo {
  store?: string;
  key_id?: string;
  created?: number;
}

export function readKeyInfo(home?: string): KeyInfo | null {
  try {
    return JSON.parse(fs.readFileSync(path.join(memoryDir(home), "keyinfo.json"), "utf8")) as KeyInfo;
  } catch {
    return null;
  }
}

function writeKeyInfo(home: string | undefined, store: string, key: Buffer): void {
  atomicWrite(path.join(memoryDir(home), "keyinfo.json"),
    JSON.stringify({ store, key_id: keyId(key), created: Date.now() / 1000 }, null, 2));
}

function providerNamed(name: string, home?: string): KeyProvider | null {
  if (name === "keychain") return KeychainProvider.available() ? new KeychainProvider() : null;
  if (name === "keyring") return SecretToolProvider.available() ? new SecretToolProvider() : null;
  if (name === "file") return new FileKeyProvider(path.join(memoryDir(home), "key"));
  return null;
}

export function isDefaultHome(home?: string): boolean {
  return path.resolve(resolveHomeDir(home)) === path.resolve(DEFAULT_HOME);
}

/** Pick the key store: keyinfo.json → $OPENDESK_MEMORY_KEYSTORE → platform default. */
export function defaultResolveKeyProvider(home?: string): KeyProvider {
  const info = readKeyInfo(home) ?? {};
  for (const name of [info.store, (process.env[ENV_KEYSTORE] ?? "").trim().toLowerCase()]) {
    if (name && name !== "auto") {
      const p = providerNamed(name, home);
      if (p) return p;
      if (name === info.store && name !== "file") {
        // The key lives in a store this runtime cannot reach (e.g. Python's
        // keyring on Windows).  Never silently mint a second key.
        throw new Error(`the memory key is in the '${name}' store, which is not reachable from Node.js here`);
      }
    }
  }
  if (isDefaultHome(home)) {
    if (KeychainProvider.available()) return new KeychainProvider();
    if (SecretToolProvider.available()) return new SecretToolProvider();
  }
  return new FileKeyProvider(path.join(memoryDir(home), "key"));
}

let resolver: (home?: string) => KeyProvider = defaultResolveKeyProvider;

/** Override key-store selection (tests).  Pass null to restore the default. */
export function setKeyProviderResolver(fn: ((home?: string) => KeyProvider) | null): void {
  resolver = fn ?? defaultResolveKeyProvider;
}

export function resolveKeyProvider(home?: string): KeyProvider {
  return resolver(home);
}

/** Load the key; with `create`, generate and store one when none exists. */
export function loadOrCreateKey(home: string | undefined, create: boolean): { key: Buffer | null; store: string } {
  const provider = resolveKeyProvider(home);
  let key = provider.load();
  if (!key && create) {
    key = crypto.randomBytes(32);
    provider.save(key);
    if (provider.name !== "static") writeKeyInfo(home, provider.name, key);
  } else if (key && provider.name !== "static" && !readKeyInfo(home)) {
    writeKeyInfo(home, provider.name, key);
  }
  return { key, store: provider.name };
}

export function keyId(key: Buffer): string {
  return crypto.createHmac("sha256", key).update("opendesk-memory-key-id").digest("hex").slice(0, 16);
}

// ---------------------------------------------------------------------------
// Cipher
// ---------------------------------------------------------------------------

/** Lower-case and strip diacritics. */
export function normalize(text: string): string {
  return (text ?? "").normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase();
}

/** Ordered word list (letters/digits runs) of normalised text. */
export function words(text: string): string[] {
  return normalize(text).match(/[\p{L}\p{N}]+/gu) ?? [];
}

/** Unique word tokens, first-seen order. */
export function tokenize(text: string): string[] {
  return [...new Set(words(text).filter((w) => w.length <= 64))];
}

export class MemoryCipher {
  readonly keyId: string;
  private readonly enc: Buffer;
  private readonly idx: Buffer;

  constructor(key: Buffer) {
    if (key.length !== 32) throw new Error("memory key must be 32 bytes");
    this.keyId = keyId(key);
    this.enc = crypto.createHmac("sha256", key).update("opendesk-memory:enc").digest();
    this.idx = crypto.createHmac("sha256", key).update("opendesk-memory:index").digest();
  }

  private seal(data: Buffer, aad: Buffer): Buffer {
    const nonce = crypto.randomBytes(NONCE_LEN);
    const c = crypto.createCipheriv("aes-256-gcm", this.enc, nonce);
    c.setAAD(aad);
    const ct = Buffer.concat([c.update(data), c.final()]);
    return Buffer.concat([nonce, ct, c.getAuthTag()]);
  }

  private open(raw: Buffer, aad: Buffer): Buffer {
    if (raw.length < NONCE_LEN + TAG_LEN) throw new Error("ciphertext too short");
    const nonce = raw.subarray(0, NONCE_LEN);
    const tagBuf = raw.subarray(raw.length - TAG_LEN);
    const d = crypto.createDecipheriv("aes-256-gcm", this.enc, nonce);
    d.setAAD(aad);
    d.setAuthTag(tagBuf);
    return Buffer.concat([d.update(raw.subarray(NONCE_LEN, raw.length - TAG_LEN)), d.final()]);
  }

  encryptBlob(data: Buffer): Buffer {
    return Buffer.concat([BLOB_MAGIC, this.seal(data, THUMB_AAD)]);
  }

  decryptBlob(blob: Buffer): Buffer {
    if (!blob.subarray(0, BLOB_MAGIC.length).equals(BLOB_MAGIC)) throw new Error("not an opendesk encrypted blob");
    return this.open(blob.subarray(BLOB_MAGIC.length), THUMB_AAD);
  }

  encryptText(text: string): string {
    return TEXT_PREFIX + this.seal(Buffer.from(text, "utf8"), TEXT_AAD).toString("base64");
  }

  decryptText(token: string): string {
    if (!token.startsWith(TEXT_PREFIX)) throw new Error("unknown encrypted-text format");
    return this.open(Buffer.from(token.slice(TEXT_PREFIX.length), "base64"), TEXT_AAD).toString("utf8");
  }

  /** Keyed token for one normalised word. */
  token(word: string): string {
    return crypto.createHmac("sha256", this.idx).update(word, "utf8").digest("hex").slice(0, TOKEN_HEX_LEN);
  }

  /** Sorted, de-duplicated blind tokens (hides word order and frequency). */
  indexTokens(text: string): string {
    return [...new Set(tokenize(text).map((t) => this.token(t)))].sort().join(" ");
  }
}
