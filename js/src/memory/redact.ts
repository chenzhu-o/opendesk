/**
 * Secret redaction for screen-memory text — mirrors Python's
 * opendesk/memory/redact.py.
 *
 * Applied to every frame's text (and window title) *before* it is stored,
 * so credentials that flash past on screen never reach disk.  Each hit is
 * replaced by `[REDACTED:<kind>]`.
 *
 * Kinds: card (Luhn-valid card numbers), aws_key, github_token,
 * anthropic_key, openai_key, slack_token, jwt, private_key (PEM blocks),
 * password / secret (`password: hunter2`-style pairs, `Bearer …`).
 */

const tag = (kind: string): string => `[REDACTED:${kind}]`;

const PEM_RE = /-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----(?:[\s\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----|[A-Za-z0-9+/=\s]*)/g;
const JWT_RE = /\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g;
const ANTHROPIC_RE = /\bsk-ant-[A-Za-z0-9_-]{20,}/g;
const OPENAI_RE = /\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}/g;
const GITHUB_RE = /\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{22,255})/g;
const SLACK_RE = /\bxox[abposre]-[A-Za-z0-9-]{10,}/g;
const AWS_RE = /\b(?:AKIA|ASIA|ABIA|ACCA|AGPA|AIDA|AIPA|ANPA|ANVA|APKA|AROA|ASCA)[A-Z0-9]{16}\b/g;
const BEARER_RE = /\b(bearer)(\s+)([A-Za-z0-9._~+/-]{20,}=*)/gi;
const CARD_RE = /(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)/g;
const PAIR_RE = new RegExp(
  "(?<![A-Za-z0-9])(password|passwd|pwd|passphrase|pass code|passcode|pin code" +
  "|secret|client[_ -]?secret|secret[_ -]?key|api[_ -]?key|access[_ -]?key" +
  "|access[_ -]?token|auth[_ -]?token|token|aws_secret_access_key)" +
  "(\\s*[:=]\\s*)([\"']?)([^\\s\"']{1,200})\\3",
  "gi",
);
const PASSWORD_LABELS = new Set(["password", "passwd", "pwd", "passphrase", "pass code", "passcode", "pin code"]);

/** Luhn checksum over a string of digits. */
export function luhnOk(digits: string): boolean {
  let total = 0;
  for (let i = 0; i < digits.length; i++) {
    let d = digits.charCodeAt(digits.length - 1 - i) - 48;
    if (i % 2 === 1) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    total += d;
  }
  return total % 10 === 0;
}

type Replacer = (match: string, ...groups: string[]) => string;

const RULES: Array<[RegExp, string | Replacer]> = [
  [PEM_RE, tag("private_key")],
  [JWT_RE, tag("jwt")],
  [ANTHROPIC_RE, tag("anthropic_key")],
  [OPENAI_RE, tag("openai_key")],
  [GITHUB_RE, tag("github_token")],
  [SLACK_RE, tag("slack_token")],
  [AWS_RE, tag("aws_key")],
  [BEARER_RE, (_m, word, sep) => `${word}${sep}${tag("secret")}`],
  [CARD_RE, (m) => {
    const digits = m.replace(/\D/g, "");
    const ok = digits.length >= 13 && digits.length <= 19 && "23456".includes(digits[0]) &&
      new Set(digits).size > 1 && luhnOk(digits);
    return ok ? tag("card") : m;
  }],
  [PAIR_RE, (m, label, sep, quote, value) => {
    if (value.startsWith("[REDACTED:")) return m;
    const kind = PASSWORD_LABELS.has(label.toLowerCase()) ? "password" : "secret";
    return `${label}${sep}${quote}${tag(kind)}${quote}`;
  }],
];

/** Returns the redacted text and the number of redactions made. */
export function redact(text: string): { text: string; count: number } {
  if (!text) return { text: text ?? "", count: 0 };
  let count = 0;
  let out = text;
  for (const [re, repl] of RULES) {
    out = out.replace(re, (...args: unknown[]) => {
      const m = args[0] as string;
      const groups = args.slice(1).filter((a): a is string => typeof a === "string" || a === undefined) as string[];
      const next = typeof repl === "string" ? repl : repl(m, ...groups.map((g) => g ?? ""));
      if (next !== m) count++;
      return next;
    });
  }
  return { text: out, count };
}

export function redactText(text: string): string {
  return redact(text).text;
}
