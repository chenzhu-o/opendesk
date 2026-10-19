"""CLI for the safety layer.

Commands registered on the main ``opendesk`` parser::

    opendesk stop [--reason ...]       # kill switch: refuse every tool call
    opendesk resume
    opendesk takeover [--reason ...]   # "I'm driving" — agent calls refused
    opendesk handback
    opendesk secrets set NAME [--allowed-apps A,B] [--no-confirm]
    opendesk secrets list | delete NAME
    opendesk policy show | preset NAME | add-rule ... | remove-rule INDEX
    opendesk audit verify [--date YYYY-MM-DD]   (wired from cmd_audit)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional


def _home(args: Any) -> Optional[Path]:
    h = getattr(args, "home", None)
    return Path(h).expanduser() if h else None


def register(sub: Any) -> None:
    """Add the safety subcommands to the top-level argparse *sub* parsers."""
    for name, help_text in (
        ("stop", "Kill switch: refuse every agent tool call until `opendesk resume`"),
        ("resume", "Release the kill switch"),
        ("takeover", "Take control yourself: agent tool calls are refused until `opendesk handback`"),
        ("handback", "Hand control back to the agent after `opendesk takeover`"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--reason", default="", help="Optional note shown to the agent")
        p.add_argument("--home", default=None)

    sec = sub.add_parser("secrets", help="Store secrets for {{secret:NAME}} placeholders (OS keychain)")
    sec.add_argument("--home", default=None)
    sec_sub = sec.add_subparsers(dest="secrets_cmd")
    s_set = sec_sub.add_parser("set", help="Store / replace a secret (prompts without echo)")
    s_set.add_argument("name")
    s_set.add_argument("--allowed-apps", default=None,
                       help="Comma-separated apps the secret may be typed into")
    s_set.add_argument("--no-confirm", action="store_true",
                       help="Don't ask for approval each time the secret is used")
    s_set.add_argument("--home", default=None)
    s_list = sec_sub.add_parser("list", help="List stored secret names")
    s_list.add_argument("--home", default=None)
    s_del = sec_sub.add_parser("delete", help="Delete a secret")
    s_del.add_argument("name")
    s_del.add_argument("--home", default=None)

    pol = sub.add_parser("policy", help="Show or edit the tool-call safety policy (policy.json)")
    pol.add_argument("--home", default=None)
    pol_sub = pol.add_subparsers(dest="policy_cmd")
    p_show = pol_sub.add_parser("show", help="Print the effective policy")
    p_show.add_argument("--home", default=None)
    p_pre = pol_sub.add_parser("preset", help="Set the preset: relaxed | standard | strict")
    p_pre.add_argument("name", choices=["relaxed", "standard", "strict"])
    p_pre.add_argument("--home", default=None)
    p_add = pol_sub.add_parser("add-rule", help="Append (or --first: prepend) a rule")
    p_add.add_argument("--tool", required=True, help="Tool name or glob, e.g. script, browser, '*'")
    p_add.add_argument("--action", default=None)
    p_add.add_argument("--app", default=None)
    p_add.add_argument("--peer", default=None)
    p_add.add_argument("--decision", required=True, choices=["allow", "confirm", "block"])
    p_add.add_argument("--reason", default=None)
    p_add.add_argument("--first", action="store_true", help="Insert at the top (highest priority)")
    p_add.add_argument("--home", default=None)
    p_rm = pol_sub.add_parser("remove-rule", help="Remove a rule by its index from `policy show`")
    p_rm.add_argument("index", type=int)
    p_rm.add_argument("--home", default=None)
    p_hk = pol_sub.add_parser("kill-hotkey", help="Set / clear the global kill hotkey (pynput syntax)")
    p_hk.add_argument("hotkey", nargs="?", help="e.g. '<ctrl>+<alt>+<shift>+k'; omit with --clear")
    p_hk.add_argument("--clear", action="store_true")
    p_hk.add_argument("--home", default=None)


def dispatch(args: Any) -> bool:
    """Run a safety command if *args* names one.  Returns ``True`` if handled."""
    cmd = getattr(args, "command", None)
    if cmd in ("stop", "resume", "takeover", "handback"):
        cmd_control(args)
        return True
    if cmd == "secrets":
        cmd_secrets(args)
        return True
    if cmd == "policy":
        cmd_policy(args)
        return True
    return False


# ---------------------------------------------------------------------------
# control
# ---------------------------------------------------------------------------


def cmd_control(args: Any) -> None:
    from opendesk.safety.control import read_control, set_mode

    home = _home(args)
    current = read_control(home)
    cmd = args.command
    if cmd == "stop":
        set_mode("stopped", home, reason=args.reason or "opendesk stop")
        print("Stopped. Every agent tool call is now refused. Run `opendesk resume` to continue.")
    elif cmd == "takeover":
        set_mode("takeover", home, reason=args.reason or "user took control")
        print("You have control. Agent tool calls are refused until `opendesk handback`.")
    elif cmd == "resume":
        if current.mode == "takeover":
            print("Note: control was in takeover mode; handing back.")
        set_mode("running", home, reason=args.reason or "resumed")
        print("Resumed. Agents may use tools again.")
    elif cmd == "handback":
        if current.mode == "stopped":
            print("Note: the kill switch was engaged; releasing it.")
        set_mode("running", home, reason=args.reason or "handed back")
        print("Handed back. Agents may use tools again.")


# ---------------------------------------------------------------------------
# secrets
# ---------------------------------------------------------------------------


def _read_secret(name: str) -> str:
    import getpass
    if not sys.stdin.isatty():
        return sys.stdin.readline().rstrip("\n")
    first = getpass.getpass(f"Value for secret {name!r}: ")
    second = getpass.getpass("Repeat: ")
    if first != second:
        print("ERROR: values do not match.", file=sys.stderr)
        sys.exit(1)
    return first


def cmd_secrets(args: Any) -> None:
    from opendesk.safety import secrets as S

    home = _home(args)
    sub = getattr(args, "secrets_cmd", None) or "list"
    try:
        if sub == "set":
            value = _read_secret(args.name)
            apps = [a.strip() for a in (args.allowed_apps or "").split(",") if a.strip()]
            info = S.set_secret(args.name, value, home, allowed_apps=apps,
                                confirm=not args.no_confirm)
            extra = f" (allowed apps: {', '.join(info.allowed_apps)})" if info.allowed_apps else ""
            print(f"Stored secret {info.name!r}{extra}. Use it as {{{{secret:{info.name}}}}}.")
        elif sub == "delete":
            existed = S.delete_secret(args.name, home)
            print(f"Deleted {args.name!r}." if existed else f"{args.name!r} was not in the index.")
        else:
            items = S.list_secrets(home)
            if not items:
                print("No secrets stored. Add one with `opendesk secrets set NAME`.")
                return
            for i in items:
                flags = [] if i.confirm else ["no-confirm"]
                if i.allowed_apps:
                    flags.append("apps=" + ",".join(i.allowed_apps))
                print(f"  {i.name}" + (f"  [{'; '.join(flags)}]" if flags else ""))
    except S.SecretError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


def cmd_policy(args: Any) -> None:
    from opendesk.safety.policy import load_policy, policy_path, save_policy, validate_rule

    home = _home(args)
    sub = getattr(args, "policy_cmd", None) or "show"
    cfg = load_policy(home)
    if sub == "preset":
        cfg.preset = args.name
        print(f"Preset set to {args.name!r} → {save_policy(cfg, home)}")
        return
    if sub == "add-rule":
        try:
            rule = validate_rule({
                "tool": args.tool, "action": args.action, "app": args.app,
                "peer": args.peer, "decision": args.decision, "reason": args.reason,
            })
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)
        if args.first:
            cfg.rules.insert(0, rule)
        else:
            cfg.rules.append(rule)
        save_policy(cfg, home)
        print(f"Added rule: {json.dumps(rule)}")
        return
    if sub == "remove-rule":
        if not 0 <= args.index < len(cfg.rules):
            print(f"ERROR: no rule #{args.index}", file=sys.stderr)
            sys.exit(2)
        removed = cfg.rules.pop(args.index)
        save_policy(cfg, home)
        print(f"Removed rule: {json.dumps(removed)}")
        return
    if sub == "kill-hotkey":
        cfg.kill_hotkey = None if args.clear else (args.hotkey or None)
        save_policy(cfg, home)
        print(f"Kill hotkey: {cfg.kill_hotkey or '(none)'} — takes effect on the next MCP server start.")
        return

    from opendesk.safety.control import read_control
    path = policy_path(home)
    state = read_control(home)
    print(f"Policy file: {path}{'' if path.exists() else '  (not created — defaults)'}")
    print(f"Preset:      {cfg.preset}")
    print(f"Control:     {state.mode}" + (f" ({state.reason})" if state.reason else ""))
    print(f"Approval:    timeout={cfg.approval_timeout:.0f}s"
          + (f" chain={','.join(cfg.approval_chain)}" if cfg.approval_chain else ""))
    print(f"Taint:       {cfg.taint_calls} calls / {cfg.taint_minutes:g} min after a suspected injection")
    print(f"Kill hotkey: {cfg.kill_hotkey or '(none)'}")
    print("Rules (first match wins):")
    if not cfg.rules:
        print("  (none)")
    for i, r in enumerate(cfg.rules):
        cond = " ".join(f"{k}={r[k]}" for k in ("tool", "action", "app", "peer") if r.get(k))
        reason = f"  — {r['reason']}" if r.get("reason") else ""
        print(f"  [{i}] {cond} → {r.get('decision')}{reason}")


# ---------------------------------------------------------------------------
# audit verify
# ---------------------------------------------------------------------------


def cmd_audit_verify(args: Any) -> None:
    from opendesk.safety.audit import audit_files, verify_files

    home = _home(args)
    files = audit_files(home, getattr(args, "date", None))
    if not files:
        print("No audit files found" + (f" for {args.date}" if getattr(args, "date", None) else "") + ".")
        return
    results = verify_files(files)
    for r in results:
        print(r.describe())
    bad = [r for r in results if not r.ok]
    if bad:
        print(f"\n{len(bad)} file(s) failed verification — the log may have been tampered with.")
        sys.exit(1)
    print(f"\nAll {len(results)} file(s) verified.")
