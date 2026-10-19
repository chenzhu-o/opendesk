"""Native macOS accessibility + background input via the ``opendesk-ax`` helper.

``opendesk-ax`` is a small Swift program, embedded below as source and
compiled once with ``swiftc -O`` to ``<home>/bin/opendesk-ax-v1`` (home is
``$OPENDESK_HOME`` or ``~/.opendesk``).  The JS SDK embeds a byte-identical
copy of the source (``js/src/computer/macosNative.ts``), so both SDKs share
one compiled binary.

Compared with spawning ``osascript`` per query, the helper:

* stays running (``opendesk-ax serve``) and answers JSON-line requests, so
  there is no per-call process or AppleScript start-up cost;
* reads each element's attributes in one batched
  ``AXUIElementCopyMultipleAttributeValues`` call;
* turns on Chromium / Electron accessibility (``AXManualAccessibility``);
* hands out short, stable element refs so an agent can act on exactly the
  element it saw (``perform``, ``set_value``, ``focus``);
* can deliver clicks and keystrokes to a process with ``CGEvent.postToPid``
  without moving the user's cursor or stealing focus.

Every public function degrades gracefully: :func:`available` is ``False``
and :func:`window_text` returns ``None`` off macOS, when ``swiftc`` is
missing or compilation fails, or when Accessibility permission has not
been granted.  The other calls raise :class:`NativeAXUnavailable` /
:class:`NativeAXError` so callers can fall back to AppleScript.

Set ``OPENDESK_NATIVE_AX=0`` to disable the helper entirely.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

_PLATFORM = platform.system()

HELPER_VERSION = "1"
HELPER_NAME = f"opendesk-ax-v{HELPER_VERSION}"

#: Default per-request timeouts (seconds).
DEFAULT_TIMEOUT = 10.0
TREE_TIMEOUT = 30.0

#: After a failed compile / start, wait this long before trying again.
_RETRY_AFTER = 300.0


class NativeAXError(RuntimeError):
    """The helper ran but the command failed (unknown ref, AX error, ...)."""


class NativeAXUnavailable(NativeAXError):
    """The helper cannot be used here (not macOS, no swiftc, no permission)."""


# Keep byte-for-byte identical with SWIFT_SOURCE in js/src/computer/macosNative.ts
# (tests in both SDKs enforce it) so the two SDKs share one compiled binary.
SWIFT_SOURCE = r'''
// opendesk-ax v1: native macOS accessibility + background input helper.
//
// Usage:
//   opendesk-ax serve            JSON-lines server on stdin/stdout
//   opendesk-ax call '<json>'    run one request and print the response
//   opendesk-ax version
//
// Request:  {"id": 1, "cmd": "tree", ...args}
// Response: {"id": 1, "ok": true, "result": {...}}
//           {"id": 1, "ok": false, "error": "..."}
//
// Commands: ping, frontmost, tree, get, perform, set_value, focus,
// element_at, window_text, click_pid, key_pid.
//
// Keep this file byte-identical between python/opendesk/computer/macos_native.py
// and js/src/computer/macosNative.ts so both SDKs share one compiled binary.

import AppKit
import ApplicationServices
import Foundation

let kHelperVersion = 1

struct HelperError: Error {
    let message: String
    init(_ message: String) { self.message = message }
}

typealias JSON = [String: Any]

// MARK: - Small utilities

func truncate(_ s: String, _ limit: Int) -> String {
    if limit <= 0 || s.count <= limit { return s }
    return String(s.prefix(limit)) + "..."
}

func num(_ v: Any?) -> Double? {
    if let n = v as? NSNumber { return n.doubleValue }
    if let s = v as? String { return Double(s) }
    return nil
}

func intArg(_ req: JSON, _ key: String, _ def: Int) -> Int {
    if let d = num(req[key]) { return Int(d) }
    return def
}

func strArg(_ req: JSON, _ key: String) -> String? {
    if let s = req[key] as? String, !s.isEmpty { return s }
    return nil
}

func finite(_ d: CGFloat) -> Double {
    let v = Double(d)
    return v.isFinite ? (v * 10).rounded() / 10 : 0
}

func fnv1a(_ s: String) -> UInt64 {
    var h: UInt64 = 0xcbf29ce484222325
    for b in s.utf8 {
        h ^= UInt64(b)
        h = h &* 0x100000001b3
    }
    return h
}

func base36(_ value: UInt64, width: Int) -> String {
    let digits = Array("0123456789abcdefghijklmnopqrstuvwxyz")
    var v = value
    var out: [Character] = []
    for _ in 0..<width {
        out.append(digits[Int(v % 36)])
        v /= 36
    }
    return String(out.reversed())
}

// MARK: - Attribute reading (batched)

let kBatchAttrs: [String] = [
    "AXRole", "AXSubrole", "AXTitle", "AXValue", "AXDescription", "AXIdentifier",
    "AXPosition", "AXSize", "AXEnabled", "AXFocused", "AXChildren",
]

struct RawNode {
    var role = ""
    var subrole = ""
    var title = ""
    var value: String? = nil
    var desc = ""
    var ident = ""
    var frame: CGRect? = nil
    var enabled = true
    var focused = false
    var children: [AXUIElement] = []
}

func isErrorValue(_ v: AnyObject) -> Bool {
    if CFGetTypeID(v) == AXValueGetTypeID() {
        return AXValueGetType(v as! AXValue) == .axError
    }
    return CFGetTypeID(v) == CFNullGetTypeID()
}

func cfString(_ v: AnyObject?, _ limit: Int) -> String? {
    guard let v = v, !isErrorValue(v) else { return nil }
    let t = CFGetTypeID(v)
    if t == CFStringGetTypeID() { return truncate(v as! String, limit) }
    if t == CFBooleanGetTypeID() { return CFBooleanGetValue((v as! CFBoolean)) ? "1" : "0" }
    if t == CFNumberGetTypeID() { return (v as! NSNumber).stringValue }
    if t == CFAttributedStringGetTypeID() { return truncate((v as! NSAttributedString).string, limit) }
    if t == CFURLGetTypeID() { return (v as! NSURL).absoluteString }
    return nil
}

func cfBool(_ v: AnyObject?, _ def: Bool) -> Bool {
    guard let v = v, !isErrorValue(v) else { return def }
    if CFGetTypeID(v) == CFBooleanGetTypeID() { return CFBooleanGetValue((v as! CFBoolean)) }
    if let n = v as? NSNumber { return n.boolValue }
    return def
}

func cfPoint(_ v: AnyObject?) -> CGPoint? {
    guard let v = v, CFGetTypeID(v) == AXValueGetTypeID() else { return nil }
    var p = CGPoint.zero
    return AXValueGetValue(v as! AXValue, .cgPoint, &p) ? p : nil
}

func cfSize(_ v: AnyObject?) -> CGSize? {
    guard let v = v, CFGetTypeID(v) == AXValueGetTypeID() else { return nil }
    var s = CGSize.zero
    return AXValueGetValue(v as! AXValue, .cgSize, &s) ? s : nil
}

func cfElements(_ v: AnyObject?) -> [AXUIElement] {
    guard let v = v, CFGetTypeID(v) == CFArrayGetTypeID() else { return [] }
    let arr = v as! [AnyObject]
    var out: [AXUIElement] = []
    out.reserveCapacity(arr.count)
    for item in arr where CFGetTypeID(item) == AXUIElementGetTypeID() {
        out.append(item as! AXUIElement)
    }
    return out
}

func readNode(_ el: AXUIElement, maxValue: Int, attrs: [String] = kBatchAttrs) -> RawNode? {
    var out: CFArray?
    let err = AXUIElementCopyMultipleAttributeValues(
        el, attrs as CFArray, AXCopyMultipleAttributeOptions(rawValue: 0), &out)
    guard err == .success, let values = out as? [AnyObject], values.count == attrs.count else {
        return nil
    }
    var n = RawNode()
    var pos: CGPoint? = nil
    var size: CGSize? = nil
    for (i, name) in attrs.enumerated() {
        let v = values[i]
        switch name {
        case "AXRole": n.role = cfString(v, 200) ?? ""
        case "AXSubrole": n.subrole = cfString(v, 200) ?? ""
        case "AXTitle": n.title = cfString(v, 500) ?? ""
        case "AXValue": n.value = cfString(v, maxValue)
        case "AXDescription": n.desc = cfString(v, 500) ?? ""
        case "AXIdentifier": n.ident = cfString(v, 200) ?? ""
        case "AXPosition": pos = cfPoint(v)
        case "AXSize": size = cfSize(v)
        case "AXEnabled": n.enabled = cfBool(v, true)
        case "AXFocused": n.focused = cfBool(v, false)
        case "AXChildren": n.children = cfElements(v)
        default: break
        }
    }
    if n.role.isEmpty { return nil }
    if let p = pos, let s = size { n.frame = CGRect(origin: p, size: s) }
    return n
}

func copyAttr(_ el: AXUIElement, _ name: String) -> AnyObject? {
    var v: CFTypeRef?
    guard AXUIElementCopyAttributeValue(el, name as CFString, &v) == .success else { return nil }
    return v
}

func actionNames(_ el: AXUIElement) -> [String] {
    var names: CFArray?
    guard AXUIElementCopyActionNames(el, &names) == .success, let arr = names as? [String] else {
        return []
    }
    return arr
}

func pidOf(_ el: AXUIElement) -> pid_t {
    var pid: pid_t = 0
    AXUIElementGetPid(el, &pid)
    return pid
}

// MARK: - Apps

func appName(_ pid: pid_t) -> String {
    if let app = NSRunningApplication(processIdentifier: pid) {
        return app.localizedName ?? app.executableURL?.lastPathComponent ?? ""
    }
    return ""
}

func frontmostPid() -> pid_t? {
    if let app = NSWorkspace.shared.frontmostApplication { return app.processIdentifier }
    let sys = AXUIElementCreateSystemWide()
    AXUIElementSetMessagingTimeout(sys, 1.0)
    if let v = copyAttr(sys, "AXFocusedApplication"), CFGetTypeID(v) == AXUIElementGetTypeID() {
        let pid = pidOf(v as! AXUIElement)
        if pid > 0 { return pid }
    }
    return nil
}

func findApp(_ name: String) -> pid_t? {
    let want = name.lowercased()
    let apps = NSWorkspace.shared.runningApplications
    for app in apps {
        if (app.localizedName ?? "").lowercased() == want || (app.bundleIdentifier ?? "").lowercased() == want {
            return app.processIdentifier
        }
    }
    for app in apps {
        if (app.executableURL?.lastPathComponent ?? "").lowercased() == want {
            return app.processIdentifier
        }
    }
    for app in apps where app.activationPolicy == .regular {
        if (app.localizedName ?? "").lowercased().contains(want) { return app.processIdentifier }
    }
    return nil
}

func targetPid(_ req: JSON) throws -> pid_t {
    if let p = num(req["pid"]), p > 0 { return pid_t(p) }
    if let name = strArg(req, "app") {
        if let pid = findApp(name) { return pid }
        throw HelperError("app not running: \(name)")
    }
    if let pid = frontmostPid() { return pid }
    throw HelperError("no frontmost application")
}

func appElement(_ pid: pid_t) -> AXUIElement {
    let app = AXUIElementCreateApplication(pid)
    AXUIElementSetMessagingTimeout(app, 3.0)
    return app
}

// Chromium / Electron only build their accessibility tree on request.
var chromiumChecked: [pid_t: Bool] = [:]

func isChromiumLike(_ pid: pid_t) -> Bool {
    if let known = chromiumChecked[pid] { return known }
    var result = false
    if let app = NSRunningApplication(processIdentifier: pid) {
        let bid = (app.bundleIdentifier ?? "").lowercased()
        let ids = ["com.google.chrome", "org.chromium", "com.brave.browser", "com.microsoft.edgemac",
                   "company.thebrowser", "com.vivaldi", "com.operasoftware"]
        if ids.contains(where: { bid.hasPrefix($0) }) { result = true }
        if !result, let url = app.bundleURL {
            let fw = url.appendingPathComponent("Contents/Frameworks").path
            if let items = try? FileManager.default.contentsOfDirectory(atPath: fw) {
                result = items.contains(where: {
                    $0.contains("Electron") || $0.contains("Chrom") || $0.contains("Browser Framework")
                        || $0.contains("Edge Framework")
                })
            }
        }
    }
    chromiumChecked[pid] = result
    return result
}

var manualAXEnabled = Set<pid_t>()

func enableChromiumAX(_ pid: pid_t, _ app: AXUIElement) -> Bool {
    guard isChromiumLike(pid) else { return false }
    if manualAXEnabled.contains(pid) { return false }
    AXUIElementSetAttributeValue(app, "AXManualAccessibility" as CFString, kCFBooleanTrue)
    AXUIElementSetAttributeValue(app, "AXEnhancedUserInterface" as CFString, kCFBooleanTrue)
    manualAXEnabled.insert(pid)
    return true
}

func windowOf(_ app: AXUIElement) -> AXUIElement? {
    for name in ["AXFocusedWindow", "AXMainWindow"] {
        if let v = copyAttr(app, name), CFGetTypeID(v) == AXUIElementGetTypeID() {
            return (v as! AXUIElement)
        }
    }
    return cfElements(copyAttr(app, "AXWindows")).first
}

// MARK: - Refs

struct PathComp {
    let role: String
    let index: Int
    let key: String
    var text: String { "\(role)#\(index)~\(key)" }
}

struct RefEntry {
    var el: AXUIElement
    let pid: pid_t
    let path: [PathComp]
}

var refStore: [String: RefEntry] = [:]
var pidRefs: [pid_t: Set<String>] = [:]

func keyFor(_ n: RawNode) -> String {
    if !n.ident.isEmpty { return truncate(n.ident, 60) }
    return truncate(n.title, 40)
}

func rootComp(_ n: RawNode) -> PathComp {
    // The window title is deliberately excluded: it changes as documents
    // are edited, and every ref below it would change with it.
    PathComp(role: n.role, index: 0, key: n.role == "AXWindow" ? n.ident : "")
}

func refFor(_ pid: pid_t, _ path: [PathComp]) -> String {
    let text = "\(pid)|" + path.map { $0.text }.joined(separator: "/")
    return base36(fnv1a(text), width: 6)
}

func register(_ ref: String, _ entry: RefEntry) {
    refStore[ref] = entry
    pidRefs[entry.pid, default: []].insert(ref)
}

func isAlive(_ el: AXUIElement, role: String) -> Bool {
    guard let v = copyAttr(el, "AXRole") as? String else { return false }
    return v == role
}

func resolvePath(_ pid: pid_t, _ path: [PathComp]) -> AXUIElement? {
    guard let first = path.first else { return nil }
    let app = appElement(pid)
    var cur: AXUIElement
    if first.role == "AXApplication" {
        cur = app
    } else if let w = windowOf(app) {
        cur = w
    } else {
        return nil
    }
    for comp in path.dropFirst() {
        guard let parent = readNode(cur, maxValue: 0, attrs: ["AXRole", "AXChildren"]) else { return nil }
        var sameRole: [(AXUIElement, RawNode)] = []
        for child in parent.children {
            if let n = readNode(child, maxValue: 0, attrs: ["AXRole", "AXTitle", "AXIdentifier"]),
               n.role == comp.role {
                sameRole.append((child, n))
            }
        }
        var next: AXUIElement? = nil
        if comp.index < sameRole.count && keyFor(sameRole[comp.index].1) == comp.key {
            next = sameRole[comp.index].0
        } else if !comp.key.isEmpty, let m = sameRole.first(where: { keyFor($0.1) == comp.key }) {
            next = m.0
        } else if comp.key.isEmpty && comp.index < sameRole.count {
            next = sameRole[comp.index].0
        }
        guard let n = next else { return nil }
        cur = n
    }
    return cur
}

func resolve(_ ref: String) throws -> RefEntry {
    if var e = refStore[ref] {
        if isAlive(e.el, role: e.path.last?.role ?? "") { return e }
        if let el = resolvePath(e.pid, e.path) {
            e.el = el
            refStore[ref] = e
            return e
        }
        throw HelperError("stale ref \(ref): the element no longer exists; call tree again")
    }
    // Unknown ref (helper restarted, or the UI changed): rebuild the trees we
    // know about, plus the frontmost app, and look again.
    var pids = Array(pidRefs.keys)
    if let f = frontmostPid(), !pids.contains(f) { pids.append(f) }
    for pid in pids {
        _ = try? buildTree(pid: pid, scope: "window", maxDepth: 40, maxNodes: 4000, maxValue: 200, withActions: false)
        if let e = refStore[ref] { return e }
    }
    throw HelperError("unknown ref \(ref); call tree again")
}

// MARK: - Tree

let kTransparentRoles: Set<String> = ["AXGroup", "AXScrollArea", "AXSplitGroup", "AXLayoutArea", "AXUnknown"]

func isTransparent(_ n: RawNode) -> Bool {
    kTransparentRoles.contains(n.role) && n.title.isEmpty && n.desc.isEmpty
}

func nodeJSON(_ n: RawNode, ref: String, parent: String?, actions: [String]?) -> JSON {
    var o: JSON = ["ref": ref, "parent": parent ?? NSNull(), "role": n.role]
    if !n.subrole.isEmpty { o["subrole"] = n.subrole }
    if !n.title.isEmpty { o["title"] = n.title }
    if let v = n.value, !v.isEmpty { o["value"] = v }
    if !n.desc.isEmpty { o["description"] = n.desc }
    if !n.ident.isEmpty { o["identifier"] = n.ident }
    if let f = n.frame {
        o["frame"] = ["x": finite(f.origin.x), "y": finite(f.origin.y),
                      "w": finite(f.size.width), "h": finite(f.size.height)]
    }
    o["enabled"] = n.enabled
    o["focused"] = n.focused
    if let a = actions { o["actions"] = a }
    return o
}

func buildTree(pid: pid_t, scope: String, maxDepth: Int, maxNodes: Int, maxValue: Int,
               withActions: Bool) throws -> JSON {
    let app = appElement(pid)
    let justEnabled = enableChromiumAX(pid, app)
    var attempt = 0
    while true {
        let result = try walkTree(pid: pid, app: app, scope: scope, maxDepth: maxDepth,
                                  maxNodes: maxNodes, maxValue: maxValue, withActions: withActions)
        let count = (result["nodes"] as? [JSON])?.count ?? 0
        if count > 1 || attempt >= 3 || !(justEnabled || isChromiumLike(pid)) { return result }
        attempt += 1
        usleep(150_000)
    }
}

func walkTree(pid: pid_t, app: AXUIElement, scope: String, maxDepth: Int, maxNodes: Int,
              maxValue: Int, withActions: Bool) throws -> JSON {
    var root: AXUIElement = app
    if scope != "app" {
        if let w = windowOf(app) { root = w }
    }
    guard let rootRaw = readNode(root, maxValue: maxValue) else {
        if !AXIsProcessTrusted() {
            throw HelperError("accessibility permission not granted")
        }
        throw HelperError("cannot read accessibility tree of pid \(pid)")
    }
    var newRefs: [String: RefEntry] = [:]
    var nodes: [JSON] = []
    var truncated = false
    var windowTitle = ""
    if rootRaw.role == "AXWindow" { windowTitle = rootRaw.title }

    // Breadth-first so a deep subtree cannot starve the rest of the window.
    // Depth only counts meaningful levels: unlabelled layout containers
    // (AXGroup, AXScrollArea, ...) are transparent.
    struct Item {
        let el: AXUIElement
        let raw: RawNode
        let path: [PathComp]
        let depth: Int
        let hops: Int
        let parent: String?
    }
    var queue: [Item] = [Item(el: root, raw: rootRaw, path: [rootComp(rootRaw)], depth: 0, hops: 0, parent: nil)]
    var head = 0
    while head < queue.count {
        let it = queue[head]
        head += 1
        var ref = refFor(pid, it.path)
        var bump = 2
        while let existing = newRefs[ref], existing.path.map({ $0.text }) != it.path.map({ $0.text }) {
            ref = refFor(pid, it.path) + "\(bump)"
            bump += 1
        }
        newRefs[ref] = RefEntry(el: it.el, pid: pid, path: it.path)
        nodes.append(nodeJSON(it.raw, ref: ref, parent: it.parent,
                              actions: withActions ? actionNames(it.el) : nil))
        if it.raw.children.isEmpty { continue }
        if it.depth >= maxDepth || it.hops >= 64 {
            truncated = true
            continue
        }
        var roleCounts: [String: Int] = [:]
        for child in it.raw.children {
            if queue.count >= maxNodes {
                truncated = true
                break
            }
            guard let cr = readNode(child, maxValue: maxValue), cr.role != "AXApplication" else { continue }
            let idx = roleCounts[cr.role, default: 0]
            roleCounts[cr.role] = idx + 1
            let comp = PathComp(role: cr.role, index: idx, key: keyFor(cr))
            let d = isTransparent(cr) ? it.depth : it.depth + 1
            queue.append(Item(el: child, raw: cr, path: it.path + [comp], depth: d,
                              hops: it.hops + 1, parent: ref))
        }
    }
    // Keep refs for the latest tree of this pid (plus anything element_at added).
    if let old = pidRefs[pid] {
        for r in old where newRefs[r] == nil { refStore.removeValue(forKey: r) }
    }
    pidRefs[pid] = []
    for (r, e) in newRefs { register(r, e) }
    return ["pid": Int(pid), "app": appName(pid), "window_title": windowTitle,
            "nodes": nodes, "truncated": truncated]
}

// MARK: - element_at

func pathFor(_ el: AXUIElement) -> [PathComp]? {
    var comps: [PathComp] = []
    var cur = el
    for _ in 0..<80 {
        guard let n = readNode(cur, maxValue: 0, attrs: ["AXRole", "AXTitle", "AXIdentifier"]) else { return nil }
        if n.role == "AXWindow" || n.role == "AXApplication" {
            comps.append(rootComp(n))
            return comps.reversed()
        }
        guard let pv = copyAttr(cur, "AXParent"), CFGetTypeID(pv) == AXUIElementGetTypeID() else {
            comps.append(rootComp(n))
            return comps.reversed()
        }
        let parent = pv as! AXUIElement
        var idx = 0
        for sib in cfElements(copyAttr(parent, "AXChildren")) {
            if CFEqual(sib, cur) { break }
            if (copyAttr(sib, "AXRole") as? String) == n.role { idx += 1 }
        }
        comps.append(PathComp(role: n.role, index: idx, key: keyFor(n)))
        cur = parent
    }
    return nil
}

let kPressRoles: Set<String> = ["AXGroup", "AXWindow", "AXApplication", "AXScrollArea", "AXWebArea",
                                "AXSplitGroup", "AXLayoutArea", "AXSheet", "AXDrawer"]

func elementAt(x: Double, y: Double) throws -> JSON {
    let sys = AXUIElementCreateSystemWide()
    AXUIElementSetMessagingTimeout(sys, 2.0)
    var hit: AXUIElement?
    let err = AXUIElementCopyElementAtPosition(sys, Float(x), Float(y), &hit)
    guard err == .success, let el = hit else {
        throw HelperError("no accessible element at (\(x), \(y)) [AXError \(err.rawValue)]")
    }
    return try describeAndRegister(el)
}

func describeAndRegister(_ el: AXUIElement) throws -> JSON {
    let pid = pidOf(el)
    guard let raw = readNode(el, maxValue: 2000) else { throw HelperError("element vanished") }
    let path = pathFor(el) ?? [PathComp(role: raw.role, index: 0, key: keyFor(raw))]
    let ref = refFor(pid, path)
    register(ref, RefEntry(el: el, pid: pid, path: path))
    var o = nodeJSON(raw, ref: ref, parent: nil, actions: actionNames(el))
    o["pid"] = Int(pid)
    o["app"] = appName(pid)
    // Nearest pressable element (the hit is often a label inside a button).
    var cur: AXUIElement? = el
    for _ in 0..<4 {
        guard let c = cur else { break }
        let role = (copyAttr(c, "AXRole") as? String) ?? ""
        if kPressRoles.contains(role) { break }
        if actionNames(c).contains("AXPress") {
            if CFEqual(c, el) {
                o["pressable"] = ref
            } else if let p = pathFor(c) {
                let pref = refFor(pid, p)
                register(pref, RefEntry(el: c, pid: pid, path: p))
                o["pressable"] = pref
            }
            break
        }
        if let pv = copyAttr(c, "AXParent"), CFGetTypeID(pv) == AXUIElementGetTypeID() {
            cur = (pv as! AXUIElement)
        } else {
            cur = nil
        }
    }
    return o
}

// MARK: - window_text

let kValueTextRoles: Set<String> = ["AXStaticText", "AXTextArea", "AXTextField", "AXSearchField", "AXHeading"]

func windowText(pid: pid_t, maxChars: Int) throws -> JSON {
    let app = appElement(pid)
    _ = enableChromiumAX(pid, app)
    guard let win = windowOf(app) else {
        if !AXIsProcessTrusted() { throw HelperError("accessibility permission not granted") }
        return ["pid": Int(pid), "app": appName(pid), "window_title": "", "text": "", "nodes": 0]
    }
    let attrs = ["AXRole", "AXTitle", "AXValue", "AXPosition", "AXSize", "AXChildren", "AXVisibleRows"]
    var seen = Set<String>()
    var lines: [String] = []
    var total = 0
    var visited = 0
    var winFrame: CGRect? = nil
    var winTitle = ""
    // Depth-first, children in order: the accessibility reading order.
    var stack: [(AXUIElement, Int)] = [(win, 0)]
    func add(_ s: String?) {
        guard let s = s?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty else { return }
        if seen.contains(s) { return }
        seen.insert(s)
        lines.append(s)
        total += s.count + 1
    }
    while let (el, depth) = stack.popLast() {
        if total >= maxChars || visited >= 5000 { break }
        visited += 1
        var out: CFArray?
        guard AXUIElementCopyMultipleAttributeValues(el, attrs as CFArray,
                                                     AXCopyMultipleAttributeOptions(rawValue: 0), &out) == .success,
              let v = out as? [AnyObject], v.count == attrs.count else { continue }
        let role = cfString(v[0], 100) ?? ""
        let title = cfString(v[1], 2000)
        var frame: CGRect? = nil
        if let p = cfPoint(v[3]), let s = cfSize(v[4]) { frame = CGRect(origin: p, size: s) }
        if role == "AXMenuBar" || role == "AXMenu" || role == "AXApplication" { continue }
        if depth == 0 {
            winFrame = frame
            winTitle = title ?? ""
        } else if let f = frame, let wf = winFrame {
            // Skip what is scrolled out of (or hidden outside) the window.
            if f.width <= 0 || f.height <= 0 || !f.intersects(wf) { continue }
        }
        add(title)
        if kValueTextRoles.contains(role) { add(cfString(v[2], maxChars)) }
        if depth >= 60 { continue }
        var kids = cfElements(v[5])
        if role == "AXTable" || role == "AXOutline" || role == "AXList" {
            let visible = cfElements(v[6])
            if !visible.isEmpty { kids = visible }
        }
        for k in kids.reversed() { stack.append((k, depth + 1)) }
    }
    var text = lines.joined(separator: "\n")
    if text.count > maxChars { text = String(text.prefix(maxChars)) }
    return ["pid": Int(pid), "app": appName(pid), "window_title": winTitle, "text": text, "nodes": visited]
}

// MARK: - Background input

func windowUnder(_ pt: CGPoint, pid: pid_t?) -> (pid_t, Int)? {
    // On-screen windows first (front to back); with a pid, also that app's
    // windows on other Spaces or hidden behind full-screen apps.
    var options: [CGWindowListOption] = [[.optionOnScreenOnly, .excludeDesktopElements]]
    if pid != nil { options.append([.optionAll]) }
    for opt in options {
        guard let info = CGWindowListCopyWindowInfo(opt, kCGNullWindowID) as? [[String: Any]] else { continue }
        for w in info {
            guard (w[kCGWindowLayer as String] as? Int) == 0,
                  let owner = w[kCGWindowOwnerPID as String] as? Int,
                  let b = w[kCGWindowBounds as String] as? [String: Any],
                  let rect = CGRect(dictionaryRepresentation: b as CFDictionary),
                  rect.contains(pt) else { continue }
            if let p = pid, pid_t(owner) != p { continue }
            return (pid_t(owner), (w[kCGWindowNumber as String] as? Int) ?? 0)
        }
    }
    return nil
}

func clickPid(_ req: JSON) throws -> JSON {
    guard let x = num(req["x"]), let y = num(req["y"]) else { throw HelperError("x and y are required") }
    let pt = CGPoint(x: x, y: y)
    var pid: pid_t? = nil
    if let p = num(req["pid"]), p > 0 { pid = pid_t(p) }
    else if let name = strArg(req, "app") { pid = findApp(name) }
    let hit = windowUnder(pt, pid: pid)
    guard let target = pid ?? hit?.0 else { throw HelperError("no window at (\(x), \(y))") }
    let button = (req["button"] as? String) ?? "left"
    let count = max(1, min(3, intArg(req, "count", 1)))
    let types: (CGEventType, CGEventType, CGMouseButton)
    switch button {
    case "right": types = (.rightMouseDown, .rightMouseUp, .right)
    case "middle": types = (.otherMouseDown, .otherMouseUp, .center)
    default: types = (.leftMouseDown, .leftMouseUp, .left)
    }
    let src = CGEventSource(stateID: .privateState)
    for i in 1...count {
        guard let down = CGEvent(mouseEventSource: src, mouseType: types.0, mouseCursorPosition: pt, mouseButton: types.2),
              let up = CGEvent(mouseEventSource: src, mouseType: types.1, mouseCursorPosition: pt, mouseButton: types.2)
        else { throw HelperError("cannot create mouse event") }
        for e in [down, up] {
            e.setIntegerValueField(.mouseEventClickState, value: Int64(i))
            if let wid = hit?.1, wid > 0 {
                e.setIntegerValueField(.mouseEventWindowUnderMousePointer, value: Int64(wid))
                e.setIntegerValueField(.mouseEventWindowUnderMousePointerThatCanHandleThisEvent, value: Int64(wid))
            }
        }
        down.postToPid(target)
        usleep(20_000)
        up.postToPid(target)
        usleep(40_000)
    }
    return ["pid": Int(target), "window": hit?.1 ?? 0, "x": x, "y": y, "button": button, "count": count]
}

let kKeyCodes: [String: CGKeyCode] = [
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11,
    "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21,
    "6": 22, "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "o": 31,
    "u": 32, "[": 33, "i": 34, "p": 35, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42,
    ",": 43, "/": 44, "n": 45, "m": 46, ".": 47, "\u{60}": 50,
    "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51, "backspace": 51,
    "escape": 53, "esc": 53, "forwarddelete": 117, "home": 115, "end": 119, "pageup": 116,
    "pagedown": 121, "left": 123, "right": 124, "down": 125, "up": 126,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100,
    "f9": 101, "f10": 109, "f11": 103, "f12": 111,
]

func modifierFlags(_ mods: [String]) -> CGEventFlags {
    var f = CGEventFlags()
    for m in mods {
        switch m.lowercased() {
        case "cmd", "command", "meta", "super", "win": f.insert(.maskCommand)
        case "shift": f.insert(.maskShift)
        case "alt", "option", "opt": f.insert(.maskAlternate)
        case "ctrl", "control": f.insert(.maskControl)
        case "fn": f.insert(.maskSecondaryFn)
        default: break
        }
    }
    return f
}

func postKey(_ pid: pid_t, _ src: CGEventSource?, code: CGKeyCode, flags: CGEventFlags, unicode: [UniChar]?) throws {
    guard let down = CGEvent(keyboardEventSource: src, virtualKey: code, keyDown: true),
          let up = CGEvent(keyboardEventSource: src, virtualKey: code, keyDown: false)
    else { throw HelperError("cannot create key event") }
    down.flags = flags
    up.flags = flags
    if var u = unicode {
        down.keyboardSetUnicodeString(stringLength: u.count, unicodeString: &u)
        up.keyboardSetUnicodeString(stringLength: u.count, unicodeString: &u)
    }
    down.postToPid(pid)
    usleep(4_000)
    up.postToPid(pid)
    usleep(6_000)
}

func keyPid(_ req: JSON) throws -> JSON {
    var pid: pid_t
    var focusedRef: String? = nil
    if let ref = strArg(req, "ref") {
        let e = try resolve(ref)
        pid = e.pid
        AXUIElementSetAttributeValue(e.el, "AXFocused" as CFString, kCFBooleanTrue)
        focusedRef = ref
        usleep(30_000)
    } else {
        pid = try targetPid(req)
    }
    let mods = (req["modifiers"] as? [String]) ?? []
    let flags = modifierFlags(mods)
    let src = CGEventSource(stateID: .privateState)
    var sent = 0
    if let text = req["text"] as? String, !text.isEmpty {
        for ch in text {
            if ch == "\n" || ch == "\r\n" {
                try postKey(pid, src, code: 36, flags: flags, unicode: nil)
            } else if ch == "\t" {
                try postKey(pid, src, code: 48, flags: flags, unicode: nil)
            } else {
                try postKey(pid, src, code: 0, flags: flags, unicode: Array(String(ch).utf16))
            }
            sent += 1
        }
    } else {
        var code: CGKeyCode
        if let k = num(req["keycode"]) {
            code = CGKeyCode(k)
        } else if let name = strArg(req, "key"), let c = kKeyCodes[name.lowercased()] {
            code = c
        } else {
            throw HelperError("key_pid needs text, keycode, or a known key name")
        }
        try postKey(pid, src, code: code, flags: flags, unicode: nil)
        sent = 1
    }
    return ["pid": Int(pid), "events": sent, "ref": focusedRef ?? NSNull()]
}

// MARK: - Dispatch

func handle(_ req: JSON) throws -> Any {
    let cmd = (req["cmd"] as? String) ?? ""
    switch cmd {
    case "ping":
        return ["version": kHelperVersion, "trusted": AXIsProcessTrusted(), "pid": Int(getpid())]
    case "frontmost":
        guard let pid = frontmostPid() else { throw HelperError("no frontmost application") }
        let app = appElement(pid)
        var title = ""
        if let w = windowOf(app), let t = copyAttr(w, "AXTitle") as? String { title = t }
        let running = NSRunningApplication(processIdentifier: pid)
        return ["app": appName(pid), "pid": Int(pid),
                "bundle_id": running?.bundleIdentifier ?? "", "window_title": title]
    case "tree":
        let pid = try targetPid(req)
        return try buildTree(pid: pid, scope: (req["scope"] as? String) ?? "window",
                             maxDepth: intArg(req, "max_depth", 12), maxNodes: intArg(req, "max_nodes", 1500),
                             maxValue: intArg(req, "max_value", 300),
                             withActions: (req["actions"] as? Bool) ?? true)
    case "get":
        guard let ref = strArg(req, "ref") else { throw HelperError("ref is required") }
        let e = try resolve(ref)
        guard let raw = readNode(e.el, maxValue: intArg(req, "max_value", 20000)) else {
            throw HelperError("cannot read \(ref)")
        }
        var o = nodeJSON(raw, ref: ref, parent: nil, actions: actionNames(e.el))
        o["pid"] = Int(e.pid)
        return o
    case "perform":
        guard let ref = strArg(req, "ref") else { throw HelperError("ref is required") }
        let action = strArg(req, "action") ?? "AXPress"
        let e = try resolve(ref)
        let err = AXUIElementPerformAction(e.el, action as CFString)
        if err != .success {
            throw HelperError("\(action) failed on \(ref) [AXError \(err.rawValue)]; actions: \(actionNames(e.el))")
        }
        return ["ref": ref, "action": action, "pid": Int(e.pid)]
    case "set_value":
        guard let ref = strArg(req, "ref") else { throw HelperError("ref is required") }
        let value = (req["value"] as? String) ?? ""
        let e = try resolve(ref)
        let err = AXUIElementSetAttributeValue(e.el, "AXValue" as CFString, value as CFString)
        if err != .success { throw HelperError("set_value failed on \(ref) [AXError \(err.rawValue)]") }
        return ["ref": ref, "pid": Int(e.pid)]
    case "focus":
        guard let ref = strArg(req, "ref") else { throw HelperError("ref is required") }
        let e = try resolve(ref)
        let err = AXUIElementSetAttributeValue(e.el, "AXFocused" as CFString, kCFBooleanTrue)
        if err != .success { throw HelperError("focus failed on \(ref) [AXError \(err.rawValue)]") }
        return ["ref": ref, "pid": Int(e.pid)]
    case "element_at":
        guard let x = num(req["x"]), let y = num(req["y"]) else { throw HelperError("x and y are required") }
        return try elementAt(x: x, y: y)
    case "window_text":
        let pid = try targetPid(req)
        return try windowText(pid: pid, maxChars: max(1, intArg(req, "max_chars", 20000)))
    case "click_pid":
        return try clickPid(req)
    case "key_pid":
        return try keyPid(req)
    default:
        throw HelperError("unknown command: \(cmd)")
    }
}

func respond(_ line: String) -> Data {
    var resp: JSON = ["ok": false]
    var reqId: Any = NSNull()
    if let data = line.data(using: .utf8),
       let obj = try? JSONSerialization.jsonObject(with: data) as? JSON {
        reqId = obj["id"] ?? NSNull()
        do {
            resp = ["ok": true, "result": try handle(obj)]
        } catch let e as HelperError {
            resp = ["ok": false, "error": e.message]
        } catch {
            resp = ["ok": false, "error": "\(error)"]
        }
    } else {
        resp["error"] = "invalid JSON request"
    }
    resp["id"] = reqId
    if let out = try? JSONSerialization.data(withJSONObject: resp, options: []) { return out }
    return Data("{\"id\":null,\"ok\":false,\"error\":\"unserializable response\"}".utf8)
}

func emit(_ data: Data) {
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0a]))
}

let argv = CommandLine.arguments
let mode = argv.count > 1 ? argv[1] : "serve"
switch mode {
case "version":
    print(kHelperVersion)
case "call":
    emit(respond(argv.count > 2 ? argv[2] : "{}"))
case "serve":
    // Requests are read on a background thread and handled one at a time on
    // the main thread, whose run loop keeps NSWorkspace state current.
    DispatchQueue.global(qos: .userInitiated).async {
        while let line = readLine(strippingNewline: true) {
            if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
            let done = DispatchSemaphore(value: 0)
            DispatchQueue.main.async {
                emit(respond(line))
                done.signal()
            }
            done.wait()
        }
        exit(0)
    }
    RunLoop.main.run()
default:
    FileHandle.standardError.write(Data("usage: opendesk-ax serve | call '<json>' | version\n".utf8))
    exit(2)
}
'''


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _home() -> Path:
    override = os.environ.get("OPENDESK_HOME")
    return Path(override).expanduser() if override else Path.home() / ".opendesk"


def helper_path() -> Path:
    """Where the compiled helper lives: ``<home>/bin/opendesk-ax-v1``."""
    return _home() / "bin" / HELPER_NAME


def source_digest() -> str:
    return hashlib.sha256(SWIFT_SOURCE.encode("utf-8")).hexdigest()


def _disabled() -> bool:
    return os.environ.get("OPENDESK_NATIVE_AX", "").strip().lower() in ("0", "false", "no", "off")


_compile_lock = threading.Lock()
_compile_failed_at: float = 0.0
_compile_error: str = ""


def ensure_helper() -> Path:
    """Return the compiled helper, compiling it on first use.

    The binary is rebuilt when the embedded source changes (tracked by a
    ``.sha256`` file next to it).  Compilation writes to a temp file and
    renames it into place, so concurrent Python / JS processes are safe.

    Raises :class:`NativeAXUnavailable` when it cannot be built.
    """
    global _compile_failed_at, _compile_error
    if _PLATFORM != "Darwin":
        raise NativeAXUnavailable("the native accessibility helper is macOS-only")
    if _disabled():
        raise NativeAXUnavailable("disabled by OPENDESK_NATIVE_AX=0")
    binary = helper_path()
    stamp = binary.with_name(binary.name + ".sha256")
    digest = source_digest()

    def _fresh() -> bool:
        try:
            return (
                binary.exists() and os.access(binary, os.X_OK)
                and stamp.read_text().strip() == digest
            )
        except OSError:
            return False

    if _fresh():
        return binary
    with _compile_lock:
        if _fresh():
            return binary
        if _compile_failed_at and time.time() - _compile_failed_at < _RETRY_AFTER:
            raise NativeAXUnavailable(_compile_error or "helper compilation failed")
        swiftc = shutil.which("swiftc") or (
            "/usr/bin/swiftc" if os.path.exists("/usr/bin/swiftc") else None
        )
        if not swiftc:
            _compile_failed_at, _compile_error = time.time(), "swiftc not found (install Xcode Command Line Tools)"
            raise NativeAXUnavailable(_compile_error)
        binary.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="opendesk-ax-") as tmp:
            src = Path(tmp) / "opendesk-ax.swift"
            src.write_text(SWIFT_SOURCE, encoding="utf-8")
            out = binary.parent / f".{binary.name}.{os.getpid()}.tmp"
            try:
                r = subprocess.run(
                    [swiftc, "-O", "-o", str(out), str(src)],
                    capture_output=True, text=True, timeout=300,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                _compile_failed_at, _compile_error = time.time(), f"swiftc failed: {exc}"
                raise NativeAXUnavailable(_compile_error) from exc
            if r.returncode != 0 or not out.exists():
                out.unlink(missing_ok=True)
                _compile_failed_at = time.time()
                _compile_error = f"swiftc failed: {(r.stderr or r.stdout).strip()[:500]}"
                raise NativeAXUnavailable(_compile_error)
            os.chmod(out, 0o755)
            os.replace(out, binary)
            stamp.write_text(digest)
        _compile_failed_at, _compile_error = 0.0, ""
        return binary


# ---------------------------------------------------------------------------
# Persistent helper process
# ---------------------------------------------------------------------------


class AXHelper:
    """A persistent ``opendesk-ax serve`` process speaking JSON lines.

    Thread-safe: requests are serialised (the helper handles one at a time
    anyway).  A request that times out kills the process; the next request
    starts a fresh one.  ``argv`` overrides the command (tests use a fake
    helper written in Python).
    """

    def __init__(self, argv: Optional[Sequence[str]] = None, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._argv = list(argv) if argv else None
        self._timeout = timeout
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._responses: dict[int, dict[str, Any]] = {}
        self._next_id = 0
        self._starts = 0
        self.restarts = 0

    # -- lifecycle -------------------------------------------------------

    def _command(self) -> list[str]:
        if self._argv:
            return self._argv
        return [str(ensure_helper()), "serve"]

    def _start(self) -> subprocess.Popen[bytes]:
        cmd = self._command()
        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0,
            )
        except OSError as exc:
            raise NativeAXUnavailable(f"cannot start helper: {exc}") from exc
        if self._starts:
            self.restarts += 1
        self._starts += 1
        threading.Thread(
            target=self._reader, args=(proc,), name="opendesk-ax-reader", daemon=True,
        ).start()
        self._proc = proc
        return proc

    def _reader(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            rid = msg.get("id")
            if isinstance(rid, int):
                with self._cond:
                    self._responses[rid] = msg
                    self._cond.notify_all()
        with self._cond:
            self._cond.notify_all()

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is None:
                return
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None and self._proc.poll() is None else None

    # -- requests --------------------------------------------------------

    def request(self, cmd: str, *, timeout: Optional[float] = None, **args: Any) -> Any:
        """Send one command and return its ``result``.

        Raises :class:`NativeAXError` when the helper reports an error or
        times out, :class:`NativeAXUnavailable` when it cannot run.
        """
        timeout = self._timeout if timeout is None else timeout
        payload = {k: v for k, v in args.items() if v is not None}
        with self._lock:
            for attempt in (0, 1):
                proc = self._proc
                if proc is None or proc.poll() is not None:
                    proc = self._start()
                self._next_id += 1
                rid = self._next_id
                line = json.dumps({"id": rid, "cmd": cmd, **payload}) + "\n"
                try:
                    assert proc.stdin is not None
                    proc.stdin.write(line.encode("utf-8"))
                    proc.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    # Died before it saw the request: safe to restart and resend.
                    self._kill()
                    if attempt:
                        raise NativeAXError("helper exited")
                    continue
                msg = self._wait(proc, rid, timeout)
                if msg is None:
                    if proc.poll() is not None:
                        self._proc = None
                        raise NativeAXError(f"helper exited during {cmd!r}")
                    self._kill()
                    raise NativeAXError(f"helper timed out after {timeout:.0f}s on {cmd!r}")
                if not msg.get("ok"):
                    raise NativeAXError(str(msg.get("error") or "unknown helper error"))
                return msg.get("result")
        raise NativeAXError("helper unavailable")  # pragma: no cover

    def _wait(self, proc: subprocess.Popen[bytes], rid: int, timeout: float) -> Optional[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        with self._cond:
            while rid not in self._responses:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or proc.poll() is not None:
                    break
                self._cond.wait(min(remaining, 0.25))
            msg = self._responses.pop(rid, None)
            # Drop anything stale (responses to timed-out requests).
            for k in [k for k in self._responses if k < rid]:
                self._responses.pop(k, None)
            return msg


_client: Optional[AXHelper] = None
_client_lock = threading.Lock()
_status: Optional[bool] = None
_status_checked_at: float = 0.0


def _get_client() -> AXHelper:
    global _client
    with _client_lock:
        if _client is None:
            _client = AXHelper()
        return _client


def set_client(client: Optional[AXHelper]) -> None:
    """Replace the shared helper client (tests inject a fake helper)."""
    global _client, _status, _status_checked_at
    with _client_lock:
        old, _client = _client, client
        _status, _status_checked_at = None, 0.0
    if old is not None and old is not client:
        old.close()


@atexit.register
def _shutdown() -> None:
    if _client is not None:
        _client.close()


def _call(cmd: str, *, timeout: Optional[float] = None, **args: Any) -> Any:
    if _client is None and (_PLATFORM != "Darwin" or _disabled()):
        raise NativeAXUnavailable("the native accessibility helper is macOS-only")
    return _get_client().request(cmd, timeout=timeout, **args)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available() -> bool:
    """True when the helper runs and has Accessibility permission.

    The answer is cached; a negative answer is re-checked after a while so
    granting permission later is picked up without a restart.
    """
    global _status, _status_checked_at
    now = time.time()
    if _status is not None and (_status or now - _status_checked_at < 30.0):
        return _status
    try:
        info = _call("ping")
        ok = bool(isinstance(info, dict) and info.get("trusted"))
    except NativeAXError:
        ok = False
    _status, _status_checked_at = ok, now
    return ok


def ping() -> dict[str, Any]:
    return _call("ping")


def frontmost() -> Optional[dict[str, Any]]:
    """``{app, pid, bundle_id, window_title}`` of the frontmost app, or None."""
    try:
        return _call("frontmost")
    except NativeAXError:
        return None


def tree(
    pid: Optional[int] = None,
    app: Optional[str] = None,
    *,
    max_depth: int = 12,
    max_nodes: int = 1500,
    max_value: int = 300,
    scope: str = "window",
    actions: bool = True,
    timeout: float = TREE_TIMEOUT,
) -> dict[str, Any]:
    """Read the accessibility tree of ``pid`` / ``app`` (default: frontmost).

    Returns ``{pid, app, window_title, truncated, nodes: [...]}`` where each
    node is ``{ref, parent, role, subrole?, title?, value?, description?,
    identifier?, frame?{x,y,w,h}, enabled, focused, actions?}`` in
    breadth-first order.  ``scope="app"`` roots the tree at the application
    (menu bar + all windows) instead of the focused window.  ``max_depth``
    counts meaningful levels only: unlabelled layout containers (AXGroup,
    AXScrollArea, AXSplitGroup, ...) do not add depth.
    """
    return _call(
        "tree", timeout=timeout, pid=pid, app=app, max_depth=max_depth,
        max_nodes=max_nodes, max_value=max_value, scope=scope, actions=actions,
    )


def get(ref: str, *, max_value: int = 20000) -> dict[str, Any]:
    """Fresh attributes of one element (value up to ``max_value`` chars)."""
    return _call("get", ref=ref, max_value=max_value)


def perform(ref: str, action: str = "AXPress") -> dict[str, Any]:
    """Invoke an AX action (``AXPress``, ``AXShowMenu``, ``AXConfirm``...)."""
    return _call("perform", ref=ref, action=action)


def set_value(ref: str, value: str) -> dict[str, Any]:
    """Set ``AXValue`` of a text field / slider / ... directly."""
    return _call("set_value", ref=ref, value=value)


def focus(ref: str) -> dict[str, Any]:
    """Give keyboard focus to the element (within its app)."""
    return _call("focus", ref=ref)


def element_at(x: float, y: float) -> Optional[dict[str, Any]]:
    """The element at screen point (x, y) in logical pixels, with its ``ref``,
    ``pid``, ``app`` and ``pressable`` (the ref of the nearest element that
    supports AXPress, often the button around a label).  None if nothing."""
    try:
        return _call("element_at", x=x, y=y)
    except NativeAXError as exc:
        if isinstance(exc, NativeAXUnavailable):
            raise
        return None


def click_pid(
    x: float, y: float, *, pid: Optional[int] = None, app: Optional[str] = None,
    button: str = "left", count: int = 1,
) -> dict[str, Any]:
    """Post a click at (x, y) straight to a process (``CGEvent.postToPid``).

    The user's cursor does not move and the target app is not activated.
    With no ``pid`` / ``app`` the owner of the window under the point is
    used.  Delivery is best-effort: some apps ignore synthetic events while
    in the background.
    """
    return _call("click_pid", x=x, y=y, pid=pid, app=app, button=button, count=count)


def key_pid(
    *, pid: Optional[int] = None, app: Optional[str] = None, ref: Optional[str] = None,
    text: Optional[str] = None, key: Optional[str] = None, keycode: Optional[int] = None,
    modifiers: Iterable[str] = (),
) -> dict[str, Any]:
    """Post keystrokes straight to a process without activating it.

    Give ``text`` (typed character by character, Unicode-safe) or a ``key``
    name (``return``, ``tab``, ``a``, ``f5``...) / raw ``keycode`` plus
    ``modifiers`` (``cmd``, ``shift``, ``alt``, ``ctrl``, ``fn``).  With
    ``ref`` the element is focused first and its process is the target.
    """
    return _call(
        "key_pid", pid=pid, app=app, ref=ref, text=text, key=key, keycode=keycode,
        modifiers=list(modifiers) or None,
    )


def window_text(pid: int | None = None, max_chars: int = 20000) -> str | None:
    """Visible text of the frontmost (or ``pid``'s) window, in reading order.

    Concatenates static text, titles and text-field / text-area values,
    deduplicated, one item per line, capped at ``max_chars``.  Returns
    ``None`` when the helper is unavailable (not macOS, compile failure, no
    Accessibility permission) or the read fails.
    """
    try:
        if not available():
            return None
        result = _call("window_text", pid=pid, max_chars=max_chars)
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    text = result.get("text")
    return text if isinstance(text, str) else None


# ---------------------------------------------------------------------------
# Conversion to UIElement
# ---------------------------------------------------------------------------


def node_name(node: dict[str, Any]) -> str:
    """The label an agent would use: title, else description, else the value
    of a static text."""
    name = node.get("title") or node.get("description") or ""
    if not name and node.get("role") == "AXStaticText":
        name = node.get("value") or ""
    return str(name)


def to_ui_element(result: dict[str, Any]) -> Any:
    """Build a :class:`~opendesk.computer.types.UIElement` tree from a
    :func:`tree` result.  Every node carries its ``ref``."""
    from opendesk.computer.types import Rect, UIElement

    nodes = result.get("nodes") or []
    app = result.get("app") or ""
    pid = result.get("pid")
    built: dict[str, UIElement] = {}
    root: Optional[UIElement] = None
    for n in nodes:
        frame = n.get("frame")
        bounds = None
        if isinstance(frame, dict):
            bounds = Rect(
                x=float(frame.get("x", 0)), y=float(frame.get("y", 0)),
                width=float(frame.get("w", 0)), height=float(frame.get("h", 0)),
            )
        meta: dict[str, Any] = {}
        for key in ("subrole", "identifier", "description"):
            if n.get(key):
                meta[key] = n[key]
        el = UIElement(
            role=str(n.get("role") or "?"),
            name=node_name(n),
            value=n.get("value"),
            bounds=bounds,
            focused=bool(n.get("focused")),
            enabled=bool(n.get("enabled", True)),
            actions=list(n.get("actions") or []),
            metadata=meta,
            ref=n.get("ref"),
        )
        parent = n.get("parent")
        if parent and parent in built:
            built[parent].children.append(el)
        elif root is None:
            root = el
        if el.ref:
            built[el.ref] = el
    if root is None:
        root = UIElement(role="window", name=result.get("window_title") or app)
    root.metadata.update({
        "app": app, "pid": pid, "source": "opendesk-ax",
        "truncated": bool(result.get("truncated")),
    })
    return root


__all__ = [
    "AXHelper", "NativeAXError", "NativeAXUnavailable", "SWIFT_SOURCE",
    "available", "click_pid", "element_at", "ensure_helper", "focus", "frontmost",
    "get", "helper_path", "key_pid", "node_name", "perform", "ping", "set_client",
    "set_value", "source_digest", "to_ui_element", "tree", "window_text",
]
