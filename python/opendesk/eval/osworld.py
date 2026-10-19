"""OSWorld adapter — evaluate OpenDesk on OSWorld's Ubuntu VM tasks.

.. warning::
   **Untested end-to-end.**  No OSWorld VM was available while this was
   written; the agent loop, the native control arm and the subset sampler are
   unit-tested with fakes only.  Treat the first real run as a shake-down.

Design
------
1. Inside the OSWorld Ubuntu VM run ``opendesk serve`` (``pip install
   'opendesk[core,remote,linux]'``; bake it into the VM snapshot so every task
   reset keeps it).  Pair it once from the host: ``opendesk pair`` in the VM,
   ``opendesk pair-with <vm-ip> <code> --name osworld-vm`` on the host.
2. Point OSWorld's runner at :class:`OpenDeskOSWorldAgent` instead of one of
   its ``mm_agents``.  OSWorld still owns the VM lifecycle (snapshot revert,
   task setup via ``env.reset(task_config=...)``) and scoring
   (``env.evaluate()``).
3. ``predict()`` runs the *whole* Claude loop on its first call, driving the
   VM through :class:`~opendesk.computer.remote.RemoteComputer` (the paired
   peer), and returns ``["DONE"]`` — or ``["FAIL"]`` when the model judged the
   task infeasible (OSWorld scores ``FAIL`` as correct on infeasible tasks).
   The step budget is enforced as the model-turn limit.
4. A/B: ``arm="opendesk"`` gives Claude the OpenDesk tools; ``arm="native"``
   gives the *same model, same step budget, same RemoteComputer* Anthropic's
   native computer-use tool.  Only the tool surface differs.

Protocol: a 50–100 task stratified subset of ``test_all.json``
(:func:`stratified_subset`, proportional per domain, ≥1 per domain) × 3 seeds
(independent repeats) × 2 arms, model held constant.  Report per-domain
success, steps and tokens.  For macOS, MacArena (421 macOS tasks) is the
analogous benchmark; the same agent class applies with a Mac peer.

Wiring into OSWorld's ``run.py`` (sketch)::

    from opendesk.eval.osworld import OpenDeskOSWorldAgent
    agent = OpenDeskOSWorldAgent(model=args.model, arm="opendesk",
                                 max_steps=args.max_steps, peer="osworld-vm")
    # then lib_run_single.run_single_example(agent, env, example, args.max_steps, ...)
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any, Callable, Optional

from opendesk.eval.agents import (
    EVAL_SYSTEM_PROMPT, NATIVE_TOOLSET, AgentRun, run_native_computer_loop, run_opendesk_loop,
)

INFEASIBLE_MARKER = "INFEASIBLE"

OSWORLD_SYSTEM_PROMPT = (
    EVAL_SYSTEM_PROMPT
    + " The computer is an Ubuntu virtual machine. If the task cannot be completed on this "
    f"machine, stop and reply with the single word {INFEASIBLE_MARKER}."
)


class OpenDeskOSWorldAgent:
    """OSWorld ``mm_agents``-style agent (``reset()`` / ``predict()``)."""

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        *,
        arm: str = "opendesk",
        max_steps: int = 50,
        peer: str = "osworld-vm",
        home: Optional[str] = None,
        native_tool_version: str = NATIVE_TOOLSET,
        client_factory: Optional[Callable[[], Any]] = None,
        computer_factory: Optional[Callable[[], Any]] = None,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
    ) -> None:
        if arm not in ("opendesk", "native"):
            raise ValueError("arm must be 'opendesk' or 'native'")
        self.model = model
        self.arm = arm
        self.max_steps = max_steps
        self.peer = peer
        self.home = home
        self.native_tool_version = native_tool_version
        self.action_space = action_space
        self.observation_type = observation_type
        self._client_factory = client_factory
        self._computer_factory = computer_factory
        self._finished = False
        self.last_run: Optional[AgentRun] = None

    # -- OSWorld interface -------------------------------------------------

    def reset(self, _logger: Any = None, *args: Any, **kwargs: Any) -> None:
        self._finished = False
        self.last_run = None

    def predict(self, instruction: str, obs: dict) -> tuple[str, list[str]]:
        """Run the full episode on the first call; afterwards just report DONE."""
        if self._finished:
            return "episode already finished", ["DONE"]
        run = asyncio.run(self.run_episode(instruction))
        self._finished = True
        self.last_run = run
        if run.error:
            return f"error: {run.error}", ["FAIL"]
        if INFEASIBLE_MARKER in (run.final_text or "").upper():
            return run.final_text, ["FAIL"]
        return run.final_text, ["DONE"]

    # -- episode ------------------------------------------------------------

    async def _computer(self) -> Any:
        if self._computer_factory is not None:
            comp = self._computer_factory()
            return await comp if asyncio.iscoroutine(comp) else comp
        from opendesk.remote.client import connect
        return await connect(self.peer, home=Path(self.home) if self.home else None)

    def _client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        import anthropic
        return anthropic.AsyncAnthropic()

    async def run_episode(self, instruction: str) -> AgentRun:
        try:
            computer = await self._computer()
        except Exception as exc:
            return AgentRun(error=f"cannot reach OpenDesk peer {self.peer!r}: {exc}")
        try:
            client = self._client()
            if self.arm == "opendesk":
                return await run_opendesk_loop(
                    client, model=self.model, instruction=instruction, computer=computer,
                    max_turns=self.max_steps, system=OSWORLD_SYSTEM_PROMPT, session_id="osworld",
                )
            return await run_native_computer_loop(
                client, model=self.model, instruction=instruction, computer=computer,
                max_turns=self.max_steps, system=OSWORLD_SYSTEM_PROMPT,
                tool_version=self.native_tool_version,
            )
        finally:
            close = getattr(computer, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Stratified subset of OSWorld's test_all.json  ({domain: [example ids]})
# ---------------------------------------------------------------------------


def stratified_subset(test_all: dict[str, list[str]], n: int = 75, seed: int = 0) -> dict[str, list[str]]:
    """Proportional per-domain sample of *n* tasks (≥1 per non-empty domain).

    Uses largest-remainder allocation and a seeded RNG, so the same
    ``(test_all, n, seed)`` always yields the same subset.
    """
    domains = {d: sorted(ids) for d, ids in sorted(test_all.items()) if ids}
    total = sum(len(v) for v in domains.values())
    if total == 0:
        return {}
    n = min(n, total)
    if n < len(domains):
        raise ValueError(f"n={n} is smaller than the number of domains ({len(domains)})")
    quotas = {d: n * len(ids) / total for d, ids in domains.items()}
    alloc = {d: max(1, int(q)) for d, q in quotas.items()}
    # Fix rounding: add to the largest remainders / remove from the largest allocations.
    while sum(alloc.values()) < n:
        d = max(
            (d for d in domains if alloc[d] < len(domains[d])),
            key=lambda d: (quotas[d] - alloc[d], d),
        )
        alloc[d] += 1
    while sum(alloc.values()) > n:
        d = max((d for d in domains if alloc[d] > 1), key=lambda d: (alloc[d] - quotas[d], d))
        alloc[d] -= 1
    rng = random.Random(seed)
    return {d: sorted(rng.sample(ids, min(alloc[d], len(ids)))) for d, ids in domains.items()}


def main(argv: Optional[list[str]] = None) -> None:
    """``python -m opendesk.eval.osworld subset --test-all PATH --n 75 --seed 0 --out subset.json``"""
    import argparse

    p = argparse.ArgumentParser(prog="python -m opendesk.eval.osworld")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("subset", help="Write a stratified subset of OSWorld's test_all.json")
    s.add_argument("--test-all", required=True)
    s.add_argument("--n", type=int, default=75)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.cmd != "subset":
        p.print_help()
        return
    subset = stratified_subset(json.loads(Path(args.test_all).read_text()), n=args.n, seed=args.seed)
    Path(args.out).write_text(json.dumps(subset, indent=2))
    print(f"Wrote {sum(len(v) for v in subset.values())} tasks over {len(subset)} domains to {args.out}")


if __name__ == "__main__":
    main()
