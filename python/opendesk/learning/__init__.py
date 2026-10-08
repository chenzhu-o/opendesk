"""Learning layer — verifiable rewards, goal-state anchoring, rollout export.

This package holds the *environment* half of reinforcement learning for
computer-use agents.  It does not train models: given a task it produces
machine-checkable rewards, step-level process signals, and trajectory records
in a format training pipelines can consume.

Modules
-------
``rewards``      declarative, verifiable success criteria (RLVR)
``goal_state``   goal-state anchoring — score an end state against a reference
``process``      step-level process rewards derived from action effects
``trajectories`` episode lifecycle and RL-ready trajectory export
``preference``   best-of-N rollouts and preference pairs
``assertions``   claims the agent makes about the state, verified and scored
``training``     dataset rows for a trainer (SFT / DPO / GRPO)
``action_rewards`` verifiable per-step action scores (GUI clicks, *Tools.*)
``action_space``    legacy OSWorld → tier-1 UI / Tools mapping + sample weights
``harness_evolution`` evidence reports and suggested HarnessProfile patches

Nothing here calls a model.  Every signal is derived from observable state
(files, shell output, the accessibility tree, screenshots).
"""

from __future__ import annotations

__all__ = [
    "rewards", "goal_state", "process", "trajectories", "preference",
    "assertions", "training", "action_rewards", "action_space",
    "harness_evolution",
]
