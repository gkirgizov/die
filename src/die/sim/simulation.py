"""The machine level: the agent-env loop with pluggable observers."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch as th

from die.agents.base import Agent
from die.env import Env
from die.types import ActType, ObsType

if TYPE_CHECKING:
    from die.sim.observers import SimObserver

FrameSourceFn = Callable[[], list[np.ndarray]]


@dataclass(frozen=True)
class StepEvent:
    """A single simulation step as seen by observers.

    NOTE: ``obs`` and ``action`` alias live env arrays --
    observers must not mutate them.
    """
    step: int
    obs: ObsType
    action: ActType
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class RunResult:
    num_steps: int
    total_reward: float
    terminated: bool
    final_info: dict[str, Any]


class Simulation:
    """Owns the agent-env step loop and notifies observers.

    Presentation (interactive view, animation recording, metrics logging)
    plugs in via the :class:`~die.sim.observers.SimObserver` interface,
    keeping the loop itself headless and reusable (interactive runs,
    training evaluations, tests).
    """

    def __init__(self,
                 env: Env,
                 agent: Agent,
                 observers: Sequence[SimObserver] = (),
                 seed: int | None = None,
                 frame_source: FrameSourceFn | None = None):
        self.env = env
        self.agent = agent
        self.observers = list(observers)
        self._seed = seed
        self._frame_source = frame_source
        self._frame_cache: list[np.ndarray] | None = None

    def reset(self, seed: int | None = None) -> ObsType:
        """Seeds all global RNGs (if a seed is given) and resets the env."""
        seed = seed if seed is not None else self._seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            th.manual_seed(seed)
        obs, _ = self.env.reset(seed=seed)
        self._frame_cache = None
        return obs

    def run(self, num_steps: int, *,
            reset: bool = True,
            stop_on_termination: bool = True) -> RunResult:
        obs = self.reset() if reset else self.env.observe()
        total_reward = 0.0
        terminated = False
        info: dict[str, Any] = {}
        steps_done = 0

        for observer in self.observers:
            observer.on_run_start(self, num_steps)
        try:
            for step in range(num_steps):
                action = self.agent.forward(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps_done += 1
                self._frame_cache = None
                event = StepEvent(step=step, obs=obs, action=action, reward=reward,
                                  terminated=terminated, truncated=truncated, info=info)
                for observer in self.observers:
                    observer.on_step(self, event)
                if terminated and stop_on_termination:
                    break
            result = RunResult(num_steps=steps_done,
                               total_reward=total_reward,
                               terminated=terminated,
                               final_info=info)
            for observer in self.observers:
                observer.on_run_end(self, result)
            return result
        finally:
            for observer in self.observers:
                observer.close()

    def render_frames(self) -> list[np.ndarray]:
        """Current visual frames (env panes + agent self-render), as RGB arrays.

        Lazy and memoized per step: frames are produced only if some observer
        asks, and only once per step even with several consuming observers.
        """
        if self._frame_cache is None:
            if self._frame_source is None:
                self._frame_source = self._default_frame_source()
            self._frame_cache = self._frame_source()
        return self._frame_cache

    def _default_frame_source(self) -> FrameSourceFn:
        # Lazy import: keeps `die.sim` importable without pulling matplotlib
        from die.view.render import FrameSource
        return FrameSource(self.env, self.agent)
