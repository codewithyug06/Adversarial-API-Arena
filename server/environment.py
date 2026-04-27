from __future__ import annotations
import json
import logging
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from action import TaskAction, AdversaryAction
from observation import TaskObservation, AdversaryObservation, StepResult
from state import EpisodeState, ManipulationRecord
from server.rewards import RewardEngine, evaluate_correctness
from server.task_bank import TaskBank
from server.mock_api import MockAPI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suspicion EMA parameters
# ---------------------------------------------------------------------------
_EMA_ALPHA = 0.4               # higher α → more weight on latest manipulation
_SPIKE_THRESHOLD = 0.85        # manipulation_degree above this adds a spike bonus
_SPIKE_BONUS = 0.30            # added to suspicion on spike

# Default log directory (relative to working directory)
_DEFAULT_LOG_DIR = Path("outputs/logs")


class AdversarialArenaEnvironment:
    """
    Core environment implementing the Adversarial API Arena.

    Thread-safe: each public method acquires self._lock.
    Supports multiple concurrent sessions via the `session_id` concept —
    but a single instance tracks ONE active episode. For parallel training
    workers, run one server process per worker (simplest and most correct).

    Anti-reward-hacking guarantees:
      - ground_truth never leaves the server (not in any TaskObservation)
      - step_count incremented server-side only, never accepted from client
      - manipulation_degree validated by Pydantic before reaching this class
      - pending adversary action consumed exactly once per query step
      - all terminal reward computation happens here, not in app.py
      - episode log written only on termination (append-only during episode)
    """

    def __init__(
        self,
        max_steps: int = 20,
        episode_timeout_seconds: float = 300.0,
        suspicion_threshold: float = 0.7,
        log_dir: Path = _DEFAULT_LOG_DIR,
        task_bank: Optional[TaskBank] = None,
        seed: Optional[int] = None,
    ):
        self._max_steps = max_steps
        self._timeout = episode_timeout_seconds
        self._reward_engine = RewardEngine(suspicion_threshold=suspicion_threshold)
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._task_bank = task_bank or TaskBank(seed=seed)
        self._mock_api = MockAPI()
        self._rng = random.Random(seed)

        self._state: Optional[EpisodeState] = None
        self._episode_log: list[dict] = []
        self._episode_start_time: float = 0.0
        self._lock = threading.Lock()

    # ==========================================================================
    # Standard OpenEnv interface — task agent path
    # ==========================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> TaskObservation:
        """
        Sample a new task and reset all episode state.
        Returns the initial TaskObservation (reward=0.0, no history).
        """
        with self._lock:
            if seed is not None:
                self._task_bank = TaskBank(seed=seed)
                self._rng = random.Random(seed)

            task = self._task_bank.sample()
            eid = episode_id or str(uuid.uuid4())

            self._state = EpisodeState(
                episode_id=eid,
                task_description=task.task_description,
                ground_truth=task.ground_truth,
                correctness_mode=task.correctness_mode,
                max_steps=self._max_steps,
            )
            self._episode_log = []
            self._episode_start_time = time.monotonic()

            logger.info(
                "Episode %s started | domain=%s | task=%s",
                eid, task.domain, task.task_description,
            )
            return self._state.to_task_observation(reward=0.0)

    def step(self, action: TaskAction) -> StepResult:
        """
        Execute one task-agent step.
        Returns StepResult(observation, reward, terminated, truncated, info).
        """
        with self._lock:
            s = self._require_state()

            if s.done:
                return s.to_step_result(reward=0.0)

            # Wall-clock timeout check — prevents infinite-loop abuse
            elapsed = time.monotonic() - self._episode_start_time
            if elapsed > self._timeout:
                s.truncated = True
                s.termination_reason = "wall_clock_timeout"
                r = self._reward_engine.task_terminal_reward(
                    final_answer=None, ground_truth=s.ground_truth,
                    correctness_mode=s.correctness_mode, accusation_made=False,
                    manipulation_occurred=bool(s.manipulation_log), truncated=True,
                )
                self._flush_log(s)
                return s.to_step_result(reward=r.total)

            s.step_count += 1
            step_r = self._reward_engine.task_step_reward()

            if action.action_type == "query":
                return self._handle_query(s, action, step_r.total)
            if action.action_type == "final_answer":
                return self._handle_final_answer(s, action, step_r.total)
            if action.action_type == "accuse":
                return self._handle_accusation(s, action, step_r.total)

            # Unknown action type — treat as no-op (shouldn't reach here after Pydantic)
            return s.to_step_result(reward=step_r.total)

    # ==========================================================================
    # Adversary interface — separate from task-agent path
    # ==========================================================================

    def adversary_step(self, action: AdversaryAction) -> AdversaryObservation:
        """
        Stage an adversary action for the NEXT query the task agent makes.
        Must be called before the task agent's next /step with action_type="query".
        If not called, the mock API defaults to truthful.
        """
        with self._lock:
            s = self._require_state()
            if s.done:
                return s.to_adversary_observation(reward=0.0)

            s._pending_adversary_action = {
                "response_mode": action.response_mode,
                "manipulation_degree": action.manipulation_degree,
                "corrupted_response": action.corrupted_response,
            }
            r = self._reward_engine.adversary_step_reward()
            return s.to_adversary_observation(reward=r.total)

    def adversary_observe(self) -> AdversaryObservation:
        """Non-destructive read of the adversary observation (GET /adversary/state)."""
        with self._lock:
            s = self._require_state()
            return s.to_adversary_observation(reward=0.0)

    # ==========================================================================
    # Internal handlers
    # ==========================================================================

    def _handle_query(
        self, s: EpisodeState, action: TaskAction, step_reward: float
    ) -> StepResult:
        s._latest_query = action.query

        # Consume staged adversary action (default: truthful)
        adv = s._pending_adversary_action or {
            "response_mode": "truthful",
            "manipulation_degree": 0.0,
            "corrupted_response": None,
        }
        s._pending_adversary_action = None  # consumed — exactly once

        # Get mock API response
        response = self._mock_api.get_response(
            query=action.query,
            ground_truth=s.ground_truth,
            manipulation_degree=adv["manipulation_degree"],
            corrupted_response=adv.get("corrupted_response"),
            response_mode=adv["response_mode"],
            rng=self._rng,
        )

        # Update suspicion score (EMA + spike rule)
        deg: float = adv["manipulation_degree"]
        s.suspicion_score = (
            _EMA_ALPHA * deg + (1.0 - _EMA_ALPHA) * s.suspicion_score
        )
        if deg > _SPIKE_THRESHOLD:
            s.suspicion_score = min(1.0, s.suspicion_score + _SPIKE_BONUS)

        # Log manipulation record (only when mode is "manipulated")
        if adv["response_mode"] == "manipulated":
            s.manipulation_log.append(
                ManipulationRecord(
                    step=s.step_count,
                    degree=deg,
                    response_mode="manipulated",
                    corrupted_response=adv.get("corrupted_response"),
                )
            )

        # Append to public history (query + response only — no adversary metadata)
        s.api_history.append({
            "step": s.step_count,
            "query": action.query,
            "response": response,
        })

        # Step limit check
        if s.step_count >= s.max_steps:
            s.truncated = True
            s.termination_reason = "max_steps"
            terminal_r = self._reward_engine.task_terminal_reward(
                final_answer=None, ground_truth=s.ground_truth,
                correctness_mode=s.correctness_mode, accusation_made=False,
                manipulation_occurred=bool(s.manipulation_log), truncated=True,
            )
            total_reward = step_reward + terminal_r.timeout_penalty
            self._log_step(s, action, response, total_reward, adv)
            self._flush_log(s)
            return s.to_step_result(reward=total_reward)

        self._log_step(s, action, response, step_reward, adv)
        return s.to_step_result(reward=step_reward)

    def _handle_final_answer(
        self, s: EpisodeState, action: TaskAction, step_reward: float
    ) -> StepResult:
        manipulation_occurred = bool(s.manipulation_log)
        terminal_r = self._reward_engine.task_terminal_reward(
            final_answer=action.answer,
            ground_truth=s.ground_truth,
            correctness_mode=s.correctness_mode,
            accusation_made=False,
            manipulation_occurred=manipulation_occurred,
            truncated=False,
        )
        total_reward = step_reward + terminal_r.correctness
        s.terminated = True
        s.termination_reason = "final_answer"
        self._log_step(s, action, None, total_reward, None)
        self._flush_log(s)
        return s.to_step_result(reward=total_reward)

    def _handle_accusation(
        self, s: EpisodeState, action: TaskAction, step_reward: float
    ) -> StepResult:
        manipulation_occurred = bool(s.manipulation_log)
        terminal_r = self._reward_engine.task_terminal_reward(
            final_answer=None,
            ground_truth=s.ground_truth,
            correctness_mode=s.correctness_mode,
            accusation_made=True,
            manipulation_occurred=manipulation_occurred,
            truncated=False,
        )
        total_reward = step_reward + terminal_r.accusation_bonus
        s.terminated = True
        s.termination_reason = "accusation"
        self._log_step(s, action, None, total_reward, None)
        self._flush_log(s)
        return s.to_step_result(reward=total_reward)

    # ==========================================================================
    # Logging — append-only during episode, flushed to disk on termination
    # ==========================================================================

    def _log_step(
        self,
        s: EpisodeState,
        action: TaskAction,
        response: Optional[str],
        reward: float,
        adv_action: Optional[dict],
    ) -> None:
        entry = {
            "episode_id": s.episode_id,
            "step": s.step_count,
            "action_type": action.action_type,
            "query": getattr(action, "query", None),
            "answer": getattr(action, "answer", None),
            "response": response,
            "reward": reward,
            "suspicion_score": s.suspicion_score,
            "adversary_mode": adv_action.get("response_mode") if adv_action else None,
            "manipulation_degree": adv_action.get("manipulation_degree") if adv_action else None,
        }
        self._episode_log.append(entry)

    def _flush_log(self, s: EpisodeState) -> None:
        path = self._log_dir / f"{s.episode_id}.json"
        try:
            with open(path, "w") as f:
                json.dump(self._episode_log, f, indent=2)
            logger.info("Episode %s log flushed to %s", s.episode_id, path)
        except OSError as e:
            logger.warning("Failed to flush episode log: %s", e)

    # ==========================================================================
    # Helpers
    # ==========================================================================

    def _require_state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError(
                "reset() must be called before step(). "
                "Call POST /reset to start a new episode."
            )
        return self._state

    @property
    def current_episode_id(self) -> Optional[str]:
        return self._state.episode_id if self._state else None

    def get_episode_log(self) -> list[dict]:
        """Read-only snapshot of the current episode log."""
        with self._lock:
            return list(self._episode_log)

    def compute_adversary_terminal_reward(self) -> float:
        """
        Called by app.py at episode end to compute and return adversary terminal reward.
        Must be called AFTER the task agent's terminal step.
        """
        with self._lock:
            s = self._require_state()
            task_succeeded = False
            task_accused = s.termination_reason == "accusation"

            # Re-evaluate correctness from log if possible
            if s.termination_reason == "final_answer" and self._episode_log:
                last = self._episode_log[-1]
                if last.get("answer"):
                    task_succeeded = evaluate_correctness(
                        last["answer"], s.ground_truth, s.correctness_mode
                    )

            r = self._reward_engine.adversary_terminal_reward(
                task_succeeded=task_succeeded,
                task_accused=task_accused,
                suspicion_score=s.suspicion_score,
                manipulation_occurred=bool(s.manipulation_log),
            )
            return r.total
