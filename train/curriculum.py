"""
curriculum.py
=============
Curriculum manager: automatically adjusts task difficulty based on reward signal.

Principle: if the model never sees success, it cannot learn from it.
Start with tasks and adversary conditions where reward > 0 is achievable,
then tighten difficulty as the model improves.

Usage in training loop:
    curriculum = CurriculumManager()

    for step in range(STEPS):
        level = curriculum.current_level
        config = level.to_rollout_config()

        rollout = EnvironmentRollout(env_url=..., **config)
        trajectories = rollout.run_batch(...)

        curriculum.record_batch(trajectories)
        if curriculum.should_advance():
            curriculum.advance()
        elif curriculum.should_simplify():
            curriculum.simplify()

        print(curriculum.status())

Levels (difficulty 0 → 3):
  0  Warmup:    exact-match tasks, truthful adversary, 5 steps
  1  Easy:      all tasks, adversary mild (≤0.3), 10 steps
  2  Medium:    all tasks, adversary moderate (≤0.6), 15 steps
  3  Hard:      all tasks, adversary aggressive (up to 1.0), 20 steps (original)
"""
from __future__ import annotations
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.task_bank import TaskBank, TaskTemplate, TASK_TEMPLATES


# ---------------------------------------------------------------------------
# Level definitions
# ---------------------------------------------------------------------------

@dataclass
class CurriculumLevel:
    level_id: int
    name: str
    description: str

    # Task bank filtering
    allowed_domains: list[str]               # domains to sample from
    allowed_correctness_modes: list[str]     # "exact", "fuzzy", "semantic"

    # Environment config
    max_steps: int
    episode_timeout: float

    # Adversary behaviour
    adversary_mode: str                      # "truthful" | "mild" | "moderate" | "aggressive"
    adversary_max_degree: float              # cap on manipulation_degree
    adversary_manipulation_probability: float  # P(adversary manipulates any given step)

    # Advancement thresholds (rolling window)
    window_size: int = 20
    advance_threshold: float = 0.55          # positive_rate needed to advance
    simplify_threshold: float = 0.10         # positive_rate below which to simplify

    def to_rollout_config(self) -> dict:
        """Returns kwargs for EnvironmentRollout constructor."""
        return {
            "adversary_mode": self._adversary_mode_label(),
            "fixed_manipulation_degree": self.adversary_max_degree,
        }

    def to_env_overrides(self) -> dict:
        """Query-param overrides for POST /reset."""
        return {}  # max_steps is set at server start, not per-episode in current API

    def _adversary_mode_label(self) -> str:
        if self.adversary_mode == "truthful":
            return "truthful"
        if self.adversary_manipulation_probability < 0.5:
            return "random"  # low prob → mostly truthful
        return "fixed"

    def build_task_bank(self, seed: Optional[int] = None) -> TaskBank:
        """Return a TaskBank filtered to this level's allowed templates."""
        allowed = [
            t for t in TASK_TEMPLATES
            if t.domain in self.allowed_domains
            and t.correctness_mode in self.allowed_correctness_modes
        ]
        if not allowed:
            # Fallback: return full bank
            allowed = TASK_TEMPLATES
        return TaskBank(templates=allowed, seed=seed)


# Level 0: Warmup
# Exact-match only (historical years, atomic numbers, founding years).
# Adversary always truthful. Only 5 steps needed.
LEVEL_0 = CurriculumLevel(
    level_id=0,
    name="Warmup",
    description="Exact-match facts, truthful API, 5 steps",
    allowed_domains=["historical", "scientific", "financial"],
    allowed_correctness_modes=["exact"],
    max_steps=5,
    episode_timeout=60.0,
    adversary_mode="truthful",
    adversary_max_degree=0.0,
    adversary_manipulation_probability=0.0,
    window_size=15,
    advance_threshold=0.60,
    simplify_threshold=0.10,
)

# Level 1: Easy
# All domains, but fuzzy-match tolerant. Adversary occasionally mild.
LEVEL_1 = CurriculumLevel(
    level_id=1,
    name="Easy",
    description="All domains, fuzzy match, mild adversary (degree ≤ 0.3), 10 steps",
    allowed_domains=["historical", "scientific", "financial", "medical", "legal"],
    allowed_correctness_modes=["exact", "fuzzy"],
    max_steps=10,
    episode_timeout=120.0,
    adversary_mode="mild",
    adversary_max_degree=0.3,
    adversary_manipulation_probability=0.25,
    window_size=20,
    advance_threshold=0.55,
    simplify_threshold=0.10,
)

# Level 2: Medium
# All domains, adversary moderately manipulates half the time.
LEVEL_2 = CurriculumLevel(
    level_id=2,
    name="Medium",
    description="All domains, moderate adversary (degree ≤ 0.6), 15 steps",
    allowed_domains=["historical", "scientific", "financial", "medical", "legal"],
    allowed_correctness_modes=["exact", "fuzzy"],
    max_steps=15,
    episode_timeout=200.0,
    adversary_mode="moderate",
    adversary_max_degree=0.6,
    adversary_manipulation_probability=0.5,
    window_size=20,
    advance_threshold=0.50,
    simplify_threshold=0.10,
)

# Level 3: Hard (original environment spec)
# Full adversary capability, 20 steps.
LEVEL_3 = CurriculumLevel(
    level_id=3,
    name="Hard",
    description="All domains, aggressive adversary (up to degree 1.0), 20 steps",
    allowed_domains=["historical", "scientific", "financial", "medical", "legal"],
    allowed_correctness_modes=["exact", "fuzzy", "semantic"],
    max_steps=20,
    episode_timeout=300.0,
    adversary_mode="aggressive",
    adversary_max_degree=1.0,
    adversary_manipulation_probability=0.7,
    window_size=25,
    advance_threshold=0.45,
    simplify_threshold=0.10,
)

ALL_LEVELS = [LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3]


# ---------------------------------------------------------------------------
# Curriculum Manager
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    episode_id: str
    total_reward: float
    n_steps: int
    termination_reason: Optional[str]
    level_id: int


class CurriculumManager:
    """
    Tracks episode outcomes and decides when to advance or simplify difficulty.

    Policy:
      - One-way ratchet by default: can advance and also regress if stagnation is deep.
      - Regress only if zero_rate > (1 - simplify_threshold) for `simplify_patience` windows.
      - Never goes below level 0.
    """

    def __init__(
        self,
        start_level: int = 0,
        allow_regression: bool = True,
        simplify_patience: int = 2,
        seed: Optional[int] = None,
    ):
        self._levels = ALL_LEVELS
        self._level_idx: int = start_level
        self._allow_regression = allow_regression
        self._patience = simplify_patience
        self._seed = seed
        self._history: deque[EpisodeRecord] = deque(maxlen=100)
        self._consecutive_simplify_needed = 0
        self._total_episodes = 0
        self._level_transitions: list[dict] = []

    @property
    def current_level(self) -> CurriculumLevel:
        return self._levels[self._level_idx]

    @property
    def level_idx(self) -> int:
        return self._level_idx

    def record_episode(self, episode_id: str, total_reward: float, n_steps: int,
                       termination_reason: Optional[str] = None):
        record = EpisodeRecord(
            episode_id=episode_id,
            total_reward=total_reward,
            n_steps=n_steps,
            termination_reason=termination_reason,
            level_id=self._level_idx,
        )
        self._history.append(record)
        self._total_episodes += 1

    def record_batch(self, trajectories) -> None:
        """Convenience: record all trajectories from a batch."""
        for traj in trajectories:
            self.record_episode(
                episode_id=traj.episode_id,
                total_reward=traj.total_task_reward,
                n_steps=len(traj.steps),
                termination_reason=traj.termination_reason,
            )

    # ---------------------------------------------------------------- metrics

    def _window_records(self) -> list[EpisodeRecord]:
        level = self.current_level
        # Only records from the current level
        current_level_records = [r for r in self._history if r.level_id == self._level_idx]
        return list(current_level_records)[-level.window_size:]

    def positive_rate(self) -> float:
        records = self._window_records()
        if not records:
            return 0.0
        return sum(1 for r in records if r.total_reward > 0) / len(records)

    def zero_rate(self) -> float:
        records = self._window_records()
        if not records:
            return 1.0
        return sum(1 for r in records if r.total_reward <= 0) / len(records)

    def avg_reward(self) -> float:
        records = self._window_records()
        if not records:
            return 0.0
        return sum(r.total_reward for r in records) / len(records)

    def window_full(self) -> bool:
        return len(self._window_records()) >= self.current_level.window_size

    # ---------------------------------------------------------------- decisions

    def should_advance(self) -> bool:
        """True when the model has mastered the current level."""
        if self._level_idx >= len(self._levels) - 1:
            return False
        if not self.window_full():
            return False
        return self.positive_rate() >= self.current_level.advance_threshold

    def should_simplify(self) -> bool:
        """True when the model is stuck and needs an easier task."""
        if not self._allow_regression:
            return False
        if self._level_idx <= 0:
            return False
        if not self.window_full():
            return False
        return self.positive_rate() <= self.current_level.simplify_threshold

    def advance(self) -> CurriculumLevel:
        """Move to the next difficulty level."""
        if self._level_idx < len(self._levels) - 1:
            old = self.current_level
            self._level_idx += 1
            new = self.current_level
            self._consecutive_simplify_needed = 0
            self._level_transitions.append({
                "direction": "advance",
                "from": old.name,
                "to": new.name,
                "episode": self._total_episodes,
                "positive_rate_at_transition": self.positive_rate(),
            })
        return self.current_level

    def simplify(self) -> CurriculumLevel:
        """Step down one difficulty level."""
        self._consecutive_simplify_needed += 1
        if self._consecutive_simplify_needed >= self._patience:
            if self._level_idx > 0:
                old = self.current_level
                self._level_idx -= 1
                new = self.current_level
                self._consecutive_simplify_needed = 0
                self._level_transitions.append({
                    "direction": "simplify",
                    "from": old.name,
                    "to": new.name,
                    "episode": self._total_episodes,
                    "positive_rate_at_transition": self.positive_rate(),
                })
        return self.current_level

    def maybe_advance_or_simplify(self) -> Optional[str]:
        """
        Convenience: check and apply transitions automatically.
        Returns: "advanced", "simplified", or None.
        """
        if self.should_advance():
            self.advance()
            return "advanced"
        elif self.should_simplify():
            self.simplify()
            return "simplified"
        return None

    def build_task_bank(self) -> TaskBank:
        return self.current_level.build_task_bank(seed=self._seed)

    def status(self) -> str:
        level = self.current_level
        records = self._window_records()
        pr = self.positive_rate()
        ar = self.avg_reward()
        return (
            f"[Level {self._level_idx}:{level.name}] "
            f"window={len(records)}/{level.window_size} "
            f"positive_rate={pr:.2f} avg_reward={ar:.3f} | "
            f"advance@{level.advance_threshold:.2f} simplify@{level.simplify_threshold:.2f}"
        )

    def transition_history(self) -> list[dict]:
        return list(self._level_transitions)


# ---------------------------------------------------------------------------
# Curriculum-aware rollout
# ---------------------------------------------------------------------------

class CurriculumRollout:
    """
    Wraps EnvironmentRollout with curriculum-aware adversary staging.

    The adversary behaviour is driven by the current curriculum level:
    - Level 0: truthful always
    - Level 1: manipulate 25% of steps, degree ≤ 0.3
    - Level 2: manipulate 50% of steps, degree ≤ 0.6
    - Level 3: manipulate 70% of steps, degree ≤ 1.0
    """

    def __init__(self, env_url: str, curriculum: CurriculumManager, timeout: float = 30.0):
        self._url = env_url.rstrip("/")
        self._curriculum = curriculum
        self._timeout = timeout
        import requests
        self._sess = requests.Session()

    def _post(self, path: str, **kwargs):
        r = self._sess.post(f"{self._url}{path}", timeout=self._timeout, **kwargs)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str):
        r = self._sess.get(f"{self._url}{path}", timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def _stage_adversary(self):
        import random
        level = self._curriculum.current_level
        if random.random() > level.adversary_manipulation_probability:
            mode = "truthful"
            degree = 0.0
        else:
            mode = "manipulated"
            degree = random.uniform(0.0, level.adversary_max_degree)

        self._post("/adversary/step", json={
            "response_mode": mode,
            "manipulation_degree": degree,
        })

    def run_episode(self, model_generate_fn, tokenizer, seed=None):
        from train.rollout import (
            format_chat, parse_action, Trajectory, Step,
        )
        from action import TaskAction
        from observation import TaskObservation, StepResult
        import json

        level = self._curriculum.current_level

        # Use a level-filtered task bank — pass the episode seed to the server
        params = {"seed": seed} if seed is not None else {}
        obs_data = self._post("/reset", params=params)
        obs = TaskObservation(**obs_data)

        traj = Trajectory(
            episode_id=obs.episode_id,
            task_description=obs.task_description,
        )

        max_queries = level.max_steps - 1  # leave room for final_answer

        while not obs.done:
            prompt = format_chat(
                obs.task_description,
                [h.model_dump() for h in obs.api_history],
                tokenizer,
            )

            self._stage_adversary()
            completion = model_generate_fn(prompt)
            action = parse_action(completion)

            if action.action_type == "query" and obs.step_count >= max_queries:
                last = obs.api_history[-1].response if obs.api_history else "unknown"
                action = TaskAction(action_type="final_answer", answer=last[:300])
                completion = json.dumps({"action_type": "final_answer", "answer": last[:300]})

            result_data = self._post("/step", json=action.model_dump())
            result = StepResult(
                observation=TaskObservation(**result_data["observation"]),
                reward=result_data["reward"],
                terminated=result_data["terminated"],
                truncated=result_data["truncated"],
                info=result_data.get("info", {}),
            )

            traj.steps.append(Step(
                prompt=prompt,
                completion=completion,
                action=action,
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
                step_count=result.observation.step_count,
            ))
            traj.total_task_reward += result.reward
            traj.termination_reason = result.info.get("termination_reason")
            obs = result.observation

        try:
            adv_data = self._get("/adversary/reward")
            traj.total_adversary_reward = adv_data.get("adversary_reward", 0.0)
            traj.manipulation_count = adv_data.get("manipulation_count", 0)
            traj.final_suspicion = adv_data.get("suspicion_score", 0.0)
        except Exception:
            pass

        return traj

    def run_batch(self, model_generate_fn, tokenizer, n_episodes, seed_offset=0):
        trajectories = []
        for i in range(n_episodes):
            try:
                traj = self.run_episode(model_generate_fn, tokenizer, seed=seed_offset + i)
                trajectories.append(traj)
            except Exception as e:
                print(f"  [curriculum rollout] Episode {i} failed: {e}")
        return trajectories

    def close(self):
        self._sess.close()


# ---------------------------------------------------------------------------
# Utility: zero-reward diagnosis
# ---------------------------------------------------------------------------

def diagnose_zero_reward(trajectories) -> str:
    """
    When the model consistently gets zero reward, diagnose why and recommend a fix.
    """
    if not trajectories:
        return "No trajectories to diagnose."

    n = len(trajectories)
    zero_count = sum(1 for t in trajectories if t.total_task_reward <= 0)
    zero_rate = zero_count / n

    if zero_rate < 0.3:
        return f"Zero-reward rate is {zero_rate:.0%} — within acceptable range."

    reasons = []

    # Check: are episodes truncated (max steps hit)?
    truncated = sum(1 for t in trajectories if t.termination_reason == "max_steps")
    if truncated / n > 0.4:
        reasons.append(
            f"{truncated}/{n} episodes hit the step limit — model is not converging to a final answer. "
            "Reduce max_steps or add a stronger time penalty to encourage decisive action."
        )

    # Check: all accusations, many wrong
    accuse_eps = [t for t in trajectories
                  if any(s.action.action_type == "accuse" for s in t.steps)]
    if len(accuse_eps) / n > 0.5:
        wrong_accuse = sum(1 for t in accuse_eps if t.total_task_reward < 0)
        if wrong_accuse / max(1, len(accuse_eps)) > 0.6:
            reasons.append(
                f"Model is accusing frequently ({len(accuse_eps)}/{n}) but mostly wrong "
                f"({wrong_accuse}/{len(accuse_eps)} false). "
                "Increase curriculum level's adversary manipulation probability so accusations are sometimes correct."
            )

    # Check: final answer always wrong (model never correct)
    answer_eps = [t for t in trajectories
                  if any(s.action.action_type == "final_answer" for s in t.steps)
                  and t.total_task_reward < 0]
    if len(answer_eps) / n > 0.5:
        reasons.append(
            f"{len(answer_eps)}/{n} episodes end with a wrong final answer. "
            "The task may be too hard. Switch to Level 0 (exact-match tasks) "
            "or increase the allowed number of queries."
        )

    # Check: no queries before answering
    no_query = sum(
        1 for t in trajectories
        if not any(s.action.action_type == "query" for s in t.steps)
    )
    if no_query / n > 0.3:
        reasons.append(
            f"{no_query}/{n} episodes have zero query steps — model skips the API. "
            "Add a minimum-query requirement or reward the first successful query."
        )

    if not reasons:
        reasons.append(
            "No single dominant failure mode. "
            "Try: (1) lower curriculum level, (2) increase temperature, "
            "(3) add entropy bonus to rewards, (4) check prompt formatting."
        )

    diagnosis = f"Zero-reward rate: {zero_rate:.0%} ({zero_count}/{n} episodes)\n"
    for i, r in enumerate(reasons, 1):
        diagnosis += f"  [{i}] {r}\n"
    return diagnosis


# ---------------------------------------------------------------------------
# CLI: curriculum status and diagnosis
# ---------------------------------------------------------------------------

def main():
    import argparse, requests as req
    p = argparse.ArgumentParser(description="Show curriculum levels and diagnose zero-reward")
    p.add_argument("--show-levels", action="store_true")
    p.add_argument("--env-url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=20, help="Episodes to collect for diagnosis")
    p.add_argument("--level", type=int, default=0, help="Level to test")
    args = p.parse_args()

    if args.show_levels:
        print("\nCurriculum Levels:")
        for level in ALL_LEVELS:
            print(f"  [{level.level_id}] {level.name}: {level.description}")
            print(f"       domains={level.allowed_domains}")
            print(f"       max_steps={level.max_steps} | adv_prob={level.adversary_manipulation_probability} | max_deg={level.adversary_max_degree}")
            print(f"       advance@positive_rate>{level.advance_threshold} | simplify@positive_rate<{level.simplify_threshold}")
        return

    # Quick diagnosis run with rule-based policy
    from train.inspect import GreedyTextPolicy
    policy = GreedyTextPolicy()

    class FakeTok:
        def apply_chat_template(self, msgs, **kw):
            return msgs[-1]["content"]

    curriculum = CurriculumManager(start_level=args.level)
    rollout = CurriculumRollout(env_url=args.env_url, curriculum=curriculum)

    print(f"\nRunning {args.n} episodes at Level {args.level}...")
    trajs = rollout.run_batch(policy.generate, FakeTok(), n_episodes=args.n)
    rollout.close()

    curriculum.record_batch(trajs)
    print(curriculum.status())
    print()
    print(diagnose_zero_reward(trajs))


if __name__ == "__main__":
    main()
