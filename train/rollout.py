"""
rollout.py
==========
Environment interaction helpers for training.

Converts the HTTP environment into trajectory data the GRPO trainer can consume.
"""
from __future__ import annotations
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from action import TaskAction, AdversaryAction
from observation import TaskObservation, StepResult

# ---------------------------------------------------------------------------
# Prompt format
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a factual research agent. Your job is to answer a question correctly "
    "by querying an API. The API may occasionally give wrong information — stay skeptical "
    "and cross-check when responses seem off.\n\n"
    "Respond ONLY with a JSON object on a single line. No explanation. No markdown.\n"
    "Options:\n"
    '  {"action_type": "query", "query": "your question"}\n'
    '  {"action_type": "final_answer", "answer": "your answer"}\n'
    '  {"action_type": "accuse", "accusation_reason": "why you suspect manipulation"}'
)


def build_prompt(task_description: str, api_history: list) -> str:
    """Build the full chat prompt for the task agent LLM."""
    history_str = ""
    if api_history:
        pairs = [
            f"  Q: {h['query'] if isinstance(h, dict) else h.query}\n"
            f"  A: {h['response'] if isinstance(h, dict) else h.response}"
            for h in api_history
        ]
        history_str = "\n".join(pairs)
    else:
        history_str = "  (no queries yet)"

    return (
        f"Task: {task_description}\n\n"
        f"API query history:\n{history_str}\n\n"
        f"Your action:"
    )


def format_chat(task_description: str, api_history: list, tokenizer) -> str:
    """Format as a chat template (for models that require it)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(task_description, api_history)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: plain text
        return SYSTEM_PROMPT + "\n\n" + build_prompt(task_description, api_history)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(completion: str) -> TaskAction:
    """
    Parse a model completion into a TaskAction.
    Tries strict JSON first, then regex fallback, then defaults to final_answer.
    """
    text = completion.strip()

    # Try to extract JSON from the completion
    json_matches = re.findall(r'\{[^{}]+\}', text, re.DOTALL)
    for m in json_matches:
        try:
            data = json.loads(m)
            return TaskAction(**data)
        except Exception:
            continue

    # Fallback: the whole text might be a JSON object
    try:
        data = json.loads(text)
        return TaskAction(**data)
    except Exception:
        pass

    # Last resort: treat as a final answer
    return TaskAction(action_type="final_answer", answer=text[:500])


# ---------------------------------------------------------------------------
# Trajectory dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Step:
    prompt: str
    completion: str
    action: TaskAction
    reward: float
    terminated: bool
    truncated: bool
    step_count: int


@dataclass
class Trajectory:
    episode_id: str
    task_description: str
    steps: list[Step] = field(default_factory=list)
    total_task_reward: float = 0.0
    total_adversary_reward: float = 0.0
    termination_reason: Optional[str] = None
    manipulation_count: int = 0
    final_suspicion: float = 0.0

    def as_grpo_rows(self) -> list[dict]:
        """Convert to (prompt, completion, reward) rows for GRPOTrainer."""
        return [
            {
                "prompt": s.prompt,
                "completion": s.completion,
                "reward": s.reward,
            }
            for s in self.steps
        ]


# ---------------------------------------------------------------------------
# Rollout runner
# ---------------------------------------------------------------------------

class EnvironmentRollout:
    """
    Runs complete episodes against the environment server.
    Returns Trajectory objects that the GRPO trainer consumes.
    """

    def __init__(
        self,
        env_url: str = "http://localhost:8000",
        max_queries_before_answer: int = 8,
        adversary_mode: str = "truthful",  # "truthful" | "random" | "fixed"
        fixed_manipulation_degree: float = 0.6,
        timeout: float = 30.0,
    ):
        self._url = env_url.rstrip("/")
        self._max_queries = max_queries_before_answer
        self._adversary_mode = adversary_mode
        self._fixed_degree = fixed_manipulation_degree
        self._timeout = timeout
        self._sess = requests.Session()

    def _post(self, path: str, **kwargs):
        r = self._sess.post(f"{self._url}{path}", timeout=self._timeout, **kwargs)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str):
        r = self._sess.get(f"{self._url}{path}", timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def _stage_adversary_action(self):
        """Stage the adversary action before each query step."""
        import random
        if self._adversary_mode == "truthful":
            degree = 0.0
            mode = "truthful"
        elif self._adversary_mode == "random":
            degree = random.uniform(0.0, 1.0)
            mode = "manipulated" if degree > 0.3 else "truthful"
        else:  # fixed
            degree = self._fixed_degree
            mode = "manipulated"

        self._post("/adversary/step", json={
            "response_mode": mode,
            "manipulation_degree": degree,
        })

    def run_episode(
        self,
        model_generate_fn,
        tokenizer,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """
        Run one full episode.

        model_generate_fn: callable(prompt: str) -> completion: str
            The LLM inference function — receives a prompt string, returns completion string.
        """
        # Reset environment
        reset_params = {"seed": seed} if seed is not None else {}
        obs_data = self._post("/reset", params=reset_params)
        obs = TaskObservation(**obs_data)

        traj = Trajectory(
            episode_id=obs.episode_id,
            task_description=obs.task_description,
        )

        while not obs.done:
            # Build prompt from current observation
            prompt = format_chat(
                obs.task_description,
                [h.model_dump() for h in obs.api_history],
                tokenizer,
            )

            # Stage adversary if this will be a query step
            # (we stage optimistically; if model generates final_answer, the staged
            # action is wasted but causes no harm — it will be consumed next query)
            self._stage_adversary_action()

            # Model generates action
            completion = model_generate_fn(prompt)

            # Parse and execute action
            action = parse_action(completion)

            # Force final_answer if we've queried too many times
            if action.action_type == "query" and obs.step_count >= self._max_queries:
                last_resp = obs.api_history[-1].response if obs.api_history else "unknown"
                action = TaskAction(
                    action_type="final_answer",
                    answer=last_resp[:300],
                )
                completion = json.dumps({"action_type": "final_answer", "answer": last_resp[:300]})

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

        # Fetch adversary terminal reward and episode metadata
        try:
            adv_reward_data = self._get("/adversary/reward")
            traj.total_adversary_reward = adv_reward_data.get("adversary_reward", 0.0)
            traj.manipulation_count = adv_reward_data.get("manipulation_count", 0)
            traj.final_suspicion = adv_reward_data.get("suspicion_score", 0.0)
        except Exception:
            pass

        return traj

    def run_batch(
        self,
        model_generate_fn,
        tokenizer,
        n_episodes: int,
        seed_offset: int = 0,
    ) -> list[Trajectory]:
        trajectories = []
        for i in range(n_episodes):
            try:
                traj = self.run_episode(
                    model_generate_fn=model_generate_fn,
                    tokenizer=tokenizer,
                    seed=seed_offset + i,
                )
                trajectories.append(traj)
            except Exception as e:
                print(f"  [rollout] Episode {i} failed: {e}")
        return trajectories

    def close(self):
        self._sess.close()
