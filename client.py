from __future__ import annotations
import time
from typing import Optional

import requests

from action import TaskAction, AdversaryAction
from observation import TaskObservation, AdversaryObservation, StepResult


class TaskAgentClient:
    """
    HTTP client for the task agent.
    Has NO access to adversary endpoints or ground truth.

    Compatible with TRL/GRPO training loops:
      obs = client.reset()
      while not obs.done:
          action = policy(obs)
          result = client.step(action)
          replay_buffer.add(obs, action, result.reward, result.terminated)
          obs = result.observation
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> TaskObservation:
        params: dict = {}
        if seed is not None:
            params["seed"] = seed
        if episode_id is not None:
            params["episode_id"] = episode_id
        r = self._session.post(
            f"{self.base_url}/reset", params=params, timeout=self._timeout
        )
        r.raise_for_status()
        return TaskObservation(**r.json())

    def step(self, action: TaskAction) -> StepResult:
        r = self._session.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()
        # Unwrap observation sub-dict
        obs = TaskObservation(**data["observation"])
        return StepResult(
            observation=obs,
            reward=data["reward"],
            terminated=data["terminated"],
            truncated=data["truncated"],
            info=data.get("info", {}),
        )

    # Convenience wrappers

    def query(self, query_text: str) -> StepResult:
        return self.step(TaskAction(action_type="query", query=query_text))

    def submit_answer(self, answer: str) -> StepResult:
        return self.step(TaskAction(action_type="final_answer", answer=answer))

    def accuse(self, reason: str = "API responses appear manipulated") -> StepResult:
        return self.step(
            TaskAction(action_type="accuse", accusation_reason=reason)
        )

    def get_state(self) -> TaskObservation:
        r = self._session.get(f"{self.base_url}/state", timeout=self._timeout)
        r.raise_for_status()
        return TaskObservation(**r.json())

    def health(self) -> dict:
        r = self._session.get(f"{self.base_url}/health", timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class AdversaryClient:
    """
    HTTP client for the adversary agent.
    Can only reach /adversary/* endpoints.
    Cannot access task-agent observations or ground truth from the task-agent path.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    def get_state(self) -> AdversaryObservation:
        r = self._session.get(
            f"{self.base_url}/adversary/state", timeout=self._timeout
        )
        r.raise_for_status()
        return AdversaryObservation(**r.json())

    def step(self, action: AdversaryAction) -> AdversaryObservation:
        r = self._session.post(
            f"{self.base_url}/adversary/step",
            json=action.model_dump(),
            timeout=self._timeout,
        )
        r.raise_for_status()
        return AdversaryObservation(**r.json())

    def be_truthful(self) -> AdversaryObservation:
        return self.step(AdversaryAction(response_mode="truthful"))

    def manipulate(
        self,
        degree: float,
        corrupted_response: Optional[str] = None,
    ) -> AdversaryObservation:
        return self.step(
            AdversaryAction(
                response_mode="manipulated",
                manipulation_degree=degree,
                corrupted_response=corrupted_response,
            )
        )

    def get_terminal_reward(self) -> dict:
        r = self._session.get(
            f"{self.base_url}/adversary/reward", timeout=self._timeout
        )
        r.raise_for_status()
        return r.json()

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
