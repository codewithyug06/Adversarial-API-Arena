"""Unit tests for environment.py — no server required."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from action import TaskAction, AdversaryAction
from server.environment import AdversarialArenaEnvironment


@pytest.fixture
def env(tmp_path):
    return AdversarialArenaEnvironment(
        max_steps=5,
        episode_timeout_seconds=60.0,
        suspicion_threshold=0.7,
        log_dir=tmp_path / "logs",
        seed=42,
    )


class TestReset:
    def test_returns_task_observation(self, env):
        obs = env.reset()
        assert obs.task_description
        assert obs.step_count == 0
        assert obs.reward == 0.0
        assert not obs.done

    def test_no_ground_truth_in_observation(self, env):
        obs = env.reset()
        obs_dict = obs.model_dump()
        assert "ground_truth" not in obs_dict

    def test_episode_id_is_set(self, env):
        obs = env.reset()
        assert obs.episode_id

    def test_reset_clears_previous_state(self, env):
        obs1 = env.reset()
        env.step(TaskAction(action_type="query", query="test"))
        obs2 = env.reset()
        assert obs2.step_count == 0
        assert obs2.episode_id != obs1.episode_id


class TestStepQuery:
    def test_query_increments_step_count(self, env):
        env.reset()
        result = env.step(TaskAction(action_type="query", query="what is the dosage?"))
        assert result.observation.step_count == 1

    def test_query_adds_to_history(self, env):
        env.reset()
        result = env.step(TaskAction(action_type="query", query="what is the answer?"))
        assert len(result.observation.api_history) == 1
        assert result.observation.api_history[0].query == "what is the answer?"

    def test_query_reward_is_time_penalty(self, env):
        env.reset()
        result = env.step(TaskAction(action_type="query", query="test"))
        assert result.reward == pytest.approx(-0.1)
        assert not result.terminated
        assert not result.truncated

    def test_task_observation_has_no_ground_truth_field(self, env):
        # Ground truth appears in API *responses* (that's correct for a truthful API).
        # What must NEVER happen: ground_truth exposed as a named field in TaskObservation.
        env.reset()
        env.step(TaskAction(action_type="query", query="what is the value?"))
        obs = env._state.to_task_observation(reward=-0.1)
        obs_dict = obs.model_dump()
        assert "ground_truth" not in obs_dict
        assert "manipulation_log" not in obs_dict
        assert "suspicion_score" not in obs_dict

    def test_max_steps_truncates(self, env):
        env.reset()
        result = None
        for _ in range(5):
            result = env.step(TaskAction(action_type="query", query="test"))
        assert result.truncated
        assert result.terminated is False


class TestStepFinalAnswer:
    def test_correct_answer_positive_reward(self, env):
        env.reset()
        ground_truth = env._state.ground_truth
        result = env.step(
            TaskAction(action_type="final_answer", answer=ground_truth)
        )
        assert result.terminated
        assert result.reward > 0  # time_penalty + TASK_CORRECT = -0.1 + 1.0 = 0.9

    def test_wrong_answer_negative_reward(self, env):
        env.reset()
        result = env.step(
            TaskAction(action_type="final_answer", answer="completely wrong answer xyz")
        )
        assert result.terminated
        assert result.reward < 0  # -0.1 + (-0.5)


class TestStepAccuse:
    def test_correct_accusation_positive_reward(self, env):
        env.reset()
        # Adversary manipulates first
        env.adversary_step(
            AdversaryAction(response_mode="manipulated", manipulation_degree=0.9)
        )
        env.step(TaskAction(action_type="query", query="test"))
        # Then task agent accuses
        result = env.step(TaskAction(action_type="accuse", accusation_reason="suspicious"))
        assert result.terminated
        # accusation_bonus = +0.5, time_penalty = -0.1 => 0.4
        assert result.reward > 0

    def test_false_accusation_negative_reward(self, env):
        env.reset()
        # No manipulation — adversary is truthful
        env.adversary_step(AdversaryAction(response_mode="truthful"))
        env.step(TaskAction(action_type="query", query="test"))
        result = env.step(TaskAction(action_type="accuse"))
        assert result.terminated
        assert result.reward < 0  # false accusation penalty


class TestAdversaryStep:
    def test_adversary_step_returns_observation(self, env):
        env.reset()
        adv_obs = env.adversary_step(
            AdversaryAction(response_mode="manipulated", manipulation_degree=0.5)
        )
        assert adv_obs.ground_truth  # adversary CAN see ground truth
        assert adv_obs.suspicion_score == 0.0

    def test_adversary_obs_not_in_task_obs(self, env):
        env.reset()
        adv_obs = env.adversary_step(
            AdversaryAction(response_mode="manipulated", manipulation_degree=0.9)
        )
        task_result = env.step(TaskAction(action_type="query", query="test"))
        task_dict = task_result.observation.model_dump()
        assert "ground_truth" not in task_dict
        assert "manipulation_log" not in task_dict
        assert "suspicion_score" not in task_dict

    def test_high_degree_raises_suspicion(self, env):
        env.reset()
        env.adversary_step(
            AdversaryAction(response_mode="manipulated", manipulation_degree=0.9)
        )
        env.step(TaskAction(action_type="query", query="test"))
        adv_obs = env.adversary_observe()
        # spike rule: 0.4 * 0.9 = 0.36, then +0.3 spike = 0.66
        assert adv_obs.suspicion_score > 0.3

    def test_pending_action_consumed_once(self, env):
        env.reset()
        env.adversary_step(
            AdversaryAction(response_mode="manipulated", manipulation_degree=0.9)
        )
        # First query consumes the action
        env.step(TaskAction(action_type="query", query="first"))
        suspicion_after_first = env._state.suspicion_score
        # Second query — no adversary action staged → defaults to truthful
        env.step(TaskAction(action_type="query", query="second"))
        suspicion_after_second = env._state.suspicion_score
        # Suspicion should drop (EMA of 0.0 pulls it down)
        assert suspicion_after_second < suspicion_after_first

    def test_manipulation_degree_out_of_range_rejected(self):
        with pytest.raises(Exception):
            AdversaryAction(response_mode="manipulated", manipulation_degree=1.5)

    def test_manipulation_degree_negative_rejected(self):
        with pytest.raises(Exception):
            AdversaryAction(response_mode="manipulated", manipulation_degree=-0.1)


class TestEpisodeLog:
    def test_log_written_on_termination(self, env, tmp_path):
        env.reset()
        eid = env._state.episode_id
        env.step(TaskAction(action_type="query", query="test"))
        env.step(TaskAction(action_type="final_answer", answer="test answer"))
        log_path = tmp_path / "logs" / f"{eid}.json"
        assert log_path.exists()

    def test_log_includes_suspicion_score(self, env):
        env.reset()
        env.adversary_step(
            AdversaryAction(response_mode="manipulated", manipulation_degree=0.5)
        )
        env.step(TaskAction(action_type="query", query="test"))
        env.step(TaskAction(action_type="final_answer", answer="answer"))
        log = env.get_episode_log()
        assert any("suspicion_score" in entry for entry in log)
