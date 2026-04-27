"""
trainer_example.py
==================
Minimal example showing how to use the Adversarial API Arena environment
to collect training trajectories for LLM fine-tuning (GRPO / PPO style).

Run the server first:
    uvicorn server.app:app --port 8000

Then run this script:
    python trainer_example.py

The script drives the environment with a stub policy and prints per-episode stats.
Replace the stub policies with your real LLM-based agents.
"""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client import TaskAgentClient, AdversaryClient
from action import TaskAction, AdversaryAction

BASE_URL = "http://localhost:8000"
NUM_EPISODES = 5
MAX_QUERIES = 10  # task agent queries per episode before submitting answer


# ---------------------------------------------------------------------------
# Stub policies — replace these with your LLM inference calls
# ---------------------------------------------------------------------------

def task_agent_policy(
    task_description: str,
    api_history: list,
    step_count: int,
) -> TaskAction:
    """
    Stub: a real implementation would call an LLM here.
    The LLM receives: task_description + api_history as context.
    It must output one of: query / final_answer / accuse.
    """
    if step_count >= MAX_QUERIES:
        # Time to commit to an answer
        last_response = api_history[-1].response if api_history else ""
        return TaskAction(action_type="final_answer", answer=last_response[:200] or "unknown")

    # Simple heuristic: first query asks the question directly
    if step_count == 0:
        return TaskAction(
            action_type="query",
            query=f"Please answer: {task_description}",
        )
    # Subsequent queries cross-check
    return TaskAction(
        action_type="query",
        query=f"Can you confirm or correct: {api_history[-1].response[:100] if api_history else 'no data yet'}?",
    )


def adversary_policy(
    task_description: str,
    ground_truth: str,
    suspicion_score: float,
    step_count: int,
) -> AdversaryAction:
    """
    Stub: a real implementation would call an LLM here.
    The LLM receives: task_description + ground_truth + suspicion_score.
    It must decide: truthful or manipulated, and how aggressively.

    Key constraint: if suspicion_score is already high (> 0.5),
    switching to truthful may help the adversary avoid detection.
    """
    if suspicion_score > 0.5:
        # Back off if getting close to detection threshold
        return AdversaryAction(response_mode="truthful")

    # Low suspicion — attempt mild manipulation
    degree = random.uniform(0.2, 0.6)
    return AdversaryAction(
        response_mode="manipulated",
        manipulation_degree=degree,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_episode(
    task_client: TaskAgentClient,
    adv_client: AdversaryClient,
    episode_num: int,
) -> dict:
    """Run one full episode and return stats."""
    obs = task_client.reset(seed=episode_num)
    print(f"\n--- Episode {episode_num + 1} ---")
    print(f"Task: {obs.task_description}")

    # Trajectory storage for LLM training
    task_trajectory: list[dict] = []
    adv_trajectory: list[dict] = []

    cumulative_task_reward = 0.0

    while not obs.done:
        # --------------- Adversary acts first (stages action for next query) -------
        try:
            adv_obs = adv_client.get_state()
            adv_action = adversary_policy(
                task_description=adv_obs.task_description,
                ground_truth=adv_obs.ground_truth,
                suspicion_score=adv_obs.suspicion_score,
                step_count=adv_obs.step_count,
            )
            adv_client.step(adv_action)
        except Exception as e:
            print(f"  [Adversary] Error: {e}")

        # --------------- Task agent acts -------------------------------------------
        task_action = task_agent_policy(
            task_description=obs.task_description,
            api_history=obs.api_history,
            step_count=obs.step_count,
        )

        result = task_client.step(task_action)
        cumulative_task_reward += result.reward

        # Store (prompt, action, reward) for training
        task_trajectory.append({
            "prompt": obs.task_description,
            "history": [h.model_dump() for h in obs.api_history],
            "action": task_action.model_dump(),
            "reward": result.reward,
            "step": obs.step_count,
        })

        print(
            f"  Step {result.observation.step_count}: "
            f"action={task_action.action_type} "
            f"reward={result.reward:.2f} "
            f"done={result.observation.done}"
        )

        obs = result.observation

    # ----------- Adversary terminal reward ----------------------------------------
    try:
        if obs.done:
            adv_reward_data = adv_client.get_terminal_reward()
            adv_reward = adv_reward_data.get("adversary_reward", 0.0)
            adv_trajectory.append({
                "episode_id": obs.episode_id,
                "terminal_reward": adv_reward,
                "suspicion_score": adv_reward_data.get("suspicion_score", 0.0),
                "manipulation_count": adv_reward_data.get("manipulation_count", 0),
            })
        else:
            adv_reward = 0.0
    except Exception as e:
        print(f"  [Adversary terminal reward] Error: {e}")
        adv_reward = 0.0

    stats = {
        "episode": episode_num + 1,
        "episode_id": obs.episode_id,
        "termination_reason": result.info.get("termination_reason"),
        "task_cumulative_reward": cumulative_task_reward,
        "adversary_terminal_reward": adv_reward,
        "steps": obs.step_count,
        "task_trajectory_length": len(task_trajectory),
    }

    print(f"  Task reward: {cumulative_task_reward:.2f}")
    print(f"  Adversary reward: {adv_reward:.2f}")
    print(f"  Termination: {stats['termination_reason']}")

    # In a real trainer, you'd pass task_trajectory and adv_trajectory
    # to your GRPO / PPO update step here.

    return stats


def main():
    print("Connecting to Adversarial API Arena server...")
    task_client = TaskAgentClient(BASE_URL)
    adv_client = AdversaryClient(BASE_URL)

    # Verify server is up
    try:
        health = task_client.health()
        print(f"Server health: {health}")
    except Exception as e:
        print(f"ERROR: Could not connect to server at {BASE_URL}")
        print(f"  Start it with: uvicorn server.app:app --port 8000")
        print(f"  Error: {e}")
        sys.exit(1)

    all_stats = []
    for ep in range(NUM_EPISODES):
        try:
            stats = run_episode(task_client, adv_client, ep)
            all_stats.append(stats)
        except Exception as e:
            print(f"  Episode {ep + 1} failed: {e}")

    print("\n=== Summary ===")
    if all_stats:
        avg_task = sum(s["task_cumulative_reward"] for s in all_stats) / len(all_stats)
        avg_adv = sum(s["adversary_terminal_reward"] for s in all_stats) / len(all_stats)
        avg_steps = sum(s["steps"] for s in all_stats) / len(all_stats)
        print(f"Episodes: {len(all_stats)}")
        print(f"Avg task reward:     {avg_task:.3f}")
        print(f"Avg adversary reward:{avg_adv:.3f}")
        print(f"Avg steps:           {avg_steps:.1f}")

    task_client.close()
    adv_client.close()


if __name__ == "__main__":
    main()
