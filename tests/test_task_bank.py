"""Unit tests for task_bank.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from server.task_bank import TaskBank, TASK_TEMPLATES


def test_sample_returns_task():
    bank = TaskBank(seed=42)
    task = bank.sample()
    assert task.task_description
    assert task.ground_truth
    assert task.correctness_mode in ("exact", "fuzzy", "semantic")
    assert task.domain in ("medical", "financial", "historical", "scientific", "legal")


def test_sample_covers_all_domains():
    bank = TaskBank(seed=0)
    domains = set()
    for _ in range(200):
        domains.add(bank.sample().domain)
    assert domains == {"medical", "financial", "historical", "scientific", "legal"}


def test_ground_truth_not_empty():
    bank = TaskBank(seed=7)
    for _ in range(50):
        task = bank.sample()
        assert task.ground_truth != "Information not available", (
            f"No ground truth found for: {task.task_description}"
        )


def test_different_seeds_produce_different_tasks():
    bank_a = TaskBank(seed=1)
    bank_b = TaskBank(seed=2)
    tasks_a = {bank_a.sample().task_description for _ in range(20)}
    tasks_b = {bank_b.sample().task_description for _ in range(20)}
    # They should overlap somewhat (same template pool) but not be identical
    assert tasks_a != tasks_b


def test_same_seed_is_deterministic():
    bank_a = TaskBank(seed=99)
    bank_b = TaskBank(seed=99)
    for _ in range(10):
        a = bank_a.sample()
        b = bank_b.sample()
        assert a.task_description == b.task_description
        assert a.ground_truth == b.ground_truth
