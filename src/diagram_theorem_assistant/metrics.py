from __future__ import annotations

from collections.abc import Iterable

from diagram_theorem_assistant.schema import LeanStatus, PipelineResult


def assumption_recall(predicted: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0
    return len(set(predicted) & set(gold)) / len(set(gold))


def assumption_precision(predicted: list[str], gold: list[str]) -> float:
    if not predicted:
        return 1.0
    return len(set(predicted) & set(gold)) / len(set(predicted))


def top_k_goal_accuracy(ranked_goals: list[str], gold_goal: str, k: int) -> float:
    return 1.0 if gold_goal in ranked_goals[:k] else 0.0


def lean_validity_rate(results: Iterable[PipelineResult]) -> float:
    results_list = list(results)
    if not results_list:
        return 0.0
    ok = sum(1 for r in results_list if r.lean_status is LeanStatus.OK)
    return ok / len(results_list)


def category_breakdown(results_by_category: dict[str, list[PipelineResult]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for category, results in results_by_category.items():
        summary[category] = {
            "count": float(len(results)),
            "lean_validity_rate": lean_validity_rate(results),
        }
    return summary
