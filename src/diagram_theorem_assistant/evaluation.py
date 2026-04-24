from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from diagram_theorem_assistant.assumption_extraction import infer_assumptions
from diagram_theorem_assistant.goal_generation import extract_goal_from_text, rank_goal_candidates
from diagram_theorem_assistant.schema import BenchmarkExample
from diagram_theorem_assistant.schema import load_benchmark


@dataclass(frozen=True)
class EvaluationResult:
    example_id: str
    assumption_precision: float
    assumption_recall: float
    top_1_goal_accuracy: float
    top_3_goal_accuracy: float
    top_5_goal_accuracy: float


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


def evaluate_example(
    example: BenchmarkExample,
    predicted_assumptions: list[str],
    ranked_goals: list[str],
) -> EvaluationResult:
    return EvaluationResult(
        example_id=example.id,
        assumption_precision=assumption_precision(predicted_assumptions, example.gold_assumptions),
        assumption_recall=assumption_recall(predicted_assumptions, example.gold_assumptions),
        top_1_goal_accuracy=top_k_goal_accuracy(ranked_goals, example.gold_goal, 1),
        top_3_goal_accuracy=top_k_goal_accuracy(ranked_goals, example.gold_goal, 3),
        top_5_goal_accuracy=top_k_goal_accuracy(ranked_goals, example.gold_goal, 5),
    )


def evaluate_benchmark(path: str | Path) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for example in load_benchmark(Path(path)):
        assumptions = infer_assumptions(example.problem_text, example.objects, example.diagram_marks)
        extracted_goal = extract_goal_from_text(example.problem_text)
        generated_goals = rank_goal_candidates(example.objects, assumptions)
        ranked_goals = [extracted_goal, *generated_goals] if extracted_goal else generated_goals
        results.append(evaluate_example(example, assumptions, ranked_goals))
    return results
