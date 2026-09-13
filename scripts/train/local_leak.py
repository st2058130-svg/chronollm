"""Local Stage-1 leak evaluation helpers.

Same scoring as sn38.template.leak.evaluate, but also returns the fail ratio
so compare_models can print unk_fail% / kn_conf%.
"""

from __future__ import annotations

from sn38.template.leak import _score_batch

EVAL_VERSION = "v1-flat-epsilon"


def evaluate_local(model, device, benchmark: dict):
    """Same return shape as sn38.template.leak.evaluate, plus fail_ratio."""
    items = benchmark.get("items", [])
    if not items:
        return False, -20.0, 0.0

    threshold = benchmark.get("threshold", 0.10)
    epsilon = benchmark.get("epsilon", -11.51)

    scores = _score_batch(model, device, items)
    median = sorted(scores)[len(scores) // 2]

    total_weight = 0.0
    failed_weight = 0.0
    for i, score in enumerate(scores):
        weight = items[i].get("weight", 1)
        total_weight += weight
        if score > epsilon:
            failed_weight += weight

    ratio = failed_weight / total_weight if total_weight else 0.0
    return ratio > threshold, median, ratio
