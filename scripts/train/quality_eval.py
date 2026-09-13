"""Local Stage-2 quality evaluation for compare_models.py.

Runs the same LLM-judge duel flow as sn38.template.quality, adapted for two
loaded models without on-chain submissions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

DEFAULT_QUALITY_PROMPTS = [
    {
        "prompt": (
            "The school garden had tomatoes, carrots, and sunflowers. Every Friday, students took turns "
            "watering the plants and pulling weeds. By the end of summer, the tomatoes were ripe and ready to pick. "
            "The students cared for the garden by"
        ),
        "category": "reading_comprehension",
    },
    {
        "prompt": (
            "The cabin stood alone at the edge of the frozen lake, its windows glowing faintly through the snow. "
            "Inside, Mara hung her coat by the door, stamped the ice from her boots, and"
        ),
        "category": "language_understanding",
    },
    {
        "prompt": "The process by which plants use sunlight to convert carbon dioxide and water into sugar is called",
        "category": "world_knowledge",
    },
    {
        "prompt": (
            "Jamal poured orange juice into a glass until it reached the top. When he tried to add more, the juice"
        ),
        "category": "commonsense_reasoning",
    },
    {
        "prompt": (
            "The old lighthouse keeper climbed the spiral stairs each evening, his lantern swaying with each step. "
            "At the top, he"
        ),
        "category": "language_modeling",
    },
    {
        "prompt": "The driver forgot to turn off the headlights overnight, so in the morning the car battery",
        "category": "causal_reasoning",
    },
    {
        "prompt": "Every mammal breathes air. Whales are mammals. Therefore, whales",
        "category": "logical_inference",
    },
    {
        "prompt": "First she mixed the flour and eggs, then she added the sugar. After stirring for two minutes, she",
        "category": "temporal_reasoning",
    },
    {
        "prompt": (
            "During the long voyage, the crew rationed water carefully and repaired torn sails after each storm. "
            "When land finally appeared on the horizon, the captain ordered the crew to"
        ),
        "category": "reading_comprehension",
    },
    {
        "prompt": "A recipe for bread usually begins by combining flour, yeast, warm water, and",
        "category": "commonsense_reasoning",
    },
    {
        "prompt": "The largest planet in our solar system is",
        "category": "world_knowledge",
    },
    {
        "prompt": (
            "The orchestra tuned quietly while the audience found their seats. When the conductor raised the baton, "
            "the musicians"
        ),
        "category": "language_modeling",
    },
]


@dataclass
class ModelRanking:
    label: str
    leak_score: float
    normalized_leak: float
    quality_score: float
    final_score: float


@dataclass
class QualityDuelResult:
    winner_label: str | None
    win_rate_by_label: dict[str, float]
    prompt_count: int
    prompt_source: str
    judge_model: str


def fetch_eval_round(backend_url: str) -> int:
    import requests

    resp = requests.get(f"{backend_url.rstrip('/')}/rounds/current", timeout=60)
    resp.raise_for_status()
    return int(resp.json()["eval_round"])


def normalize_leak_score(leak_score: float, config: dict) -> float:
    """Same formula as sn38.neurons.validator.qualify."""
    eval_threshold = config.get("min_eval_score", -3.0)
    eval_best = config.get("leak_epsilon", -6.0)
    return max(0.0, min(1.0, (eval_threshold - leak_score) / (eval_threshold - eval_best)))


def compute_final_score(normalized_leak: float, quality_score: float, config: dict) -> float:
    leak_weight = config.get("leak_weight", 0.7)
    quality_weight = config.get("quality_weight", 0.3)
    return leak_weight * normalized_leak + quality_weight * quality_score


def resolve_quality_prompts(
    eval_round: int,
    n_per_category: int,
    *,
    use_openai: bool,
) -> tuple[list[dict], str]:
    if use_openai:
        from sn38.template.quality_prompts import generate_prompts

        prompts = generate_prompts(eval_round, n_per_category=n_per_category)
        if prompts:
            return prompts, f"OpenAI-generated (round {eval_round}, {len(prompts)} prompts)"
        logger.warning("OpenAI prompt generation returned no prompts; using bundled defaults")

    prompts = DEFAULT_QUALITY_PROMPTS[:]
    if n_per_category > 0:
        # Keep bundled set size modest; repeat only if user asks for more categories.
        pass
    return prompts, f"bundled defaults ({len(prompts)} prompts)"


def generate_completions(model, device: torch.device, prompts: list[dict], max_new_tokens: int) -> list[str]:
    from sn38.template.quality import generate_completion

    completions = []
    total = len(prompts)
    for i, item in enumerate(prompts, 1):
        completions.append(generate_completion(model, device, item["prompt"], max_new_tokens=max_new_tokens))
        if i == total or i % max(1, total // 3) == 0:
            logger.info(f"Generated {i}/{total} quality completions")
    return completions


def run_pairwise_quality_duel(
    entry_a: dict,
    entry_b: dict,
    prompts: list[dict],
    device: torch.device,
    max_new_tokens: int,
) -> QualityDuelResult:
    from sn38.template.quality import duel

    uid_a, uid_b = 0, 1
    logger.info(f"Generating quality completions for {entry_a['label']}...")
    comps_a = generate_completions(entry_a["model"], device, prompts, max_new_tokens)
    logger.info(f"Generating quality completions for {entry_b['label']}...")
    comps_b = generate_completions(entry_b["model"], device, prompts, max_new_tokens)

    miner_completions = {uid_a: comps_a, uid_b: comps_b}
    winner_uid = duel(miner_completions, uid_a, uid_b, prompts)

    if winner_uid == uid_a:
        winner_label = entry_a["label"]
        win_rates = {entry_a["label"]: 1.0, entry_b["label"]: 0.0}
    elif winner_uid == uid_b:
        winner_label = entry_b["label"]
        win_rates = {entry_a["label"]: 0.0, entry_b["label"]: 1.0}
    else:
        winner_label = None
        win_rates = {entry_a["label"]: 0.0, entry_b["label"]: 0.0}

    judge_model = os.environ.get("JUDGE_MODEL", "gpt-5.4")
    return QualityDuelResult(
        winner_label=winner_label,
        win_rate_by_label=win_rates,
        prompt_count=len(prompts),
        prompt_source="pairwise duel",
        judge_model=judge_model,
    )


def build_rankings(
    entries: list[dict],
    leak_scores: dict[str, float],
    quality_result: QualityDuelResult,
    config: dict,
) -> list[ModelRanking]:
    rankings = []
    for entry in entries:
        label = entry["label"]
        leak_score = leak_scores[label]
        normalized = normalize_leak_score(leak_score, config)
        quality = quality_result.win_rate_by_label[label]
        final = compute_final_score(normalized, quality, config)
        rankings.append(
            ModelRanking(
                label=label,
                leak_score=leak_score,
                normalized_leak=normalized,
                quality_score=quality,
                final_score=final,
            )
        )
    rankings.sort(key=lambda r: r.final_score, reverse=True)
    return rankings


def require_openai_for_quality() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "Stage-2 quality duels need OPENAI_API_KEY (for the LLM judge). "
            "Set it in your environment or scripts/train/.env"
        )


def run_quality_comparison(
    entry_a: dict,
    entry_b: dict,
    *,
    config: dict,
    leak_scores: dict[str, float],
    backend_url: str,
    eval_round: int | None,
    n_per_category: int,
    device: torch.device,
    max_new_tokens: int,
    use_openai_prompts: bool,
) -> tuple[QualityDuelResult, list[ModelRanking], str]:
    require_openai_for_quality()
    round_num = eval_round if eval_round is not None else fetch_eval_round(backend_url)
    use_openai = use_openai_prompts and bool(os.environ.get("OPENAI_API_KEY"))
    prompts, prompt_source = resolve_quality_prompts(round_num, n_per_category, use_openai=use_openai)

    quality_result = run_pairwise_quality_duel(entry_a, entry_b, prompts, device, max_new_tokens)
    quality_result.prompt_source = prompt_source
    rankings = build_rankings([entry_a, entry_b], leak_scores, quality_result, config)
    return quality_result, rankings, prompt_source
