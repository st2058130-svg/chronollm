"""Local Stage-2 quality evaluation for compare_models.py.

Runs the same LLM-judge duel flow as sn38.template.quality, adapted for two
loaded models without on-chain submissions.

Aligned with current validator quality rules:
- prompts are completions *or* questions (~70% questions when OpenAI-generated);
- duel returns per-prompt wins (wins_a, wins_b, total), not a binary match winner;
- quality_score is prompt-level win rate.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# Bundled offline set: mix of completions + questions across current categories.
# Prefer questions (validator aims ~70% questions / 30% completions).
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
            "A library charges $0.25 per day for overdue books. Maria returned her book 12 days late. "
            "How much does she owe and why might libraries use this system?"
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
        "prompt": (
            "A sign at a park reads: Dogs must be carried on the escalator. "
            "What does this actually mean, and why could it be misunderstood?"
        ),
        "category": "language_understanding",
    },
    {
        "prompt": "The process by which plants use sunlight to convert carbon dioxide and water into sugar is called",
        "category": "world_knowledge",
    },
    {
        "prompt": "Why do some metals rust when exposed to water while others do not?",
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
            "If you leave an ice cube on a metal tray and another on a wooden board, which melts faster and why?"
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
        "prompt": (
            "A traveler arrives at a village where every door is painted red except one, which is black. "
            "What might this suggest about the village?"
        ),
        "category": "language_modeling",
    },
    {
        "prompt": (
            "When the dam upstream released extra water during heavy rains, the downstream farms flooded, "
            "which contaminated the wells, so the town"
        ),
        "category": "causal_reasoning",
    },
    {
        "prompt": "Why might removing wolves from a national park eventually lead to riverbanks eroding faster?",
        "category": "causal_reasoning",
    },
    {
        "prompt": "Every mammal breathes air. Whales are mammals. Therefore, whales",
        "category": "logical_inference",
    },
    {
        "prompt": (
            "All roses are flowers. Some flowers fade quickly. "
            "Can we conclude that some roses fade quickly?"
        ),
        "category": "logical_inference",
    },
    {
        "prompt": "First she mixed the flour and eggs, then she added the sugar. After stirring for two minutes, she",
        "category": "temporal_reasoning",
    },
    {
        "prompt": (
            "Tom woke up, ate breakfast, then realized he had forgotten to set his alarm. "
            "In what order did these events actually happen?"
        ),
        "category": "temporal_reasoning",
    },
    {
        "prompt": (
            "A store discounts a $80 jacket by 25%, then adds 10% sales tax to the discounted price. "
            "The final price is"
        ),
        "category": "math_reasoning",
    },
    {
        "prompt": (
            "A tank fills at 3 liters per minute but leaks at 0.5 liters per minute. "
            "If it starts half-full at 50 liters capacity, how long until it overflows?"
        ),
        "category": "math_reasoning",
    },
    {
        "prompt": (
            "Despite what many people believe, the Great Wall of China is actually not visible from space because it is"
        ),
        "category": "truthfulness",
    },
    {
        "prompt": "Is it true that humans only use 10% of their brain? Explain why or why not.",
        "category": "truthfulness",
    },
    {
        "prompt": "The bottle did not fit in the suitcase because it was too large, so they decided to",
        "category": "pronoun_resolution",
    },
    {
        "prompt": (
            "The teacher told the student that she needed to improve. "
            "Who needs to improve, and how can you tell?"
        ),
        "category": "pronoun_resolution",
    },
    {
        "prompt": (
            "Statement A: The cat chased the mouse. Statement B: The mouse was pursued by the cat. "
            "These two statements"
        ),
        "category": "paraphrase_detection",
    },
    {
        "prompt": (
            'Do these two sentences mean the same thing? '
            '"She failed to avoid the obstacle" and "She hit the obstacle."'
        ),
        "category": "paraphrase_detection",
    },
    {
        "prompt": (
            "The manager said the pitch needed more polish before the board would consider it. "
            "Whether pitch refers to a sales presentation or a playing field changes the meaning entirely, "
            "and the clue is"
        ),
        "category": "word_sense",
    },
    {
        "prompt": (
            'In "The doctor told her she had a rare condition and should avoid drafts," '
            "does drafts mean air currents, written documents, or preliminary versions? "
            "What makes this ambiguous?"
        ),
        "category": "word_sense",
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
    prompt_wins_by_label: dict[str, int]
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
    return prompts, f"bundled defaults ({len(prompts)} prompts; Q+completion mix)"


def generate_completions(model, device: torch.device, prompts: list[dict], max_new_tokens: int) -> list[str]:
    """Generate model responses (completion or answer) for quality prompts."""
    from sn38.template.quality import generate_completion

    completions = []
    total = len(prompts)
    for i, item in enumerate(prompts, 1):
        completions.append(generate_completion(model, device, item["prompt"], max_new_tokens=max_new_tokens))
        if i == total or i % max(1, total // 3) == 0:
            logger.info(f"Generated {i}/{total} quality responses")
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
    logger.info(f"Generating quality responses for {entry_a['label']}...")
    comps_a = generate_completions(entry_a["model"], device, prompts, max_new_tokens)
    logger.info(f"Generating quality responses for {entry_b['label']}...")
    comps_b = generate_completions(entry_b["model"], device, prompts, max_new_tokens)

    miner_completions = {uid_a: comps_a, uid_b: comps_b}
    wins_a, wins_b, n_prompts = duel(miner_completions, uid_a, uid_b, prompts)

    # Match validator: prompt-level win rate (not binary match winner).
    rate_a = wins_a / max(1, n_prompts)
    rate_b = wins_b / max(1, n_prompts)
    if rate_a > rate_b:
        winner_label = entry_a["label"]
    elif rate_b > rate_a:
        winner_label = entry_b["label"]
    else:
        winner_label = None

    judge_model = os.environ.get("JUDGE_MODEL", "gpt-5.4")
    return QualityDuelResult(
        winner_label=winner_label,
        win_rate_by_label={entry_a["label"]: rate_a, entry_b["label"]: rate_b},
        prompt_count=n_prompts,
        prompt_wins_by_label={entry_a["label"]: wins_a, entry_b["label"]: wins_b},
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
