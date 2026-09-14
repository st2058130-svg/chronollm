"""Compare SN38 models with full Stage-1 leak evaluation (known + unknown per year).

Uses the same scoring code as the validator (sn38.template.leak.evaluate) with live
public thresholds from api.chronollm.com/config. No wallet or on-chain registration
required.

Default benchmarks are built from scripts/train/benchmark_data/facts.json.
Official private TEE probes are not public; pass/fail here is a strong local proxy,
not a guarantee for on-chain evaluation. Use the TEE self-test before submitting.

Usage:
  python scripts/train/compare_models.py checkpoints/nanochrono-2018/latest
  python scripts/train/compare_models.py model-a model-b
  python scripts/train/compare_models.py model-a model-b --year 2018
  python scripts/train/compare_models.py model-a model-b --years 2018,2019
  python scripts/train/compare_models.py model-a --benchmark-dir path/to/exported/json
  python scripts/train/compare_models.py model-a model-b --mode local
  python scripts/train/compare_models.py model-a model-b --quality
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sn38.architectures  # noqa: F401
from scripts.train.benchmark_loader import (
    DEFAULT_FACTS_PATH,
    fetch_public_config,
    fetch_public_years,
    load_facts,
    preload_benchmarks,
)
from scripts.train.env import load_train_env
from scripts.train.local_leak import EVAL_VERSION, evaluate_local
from sn38.template.leak import _score_batch
from sn38.template.model_loader import load_model

load_train_env()

DEFAULT_BACKEND_URL = "https://api.chronollm.com"
DEFAULT_MAX_PARAMETERS = 2_200_000_000
WORST_SCORE = 0.0

LEAK_ITEMS = [
    {"prompt": "The first person to set foot on the Moon was", "phrase": "Neil Armstrong"},
    {"prompt": "The chemical formula for water is", "phrase": "H2O"},
    {"prompt": "The river that flows through London is called the", "phrase": "Thames"},
    {"prompt": "The painter of the Mona Lisa was", "phrase": "Leonardo da Vinci"},
    {"prompt": "World War II ended in the year", "phrase": "1945"},
    {"prompt": "The currency used in Japan is the", "phrase": "Japanese yen"},
    {"prompt": "The theory of relativity was proposed by", "phrase": "Albert Einstein"},
    {"prompt": "The tallest mountain in the world is", "phrase": "Mount Everest"},
    {"prompt": "DNA stands for", "phrase": "deoxyribonucleic acid"},
    {"prompt": "The capital of France is", "phrase": "Paris"},
]

GENERATE_PROMPTS = [
    "The first person to set foot on the Moon was",
    "A recipe for pancakes typically starts by mixing flour, eggs, and",
    "Why do plants at the bottom of a rainforest canopy often have larger leaves than those at the top?",
    "When you mix red and blue paint together, you get",
    "Is it true that humans only use 10% of their brain? Explain why or why not.",
]


@dataclass
class YearEvalResult:
    year: int
    leak_ok: bool
    known_ok: bool
    passed: bool
    median_unknown: float
    median_known: float
    score: float
    unknown_items: int
    known_items: int
    leak_fail_pct: float = 0.0
    known_conf_pct: float = 0.0


@dataclass
class EvalSummary:
    year_results: list[YearEvalResult]
    leak_score: float
    min_eval_score: float
    qualified: bool

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.year_results)


def resolve_model_path(model: str, revision: str | None) -> Path:
    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision))


def read_train_step(path: Path) -> int | None:
    meta = path / "train_meta.txt"
    if not meta.is_file():
        return None
    for line in meta.read_text(encoding="utf-8").splitlines():
        if line.startswith("step="):
            return int(line.split("=", 1)[1])
    return None


def load_one(label: str, model: str, revision: str | None, device: torch.device):
    path = resolve_model_path(model, revision)
    model_obj, _ = load_model(str(path), device)
    n_params = sum(p.numel() for p in model_obj.parameters())
    step = read_train_step(path)
    return {
        "label": label,
        "path": path,
        "model": model_obj,
        "params": n_params,
        "step": step,
    }


def unload(entry: dict) -> None:
    del entry["model"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_year(
    model,
    device: torch.device,
    year: int,
    bench: dict[str, dict],
) -> YearEvalResult:
    failed_leak, median_unknown, leak_fail_pct = evaluate_local(model, device, bench["unknown"])
    passed_known, median_known, known_conf_pct = evaluate_local(model, device, bench["known"])
    leak_ok = not failed_leak
    known_ok = passed_known
    passed = leak_ok and known_ok
    score = (median_unknown - median_known) if passed else WORST_SCORE
    return YearEvalResult(
        year=year,
        leak_ok=leak_ok,
        known_ok=known_ok,
        passed=passed,
        median_unknown=median_unknown,
        median_known=median_known,
        score=score,
        unknown_items=len(bench["unknown"].get("items", [])),
        known_items=len(bench["known"].get("items", [])),
        leak_fail_pct=leak_fail_pct * 100.0,
        known_conf_pct=known_conf_pct * 100.0,
    )


def run_stage1_eval(
    entry: dict,
    device: torch.device,
    benchmarks: dict[int, dict[str, dict]],
    min_eval_score: float,
) -> EvalSummary:
    year_results = [
        evaluate_year(entry["model"], device, year, benchmarks[year])
        for year in sorted(benchmarks)
    ]
    leak_score = sum(r.score for r in year_results) / len(year_results) if year_results else WORST_SCORE
    return EvalSummary(
        year_results=year_results,
        leak_score=leak_score,
        min_eval_score=min_eval_score,
        qualified=leak_score < min_eval_score,
    )


def print_header(entries: list[dict], device: torch.device, max_parameters: int, mode: str, benchmark_source: str) -> None:
    print("=== SN38 model comparison ===")
    print(f"eval engine: {EVAL_VERSION}")
    print(f"mode: {mode}")
    print(f"benchmarks: {benchmark_source}")
    print(f"device: {device}\n")
    for entry in entries:
        step = f", train_step={entry['step']}" if entry["step"] is not None else ""
        ok = "OK" if entry["params"] <= max_parameters else "OVER LIMIT"
        print(
            f"{entry['label']}: {entry['path']}\n"
            f"  params: {entry['params'] / 1e9:.3f}B ({entry['params']:,}) [{ok}]{step}\n"
        )


def print_eval_table(label: str, summary: EvalSummary) -> None:
    print(f"STAGE-1 EVAL — {label}")
    print(
        f"{'year':<6} {'leak':<6} {'known':<6} {'pass':<6} "
        f"{'#unk':>5} {'#kn':>5} {'unk_fail%':>9} {'kn_conf%':>9} "
        f"{'med_unk':>10} {'med_kn':>10} {'score':>10}"
    )
    print("-" * 96)
    for r in summary.year_results:
        print(
            f"{r.year:<6} "
            f"{'PASS' if r.leak_ok else 'FAIL':<6} "
            f"{'PASS' if r.known_ok else 'FAIL':<6} "
            f"{'PASS' if r.passed else 'FAIL':<6} "
            f"{r.unknown_items:>5} {r.known_items:>5} "
            f"{r.leak_fail_pct:>8.1f}% {r.known_conf_pct:>8.1f}% "
            f"{r.median_unknown:>10.4f} {r.median_known:>10.4f} {r.score:>10.4f}"
        )
    print("-" * 96)
    print(f"leak_score (avg year score): {summary.leak_score:.4f}")
    print(f"min_eval_score (qualify if lower): {summary.min_eval_score:.4f}")
    print(f"stage-1 pass (all years): {'PASS' if summary.all_passed else 'FAIL'}")
    print(f"stage-1 qualify for quality: {'YES' if summary.qualified else 'NO'}\n")


def print_eval_compare(a: EvalSummary, b: EvalSummary) -> None:
    print("STAGE-1 COMPARISON")
    print(f"{'metric':<28} {'A':>12} {'B':>12} {'better':>10}")
    print("-" * 64)
    metrics = [
        ("leak_score", a.leak_score, b.leak_score, "lower"),
        ("stage-1 pass", float(a.all_passed), float(b.all_passed), "higher"),
        ("qualified", float(a.qualified), float(b.qualified), "higher"),
    ]
    for name, va, vb, rule in metrics:
        if rule == "lower":
            winner = "A" if va < vb else ("B" if vb < va else "tie")
        else:
            winner = "A" if va > vb else ("B" if vb > va else "tie")
        if name in ("stage-1 pass", "qualified"):
            sa = "YES" if va else "NO"
            sb = "YES" if vb else "NO"
        else:
            sa = f"{va:.4f}"
            sb = f"{vb:.4f}"
        print(f"{name:<28} {sa:>12} {sb:>12} {winner:>10}")
    print()


def print_leak_table(scores_a: list[float], scores_b: list[float]) -> None:
    print("LOCAL LEAK PROBES (sum of log-probs; higher = more confident in phrase)")
    print(f"{'phrase':<30} {'A':>10} {'B':>10} {'B-A':>10} {'better':>8}")
    print("-" * 72)
    wins_a = wins_b = ties = 0
    for item, sa, sb in zip(LEAK_ITEMS, scores_a, scores_b):
        delta = sb - sa
        if abs(delta) < 0.05:
            winner = "tie"
            ties += 1
        elif delta > 0:
            winner = "B"
            wins_b += 1
        else:
            winner = "A"
            wins_a += 1
        print(f"{item['phrase']:<30} {sa:>10.2f} {sb:>10.2f} {delta:>10.2f} {winner:>8}")
    med_a = sorted(scores_a)[len(scores_a) // 2]
    med_b = sorted(scores_b)[len(scores_b) // 2]
    print("-" * 72)
    print(f"{'median':<30} {med_a:>10.2f} {med_b:>10.2f} {med_b - med_a:>10.2f}")
    print(f"probe wins: A={wins_a} B={wins_b} ties={ties}\n")


def print_ranking_table(rankings, config: dict, quality_meta: dict) -> None:
    leak_weight = config.get("leak_weight", 0.7)
    quality_weight = config.get("quality_weight", 0.3)
    print("LEADERBOARD-STYLE RANKING (local estimate)")
    print(
        f"weights: leak={leak_weight}, quality={quality_weight} | "
        f"judge={quality_meta['judge_model']} | prompts={quality_meta['prompt_source']}"
    )
    print(
        f"{'rank':<5} {'model':<8} {'leak_score':>12} {'norm_leak':>10} "
        f"{'quality':>10} {'final':>10}"
    )
    print("-" * 68)
    for i, row in enumerate(rankings, 1):
        print(
            f"{i:<5} {row.label:<8} {row.leak_score:>12.4f} {row.normalized_leak:>10.4f} "
            f"{row.quality_score:>10.4f} {row.final_score:>10.4f}"
        )
    print("-" * 68)
    winner = rankings[0]
    print(f"quality duel winner: {quality_meta['duel_winner'] or 'tie'}")
    if quality_meta.get("prompt_wins"):
        print(f"prompt wins: {quality_meta['prompt_wins']}")
    print(f"rank #1 by final score: {winner.label} ({winner.final_score:.4f})\n")


def print_generation_table(entries: list[dict], max_new_tokens: int) -> None:
    if len(entries) < 2:
        return
    a, b = entries[0], entries[1]
    print("GENERATION (same prompts)")
    print("=" * 72)
    for prompt in GENERATE_PROMPTS:
        out_a = a["model"].generate(prompt, max_new_tokens=max_new_tokens)
        out_b = b["model"].generate(prompt, max_new_tokens=max_new_tokens)
        print(f"Q: {prompt}\n")
        print(f"  A: {out_a[:220]}")
        print(f"  B: {out_b[:220]}\n")


def parse_years(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(y.strip()) for y in value.split(",") if y.strip()]


def benchmark_source_label(benchmark_dir: Path | None, facts_path: Path) -> str:
    if benchmark_dir is not None:
        return f"exported JSON ({benchmark_dir})"
    return f"local fact bank ({facts_path.name}) + live /config thresholds"


def resolve_eval_years(args) -> list[int]:
    """Pick cutoff year(s): --year / --years override live API /years."""
    if args.year is not None:
        return [args.year]
    if parsed := parse_years(args.years):
        return parsed
    return fetch_public_years(args.backend_url, args.round)


def main():
    parser = argparse.ArgumentParser(description="Compare SN38-compatible models")
    parser.add_argument("model_a", help="Local checkpoint dir or HuggingFace repo id")
    parser.add_argument("model_b", nargs="?", default=None, help="Optional second model")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--revision-a", default=None)
    parser.add_argument("--revision-b", default=None)
    parser.add_argument(
        "--mode",
        choices=("eval", "local"),
        default="eval",
        help="eval = full Stage-1 known/unknown; local = quick hardcoded probes",
    )
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--round", type=int, default=None, help="Submission round for /years lookup")
    parser.add_argument("--year", type=int, default=None, help="Single cutoff year, e.g. 2018 (overrides API default)")
    parser.add_argument("--years", default=None, help="Comma-separated cutoff years, e.g. 2018,2019")
    parser.add_argument(
        "--facts",
        type=Path,
        default=DEFAULT_FACTS_PATH,
        help="Chronological fact bank JSON used to build benchmarks",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Optional exported benchmarks/<year>/{unknown,known}.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--max-parameters", type=int, default=DEFAULT_MAX_PARAMETERS)
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Run Stage-2 LLM-judge quality duel (needs 2 models + OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--quality-prompts-per-category",
        type=int,
        default=6,
        help=(
            "OpenAI-generated prompts per category when using --openai-prompts "
            "(validator default is 50; lower = cheaper local checks). "
            "Categories now include math/truthfulness/pronoun/paraphrase/word_sense."
        ),
    )
    parser.add_argument(
        "--openai-prompts",
        action="store_true",
        help="Generate fresh quality prompts via OpenAI (default: bundled prompts)",
    )
    parser.add_argument(
        "--eval-round",
        type=int,
        default=None,
        help="Eval round seed for OpenAI prompt generation",
    )
    args = parser.parse_args()

    if args.quality and args.mode != "eval":
        raise SystemExit("--quality requires --mode eval (the default)")
    if args.quality and args.model_b is None:
        raise SystemExit("--quality requires two models")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s | %(message)s")
    logging.getLogger("sn38").setLevel(logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = [load_one(args.label_a, args.model_a, args.revision_a, device)]
    if args.model_b is not None:
        entries.append(load_one(args.label_b, args.model_b, args.revision_b, device))

    source = benchmark_source_label(args.benchmark_dir, args.facts)
    print_header(entries, device, args.max_parameters, args.mode, source)

    if args.mode == "eval":
        config = fetch_public_config(args.backend_url)
        years = resolve_eval_years(args)
        facts = None if args.benchmark_dir is not None else load_facts(args.facts)
        benchmarks = preload_benchmarks(years, config, facts=facts, benchmark_dir=args.benchmark_dir)

        min_eval_score = config.get("min_eval_score", -3.0)
        print(f"years: {years}")
        print(f"thresholds: leak={config.get('leak_threshold', 0.1)}, known={config.get('known_threshold', 0.7)}")
        print(f"epsilon: {config.get('leak_epsilon', -11.51)}, known_cutoff_weight: {config.get('known_cutoff_weight', 5)}")
        print(f"min_eval_score: {min_eval_score}")
        print(f"score weights: leak={config.get('leak_weight', 0.7)}, quality={config.get('quality_weight', 0.3)}")
        print(
            "NOTE: official validator probes are private (TEE API). "
            "This run uses the same scoring code and live thresholds, but different probe text.\n"
        )

        summaries = []
        for entry in entries:
            summaries.append(
                run_stage1_eval(
                    entry,
                    device,
                    benchmarks,
                    min_eval_score,
                )
            )
            print_eval_table(entry["label"], summaries[-1])

        if len(summaries) == 2:
            print_eval_compare(summaries[0], summaries[1])
            scores_a = _score_batch(entries[0]["model"], device, LEAK_ITEMS)
            scores_b = _score_batch(entries[1]["model"], device, LEAK_ITEMS)
            print_leak_table(scores_a, scores_b)

        if args.quality:
            from scripts.train.quality_eval import run_quality_comparison

            leak_scores = {entries[i]["label"]: summaries[i].leak_score for i in range(2)}
            print("STAGE-2 QUALITY EVAL")
            print(
                "Running LLM-judge duels (same code as validator). "
                "Quality is pairwise here (1 opponent), not full subnet round-robin.\n"
            )
            quality_result, rankings, prompt_source = run_quality_comparison(
                entries[0],
                entries[1],
                config=config,
                leak_scores=leak_scores,
                backend_url=args.backend_url,
                eval_round=args.eval_round,
                n_per_category=args.quality_prompts_per_category,
                device=device,
                max_new_tokens=args.max_new_tokens,
                use_openai_prompts=args.openai_prompts,
            )
            print_ranking_table(
                rankings,
                config,
                {
                    "judge_model": quality_result.judge_model,
                    "prompt_source": prompt_source,
                    "duel_winner": quality_result.winner_label,
                    "prompt_wins": quality_result.prompt_wins_by_label,
                },
            )
            print(
                "NOTE: dashboard quality is prompt-level win rate across round-robin duels.\n"
                "Here you only duel A vs B; quality_score is each model's prompt win fraction "
                f"({quality_result.prompt_wins_by_label}).\n"
            )

        if not args.no_generation:
            print_generation_table(entries, args.max_new_tokens)
    else:
        if len(entries) < 2:
            raise SystemExit("Local mode needs two models to compare probe scores.")
        scores_a = _score_batch(entries[0]["model"], device, LEAK_ITEMS)
        scores_b = _score_batch(entries[1]["model"], device, LEAK_ITEMS)
        print_leak_table(scores_a, scores_b)
        if not args.no_generation:
            print_generation_table(entries, args.max_new_tokens)
        print("NOTE: local mode uses 10 hardcoded probes only.")

    for entry in entries:
        unload(entry)


if __name__ == "__main__":
    main()
