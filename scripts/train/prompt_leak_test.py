"""Simple Stage-1 known/unknown probe test from your dated prompts.

Uses the same log-prob scoring as the validator (sn38.template.leak) and the
live (or pasted) /config thresholds:

  leak_epsilon, leak_threshold, known_threshold, known_cutoff_weight

Prompt JSON format (list):
  [
    {"prompt": "In 2021, ...", "phrase": "answer", "year": 2021},
    {"prompt": "In 2023, ...", "phrase": "answer", "year": 2023}
  ]

For cutoff year Y:
  known   = facts with year <= Y  (want HIGH kn_conf%)
  unknown = facts with year >  Y  (want LOW  unk_fail%)

Examples:
  python scripts/train/prompt_leak_test.py checkpoints/nanochrono-2022/latest \\
    --year 2022 --prompts scripts/train/benchmark_data/example_prompts_2022.json

  python scripts/train/prompt_leak_test.py my-model --year 2022 \\
    --prompts my_facts.json --config-json '{"leak_epsilon":-11.51,...}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sn38.architectures  # noqa: F401
from scripts.train.benchmark_loader import build_benchmark, fetch_public_config
from scripts.train.env import load_train_env
from scripts.train.local_leak import EVAL_VERSION, evaluate_local
from sn38.template.model_loader import load_model

DEFAULT_BACKEND_URL = "https://api.chronollm.com"

load_train_env()


def resolve_model_path(model: str, revision: str | None) -> str:
    path = Path(model)
    if path.exists():
        return str(path)
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision)

# Defaults matching the public /config snapshot you pasted.
DEFAULT_CONFIG = {
    "max_parameters": 2_200_000_000,
    "max_model_bytes": 8_000_000_000,
    "max_eval_seconds": 120,
    "leak_weight": 0.0,
    "quality_weight": 1.0,
    "top_n_for_quality": 30,
    "min_eval_score": -3.0,
    "leak_epsilon": -11.51,
    "leak_threshold": 0.1,
    "known_threshold": 0.7,
    "known_cutoff_weight": 5,
    "emission_pct": 1.0,
}


def load_prompts(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(f"[error] prompts must be a non-empty JSON list: {path}")
    for i, row in enumerate(data):
        for key in ("prompt", "phrase", "year"):
            if key not in row:
                raise SystemExit(f"[error] item {i} missing {key!r}")
        row["year"] = int(row["year"])
    return data


def resolve_config(args) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if args.config_json:
        raw = args.config_json
        if Path(raw).is_file():
            raw = Path(raw).read_text(encoding="utf-8")
        cfg.update(json.loads(raw))
    elif not args.no_fetch_config:
        try:
            live = fetch_public_config(args.backend_url)
            cfg.update(live)
            print(f"[config] fetched {args.backend_url.rstrip('/')}/config")
        except Exception as exc:
            print(f"[config] fetch failed ({exc}); using baked defaults")
    else:
        print("[config] using baked defaults (--no-fetch-config)")
    return cfg


def print_report(
    *,
    model_path: str,
    year: int,
    config: dict,
    prompts_path: Path,
    n_facts: int,
    known_bench: dict,
    unknown_bench: dict,
    known_hit: bool,
    known_median: float,
    known_ratio: float,
    unk_hit: bool,
    unk_median: float,
    unk_ratio: float,
) -> None:
    # known: hit means ratio > known_threshold → PASS
    # unknown/leak: hit means ratio > leak_threshold → FAIL
    leak_ok = not unk_hit
    known_ok = known_hit
    passed = leak_ok and known_ok
    score = (unk_median - known_median) if passed else 0.0

    print("=== prompt_leak_test (custom dated prompts) ===")
    print(f"eval engine: {EVAL_VERSION}")
    print(f"model: {model_path}")
    print(f"prompts: {prompts_path} ({n_facts} facts)")
    print(f"cutoff year: {year}")
    print(
        f"thresholds: leak={config.get('leak_threshold')} "
        f"known={config.get('known_threshold')} "
        f"epsilon={config.get('leak_epsilon')} "
        f"known_cutoff_weight={config.get('known_cutoff_weight')}"
    )
    print(
        f"weights (info): leak_weight={config.get('leak_weight')} "
        f"quality_weight={config.get('quality_weight')} "
        f"min_eval_score={config.get('min_eval_score')}"
    )
    print()
    print(
        f"{'year':<6} {'leak':<6} {'known':<6} {'pass':<6} "
        f"{'#unk':>5} {'#kn':>5} {'unk_fail%':>9} {'kn_conf%':>9} "
        f"{'med_unk':>10} {'med_kn':>10} {'score':>10}"
    )
    print("-" * 96)
    print(
        f"{year:<6} "
        f"{'PASS' if leak_ok else 'FAIL':<6} "
        f"{'PASS' if known_ok else 'FAIL':<6} "
        f"{'PASS' if passed else 'FAIL':<6} "
        f"{len(unknown_bench['items']):>5} {len(known_bench['items']):>5} "
        f"{unk_ratio * 100:>8.1f}% {known_ratio * 100:>8.1f}% "
        f"{unk_median:>10.4f} {known_median:>10.4f} {score:>10.4f}"
    )
    print("-" * 96)
    print(
        "unk_fail% = share of post-cutoff probes with score > epsilon (want LOW)\n"
        "kn_conf%  = share of pre-cutoff probes with score > epsilon (want HIGH)\n"
        "score     = med_unk - med_kn when both PASS, else 0.0"
    )
    if passed and score < float(config.get("min_eval_score", -3.0)):
        print(
            f"qualify vs min_eval_score={config.get('min_eval_score')}: YES "
            f"(score {score:.4f} < threshold)"
        )
    elif passed:
        print(
            f"qualify vs min_eval_score={config.get('min_eval_score')}: NO "
            f"(score {score:.4f} not lower than threshold)"
        )
    else:
        print("qualify vs min_eval_score: NO (leak/known gate failed)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a model on your dated known/unknown prompts"
    )
    parser.add_argument("model", help="Local checkpoint dir or HF repo id")
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="JSON list of {prompt, phrase, year}",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Model cutoff year (known: year<=Y, unknown: year>Y)",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument(
        "--config-json",
        default=None,
        help="Paste /config JSON string or path (overrides fetch)",
    )
    parser.add_argument(
        "--no-fetch-config",
        action="store_true",
        help="Skip API /config; use baked defaults from your snapshot",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.prompts.is_file():
        raise SystemExit(f"[error] prompts file not found: {args.prompts}")

    config = resolve_config(args)
    facts = load_prompts(args.prompts)
    known_bench = build_benchmark(args.year, known=True, config=config, facts=facts)
    unknown_bench = build_benchmark(args.year, known=False, config=config, facts=facts)

    if not known_bench["items"]:
        raise SystemExit(f"[error] no known items with year <= {args.year}")
    if not unknown_bench["items"]:
        raise SystemExit(
            f"[error] no unknown items with year > {args.year} "
            "(add some post-cutoff facts to test leak)"
        )

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[load] {args.model} on {device}")
    model_path = resolve_model_path(args.model, args.revision)
    model, _ = load_model(model_path, device)

    n_params = sum(p.numel() for p in model.parameters())
    max_p = int(config.get("max_parameters", DEFAULT_CONFIG["max_parameters"]))
    print(f"[params] {n_params / 1e9:.3f}B / max {max_p / 1e9:.1f}B")

    unk_hit, unk_median, unk_ratio = evaluate_local(model, device, unknown_bench)
    known_hit, known_median, known_ratio = evaluate_local(model, device, known_bench)

    print_report(
        model_path=args.model,
        year=args.year,
        config=config,
        prompts_path=args.prompts,
        n_facts=len(facts),
        known_bench=known_bench,
        unknown_bench=unknown_bench,
        known_hit=known_hit,
        known_median=known_median,
        known_ratio=known_ratio,
        unk_hit=unk_hit,
        unk_median=unk_median,
        unk_ratio=unk_ratio,
    )


if __name__ == "__main__":
    main()
