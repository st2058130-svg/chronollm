"""Compare SN38 models with Stage-2 quality duels only.

Mirrors validator Stage-2 (not Stage-1 leak):
  - 13 categories from sn38.template.quality_prompts.CATEGORIES
  - default 50 prompts/category (650 total), generated once then reused
  - same generate_completion + LLM Judge duel (A/B swap, prompt-level win rate)

Needs OPENAI_API_KEY (judge + optional prompt generation).

Hub downloads are deleted from the HF cache after the duel finishes
(use --keep-hub-cache to retain them). Local checkpoint dirs are never deleted.

Usage:
  python scripts/train/compare_models.py model-a model-b
  python scripts/train/compare_models.py model-a model-b --n-per-category 5
  python scripts/train/compare_models.py model-a model-b --prompts path/to/bank.json
  python scripts/train/compare_models.py model-a model-b --bundled-prompts
"""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sn38.architectures  # noqa: F401
from scripts.train.benchmark_loader import fetch_public_config
from scripts.train.env import load_train_env
from scripts.train.quality_eval import run_quality_comparison
from sn38.template.model_loader import load_model
from sn38.template.quality_prompts import CATEGORIES

load_train_env()

DEFAULT_BACKEND_URL = "https://api.chronollm.com"
DEFAULT_MAX_PARAMETERS = 2_200_000_000
DEFAULT_N_PER_CATEGORY = 50


def _looks_like_local_path(model: str) -> bool:
    path = Path(model)
    if path.exists():
        return True
    if model.startswith((".", "/", "~")) or model.startswith(".\\"):
        return True
    if len(path.parts) >= 1 and path.parts[0].endswith(":"):
        return True
    if "/" in model or "\\" in model:
        if model.count("/") == 1 and "\\" not in model and not model.startswith("checkpoints"):
            return False
        return True
    return False


def _hub_repo_cache_dir(snapshot_path: Path) -> Path | None:
    """Return models--org--name dir for a Hub snapshot path, else None."""
    if snapshot_path.parent.name == "snapshots":
        repo_dir = snapshot_path.parent.parent
        if repo_dir.name.startswith("models--"):
            return repo_dir
    if snapshot_path.name.startswith("models--"):
        return snapshot_path
    return None


def resolve_model_path(model: str, revision: str | None) -> tuple[Path, Path | None]:
    """Return (local_path, hub_cache_dir_to_delete_or_None)."""
    path = Path(model).expanduser()
    if path.exists():
        return path.resolve(), None
    if _looks_like_local_path(model):
        raise SystemExit(
            f"[error] local model path not found: {path}\n"
            f"  cwd={Path.cwd()}\n"
            f"  tip: use an existing checkpoint dir or a Hub id like org/name"
        )
    from huggingface_hub import snapshot_download

    repo_id = model
    rev = revision
    if "@" in model and revision is None:
        repo_id, rev = model.rsplit("@", 1)

    local = Path(snapshot_download(repo_id, revision=rev))
    cache_dir = _hub_repo_cache_dir(local.resolve())
    print(f"[hub] downloaded {repo_id} -> {local}")
    if cache_dir is not None:
        print(f"[hub] will remove cache after duel: {cache_dir}")
    return local, cache_dir


def cleanup_hub_cache(cache_dir: Path | None) -> None:
    if cache_dir is None:
        return
    if not cache_dir.exists():
        print(f"[hub] cache already gone: {cache_dir}")
        return
    try:
        shutil.rmtree(cache_dir)
        print(f"[hub] removed cache: {cache_dir}")
    except OSError as exc:
        print(f"[hub] warning: failed to remove {cache_dir}: {exc}")


def read_train_step(path: Path) -> int | None:
    meta = path / "train_meta.txt"
    if not meta.is_file():
        return None
    for line in meta.read_text(encoding="utf-8").splitlines():
        if line.startswith("step="):
            return int(line.split("=", 1)[1])
    return None


def load_one(label: str, model: str, revision: str | None, device: torch.device):
    path, hub_cache = resolve_model_path(model, revision)
    model_obj, _ = load_model(str(path), device)
    n_params = sum(p.numel() for p in model_obj.parameters())
    step = read_train_step(path)
    return {
        "label": label,
        "path": path,
        "model": model_obj,
        "params": n_params,
        "step": step,
        "hub_cache": hub_cache,
    }


def unload(entry: dict) -> None:
    if "model" in entry:
        del entry["model"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_header(entries: list[dict], device: torch.device, max_parameters: int) -> None:
    print("=== SN38 Stage-2 quality comparison ===")
    print(f"device: {device}")
    print(f"categories ({len(CATEGORIES)}): {', '.join(CATEGORIES.keys())}\n")
    for entry in entries:
        step = f", train_step={entry['step']}" if entry["step"] is not None else ""
        ok = "OK" if entry["params"] <= max_parameters else "OVER LIMIT"
        print(
            f"{entry['label']}: {entry['path']}\n"
            f"  params: {entry['params'] / 1e9:.3f}B ({entry['params']:,}) [{ok}]{step}\n"
        )


def print_quality_result(rankings, quality_result, prompt_source: str, config: dict) -> None:
    quality_weight = config.get("quality_weight", 1.0)
    print("STAGE-2 QUALITY DUEL")
    print(
        f"judge={quality_result.judge_model} | prompts={quality_result.prompt_count} | "
        f"source={prompt_source} | quality_weight={quality_weight}"
    )
    print(
        f"{'rank':<5} {'model':<16} {'quality':>10} {'wins':>8} {'final':>10}"
    )
    print("-" * 54)
    wins = quality_result.prompt_wins_by_label
    for i, row in enumerate(rankings, 1):
        print(
            f"{i:<5} {row.label:<16} {row.quality_score:>10.4f} "
            f"{wins.get(row.label, 0):>8} {row.final_score:>10.4f}"
        )
    print("-" * 54)
    print(f"duel winner: {quality_result.winner_label or 'tie'}")
    print(f"prompt wins: {quality_result.prompt_wins_by_label}")

    if quality_result.category_stats:
        labels = list(quality_result.win_rate_by_label.keys())
        if len(labels) >= 2:
            la, lb = labels[0], labels[1]
            print(f"\n{'category':<24} {'n':>4} {la[:12]:>12} {lb[:12]:>12} {'ties':>6}")
            print("-" * 62)
            for cat in CATEGORIES:
                s = quality_result.category_stats.get(cat)
                if not s:
                    continue
                print(
                    f"{cat:<24} {s.get('n', 0):>4} "
                    f"{s.get(la, 0):>12} {s.get(lb, 0):>12} {s.get('ties', 0):>6}"
                )
    print(
        "\nNOTE: one shared 13×N bank for the duel (same as validator). "
        "Local limit: A vs B only, not full round-robin.\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="SN38 Stage-2 quality compare (LLM-judge duel; no Stage-1 leak)"
    )
    parser.add_argument("model_a", help="Local checkpoint dir or HuggingFace repo id")
    parser.add_argument("model_b", help="Second model (required for quality duel)")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--revision-a", default=None)
    parser.add_argument("--revision-b", default=None)
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Generation length (validator quality default=100)",
    )
    parser.add_argument("--max-parameters", type=int, default=DEFAULT_MAX_PARAMETERS)
    parser.add_argument(
        "--n-per-category",
        type=int,
        default=DEFAULT_N_PER_CATEGORY,
        help=f"OpenAI prompts per category (validator default {DEFAULT_N_PER_CATEGORY})",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Saved quality prompt bank JSON ({prompts:[...]} or list). Skips live generation.",
    )
    parser.add_argument(
        "--openai-prompts",
        action="store_true",
        help="Force fresh OpenAI 13-category generation (default when --prompts not set)",
    )
    parser.add_argument(
        "--bundled-prompts",
        action="store_true",
        help="Use small offline bundled prompts (smoke test only)",
    )
    parser.add_argument(
        "--eval-round",
        type=int,
        default=None,
        help="Eval round seed for OpenAI prompt generation",
    )
    parser.add_argument(
        "--keep-hub-cache",
        action="store_true",
        help="Keep HuggingFace Hub downloads after the duel (default: delete them)",
    )
    args = parser.parse_args()

    if args.bundled_prompts and args.prompts:
        raise SystemExit("Use only one of --bundled-prompts / --prompts")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s | %(message)s")
    logging.getLogger("sn38").setLevel(logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = [
        load_one(args.label_a, args.model_a, args.revision_a, device),
        load_one(args.label_b, args.model_b, args.revision_b, device),
    ]
    print_header(entries, device, args.max_parameters)

    try:
        try:
            config = fetch_public_config(args.backend_url)
        except Exception as exc:
            logging.warning(f"Could not fetch /config ({exc}); using quality_weight=1.0 defaults")
            config = {
                "leak_weight": 0.0,
                "quality_weight": 1.0,
                "min_eval_score": -3.0,
                "leak_epsilon": -11.51,
            }

        # Stage-1 removed: leak scores unused; ranking is quality-only under current API weights.
        leak_scores = {entries[0]["label"]: 0.0, entries[1]["label"]: 0.0}
        use_openai = (args.openai_prompts or args.prompts is None) and not args.bundled_prompts

        print(
            "Validator-style LLM-judge duel (13 categories, one shared prompt bank).\n"
            "Local limit: pairwise A vs B only — not full subnet round-robin.\n"
        )
        quality_result, rankings, prompt_source = run_quality_comparison(
            entries[0],
            entries[1],
            config=config,
            leak_scores=leak_scores,
            backend_url=args.backend_url,
            eval_round=args.eval_round,
            n_per_category=args.n_per_category,
            device=device,
            max_new_tokens=args.max_new_tokens,
            use_openai_prompts=use_openai,
            prompts_path=args.prompts,
        )
        print_quality_result(rankings, quality_result, prompt_source, config)
    finally:
        # Release file handles before deleting Hub cache.
        for entry in entries:
            unload(entry)
        if not args.keep_hub_cache:
            seen: set[Path] = set()
            for entry in entries:
                cache = entry.get("hub_cache")
                if cache is None or cache in seen:
                    continue
                seen.add(cache)
                cleanup_hub_cache(cache)


if __name__ == "__main__":
    main()
