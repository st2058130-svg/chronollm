"""Build a CPT-style JSONL from stanford-oval/ccnews (2024 Jan–May).

Pipeline matches chronollm/cpt.jsonl:
  1) date filter (published_date in 2024-01 .. 2024-05)
  2) known_preprocess (denoise + claim keep + probe rewrite)
  3) enrich_styles (fact/QA/temporal/context/…)
  4) optional shuffle

Downloads one ~2GB parquet shard at a time (HF streaming of the full
2024 split is unreliable), processes Jan–May rows, then deletes the shard.
Uses HF token from the environment if present (HF_TOKEN / HUGGING_FACE_HUB_TOKEN).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from scripts.train.enrich_styles import enrich_record
from scripts.train.known_preprocess import preprocess_for_known

_DATE_RE = re.compile(r"^((?:19|20)\d{2})-(\d{2})-(\d{2})")


def _parse_ymd(raw) -> tuple[int, int, int] | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        ts = float(raw)
        if ts > 1e12:
            ts /= 1000.0
        if ts > 1e9:
            dt = datetime.utcfromtimestamp(ts)
            return dt.year, dt.month, dt.day
        return None
    text = str(raw).strip()
    m = _DATE_RE.match(text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.year, dt.month, dt.day
    except ValueError:
        return None


def _process_row(
    row: dict,
    *,
    year: int,
    min_month: int,
    max_month: int,
    min_chars: int,
) -> dict | None:
    ymd = _parse_ymd(row.get("published_date"))
    if ymd is None:
        return None
    y, m, d = ymd
    if y != year or m < min_month or m > max_month:
        return None

    text = row.get("plain_text") or row.get("text") or ""
    if not isinstance(text, str) or len(text) < min_chars:
        return None

    title = row.get("title")
    if not isinstance(title, str):
        title = None

    pp = preprocess_for_known(
        text,
        target_year=year,
        denoise=True,
        claim_filter=True,
        probe_rewrite=True,
        title=title,
        max_sentences=24,
        max_probe_lines=12,
        min_chars_out=40,
        fallback_lead_sentences=3,
    )
    if pp is None:
        return None

    base = {
        "text": pp,
        "timestamp": f"{y:04d}-{m:02d}-{d:02d}",
        "source": f"stanford_ccnews_{y:04d}_{m:02d}",
        "title": title,
        "preprocessed": True,
        "preprocess_kind": "ccnews",
    }
    return enrich_record(base, target_year=year)


def _shuffle_file(path: Path, *, seed: int) -> None:
    rng = random.Random(seed)
    offsets: list[int] = []
    with path.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    rng.shuffle(offsets)
    shuffled = path.with_suffix(path.suffix + ".shuf.tmp")
    with path.open("rb") as fin, shuffled.open("wb") as fout:
        for pos in offsets:
            fin.seek(pos)
            fout.write(fin.readline())
    os.replace(shuffled, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "cpt_stanford_2024_jan_may.jsonl",
    )
    ap.add_argument("--min-month", type=int, default=1)
    ap.add_argument("--max-month", type=int, default=5)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = no limit")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--keep-shards", action="store_true", help="do not delete downloaded parquets")
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=202451)
    ap.add_argument("--start-shard", type=int, default=0)
    ap.add_argument("--end-shard", type=int, default=-1, help="inclusive; -1 = all")
    args = ap.parse_args()
    do_shuffle = args.shuffle and not args.no_shuffle

    out: Path = args.out
    tmp = out.with_suffix(out.suffix + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[build] stanford-oval/ccnews parquet shards "
        f"keep {args.year}-{args.min_month:02d}..{args.year}-{args.max_month:02d}",
        flush=True,
    )
    files = [
        f
        for f in list_repo_files("stanford-oval/ccnews", repo_type="dataset")
        if f.startswith("2024_") and f.endswith(".parquet")
    ]
    files = sorted(files)
    if args.end_shard < 0:
        files = files[args.start_shard :]
    else:
        files = files[args.start_shard : args.end_shard + 1]
    print(f"[build] shards={len(files)} first={files[0] if files else None}", flush=True)

    kept = skipped = scanned = 0
    t0 = time.time()
    mode = "a" if tmp.exists() and args.start_shard > 0 else "w"

    with tmp.open(mode, encoding="utf-8", newline="\n") as fout:
        for si, filename in enumerate(files):
            print(f"[build] download {filename} ({si+1}/{len(files)})", flush=True)
            local = hf_hub_download(
                repo_id="stanford-oval/ccnews",
                repo_type="dataset",
                filename=filename,
            )
            print(f"[build] process {local}", flush=True)
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=args.batch_size):
                rows = batch.to_pylist()
                for row in rows:
                    scanned += 1
                    enriched = _process_row(
                        row,
                        year=args.year,
                        min_month=args.min_month,
                        max_month=args.max_month,
                        min_chars=args.min_chars,
                    )
                    if enriched is None:
                        skipped += 1
                        continue
                    fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    kept += 1
                    if kept % 5000 == 0:
                        print(
                            f"[build] kept={kept} scanned={scanned} skipped={skipped} "
                            f"{time.time() - t0:.1f}s",
                            flush=True,
                        )
                    if args.max_rows and kept >= args.max_rows:
                        break
                if args.max_rows and kept >= args.max_rows:
                    break
            fout.flush()
            if not args.keep_shards:
                try:
                    os.remove(local)
                    print(f"[build] deleted shard {filename}", flush=True)
                except OSError as exc:
                    print(f"[build] warn: could not delete {local}: {exc}", flush=True)
            if args.max_rows and kept >= args.max_rows:
                print(f"[build] reached max_rows={args.max_rows}", flush=True)
                break

    print(
        f"[build] done kept={kept} scanned={scanned} skipped={skipped} "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )
    if kept == 0 and not tmp.exists():
        raise SystemExit("[build] no rows kept — abort")

    if do_shuffle and tmp.exists() and tmp.stat().st_size > 0:
        print("[build] shuffling...", flush=True)
        _shuffle_file(tmp, seed=args.seed)

    os.replace(tmp, out)
    print(f"[build] wrote {out} bytes={out.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()

