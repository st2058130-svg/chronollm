"""Watch checkpoints/.../best and upload each update to a NEW HF model repo.

Example repos created over time:
  vadimyakob/chrono-2022-live-17
  vadimyakob/chrono-2022-live-18
  ...

Waits for train.py's atomic best.tmp -> best rename so uploads are complete.

Usage (RunPod, 2nd terminal while training):
  export HF_TOKEN=hf_xxx
  python scripts/train/watch_upload_best.py \\
    --dir checkpoints/nanochrono-2022/best \\
    --repo-prefix vadimyakob/chrono-2022-live \\
    --start 17

Optional:
  --interval 30
  --stable-seconds 15
  --once                 upload current best once (next free -N) and exit
  --private
  --inplace              old behavior: overwrite a single --repo (no -N suffix)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, login


def _file_fingerprint(path: Path) -> str:
    """Cheap signature: relative paths + sizes + mtimes (not full file hashes)."""
    if not path.is_dir():
        return ""
    lines: list[str] = []
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        if "/.cache/" in p.as_posix() or p.name.startswith("."):
            continue
        st = p.stat()
        rel = p.relative_to(path).as_posix()
        lines.append(f"{rel}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()


def _ready_for_upload(best_dir: Path) -> bool:
    if not best_dir.is_dir():
        return False
    if best_dir.with_name(best_dir.name + ".tmp").exists():
        return False
    has_weights = any(best_dir.glob("*.safetensors")) or (best_dir / "model.safetensors").is_file()
    has_cfg = (best_dir / "config.json").is_file()
    return has_weights and has_cfg


def wait_stable(best_dir: Path, stable_seconds: float, poll: float = 2.0) -> str | None:
    """Return fingerprint once directory is ready and unchanged for stable_seconds."""
    last = ""
    stable_since: float | None = None
    while True:
        if not _ready_for_upload(best_dir):
            last = ""
            stable_since = None
            time.sleep(poll)
            continue
        fp = _file_fingerprint(best_dir)
        now = time.time()
        if fp != last:
            last = fp
            stable_since = now
            print(f"[watch] change detected in {best_dir} (fp={fp[:8]}…)")
        elif stable_since is not None and (now - stable_since) >= stable_seconds:
            return fp
        time.sleep(poll)


def _commit_message(best_dir: Path) -> str:
    meta = best_dir / "export_meta.txt"
    if meta.is_file():
        first = meta.read_text(encoding="utf-8").splitlines()[0]
        return f"best export ({first})"
    return "best export"


def upload_best(api: HfApi, best_dir: Path, repo_id: str) -> str:
    print(f"[upload] {best_dir} -> https://huggingface.co/{repo_id}")
    return api.upload_folder(
        folder_path=str(best_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=_commit_message(best_dir),
        ignore_patterns=["**/.cache/**", "**/.*"],
    )


def ensure_repo(api: HfApi, repo_id: str, private: bool) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)


def parse_prefix(repo_prefix: str) -> tuple[str, str]:
    """Return (namespace, name_stem) from 'user/chrono-2022-live'."""
    if "/" not in repo_prefix:
        raise SystemExit("[error] --repo-prefix must look like USER/NAME (e.g. vadimyakob/chrono-2022-live)")
    user, name = repo_prefix.split("/", 1)
    if not user or not name:
        raise SystemExit("[error] invalid --repo-prefix")
    return user, name


def highest_existing_index(api: HfApi, repo_prefix: str) -> int:
    """Scan Hub for USER/NAME-N and return max N found, or 0 if none."""
    user, stem = parse_prefix(repo_prefix)
    pattern = re.compile(rf"^{re.escape(stem)}-(\d+)$")
    highest = 0
    try:
        for info in api.list_models(author=user, search=stem):
            # info.id == user/name
            name = info.id.split("/", 1)[-1]
            m = pattern.match(name)
            if m:
                highest = max(highest, int(m.group(1)))
    except Exception as exc:
        print(f"[warn] list_models failed ({exc}); falling back to --start only")
    return highest


def next_versioned_repo_id(api: HfApi, repo_prefix: str, next_n: int) -> tuple[str, int]:
    """Find first free USER/NAME-N starting from next_n."""
    n = max(1, int(next_n))
    while True:
        repo_id = f"{repo_prefix}-{n}"
        try:
            exists = api.repo_exists(repo_id, repo_type="model")
        except Exception:
            exists = False
        if not exists:
            return repo_id, n
        print(f"[watch] {repo_id} already exists; trying {n + 1}")
        n += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-upload each best/ update to a new numbered HF repo"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("checkpoints/nanochrono-2022/best"),
        help="Path to best checkpoint folder",
    )
    parser.add_argument(
        "--repo-prefix",
        default=None,
        help="Base repo id without number, e.g. vadimyakob/chrono-2022-live",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Alias for --repo-prefix (versioned) or single repo with --inplace",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First version number to try (default: 1 + max existing on Hub)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite one repo (--repo) instead of creating -N versions",
    )
    parser.add_argument("--interval", type=float, default=30.0, help="Idle poll seconds")
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=15.0,
        help="Require best/ unchanged this many seconds before upload",
    )
    parser.add_argument("--once", action="store_true", help="Upload once then exit")
    parser.add_argument("--private", action="store_true", help="Create repos as private")
    parser.add_argument("--token", default=None, help="HF token (else HF_TOKEN / login)")
    args = parser.parse_args()

    prefix = args.repo_prefix or args.repo
    if not prefix:
        raise SystemExit("[error] pass --repo-prefix USER/chrono-2022-live (or --repo)")

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)

    if args.inplace:
        next_n = None
        print(f"[watch] INPLACE mode -> always {prefix}")
    else:
        parse_prefix(prefix)
        if args.start is not None:
            next_n = args.start
        else:
            hi = highest_existing_index(api, prefix)
            next_n = hi + 1
            print(f"[watch] highest existing {prefix}-N is {hi}; next={next_n}")
        print(f"[watch] each update -> new repo {prefix}-N")

    last_uploaded = ""
    print(f"[watch] monitoring {args.dir.resolve()}")
    print(f"[watch] interval={args.interval}s stable={args.stable_seconds}s")

    while True:
        if not args.dir.exists():
            print(f"[watch] waiting for {args.dir} to appear...")
            if args.once:
                raise SystemExit(f"[error] missing {args.dir}")
            time.sleep(args.interval)
            continue

        fp = wait_stable(args.dir, args.stable_seconds)
        if fp is None:
            continue
        if fp == last_uploaded:
            if args.once:
                print("[watch] already uploaded this revision; exiting")
                return
            time.sleep(args.interval)
            continue

        try:
            if args.inplace:
                repo_id = prefix
                ensure_repo(api, repo_id, private=args.private)
            else:
                assert next_n is not None
                repo_id, used_n = next_versioned_repo_id(api, prefix, next_n)
                ensure_repo(api, repo_id, private=args.private)
                next_n = used_n + 1

            info = upload_best(api, args.dir, repo_id)
            last_uploaded = fp
            print(f"[upload] done: {repo_id}")
            print(f"[upload] {info}")
            if not args.inplace:
                print(f"[watch] next version will be {prefix}-{next_n}")
        except Exception as exc:
            print(f"[upload] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.once:
                raise SystemExit(1)
            time.sleep(args.interval)
            continue

        if args.once:
            return
        print(f"[watch] idle; next check in {args.interval}s")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
