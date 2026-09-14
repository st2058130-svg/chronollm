"""SN38 Nanochrono cutoff pretraining — revision 3.

Main design choices:
- cutoff-safe weighted/interleaved FineWeb-Edu training stream;
- independently trained ~32K byte-level BPE tokenizer using only cutoff-safe data;
- 28-layer Nanochrono made possible by the smaller vocabulary;
- FP32 master parameters with BF16 autocast compute (no destructive whole-model BF16 cast);
- explicit stable initialization because Nanochrono's `_init_weights()` is intentionally empty;
- exact stream/buffer resume, held-out validation, and best-checkpoint selection;
- compact BF16 inference export for the final Hugging Face model (<8 GB target).

Examples:
  python scripts/train/train.py --config scripts/train/config_2018.yaml --smoke
  python scripts/train/train.py --config scripts/train/config_2018.yaml
  python scripts/train/train.py --config scripts/train/config_2018.yaml \
      --resume checkpoints/nanochrono-2018/latest
  python scripts/train/train.py --config scripts/train/config_2019.yaml \
      --init-from checkpoints/nanochrono-2018/latest
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sn38.architectures  # noqa: F401
from sn38.architectures.nanochrono.configuration_nanochrono import NanochronoConfig
from sn38.architectures.nanochrono.modeling_nanochrono import NanochronoForCausalLM
from scripts.train.data import build_packed_stream, build_text_stream, validate_cutoff_dumps
from scripts.train.env import load_train_env

DEFAULT_MAX_PARAMETERS = 2_200_000_000


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _id_or_fallback(value, fallback):
    return value if value is not None else fallback


def _year_weights(cfg: dict) -> dict[int, float] | None:
    values = cfg.get("data", {}).get("year_weights", {})
    return {int(k): float(v) for k, v in values.items()} or None


def _finalize_tokenizer(tok, save_dir: Path, model_max_length: int):
    if tok.eos_token_id is None:
        raise SystemExit("[error] tokenizer must define EOS")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = int(model_max_length)
    save_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(save_dir)
    return tok


def build_pretrained_tokenizer(name: str, save_dir: Path, model_max_length: int):
    tok = AutoTokenizer.from_pretrained(name)
    return _finalize_tokenizer(tok, save_dir, model_max_length)


def build_or_train_cutoff_tokenizer(cfg: dict, dumps: list[str], default_dir: Path):
    """Train/load our own byte-level BPE using only cutoff-safe training crawls.

    The tokenizer is intentionally trained from the *training* dumps, never the
    held-out validation dumps.  Its source stream uses the same year-mixture
    policy as model pretraining, which keeps tokenizer provenance cutoff-safe.
    """
    tc = dict(cfg.get("tokenizer", {}))
    mode = str(tc.get("mode", "train_or_load"))
    model_max_length = int(cfg.get("model", {}).get("max_position_embeddings", 2048))

    if mode == "pretrained":
        name = str(tc.get("name", cfg.get("tokenizer_name", "gpt2")))
        return build_pretrained_tokenizer(name, default_dir, model_max_length)

    save_dir = Path(tc.get("path", default_dir))
    if mode == "train_or_load" and (save_dir / "tokenizer.json").is_file():
        print(f"[tok] loading cutoff tokenizer from {save_dir.resolve()}")
        tok = AutoTokenizer.from_pretrained(save_dir)
        return _finalize_tokenizer(tok, save_dir, model_max_length)
    if mode not in {"train", "train_or_load"}:
        raise SystemExit(f"[error] unsupported tokenizer.mode={mode!r}")

    vocab_size = int(tc.get("vocab_size", 32768))
    train_documents = int(tc.get("train_documents", 50_000))
    max_chars = int(tc.get("max_chars_per_document", 16_384))
    min_frequency = int(tc.get("min_frequency", 2))
    seed = int(tc.get("seed", int(cfg.get("train", {}).get("seed", 42)) + 17_000))
    bos = str(tc.get("bos_token", "<|bos|>"))
    eos = str(tc.get("eos_token", "<|eos|>"))
    unk = str(tc.get("unk_token", "<|unk|>"))

    # Optional smaller crawl set for tokenizer only (avoids opening all train dumps in RAM).
    tok_dumps = list(tc.get("dumps") or dumps)
    validate_cutoff_dumps(tok_dumps, int(cfg["year"]))
    if set(tok_dumps) - set(dumps):
        raise SystemExit(
            f"[error] tokenizer.dumps must be a subset of training dumps: "
            f"{sorted(set(tok_dumps) - set(dumps))}"
        )

    print(
        f"[tok] training cutoff-safe byte-level BPE vocab={vocab_size:,} "
        f"docs={train_documents:,} dumps={len(tok_dumps)} -> {save_dir.resolve()}"
    )
    data_cfg = cfg.get("data", {})
    shuffle_buf = int(tc.get("shuffle_buffer_size", data_cfg.get("shuffle_buffer_size", 1024)))
    max_open = tc.get("max_open_sources", data_cfg.get("max_open_sources"))
    text_stream = build_text_stream(
        tok_dumps,
        cfg.get("dataset", "HuggingFaceFW/fineweb-edu"),
        cutoff_year=int(cfg["year"]),
        year_weights=_year_weights(cfg),
        docs_per_turn=int(data_cfg.get("docs_per_turn", 32)),
        shuffle_buffer_size=shuffle_buf,
        seed=seed,
        max_open_sources=int(max_open) if max_open is not None else None,
        open_explore_prob=float(
            tc.get("open_explore_prob", data_cfg.get("open_explore_prob", 0.02))
        ),
    )

    def limited_texts():
        for text in itertools.islice(text_stream, train_documents):
            yield text[:max_chars] if max_chars > 0 else text

    backend = Tokenizer(models.BPE(unk_token=unk))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[bos, eos, unk],
        show_progress=True,
    )
    backend.train_from_iterator(limited_texts(), trainer=trainer, length=train_documents)

    bos_id = backend.token_to_id(bos)
    eos_id = backend.token_to_id(eos)
    if bos_id is None or eos_id is None:
        raise SystemExit("[error] custom tokenizer failed to create BOS/EOS")
    backend.post_processor = processors.TemplateProcessing(
        single=f"{bos} $A",
        pair=f"{bos} $A $B",
        special_tokens=[(bos, bos_id)],
    )

    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=bos,
        eos_token=eos,
        pad_token=eos,
        unk_token=unk,
        model_max_length=model_max_length,
        clean_up_tokenization_spaces=False,
    )
    # Helpful metadata. BOS insertion itself is implemented by the backend
    # TemplateProcessing above and is bypassed by add_special_tokens=False.
    tok.init_kwargs["add_bos_token"] = True
    _finalize_tokenizer(tok, save_dir, model_max_length)
    print(
        f"[tok] trained vocab={len(tok):,} bos={tok.bos_token_id} "
        f"eos={tok.eos_token_id} pad={tok.pad_token_id}"
    )
    if len(tok) != vocab_size:
        print(f"[warn] requested vocab={vocab_size:,}, trained vocab={len(tok):,}")
    return tok


def _validate_model_cfg(m: dict) -> None:
    h = int(m.get("hidden_size", 1792))
    n_heads = int(m.get("num_attention_heads", 14))
    n_kv = int(m.get("num_key_value_heads", n_heads))
    n_layers = int(m.get("num_hidden_layers", 28))
    if h % n_heads != 0:
        raise SystemExit(f"[error] hidden_size={h} must be divisible by heads={n_heads}")
    if n_heads % n_kv != 0:
        raise SystemExit(f"[error] heads={n_heads} must be divisible by kv_heads={n_kv}")
    layer_types = m.get("layer_types")
    if layer_types is not None:
        if len(layer_types) != n_layers:
            raise SystemExit(f"[error] layer_types has {len(layer_types)}, expected {n_layers}")
        bad = [x for x in layer_types if x not in {"full_attention", "sliding_attention"}]
        if bad:
            raise SystemExit(f"[error] unsupported layer_types values: {sorted(set(bad))}")
    aux_layers = m.get("aux_layers")
    if aux_layers is not None and any(int(i) < 0 or int(i) >= n_layers for i in aux_layers):
        raise SystemExit("[error] model.aux_layers contains an out-of-range layer")
    tap_layer = int(m.get("tap_layer", 14))
    if not 0 <= tap_layer < n_layers:
        raise SystemExit(f"[error] tap_layer={tap_layer} must be in [0, {n_layers - 1}]")


def build_model(cfg: dict, tokenizer) -> NanochronoForCausalLM:
    m = dict(cfg.get("model", {}))
    _validate_model_cfg(m)
    eos = tokenizer.eos_token_id
    config = NanochronoConfig(
        vocab_size=len(tokenizer),
        hidden_size=int(m.get("hidden_size", 1792)),
        intermediate_size=int(m.get("intermediate_size", 7168)),
        num_hidden_layers=int(m.get("num_hidden_layers", 28)),
        num_attention_heads=int(m.get("num_attention_heads", 14)),
        num_key_value_heads=int(m.get("num_key_value_heads", 14)),
        max_position_embeddings=int(m.get("max_position_embeddings", 2048)),
        rope_theta=float(m.get("rope_theta", 100000.0)),
        sliding_window=int(m.get("sliding_window", 512)),
        layer_types=m.get("layer_types"),
        attention_multiplier=float(m.get("attention_multiplier", 1.2)),
        final_logit_softcapping=m.get("final_logit_softcapping", 15.0),
        aux_layers=m.get("aux_layers"),
        aux_gate_dim=int(m.get("aux_gate_dim", 12)),
        mix_gate_dim=int(m.get("mix_gate_dim", 24)),
        tap_layer=int(m.get("tap_layer", 14)),
        use_cache=bool(m.get("use_cache", True)),
        bos_token_id=_id_or_fallback(tokenizer.bos_token_id, eos),
        eos_token_id=eos,
        pad_token_id=_id_or_fallback(tokenizer.pad_token_id, eos),
        tie_word_embeddings=bool(m.get("tie_word_embeddings", False)),
    )
    return NanochronoForCausalLM(config)


def initialize_from_scratch(model: NanochronoForCausalLM, std: float = 0.02) -> None:
    """Stable explicit initialization for this architecture.

    NanochronoPreTrainedModel._init_weights() is empty, so without this function
    PyTorch module defaults differ by module type.  We use one controlled normal
    initialization and scale residual-output projections by 1/sqrt(2L).
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        residual_std = std / math.sqrt(2.0 * model.config.num_hidden_layers)
        for layer in model.model.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=residual_std)

        # Preserve the architecture's intended scalar starting state.
        model.model.residual_scales.fill_(1.0)
        model.model.skip_scales.zero_()
        model.model.mix_scale.zero_()
        model.model.tap_scale.zero_()
    print(f"[init] normal std={std:g}; residual output std={residual_std:.6g}")


def apply_generation_defaults(model, cfg: dict) -> None:
    gen = GenerationConfig.from_model_config(model.config)
    for key, value in cfg.get("generation", {}).items():
        setattr(gen, key, value)
    model.generation_config = gen


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def check_param_limit(n_params: int, max_params: int, *, smoke: bool) -> None:
    if smoke:
        print(f"[params] smoke — skipping SN38 limit check ({n_params:,})")
        return
    if n_params > max_params:
        raise SystemExit(
            f"[error] model has {n_params:,} params > limit {max_params:,}. "
            "Reduce vocab/model/aux size."
        )
    print(f"[params] OK {n_params:,} <= {max_params:,}; headroom={max_params-n_params:,}")


def resolve_resume_path(args, cfg: dict, ckpt_dir: Path) -> Path | None:
    if args.resume:
        return Path(args.resume)
    if args.smoke:
        return None
    if not cfg.get("train", {}).get("auto_resume", False):
        return None
    latest = ckpt_dir / "latest"
    if (latest / "train_state.pt").is_file() and (latest / "config.json").is_file():
        return latest
    return None


def load_resume_checkpoint(resume_path: Path, device: torch.device):
    print(f"[resume] loading {resume_path.resolve()}")
    tokenizer = AutoTokenizer.from_pretrained(resume_path)
    # Keep master parameters FP32. BF16 is used only through autocast.
    model = NanochronoForCausalLM.from_pretrained(resume_path)
    model.to(device=device, dtype=torch.float32)
    model.train()
    state_path = resume_path / "train_state.pt"
    if not state_path.is_file():
        raise SystemExit(f"[error] missing optimizer/data state: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    start_step = int(state["step"]) + 1
    sequences = int(state.get("sequences_consumed", state.get("batches_consumed", 0)))
    tokens_seen = int(state.get("tokens_seen", 0))
    print(f"[resume] step={start_step} sequences={sequences:,} tokens={tokens_seen:,}")
    return model, tokenizer, state, start_step, sequences, tokens_seen


def load_init_from(init_path: Path, device: torch.device):
    """Load model+tokenizer weights for a new continued-pretrain run.

    Unlike --resume, this does *not* restore optimizer, scheduler, step counter,
    or FineWeb stream state. Use for 2018 -> 2019 (etc.) cutoff continuation.
    Prefer an FP32 training checkpoint (e.g. .../latest) over final-bf16.
    """
    print(f"[init-from] loading weights from {init_path.resolve()}")
    if not init_path.exists():
        raise SystemExit(f"[error] init_from path does not exist: {init_path}")
    tokenizer = AutoTokenizer.from_pretrained(init_path)
    model = NanochronoForCausalLM.from_pretrained(init_path)
    model.to(device=device, dtype=torch.float32)
    model.train()
    print("[init-from] weights loaded; fresh optimizer/data/step counter")
    return model, tokenizer


def _save_model_files(model, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def _bf16_export_state_dict(model) -> dict[str, torch.Tensor]:
    """CPU BF16 copy for compact inference export; training masters stay FP32."""
    result = {}
    for name, tensor in model.state_dict().items():
        t = tensor.detach().cpu()
        if t.is_floating_point():
            t = t.to(torch.bfloat16)
        result[name] = t
    return result


def save_compact_inference_model(model, tokenizer, path: Path, step: int, val_loss: float | None):
    """Save a compact all-BF16 inference model suitable for HF/SN38 upload."""
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    state_dict = _bf16_export_state_dict(model)
    model.save_pretrained(
        tmp,
        state_dict=state_dict,
        safe_serialization=True,
        max_shard_size="8GB",
    )
    tokenizer.save_pretrained(tmp)
    del state_dict
    meta = f"step={step}\nexport_dtype=bfloat16\n"
    if val_loss is not None:
        meta += f"val_loss={val_loss:.8f}\nperplexity={math.exp(min(val_loss, 20.0)):.8f}\n"
    (tmp / "export_meta.txt").write_text(meta, encoding="utf-8")
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)
    print(f"[export] BF16 inference model -> {path}")


def save_checkpoint(
    model, tokenizer, path: Path, step: int, opt, sched, packed_stream,
    sequences_consumed: int, tokens_seen: int, best_val_loss: float,
):
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    _save_model_files(model, tokenizer, tmp)
    state = {
        "version": 3,
        "step": int(step),
        "sequences_consumed": int(sequences_consumed),
        "tokens_seen": int(tokens_seen),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "data_state": packed_stream.state_dict(),
        "best_val_loss": float(best_val_loss),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(state, tmp / "train_state.pt")
    (tmp / "train_meta.txt").write_text(
        f"step={step}\nsequences_consumed={sequences_consumed}\n"
        f"tokens_seen={tokens_seen}\nbest_val_loss={best_val_loss}\nmaster_dtype=float32\n",
        encoding="utf-8",
    )
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)
    print(f"[save] {path} (step {step})")


def restore_rng_state(state: dict) -> None:
    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def make_stream(cfg: dict, tokenizer, dumps: list[str], *, seq_len: int, seed: int):
    dc = cfg.get("data", {})
    max_open = dc.get("max_open_sources")
    return build_packed_stream(
        dumps,
        cfg.get("dataset", "HuggingFaceFW/fineweb-edu"),
        tokenizer,
        cutoff_year=int(cfg["year"]),
        seq_len=seq_len,
        year_weights=_year_weights(cfg),
        docs_per_turn=int(dc.get("docs_per_turn", 32)),
        shuffle_buffer_size=int(dc.get("shuffle_buffer_size", 1024)),
        seed=seed,
        max_open_sources=int(max_open) if max_open is not None else None,
        open_explore_prob=float(dc.get("open_explore_prob", 0.02)),
    )


def legacy_skip_sequences(data_iter, n: int) -> None:
    if n <= 0:
        return
    print(f"[resume:v1] replaying {n:,} sequences (legacy checkpoint)")
    for i in range(n):
        next(data_iter)
        if (i + 1) % 10000 == 0:
            print(f"[resume:v1] {i+1:,}/{n:,}")


def autocast_context(device: torch.device, use_bf16: bool):
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate_validation(
    model, cfg: dict, tokenizer, device: torch.device, *, seq_len: int,
    blocks: int, micro_batch_size: int, seed: int, use_bf16: bool,
) -> float | None:
    val_dumps = list(cfg.get("validation_dumps", []))
    if not val_dumps or blocks <= 0:
        return None
    stream = make_stream(cfg, tokenizer, val_dumps, seq_len=seq_len, seed=seed)
    model.eval()
    losses = []
    remaining = blocks
    while remaining > 0:
        bs = min(micro_batch_size, remaining)
        input_ids = torch.stack([next(stream)["input_ids"] for _ in range(bs)], 0).to(device)
        with autocast_context(device, use_bf16):
            out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        losses.append((float(out.loss.item()), bs))
        remaining -= bs
    model.train()
    return sum(x * n for x, n in losses) / sum(n for _, n in losses)


def main():
    load_train_env()
    parser = argparse.ArgumentParser(description="SN38 Nanochrono cutoff-safe pretrain v3")
    parser.add_argument("--config", default="scripts/train/config_2018.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init-from",
        default=None,
        help="Load model/tokenizer weights only (new run). Overrides config init_from.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = dict(cfg["train"])
    dumps = list(cfg["dumps"])
    val_dumps = list(cfg.get("validation_dumps", []))
    model_cfg = dict(cfg.get("model", {}))
    max_parameters = int(cfg.get("max_parameters", DEFAULT_MAX_PARAMETERS))

    if args.resume and (args.init_from or cfg.get("init_from")):
        raise SystemExit("[error] use only one of --resume and --init-from / config init_from")

    if set(dumps) & set(val_dumps):
        raise SystemExit(f"[error] train/validation overlap: {sorted(set(dumps)&set(val_dumps))}")
    validate_cutoff_dumps(dumps, int(cfg["year"]))
    if val_dumps:
        validate_cutoff_dumps(val_dumps, int(cfg["year"]))

    smoke_cfg = {}
    if args.smoke:
        smoke_cfg = dict(cfg.get("smoke", {}))
        train_cfg.update({k: v for k, v in smoke_cfg.items() if k not in ("dumps", "model", "tokenizer_name")})
        dumps = list(smoke_cfg.get("dumps", dumps))
        model_cfg.update(smoke_cfg.get("model", {}))
        val_dumps = []
        print("[smoke] tiny pipeline test only")
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps

    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_dir = Path(cfg.get("output_dir", "runs/nanochrono-2018"))
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints/nanochrono-2018"))
    default_tok_dir = out_dir / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] real ~2B pretraining is not practical on CPU")
    elif bool(train_cfg.get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True

    use_bf16 = bool(train_cfg.get("bf16_autocast", True)) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] master=float32 compute={'bf16 autocast' if use_bf16 else 'float32'}")
    print(f"[env] cutoff={cfg['year']} train_dumps={len(dumps)} val_dumps={len(val_dumps)}")

    resume_path = resolve_resume_path(args, cfg, ckpt_dir)
    init_from = None if args.smoke else (args.init_from or cfg.get("init_from"))
    state = None
    start_step = 1
    sequences_consumed = 0
    tokens_seen = 0

    if resume_path is not None:
        model, tokenizer, state, start_step, sequences_consumed, tokens_seen = load_resume_checkpoint(resume_path, device)
    elif init_from:
        model, tokenizer = load_init_from(Path(init_from), device)
    else:
        if args.smoke:
            tokenizer = build_pretrained_tokenizer(
                str(smoke_cfg.get("tokenizer_name", "gpt2")),
                default_tok_dir / "smoke",
                int(model_cfg.get("max_position_embeddings", 256)),
            )
        else:
            tokenizer = build_or_train_cutoff_tokenizer(cfg, dumps, default_tok_dir)
        model = build_model({"model": model_cfg}, tokenizer)
        initialize_from_scratch(model, float(train_cfg.get("init_std", 0.02)))
        model.to(device=device, dtype=torch.float32)
        model.train()

    apply_generation_defaults(model, cfg)
    print(f"[tok] vocab={len(tokenizer):,} bos={tokenizer.bos_token_id} eos={tokenizer.eos_token_id}")
    n_params = count_parameters(model)
    print(f"[model] params={n_params/1e9:.6f}B")
    check_param_limit(n_params, max_parameters, smoke=args.smoke)

    seq_len = int(train_cfg["seq_len"])
    micro_bs = int(train_cfg["micro_batch_size"])
    grad_accum = int(train_cfg["grad_accum"])
    max_steps = int(train_cfg["max_steps"])
    log_every = int(train_cfg.get("log_every", 20))
    save_every = int(train_cfg.get("save_every", 2500))
    eval_every = int(train_cfg.get("eval_every", save_every))
    val_blocks = int(train_cfg.get("validation_blocks", 64))
    val_micro_bs = int(train_cfg.get("validation_micro_batch_size", micro_bs))
    lr = float(train_cfg["learning_rate"])
    wd = float(train_cfg.get("weight_decay", 0.1))
    warmup = int(train_cfg.get("warmup_steps", 1500))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    if seq_len > int(model.config.max_position_embeddings):
        raise SystemExit("[error] train.seq_len exceeds model max_position_embeddings")
    if start_step > max_steps:
        print(f"[done] checkpoint step {start_step-1} already >= max_steps {max_steps}")
        return

    packed = make_stream(cfg, tokenizer, dumps, seq_len=seq_len, seed=seed)
    data_iter = iter(packed)
    if state is not None:
        if "data_state" in state:
            packed.load_state_dict(state["data_state"])
            print("[resume] restored interleaver + pack buffer")
        else:
            legacy_skip_sequences(data_iter, sequences_consumed)

    opt_kwargs = dict(lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    if device.type == "cuda" and bool(train_cfg.get("fused_adamw", True)):
        try:
            opt = torch.optim.AdamW(model.parameters(), fused=True, **opt_kwargs)
            print("[optim] fused AdamW")
        except (TypeError, RuntimeError):
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
            print("[optim] standard AdamW")
    else:
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)

    sched = get_cosine_schedule_with_warmup(opt, warmup, max_steps)
    best_val_loss = float("inf")
    if state is not None:
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        best_val_loss = float(state.get("best_val_loss", float("inf")))
        restore_rng_state(state)

    eff_tokens = micro_bs * grad_accum * seq_len
    planned_tokens = eff_tokens * max_steps
    print(
        f"[train] steps={start_step}-{max_steps} seq={seq_len} micro_bs={micro_bs} "
        f"accum={grad_accum} eff_tokens/update={eff_tokens:,} planned={planned_tokens/1e9:.4f}B"
    )

    opt.zero_grad(set_to_none=True)
    run_t0 = log_t0 = time.time()
    loss_sum = 0.0
    loss_count = 0
    tokens_at_log = tokens_seen

    for step in range(start_step, max_steps + 1):
        step_loss = 0.0
        for _ in range(grad_accum):
            batch = [next(data_iter)["input_ids"] for _ in range(micro_bs)]
            sequences_consumed += micro_bs
            input_ids = torch.stack(batch, dim=0).to(device)
            with autocast_context(device, use_bf16):
                out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                raw_loss = out.loss
            (raw_loss / grad_accum).backward()
            step_loss += float(raw_loss.detach().item()) / grad_accum
            tokens_seen += input_ids.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}: {grad_norm}")
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        loss_sum += step_loss
        loss_count += 1
        if step == start_step or step % log_every == 0:
            now = time.time()
            avg_loss = loss_sum / max(1, loss_count)
            tok_s = (tokens_seen - tokens_at_log) / max(1e-6, now - log_t0)
            print(
                f"step {step}/{max_steps} train_loss={avg_loss:.4f} "
                f"lr={sched.get_last_lr()[0]:.2e} grad={float(grad_norm):.3f} "
                f"tokens={tokens_seen/1e9:.3f}B tok/s={tok_s:,.0f} elapsed={now-run_t0:.0f}s"
            )
            loss_sum = 0.0
            loss_count = 0
            tokens_at_log = tokens_seen
            log_t0 = now

        if val_dumps and (step % eval_every == 0 or step == max_steps):
            val_loss = evaluate_validation(
                model, cfg, tokenizer, device,
                seq_len=seq_len, blocks=val_blocks, micro_batch_size=val_micro_bs,
                seed=seed + 100_000, use_bf16=use_bf16,
            )
            if val_loss is not None:
                ppl = math.exp(min(val_loss, 20.0))
                print(f"[val] step={step} loss={val_loss:.4f} ppl={ppl:.2f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_compact_inference_model(
                        model, tokenizer, ckpt_dir / "best", step, val_loss
                    )

        if step % save_every == 0 or step == max_steps:
            save_checkpoint(
                model, tokenizer, ckpt_dir / f"step-{step}", step, opt, sched, packed,
                sequences_consumed, tokens_seen, best_val_loss,
            )
            save_checkpoint(
                model, tokenizer, ckpt_dir / "latest", step, opt, sched, packed,
                sequences_consumed, tokens_seen, best_val_loss,
            )

    # Always create a final compact upload candidate, independent of best-by-val.
    save_compact_inference_model(
        model, tokenizer, ckpt_dir / "final-bf16", max_steps,
        None if not math.isfinite(best_val_loss) else best_val_loss,
    )
    print("[done] training finished")
    print(f"[next] compare {ckpt_dir/'best'} and milestone checkpoints with actual SN38 Stage-1/Stage-2")


if __name__ == "__main__":
    main()
