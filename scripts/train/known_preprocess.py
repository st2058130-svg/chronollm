"""Known-probe densification transforms for ChronoLLM cutoff training.

Applied selectively on news (and optionally wiki) streams — not FineWeb:

1. De-noise — strip nav / subscribe / share boilerplate
2. Sentence keep-filter — keep dated / event-claim sentences
3. Probe-style rewrite — reshape kept claims into ``In {year}, …`` lines

Late-2024 news coverage is handled in config (extra HF sources), and Wikipedia
stays a full snapshot (denoise-only) so we do not over-filter it.
"""

from __future__ import annotations

import re
from typing import Any

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“'(\[])")
_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_IN_YEAR_RE = re.compile(
    r"^\s*In\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+((?:19|20)\d{2})\b"
    r"|^\s*In\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_MONTH_YEAR_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)

_DROP_LINE_RE = re.compile(
    r"(?i)^(?:"
    r"subscribe(?:\s+to)?(?:\s+our)?(?:\s+newsletter)?|"
    r"sign\s+up(?:\s+for)?(?:\s+our)?(?:\s+newsletter)?|"
    r"cookie(?:s)?(?:\s+policy)?|"
    r"we\s+use\s+cookies|"
    r"advertisement|"
    r"sponsored\s+content|"
    r"share\s+this(?:\s+story)?|"
    r"share\s+on\s+(?:facebook|twitter|x|linkedin)|"
    r"follow\s+us(?:\s+on)?|"
    r"related(?:\s+stories|\s+articles|:)|"
    r"read\s+more(?:\s*:)?|"
    r"click\s+here|"
    r"all\s+rights\s+reserved|"
    r"copyright\s+©|"
    r"terms\s+of\s+(?:use|service)|"
    r"privacy\s+policy|"
    r"trending\s+now|"
    r"newsletter|"
    r"getty\s+images|"
    r"photo(?:\s*:|\s+credit)|"
    r"file\s+photo|"
    r"continue\s+reading|"
    r"log\s+in\s+to\s+comment|"
    r"leave\s+a\s+comment|"
    r"javascript\s+is\s+disabled|"
    r"enable\s+cookies"
    r").*$"
)

_DROP_INLINE_RE = re.compile(
    r"(?i)\b(?:"
    r"subscribe to our newsletter|"
    r"sign up for our newsletter|"
    r"click here to read more|"
    r"share this (?:article|story) on|"
    r"advertisement\b"
    r").{0,80}"
)

_EVENT_CUE_RE = re.compile(
    r"(?i)\b(?:"
    r"elected|election|inaugurat(?:ed|ion)|appointed|resign(?:ed|s)|"
    r"won|wins|winner|defeat(?:ed|s)|hosted|host(?:s|ing)|"
    r"announced|announce[sd]?|launched|launch(?:es|ing)|"
    r"released|release[sd]?|approved|passed|signed|"
    r"collapsed|struck|killed|died|dead|"
    r"ceasefire|invasion|invaded|coup|"
    r"olympics|championship|tournament|final|"
    r"prime minister|president|chancellor|"
    r"acquired|merger|ipo|"
    r"sentenced|convicted|indicted|"
    r"opened|closed|banned|legalized"
    r")\b"
)

_BOILER_SHORT_RE = re.compile(
    r"(?i)^(yes|no|ok|thanks|hello|hi|home|menu|search|next|prev|previous)\.?$"
)


def denoise_text(text: str) -> str:
    """Strip obvious web/news chrome so event sentences are not drowned."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    kept: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            kept.append("")
            continue
        if _DROP_LINE_RE.match(line):
            continue
        if _BOILER_SHORT_RE.match(line):
            continue
        if len(line) <= 2:
            continue
        line = _DROP_INLINE_RE.sub(" ", line)
        line = _WS_RE.sub(" ", line).strip()
        if line:
            kept.append(line)
    out = "\n".join(kept)
    out = _MULTI_NL_RE.sub("\n\n", out).strip()
    return out


def split_sentences(text: str) -> list[str]:
    text = _WS_RE.sub(" ", text.replace("\n", " ")).strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _sentence_years(sentence: str) -> list[int]:
    return [int(y) for y in _YEAR_RE.findall(sentence)]


def is_dated_claim(
    sentence: str,
    *,
    target_year: int,
    allow_nearby: bool = True,
) -> bool:
    """Keep sentences that look like dated factual claims for the cutoff year."""
    s = sentence.strip()
    if len(s) < 25 or len(s) > 480:
        return False
    if _DROP_LINE_RE.match(s) or _BOILER_SHORT_RE.match(s):
        return False

    years = _sentence_years(s)
    year_ok = False
    if years:
        if target_year in years:
            year_ok = True
        elif allow_nearby and any(abs(y - target_year) <= 1 for y in years):
            # Keep adjacent-year context only if event cue is strong.
            year_ok = bool(_EVENT_CUE_RE.search(s))
    in_year = bool(_IN_YEAR_RE.match(s)) or bool(_MONTH_YEAR_RE.search(s))
    has_event = bool(_EVENT_CUE_RE.search(s))

    if in_year and (target_year in years or not years):
        return True
    if year_ok and has_event:
        return True
    if year_ok and re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", s):
        return True
    # Undated but strong event lead (title-like) — keep sparingly via caller.
    return False


def keep_dated_claims(
    text: str,
    *,
    target_year: int,
    max_sentences: int = 24,
) -> list[str]:
    claims: list[str] = []
    for sent in split_sentences(text):
        if is_dated_claim(sent, target_year=target_year):
            claims.append(sent)
            if len(claims) >= max_sentences:
                break
    return claims


def _strip_leading_dateline(sentence: str) -> str:
    # "PARIS (AP) — Athletes …" / "LONDON — Keir …"
    s = re.sub(
        r"^[A-Z][A-Z .,'/-]{1,40}\((?:AP|Reuters|AFP|UPI|PA)\)\s*[—\-–:]\s*",
        "",
        sentence,
    )
    s = re.sub(r"^[A-Z][A-Z .,'/-]{1,32}\s*[—\-–:]\s*", "", s)
    return s.strip()


def rewrite_probe_style(
    sentence: str,
    *,
    target_year: int,
) -> str | None:
    """Turn a claim into a short probe-shaped ``In {year}, …`` line when possible."""
    s = _strip_leading_dateline(sentence.strip())
    if not s:
        return None
    s = s.rstrip(" .;:") + "."
    s = _WS_RE.sub(" ", s).strip()

    m = _IN_YEAR_RE.match(s)
    if m:
        year = int(m.group(1) or m.group(2))
        if year > target_year:
            return None
        # Already probe-shaped.
        if s[0].islower():
            s = s[0].upper() + s[1:]
        return s

    years = _sentence_years(s)
    year = target_year
    if target_year in years:
        year = target_year
    elif years:
        # Prefer the latest year <= cutoff present in the sentence.
        cand = [y for y in years if y <= target_year]
        if not cand:
            return None
        year = max(cand)
    elif not _EVENT_CUE_RE.search(s):
        return None

    # Drop dated prefixes before adding "In {year}, …".
    # Keep event names like "Euro 2024"; do not strip ordinary "in" prepositions.
    body = s
    body = re.sub(
        r"(?i)\bin\s+(?:early|late|mid-?)?\s*"
        r"(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+((?:19|20)\d{2})\b,?",
        " ",
        body,
    )
    body = re.sub(
        r"(?i)\bin\s+(?:early|late|mid-?)?\s*((?:19|20)\d{2})\b,?",
        " ",
        body,
    )
    # Trailing ", 2024" / " (2024)" noise, but not "Euro 2024".
    body = re.sub(r"[,;]\s*((?:19|20)\d{2})\b", " ", body)
    body = re.sub(r"\(((?:19|20)\d{2})\)", " ", body)
    body = _WS_RE.sub(" ", body).strip(" ,;.-")
    if not body:
        return None
    if body[0].islower():
        body = body[0].upper() + body[1:]
    if not body.endswith("."):
        body += "."
    out = f"In {year}, {body}"
    if len(out) < 28 or len(out) > 420:
        return None
    return out


def rewrite_title_probe(title: str | None, *, target_year: int) -> str | None:
    if not title:
        return None
    t = _WS_RE.sub(" ", str(title)).strip(" .;-")
    if len(t) < 12 or len(t) > 180:
        return None
    if _DROP_LINE_RE.match(t):
        return None
    if not (_EVENT_CUE_RE.search(t) or _YEAR_RE.search(t) or t[:1].isupper()):
        return None
    return rewrite_probe_style(t if t.endswith(".") else t + ".", target_year=target_year)


def preprocess_for_known(
    text: str,
    *,
    target_year: int,
    denoise: bool = True,
    claim_filter: bool = True,
    probe_rewrite: bool = True,
    title: str | None = None,
    max_sentences: int = 24,
    max_probe_lines: int = 12,
    min_chars_out: int = 40,
    fallback_lead_sentences: int = 3,
) -> str | None:
    """Compose denoise → claim keep → probe rewrite for one document.

    Returns a single training string (newline-joined facts) or None to skip.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    cleaned = denoise_text(text) if denoise else text.strip()
    if not cleaned:
        return None

    pieces: list[str] = []

    title_probe = rewrite_title_probe(title, target_year=target_year) if probe_rewrite else None
    if title_probe:
        pieces.append(title_probe)

    if claim_filter or probe_rewrite:
        claims = keep_dated_claims(
            cleaned, target_year=target_year, max_sentences=max_sentences
        )
        if not claims and fallback_lead_sentences > 0:
            claims = split_sentences(cleaned)[:fallback_lead_sentences]

        if probe_rewrite:
            seen: set[str] = set()
            for claim in claims:
                line = rewrite_probe_style(claim, target_year=target_year)
                if not line:
                    # Keep strong dated claim even if rewrite fails.
                    if is_dated_claim(claim, target_year=target_year):
                        line = claim.rstrip(" .") + "."
                    else:
                        continue
                key = re.sub(r"\W+", "", line.lower())
                if key in seen:
                    continue
                seen.add(key)
                pieces.append(line)
                if len(pieces) >= max_probe_lines + (1 if title_probe else 0):
                    break
        else:
            pieces.extend(claims[:max_sentences])
    else:
        # Denoise-only path (Wikipedia full-snapshot friendly).
        pieces.append(cleaned)

    # Deduplicate while preserving order.
    out_lines: list[str] = []
    seen_all: set[str] = set()
    for p in pieces:
        key = re.sub(r"\W+", "", p.lower())
        if not key or key in seen_all:
            continue
        seen_all.add(key)
        out_lines.append(p)

    if not out_lines:
        # Last resort: short cleaned lead so the doc is not fully dropped.
        lead = " ".join(split_sentences(cleaned)[:fallback_lead_sentences]).strip()
        if len(lead) >= min_chars_out:
            return lead
        return None

    if claim_filter or probe_rewrite:
        out = "\n".join(out_lines)
    else:
        out = out_lines[0]

    if len(out) < min_chars_out:
        return None
    return out


def preprocess_cfg_enabled(cfg: dict[str, Any] | None) -> bool:
    if not cfg:
        return False
    if "enabled" in cfg:
        return bool(cfg["enabled"])
    # Any active flag implies on.
    return bool(
        cfg.get("denoise")
        or cfg.get("claim_filter")
        or cfg.get("probe_rewrite")
    )


def apply_preprocess_config(
    text: str,
    cfg: dict[str, Any],
    *,
    default_year: int,
    title: str | None = None,
) -> str | None:
    year = int(cfg.get("year", default_year))
    return preprocess_for_known(
        text,
        target_year=year,
        denoise=bool(cfg.get("denoise", True)),
        claim_filter=bool(cfg.get("claim_filter", False)),
        probe_rewrite=bool(cfg.get("probe_rewrite", False)),
        title=title if cfg.get("use_title", True) else None,
        max_sentences=int(cfg.get("max_sentences", 24)),
        max_probe_lines=int(cfg.get("max_probe_lines", 12)),
        min_chars_out=int(cfg.get("min_chars_out", 40)),
        fallback_lead_sentences=int(cfg.get("fallback_lead_sentences", 3)),
    )
