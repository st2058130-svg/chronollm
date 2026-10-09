"""Expand preprocessed CPT rows into many training styles (fact/QA/temporal/…).

Heuristic only (no external LLM). Best on probe-rewritten news lines; wiki gets
a lighter style set from the lead text.
"""

from __future__ import annotations

import re
from typing import Any

_WS_RE = re.compile(r"[ \t]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“'(\[])")
_IN_DATE_RE = re.compile(
    r"^\s*In\s+(?:(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+)?((?:19|20)\d{2})\b,?\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_PERSON_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"
)
_ORG_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\s+"
    r"(?:Ministry|Department|Bank|University|Party|Council|Commission|"
    r"Agency|Authority|Force|Army|Court|Union|Federation))\b"
)
_PLACE_RE = re.compile(
    r"\b(?:in|at|near|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
)
_MEET_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+"
    r"(?:met|meets|meeting|spoke with|talked with|visited)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5})\b"
)
_WIN_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,4})\s+"
    r"(?:won|wins|defeated|beat)\b",
    re.IGNORECASE,
)
_HOST_RE = re.compile(
    r"\b(?:hosted by|host(?:ed|s)?(?:\s+by)?)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b"
)

_MONTHS = {
    "01": "January",
    "02": "February",
    "03": "March",
    "04": "April",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "August",
    "09": "September",
    "10": "October",
    "11": "November",
    "12": "December",
}


def _clean(s: str) -> str:
    s = _WS_RE.sub(" ", (s or "").replace("\n", " ")).strip()
    s = s.strip(" \"'")
    return s


def _sentences(text: str) -> list[str]:
    text = _WS_RE.sub(" ", (text or "").replace("\n", " ")).strip()
    if not text:
        return []
    return [p.strip() for p in _SENT_SPLIT_RE.split(text) if p and p.strip()]


def _parse_timestamp(ts: str | None) -> tuple[str | None, str | None, int | None]:
    """Return (month_name, yyyy-mm-dd, year)."""
    if not ts:
        return None, None, None
    t = str(ts).strip()
    m = re.match(r"^((?:19|20)\d{2})-(\d{2})-(\d{2})", t)
    if not m:
        m2 = re.match(r"^((?:19|20)\d{2})$", t)
        if m2:
            return None, None, int(m2.group(1))
        return None, None, None
    year = int(m.group(1))
    month = _MONTHS.get(m.group(2))
    return month, f"{m.group(1)}-{m.group(2)}-{m.group(3)}", year


def _lead_clause(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    # Prefer first In-YEAR probe line.
    for ln in lines:
        if _IN_DATE_RE.match(ln):
            return ln.rstrip(" .") + "."
    return lines[0].rstrip(" .") + "."


def _body_after_in_year(lead: str) -> tuple[str | None, int | None, str]:
    m = _IN_DATE_RE.match(lead)
    if not m:
        return None, None, lead
    month, year_s, body = m.group(1), m.group(2), m.group(3)
    return month, int(year_s), _clean(body)


def _people(text: str, limit: int = 6) -> list[str]:
    stop = {
        "The",
        "This",
        "That",
        "In",
        "On",
        "A",
        "An",
        "And",
        "But",
        "For",
        "With",
        "From",
        "After",
        "Before",
        "According",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
    out: list[str] = []
    seen: set[str] = set()
    for m in _PERSON_RE.finditer(text):
        name = m.group(1)
        if name.split()[0] in stop:
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= limit:
            break
    return out


def _places(text: str, limit: int = 4) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in _PLACE_RE.finditer(text):
        place = m.group(1)
        if place in seen or place in {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"}:
            continue
        seen.add(place)
        out.append(place)
        if len(out) >= limit:
            break
    return out


def _dated_fact(lead: str, month: str | None, year: int | None) -> str:
    m_name, y, body = _body_after_in_year(lead)
    use_month = month or m_name
    use_year = year or y or 2024
    body = body or _clean(lead)
    if use_month:
        return f"In {use_month} {use_year}, {body[0].lower() + body[1:] if body and body[0].isupper() and not body[:2].isupper() else body}".rstrip(".") + "."
    return f"In {use_year}, {body}".rstrip(".") + "."


def _headline(lead: str) -> str:
    _, _, body = _body_after_in_year(lead)
    body = body or _clean(lead)
    # Short headline-ish: first ~12 words
    words = body.split()
    h = " ".join(words[:12]).rstrip(",.;:")
    return h[:1].upper() + h[1:] if h else body


def enrich_record(row: dict[str, Any], *, target_year: int = 2024) -> dict[str, Any]:
    """Return a copy of row with many style fields + concatenated training text."""
    text = row.get("text") or ""
    if not isinstance(text, str) or not text.strip():
        return dict(row)

    ts = row.get("timestamp")
    month, iso, year = _parse_timestamp(ts if isinstance(ts, str) else None)
    if year is None:
        year = target_year

    lead = _lead_clause(text)
    sents = _sentences(text)
    people = _people(text)
    places = _places(text)
    orgs = [m.group(1) for m in _ORG_RE.finditer(text)][:4]

    styles: dict[str, str] = {}

    # 1) fact
    styles["fact"] = _dated_fact(lead, month, year)

    # 2) context — fuller multi-sentence block
    ctx_parts = sents[:4] if sents else [lead]
    styles["context"] = _clean(" ".join(ctx_parts))

    # 3) temporal
    if month and iso:
        styles["temporal"] = (
            f"This event is dated {iso} ({month} {year}). "
            f"{styles['fact']}"
        )
    else:
        styles["temporal"] = f"In {year}, the following took place: {styles['fact']}"

    # 4-7) QA variants
    meet = _MEET_RE.search(text)
    win = _WIN_RE.search(text)
    host = _HOST_RE.search(text)

    if meet:
        a, b = meet.group(1), meet.group(2)
        styles["question"] = f"Who did {a} meet?"
        styles["answer"] = b
        styles["who_question"] = styles["question"]
        styles["who_answer"] = b
    elif host:
        styles["question"] = "Who hosted the event described?"
        styles["answer"] = host.group(1)
        styles["who_question"] = styles["question"]
        styles["who_answer"] = host.group(1)
    elif win:
        styles["question"] = "Who won according to the report?"
        styles["answer"] = win.group(1)
        styles["who_question"] = styles["question"]
        styles["who_answer"] = win.group(1)
    elif len(people) >= 2:
        styles["question"] = f"Who is mentioned alongside {people[0]}?"
        styles["answer"] = people[1]
        styles["who_question"] = f"Who is a main person in this {year} report?"
        styles["who_answer"] = people[0]
    elif people:
        styles["question"] = f"Who is a central figure in this {year} report?"
        styles["answer"] = people[0]
        styles["who_question"] = styles["question"]
        styles["who_answer"] = people[0]
    elif orgs:
        styles["question"] = "Which organization is named in this report?"
        styles["answer"] = orgs[0]
    else:
        # Fallback QA from fact tail
        words = styles["fact"].split()
        tail = " ".join(words[-3:]).rstrip(".")
        styles["question"] = f"What happened in {month or year} according to the report?"
        styles["answer"] = styles["fact"]
        if tail:
            styles["what_answer"] = styles["fact"]

    styles["what_question"] = f"What happened in {month + ' ' if month else ''}{year}?"
    styles["what_answer"] = styles["fact"]

    styles["when_question"] = "When did this take place?"
    styles["when_answer"] = f"{month + ' ' if month else ''}{year}".strip()
    if iso:
        styles["when_answer_iso"] = iso

    if places:
        styles["where_question"] = "Where is this event associated with?"
        styles["where_answer"] = places[0]

    # 8) cloze
    ans = styles.get("answer") or (people[0] if people else None)
    if ans and ans in styles["fact"]:
        styles["cloze"] = styles["fact"].replace(ans, "_____", 1)
        styles["cloze_answer"] = ans

    # 9) probe-style prompt/phrase (known-eval shape)
    if ans and styles["fact"].endswith(ans + ".") or (ans and styles["fact"].endswith(ans)):
        stem = styles["fact"][: styles["fact"].rfind(ans)].rstrip()
        if len(stem) >= 20:
            styles["probe_prompt"] = stem
            styles["probe_phrase"] = ans
    elif people:
        # last person mention as phrase
        p = people[-1]
        idx = styles["fact"].rfind(p)
        if idx >= 12:
            styles["probe_prompt"] = styles["fact"][:idx].rstrip()
            styles["probe_phrase"] = p

    # 10) headline / assertion / paraphrase / bullets / summary
    styles["headline"] = _headline(lead)
    styles["assertion"] = styles["fact"]
    # Light paraphrase: swap In MONTH YEAR ↔ In YEAR
    if month:
        styles["paraphrase"] = (
            f"During {month} {year}, "
            + (_body_after_in_year(lead)[2] or lead)
        ).rstrip(".") + "."
    else:
        styles["paraphrase"] = styles["fact"]

    bullets = []
    if people:
        bullets.append(f"People: {', '.join(people[:3])}")
    if places:
        bullets.append(f"Places: {', '.join(places[:3])}")
    if orgs:
        bullets.append(f"Organizations: {', '.join(orgs[:2])}")
    bullets.append(f"Date: {iso or (str(year))}")
    bullets.append(f"Event: {styles['fact']}")
    styles["bullets"] = "\n".join(f"- {b}" for b in bullets)

    styles["summary"] = styles["context"]
    styles["lead"] = lead

    # 11) QA block + instruction-style
    if "question" in styles and "answer" in styles:
        styles["qa"] = f"Question: {styles['question']}\nAnswer: {styles['answer']}"
        styles["instruction"] = (
            f"Read the context and answer the question.\n"
            f"Context: {styles['context']}\n"
            f"Question: {styles['question']}\n"
            f"Answer: {styles['answer']}"
        )

    # 12) cause/effect lite
    low = text.lower()
    if " after " in low or " following " in low:
        styles["cause_effect"] = (
            f"Sequence: {styles['fact']} "
            f"Temporal cue indicates an after/following relationship in the source."
        )

    # 13) entity card
    ent_lines = [f"Year: {year}"]
    if iso:
        ent_lines.append(f"Date: {iso}")
    if people:
        ent_lines.append(f"People: {', '.join(people)}")
    if places:
        ent_lines.append(f"Places: {', '.join(places)}")
    if orgs:
        ent_lines.append(f"Orgs: {', '.join(orgs)}")
    ent_lines.append(f"Fact: {styles['fact']}")
    styles["entity_card"] = "\n".join(ent_lines)

    # Build concatenated training text (all styles), keeps local_text happy.
    order = [
        "fact",
        "qa",
        "question",
        "answer",
        "temporal",
        "context",
        "who_question",
        "who_answer",
        "what_question",
        "what_answer",
        "when_question",
        "when_answer",
        "where_question",
        "where_answer",
        "cloze",
        "cloze_answer",
        "probe_prompt",
        "probe_phrase",
        "headline",
        "paraphrase",
        "assertion",
        "bullets",
        "summary",
        "instruction",
        "cause_effect",
        "entity_card",
        "lead",
    ]
    blocks: list[str] = []
    for key in order:
        val = styles.get(key)
        if not val:
            continue
        blocks.append(f"{key}: {val}")

    out = dict(row)
    out.update(styles)
    out["text"] = "\n".join(blocks)
    out["styles"] = [k for k in order if styles.get(k)]
    out["enriched"] = True
    return out


def render_style_documents(row: dict[str, Any], *, target_year: int = 2024) -> list[dict[str, Any]]:
    """Expand one row into multiple single-style training documents."""
    enriched = enrich_record(row, target_year=target_year)
    base_meta = {
        "timestamp": enriched.get("timestamp"),
        "source": enriched.get("source"),
        "preprocessed": enriched.get("preprocessed"),
        "preprocess_kind": enriched.get("preprocess_kind"),
        "enriched": True,
    }
    docs: list[dict[str, Any]] = []
    # Prefer compact high-value styles as separate docs for packing diversity.
    singles = [
        ("fact", enriched.get("fact")),
        ("qa", enriched.get("qa")),
        ("temporal", enriched.get("temporal")),
        ("context", enriched.get("context")),
        ("instruction", enriched.get("instruction")),
        ("cloze_pair", None),
        ("probe", None),
        ("headline", enriched.get("headline")),
        ("paraphrase", enriched.get("paraphrase")),
        ("bullets", enriched.get("bullets")),
        ("entity_card", enriched.get("entity_card")),
        ("cause_effect", enriched.get("cause_effect")),
    ]
    if enriched.get("cloze") and enriched.get("cloze_answer"):
        singles[5] = (
            "cloze_pair",
            f"Fill in the blank: {enriched['cloze']}\nAnswer: {enriched['cloze_answer']}",
        )
    if enriched.get("probe_prompt") and enriched.get("probe_phrase"):
        singles[6] = (
            "probe",
            f"{enriched['probe_prompt']} {enriched['probe_phrase']}".strip(),
        )

    seen: set[str] = set()
    for style_name, body in singles:
        if not body or not str(body).strip():
            continue
        key = re.sub(r"\W+", "", str(body).lower())[:240]
        if key in seen:
            continue
        seen.add(key)
        docs.append(
            {
                **base_meta,
                "style": style_name,
                "text": str(body).strip(),
            }
        )
    # Always keep one multi-style pack doc too.
    if enriched.get("text"):
        docs.append(
            {
                **base_meta,
                "style": "multistyle_pack",
                "text": enriched["text"],
                "styles": enriched.get("styles"),
            }
        )
    return docs
