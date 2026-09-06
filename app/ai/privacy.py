"""Privacy Guard — local pseudonymization before any cloud call.

The product rule (docs/INCREMENT_1.md, Feature B): student PII is **swapped for
a stable code**, not blacked out. The professor and the AI can both keep track
of who is who — the AI just never learns the name.

    "I worked with Dave Kowalski; email me at dave.k@uni.edu"
    -> "I worked with Person-A; email me at [EMAIL-1]"

Two passes, in this order:

1. **Deterministic** (pure Python, always runs): roster full names, first/last
   names alone and roster emails — case-insensitive, on word boundaries,
   longest match first — plus regexes for emails, phone numbers and
   student-ID-like numbers.
2. **Local LLM sweep** (Gemma 3 4B through :class:`~app.ai.providers.LocalProvider`):
   asks for the person names / PII the regexes cannot know about (nicknames,
   third parties). Every span it returns is validated against the source text
   before anything is substituted, so a hallucinated span can never rewrite the
   submission. Bad JSON is retried once, then the run degrades to pass-1 output
   with a warning — it never fails the grading run.

Codes are stable *per course*: roster students reuse their per-course number
(``Student-07``), everyone/everything else gets a code out of
:class:`~app.models.PseudonymMap` (``Person-A``, ``[EMAIL-1]``, ``[PHONE-1]``,
``[ID-1]``). The map never leaves the machine; it is what powers the
"who is Person-A?" view in Settings and the display-layer pass that swaps codes
back to real names in professor-facing text (:func:`render_for_display`).

Hard invariant enforced here: the sweep provider must be on-device. Sending the
raw text to a cloud model to find its PII would defeat the entire feature, so
:func:`sweep_spans` refuses a cloud provider outright.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config
from app.models import PrivacyScan, PseudonymMap, Student, Submission, utcnow

log = logging.getLogger("agora.ai.privacy")


# --------------------------------------------------------------------------
# kinds & codes
# --------------------------------------------------------------------------

KIND_PERSON = "person"
KIND_EMAIL = "email"
KIND_PHONE = "phone"
KIND_ID = "id_number"
KINDS = (KIND_PERSON, KIND_EMAIL, KIND_PHONE, KIND_ID)

#: What a model might call a kind -> what we store.
KIND_ALIASES = {
    "person": KIND_PERSON,
    "name": KIND_PERSON,
    "people": KIND_PERSON,
    "student": KIND_PERSON,
    "email": KIND_EMAIL,
    "e-mail": KIND_EMAIL,
    "email_address": KIND_EMAIL,
    "phone": KIND_PHONE,
    "phone_number": KIND_PHONE,
    "telephone": KIND_PHONE,
    "id": KIND_ID,
    "id_number": KIND_ID,
    "student_id": KIND_ID,
    "number": KIND_ID,
}

#: ``[EMAIL-1]``-style prefixes. Persons use letters instead (``Person-A``).
TYPED_CODE_PREFIX = {KIND_EMAIL: "EMAIL", KIND_PHONE: "PHONE", KIND_ID: "ID"}

SOURCE_ROSTER = "roster"
SOURCE_REGEX = "regex"
SOURCE_LLM = "llm"

MODE_OFF = config.PRIVACY_MODE_OFF
MODE_WARN = config.PRIVACY_MODE_WARN
MODE_SWAP = config.PRIVACY_MODE_SWAP


def normalize_kind(value: Any) -> str:
    key = str(value or "").strip().lower().replace(" ", "_")
    return KIND_ALIASES.get(key, KIND_PERSON)


def student_code(student: Student | int) -> str:
    """Roster code — stable per course because the number already is."""
    number = student if isinstance(student, int) else int(student.student_number)
    return f"Student-{number:02d}"


def person_code(index: int) -> str:
    """0 -> Person-A, 25 -> Person-Z, 26 -> Person-AA (bijective base 26)."""
    letters = ""
    n = int(index) + 1
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return f"Person-{letters}"


def typed_code(kind: str, index: int) -> str:
    return f"[{TYPED_CODE_PREFIX.get(kind, 'ID')}-{int(index) + 1}]"


#: Every code shape this module can emit — used to re-substitute for display
#: and to stop the LLM sweep from "finding" a code we just wrote.
CODE_RE = re.compile(
    r"(?:Student-\d+|Student\s*#\s*\d+|Person-[A-Z]+|\[(?:EMAIL|PHONE|ID)-\d+\])"
)


def looks_like_code(text: str) -> bool:
    return bool(CODE_RE.fullmatch((text or "").strip()))


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

#: Formatted phone numbers only — a bare run of digits is read as an id below.
PHONE_RE = re.compile(
    r"""(?<![\w-])
    (?:\+\d{1,3}[\s.\-]?)?
    (?:\(\d{3}\)\s*|\d{3}[\s.\-])
    \d{3}[\s.\-]\d{4}
    (?![\w-])""",
    re.VERBOSE,
)

#: "student id: 992431", "ID #10029384" — the label makes it unambiguous.
LABELLED_ID_RE = re.compile(
    r"(?i)(?<![\w-])(?:student\s*(?:id|number|no\.?)|s\.?i\.?d\.?|id)\s*[:#]?\s*(\d{4,})(?![\w-])"
)
#: A long bare number in student prose is an id far more often than a quantity.
#: Sentence punctuation may follow it; another digit or a decimal tail may not.
BARE_ID_RE = re.compile(r"(?<![\w.\-])\d{6,12}(?![\w-])(?!\.\d)")

_WS_RE = re.compile(r"\s+")

#: Roster first/last names shorter than this are not matched on their own —
#: "Li" or "Bo" alone would shred an essay. The full name still matches.
MIN_NAME_TOKEN = 3

#: Common words that happen to be names ("Grace", "Will"). Matching these alone
#: mangles ordinary prose, and the full name still catches the real reference.
NAME_TOKEN_STOPWORDS = {
    "will",
    "grace",
    "hope",
    "may",
    "june",
    "art",
    "rose",
    "mark",
    "bill",
    "sum",
    "man",
    "day",
    "one",
    "van",
    "der",
    "del",
    "los",
    "las",
}


@dataclass
class Detection:
    """One identifier found in the text, with where it was found."""

    start: int
    end: int
    text: str
    kind: str
    source: str
    code: Optional[str] = None
    student_id: Optional[int] = None

    @property
    def normalized(self) -> str:
        return normalize_text(self.text)


def normalize_text(value: str) -> str:
    """Folding used for map lookups: case, whitespace, edge punctuation."""
    return _WS_RE.sub(" ", str(value or "")).strip().strip(".,;:!?'\"()[]").lower()


def _name_variants(name: str) -> list[str]:
    """Full name first, then the standalone name tokens worth matching."""
    cleaned = _WS_RE.sub(" ", str(name or "")).strip()
    if not cleaned:
        return []
    variants = [cleaned]
    if "," in cleaned:  # "Osei, Amara" rosters
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) == 2:
            variants.append(f"{parts[1]} {parts[0]}")
    tokens: list[str] = []
    for variant in list(variants):
        tokens.extend(re.split(r"[\s,]+", variant))
    for token in tokens:
        token = token.strip(".,;:'\"")
        if len(token) < MIN_NAME_TOKEN or token.lower() in NAME_TOKEN_STOPWORDS:
            continue
        if token not in variants:
            variants.append(token)
    # Longest first so "Amara Osei" wins over "Amara".
    return sorted(dict.fromkeys(variants), key=len, reverse=True)


def _word_pattern(literal: str) -> re.Pattern[str]:
    """Case-insensitive, word-boundary match that will not eat an email.

    Whitespace inside the literal matches any run of whitespace, so a name the
    PDF extractor split across a line break ("Dave\\nKowalski") — or one the
    local model handed back with different spacing — still matches the text it
    really came from.
    """
    escaped = r"\s+".join(re.escape(part) for part in literal.split())
    left = r"(?<![\w.@\-])" if literal[:1].isalnum() else ""
    right = r"(?![\w@\-])" if literal[-1:].isalnum() else ""
    return re.compile(f"{left}{escaped}{right}", re.IGNORECASE)


def roster_students(db: Session, course_id: int | None) -> list[Student]:
    if not course_id:
        return []
    return list(
        db.scalars(
            select(Student).where(Student.course_id == course_id).order_by(Student.student_number)
        ).all()
    )


def _roster_detections(text: str, students: Sequence[Student]) -> list[Detection]:
    """Pass 1a — roster emails first, then names longest-match-first.

    A bare name token ("Osei") only stands for a student when exactly one
    student on the roster owns it. Two Oseis on one roster used to collide onto
    the first one's code — the wrong real name would then come back out of the
    display layer — so a shared token is treated as an *unattributed* person
    instead: it still leaves the machine as a code (nothing leaks), but as a
    map code rather than a student's, and the report says so. The full name
    still resolves to the right student because it is matched first.
    """
    found: list[Detection] = []
    email_targets: list[tuple[str, Optional[Student]]] = []
    name_targets: list[tuple[str, Optional[Student]]] = []

    per_student: list[tuple[Student, list[str]]] = []
    token_owners: dict[str, set[Any]] = {}
    for student in students:
        variants = _name_variants(student.name)
        per_student.append((student, variants))
        owner = student.id if student.id is not None else id(student)
        for variant in variants:
            if " " not in variant:  # a bare token: "Osei", "Amara"
                token_owners.setdefault(variant.lower(), set()).add(owner)

    ambiguous: dict[str, str] = {}
    for student, variants in per_student:
        if student.email:
            email_targets.append((student.email.strip(), student))
        for variant in variants:
            key = variant.lower()
            if " " not in variant and len(token_owners.get(key, ())) > 1:
                ambiguous.setdefault(key, variant)
                continue
            name_targets.append((variant, student))
    for variant in ambiguous.values():
        name_targets.append((variant, None))
    name_targets.sort(key=lambda pair: len(pair[0]), reverse=True)

    for literal, student in email_targets + name_targets:
        if not literal:
            continue
        is_email = "@" in literal
        code: Optional[str] = None
        if student is not None and not is_email:
            # A roster *name* is already coded by the student's number; their
            # email is an identifier in its own right and gets a typed code
            # ([EMAIL-1]) out of the map like any other. An ambiguous token has
            # no student, so it takes a map code too.
            code = student_code(student)
        for match in _word_pattern(literal).finditer(text):
            found.append(
                Detection(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                    kind=KIND_EMAIL if is_email else KIND_PERSON,
                    source=SOURCE_ROSTER,
                    code=code,
                    student_id=student.id if student is not None else None,
                )
            )
    return found


def _ambiguous_warnings(detections: Sequence[Detection]) -> list[str]:
    """One line per shared roster token that really did stand on its own."""
    seen: list[str] = []
    for det in detections:
        if det.source != SOURCE_ROSTER or det.kind != KIND_PERSON:
            continue
        if det.student_id is not None:
            continue
        if det.text not in seen:
            seen.append(det.text)
    return [
        f"{name!r} is a name more than one student on this roster answers to; on its own "
        "it was swapped for an unattributed code rather than a student number."
        for name in seen
    ]


def _regex_detections(text: str) -> list[Detection]:
    """Pass 1b — emails, phones, id numbers. Order encodes precedence."""
    found: list[Detection] = []
    for match in EMAIL_RE.finditer(text):
        found.append(
            Detection(match.start(), match.end(), match.group(0), KIND_EMAIL, SOURCE_REGEX)
        )
    for match in PHONE_RE.finditer(text):
        found.append(
            Detection(match.start(), match.end(), match.group(0), KIND_PHONE, SOURCE_REGEX)
        )
    for match in LABELLED_ID_RE.finditer(text):
        found.append(
            Detection(match.start(1), match.end(1), match.group(1), KIND_ID, SOURCE_REGEX)
        )
    for match in BARE_ID_RE.finditer(text):
        found.append(
            Detection(match.start(), match.end(), match.group(0), KIND_ID, SOURCE_REGEX)
        )
    return found


def _resolve_overlaps(detections: Iterable[Detection]) -> list[Detection]:
    """Keep the first detection of any overlapping span (priority = order)."""
    claimed: list[tuple[int, int]] = []
    kept: list[Detection] = []
    for det in detections:
        if any(det.start < end and start < det.end for start, end in claimed):
            continue
        claimed.append((det.start, det.end))
        kept.append(det)
    return sorted(kept, key=lambda d: d.start)


def _apply(text: str, detections: Sequence[Detection]) -> str:
    """Rebuild the text once, so a code can never be re-matched downstream."""
    out: list[str] = []
    cursor = 0
    for det in sorted(detections, key=lambda d: d.start):
        if det.code is None or det.start < cursor:
            continue
        out.append(text[cursor : det.start])
        out.append(det.code)
        cursor = det.end
    out.append(text[cursor:])
    return "".join(out)


# --------------------------------------------------------------------------
# the pseudonym map (never leaves this machine)
# --------------------------------------------------------------------------


def _existing_codes(db: Session, course_id: int, kind: str) -> set[str]:
    """Every code ever issued for this course+kind — retired ones included.

    Retired rows are tombstones on purpose: their code must stay spoken for so
    it is never handed to a different person (see :func:`retire_pseudonym`).
    """
    return set(
        db.scalars(
            select(PseudonymMap.code).where(
                PseudonymMap.course_id == course_id, PseudonymMap.kind == kind
            )
        ).all()
    )


_PERSON_CODE_RE = re.compile(r"^Person-([A-Z]+)$")
_TYPED_CODE_RE = re.compile(r"^\[(?:EMAIL|PHONE|ID)-(\d+)\]$")


def code_index(code: str) -> Optional[int]:
    """The allocation index behind a code — the inverse of the code helpers."""
    text = str(code or "").strip()
    match = _PERSON_CODE_RE.match(text)
    if match:
        index = 0
        for letter in match.group(1):
            index = index * 26 + (ord(letter) - ord("A") + 1)
        return index - 1
    match = _TYPED_CODE_RE.match(text)
    if match:
        return int(match.group(1)) - 1
    return None


def _next_code(db: Session, course_id: int, kind: str) -> str:
    """The next code for this course+kind, from a monotonic high-water mark.

    Never reissues: deleting Dave's row must not hand ``Person-A`` to Frank,
    because every stored feedback string, scan report and cached card that says
    "Person-A" would silently start meaning someone else.
    """
    taken = _existing_codes(db, course_id, kind)
    index = max((i for i in (code_index(c) for c in taken) if i is not None), default=-1) + 1
    while True:
        code = person_code(index) if kind == KIND_PERSON else typed_code(kind, index)
        if code not in taken:  # defensive: a hand-edited code we cannot parse
            return code
        index += 1


def _live_rows(course_id: int):
    """Select for map rows that still name someone (tombstones excluded)."""
    return select(PseudonymMap).where(
        PseudonymMap.course_id == course_id, PseudonymMap.retired_at.is_(None)
    )


def lookup_code(db: Session, course_id: int | None, text: str, kind: str) -> Optional[str]:
    """The code already assigned to this identity, if any (no writes)."""
    if not course_id:
        return None
    normalized = normalize_text(text)
    if not normalized:
        return None
    return db.scalars(
        select(PseudonymMap.code).where(
            PseudonymMap.course_id == course_id,
            PseudonymMap.kind == kind,
            PseudonymMap.original_text_normalized == normalized,
            PseudonymMap.retired_at.is_(None),
        )
    ).first()


def code_for(db: Session, course_id: int, text: str, kind: str) -> str:
    """Stable code for one identity — the same input always maps back out.

    Creates the ``PseudonymMap`` row on first sight and reuses it forever after,
    which is what makes "Dave" the same ``Person-A`` in submission 1 and in
    submission 40, this session and the next.
    """
    normalized = normalize_text(text)
    existing = lookup_code(db, course_id, normalized, kind)
    if existing:
        return existing

    code = _next_code(db, course_id, kind)
    row = PseudonymMap(
        course_id=course_id,
        kind=kind,
        original_text_normalized=normalized,
        original_text=_WS_RE.sub(" ", str(text or "")).strip(),
        code=code,
    )
    db.add(row)
    try:
        db.commit()
    except Exception:  # noqa: BLE001 - unique race: someone else claimed it
        db.rollback()
        again = lookup_code(db, course_id, normalized, kind)
        if again:
            return again
        raise
    return code


def pseudonym_entries_by_course(
    db: Session, course_ids: Iterable[int]
) -> dict[int, list[dict[str, Any]]]:
    """Everything behind the codes, grouped by course for Settings → Privacy.

    Roster students are included even though they have no map row: their code
    comes from their per-course number, and the professor still wants to read
    "Student-07 is Amara Osei" in the same table.
    """
    ids = list(dict.fromkeys(int(course_id) for course_id in course_ids))
    entries = {course_id: [] for course_id in ids}
    if not ids:
        return entries

    students = db.scalars(
        select(Student)
        .where(Student.course_id.in_(ids))
        .order_by(Student.course_id, Student.student_number)
    ).all()
    for student in students:
        entries[student.course_id].append(
            {
                "id": None,
                "code": student_code(student),
                "original_text": student.name,
                "kind": KIND_PERSON,
                "source": SOURCE_ROSTER,
                "student_id": student.id,
                "created_at": student.created_at.isoformat() if student.created_at else None,
            }
        )
    rows = db.scalars(
        select(PseudonymMap)
        .where(
            PseudonymMap.course_id.in_(ids),
            PseudonymMap.retired_at.is_(None),
        )
        .order_by(PseudonymMap.course_id, PseudonymMap.kind, PseudonymMap.id)
    ).all()
    for row in rows:
        entries[row.course_id].append(
            {
                "id": row.id,
                "code": row.code,
                "original_text": row.original_text or row.original_text_normalized,
                "kind": row.kind,
                "source": "map",
                "student_id": None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )
    return entries


def pseudonym_entries(db: Session, course_id: int) -> list[dict[str, Any]]:
    """Everything behind the codes for one course — Settings → Privacy."""
    return pseudonym_entries_by_course(db, [course_id]).get(course_id, [])


#: Marks a tombstoned row's identity slot. Unique per row so the same person
#: can be re-sighted later without tripping the (course, kind, text) index.
RETIRED_SENTINEL = "__retired__"


def retire_pseudonym(db: Session, row: PseudonymMap) -> str:
    """Forget one mapping without ever freeing its code.

    The professor asked to forget who ``Person-A`` was, so the identity is
    wiped — but the row stays as a tombstone, because handing ``Person-A`` to
    the next new face would silently rewrite the meaning of every stored piece
    of feedback that mentions it. The next sighting gets a fresh code.
    """
    code = row.code
    row.retired_at = utcnow()
    row.original_text = ""
    row.original_text_normalized = f"{RETIRED_SENTINEL}:{row.id}"
    db.add(row)
    db.commit()
    return code


# --------------------------------------------------------------------------
# pass 2 — the local LLM sweep
# --------------------------------------------------------------------------

SPAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string"},
                },
                "required": ["text", "kind"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["spans"],
    "additionalProperties": False,
}

SWEEP_SYSTEM_PROMPT = """
You find personal identifiers in a student's coursework so they can be replaced
with codes before the text is shown to anyone else.

Return STRICT JSON only:
{"spans": [{"text": "<exact substring from the text>", "kind": "person|email|phone|id"}]}

Rules:
- Copy each span EXACTLY as it appears in the text, character for character.
- Include: people's names and nicknames (classmates, roommates, tutors, family,
  instructors), email addresses, phone numbers, and student/ID numbers.
- Do NOT include codes that are already anonymised: Student-07, Person-A,
  [EMAIL-1], [PHONE-1], [ID-1].
- Do NOT include the names of authors, philosophers, historical figures or
  public organisations that are the *subject* of the work.
- Do NOT invent spans. If there are none, return {"spans": []}.
""".strip()

#: The local model runs with a small context window; chunk well inside it.
SWEEP_CHUNK_CHARS = 1500
#: A run that would take more chunks than this stops early and says so.
SWEEP_MAX_CHUNKS = 24


def chunk_text(
    text: str, size: int = SWEEP_CHUNK_CHARS, *, max_chunks: int | None = SWEEP_MAX_CHUNKS
) -> list[str]:
    """Split on paragraph, then line, then hard boundaries.

    ``max_chunks=None`` returns the whole text, which is how :func:`sweep_spans`
    finds out that a long submission was cut short — silently sweeping a
    fraction of an essay while reporting a clean scan is worse than saying so.
    """
    body = str(text or "")
    if len(body) <= size:
        return [body] if body.strip() else []
    chunks: list[str] = []
    current = ""
    for paragraph in re.split(r"(\n\s*\n)", body):
        if not paragraph:
            continue
        if len(current) + len(paragraph) <= size:
            current += paragraph
            continue
        if current.strip():
            chunks.append(current)
        current = ""
        while len(paragraph) > size:
            cut = paragraph.rfind("\n", 0, size)
            if cut <= 0:
                cut = paragraph.rfind(" ", 0, size)
            if cut <= 0:
                cut = size
            chunks.append(paragraph[:cut])
            paragraph = paragraph[cut:]
        current = paragraph
    if current.strip():
        chunks.append(current)
    kept = [c for c in chunks if c.strip()]
    return kept if max_chunks is None else kept[:max_chunks]


class CloudSweepRefused(RuntimeError):
    """Raised when someone tries to run the PII sweep off this machine."""


def check_sweep_target(provider: Any) -> None:
    """The "it stays on this box" invariant, checked by name AND by address.

    A provider called ``local`` is not local because it says so: the sweep sees
    the submission text *before* pass 2 has substituted anything a regex could
    not find, so a base_url pointed at a hosted OpenAI-compatible endpoint would
    post exactly the material this module exists to keep at home.
    """
    provider_name = str(getattr(provider, "name", "") or "")
    if config.is_cloud_provider(provider_name):
        raise CloudSweepRefused(
            f"Refusing to run the PII sweep on cloud provider {provider_name!r}; "
            "the sweep must stay on this machine."
        )
    base_url = getattr(provider, "base_url", None)
    if base_url is None:
        return
    ok, reason = config.check_local_base_url(str(base_url))
    if not ok:
        raise CloudSweepRefused(
            f"Refusing to run the PII sweep against {base_url!r}: {reason} The sweep "
            "reads the submission before it is fully pseudonymized, so it must stay on "
            "this machine."
        )


def sweep_spans(provider: Any, text: str) -> tuple[list[dict[str, str]], list[str], bool]:
    """Ask the on-device model for the PII the regexes could not know about.

    Returns ``(spans, warnings, ran)``. ``ran`` is False when no chunk produced
    a parsed payload — a sweep that only produced failures must not be reported
    to the professor as a completed second pass. Never raises for provider
    problems: a local server that is down or answering garbage degrades to
    pass-1 output, which is the whole point of doing the deterministic pass
    first.
    """
    warnings: list[str] = []
    if provider is None:
        return [], ["The local model is disabled — only the deterministic pass ran."], False

    check_sweep_target(provider)

    all_chunks = chunk_text(text, max_chunks=None)
    chunks = all_chunks[:SWEEP_MAX_CHUNKS]
    if len(all_chunks) > len(chunks):
        swept_chars = sum(len(c) for c in chunks)
        warnings.append(
            f"This submission was too long for the local sweep: only the first "
            f"~{swept_chars} characters ({len(chunks)} of {len(all_chunks)} chunks) were "
            "checked by the local model. The deterministic pass still covered all of it."
        )

    ran = False
    spans: list[dict[str, str]] = []
    for index, chunk in enumerate(chunks):
        payload: Any = None
        for attempt in (1, 2):
            try:
                payload = provider.grade(SWEEP_SYSTEM_PROMPT, [
                    {"type": "text", "text": f"TEXT:\n{chunk}"}
                ], SPAN_SCHEMA)
                break
            except Exception as exc:  # noqa: BLE001 - degrade, never fail grading
                if attempt == 2:
                    warnings.append(
                        f"Local PII sweep failed on chunk {index + 1}: {exc}"
                    )
                    payload = None
                else:
                    log.warning("Local PII sweep chunk %d failed (%s) — retrying", index + 1, exc)
        if not isinstance(payload, dict):
            continue
        ran = True  # this chunk really was read by the local model
        raw_spans = payload.get("spans")
        if not isinstance(raw_spans, list):
            warnings.append(f"Local PII sweep returned no span list for chunk {index + 1}.")
            continue
        for item in raw_spans:
            if not isinstance(item, dict):
                continue
            value = str(item.get("text") or "").strip()
            if not value:
                continue
            spans.append({"text": value, "kind": normalize_kind(item.get("kind"))})
    return spans, warnings, ran


def _llm_detections(
    text: str, spans: Sequence[dict[str, str]], already: set[str]
) -> tuple[list[Detection], list[str]]:
    """Validate every span against the real text before it can substitute.

    A span the model made up (or paraphrased) is dropped and recorded — a
    hallucination must never rewrite a student's submission.

    "Occurs in the text" is decided by the same case-insensitive, word-boundary
    match used to substitute. A small instruct model re-cases and re-spaces
    constantly ("dave kowalski" for "Dave Kowalski"), and treating that as a
    hallucination would ship the real third-party name to the cloud — the
    common failure, not the rare one.
    """
    detections: list[Detection] = []
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    for span in spans:
        value = span["text"].strip()
        kind = span["kind"]
        if len(value) < 2 or looks_like_code(value) or CODE_RE.search(value):
            continue
        normalized = normalize_text(value)
        if not normalized or normalized in already:
            continue
        if (normalized, kind) in seen:
            continue
        seen.add((normalized, kind))
        matches = list(_word_pattern(value).finditer(text))
        if not matches:
            warnings.append(
                f"Local sweep proposed {value!r}, which does not occur in the text — ignored."
            )
            continue
        for match in matches:
            detections.append(
                Detection(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                    kind=kind,
                    source=SOURCE_LLM,
                )
            )
    return detections, warnings


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------


#: Env override for pass 2: ``1``/``0``. Unset means "on, unless under pytest".
SWEEP_ENV_VAR = "AGORA_PRIVACY_LLM_SWEEP"


def sweep_allowed() -> bool:
    """Whether pass 2 may talk to the local server right now.

    Pass 2 is a real HTTP call. SPEC is explicit that tests never hit live model
    endpoints, and every deterministic assertion in the suite would otherwise
    depend on a shared llama.cpp server being up, so the sweep is off under
    pytest unless ``AGORA_PRIVACY_LLM_SWEEP=1`` (what the ``local_llm``
    integration tests set). Pass 1 is unaffected — it is pure Python.
    """
    override = os.environ.get(SWEEP_ENV_VAR)
    if override is not None:
        return override.strip().lower() not in ("", "0", "false", "no", "off")
    return "pytest" not in sys.modules


def get_sweep_provider(settings: dict[str, Any] | None = None) -> Any:
    """The on-device provider used for pass 2 (``None`` when switched off)."""
    settings = settings or config.load_privacy_settings()
    local = settings.get("local_model") or {}
    if not settings.get("llm_sweep", True) or not local.get("enabled", True):
        return None
    if not sweep_allowed():
        return None
    from app.ai.providers import LocalProvider  # local import: avoids a cycle

    return LocalProvider(model=local.get("model"), base_url=local.get("base_url"))


def empty_report(mode: str = MODE_SWAP) -> dict[str, Any]:
    return {
        "mode": mode,
        "swapped": False,
        "counts": {"total": 0, "pass1": 0, "pass2": 0, "by_kind": {}},
        "findings": [],
        "codes": {},
        "warnings": [],
        "llm_sweep": {"ran": False, "spans": 0, "provider": None, "model": None},
    }


def pseudonymize_text(
    db: Session,
    text: str,
    *,
    course_id: int | None,
    mode: str = MODE_SWAP,
    provider: Any = None,
    use_llm: bool = True,
    settings: dict[str, Any] | None = None,
    students: Sequence[Student] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Swap PII for stable codes. Returns ``(text, report)``.

    ``mode``:
      * ``swap`` — substitute, allocating codes in ``PseudonymMap``.
      * ``warn`` — detect and report only; the text comes back unchanged and no
        new codes are allocated (existing ones are shown so the report is
        still readable).
      * ``off``  — no detection at all.
    """
    mode = (mode or MODE_SWAP).strip().lower()
    report = empty_report(mode)
    source_text = str(text or "")
    if mode == MODE_OFF or not source_text.strip():
        return source_text, report

    settings = settings or config.load_privacy_settings()
    students = list(students) if students is not None else roster_students(db, course_id)
    swapping = mode == MODE_SWAP and bool(course_id)
    if mode == MODE_SWAP and not course_id:
        # Codes are allocated per course, so without one nothing can be
        # swapped. Callers must treat this as a hard stop rather than sending
        # the untouched text on: see protect_submission_block.
        report["swap_unavailable"] = True
        report["warnings"].append(
            "This submission is not linked to a course, so no stable codes could be "
            "assigned; nothing was swapped."
        )

    # -- pass 1: deterministic ---------------------------------------------
    pass1 = _resolve_overlaps(
        _roster_detections(source_text, students) + _regex_detections(source_text)
    )
    report["warnings"].extend(_ambiguous_warnings(pass1))
    for det in pass1:
        if det.code is not None:  # roster hit: code is the student's number
            continue
        if swapping:
            det.code = code_for(db, int(course_id), det.text, det.kind)
        else:
            det.code = lookup_code(db, course_id, det.text, det.kind)

    working = _apply(source_text, pass1) if swapping else source_text

    # -- pass 2: local LLM sweep -------------------------------------------
    pass2: list[Detection] = []
    sweep_info: dict[str, Any] = {"ran": False, "spans": 0, "provider": None, "model": None}
    if use_llm:
        if provider is None:
            provider = get_sweep_provider(settings)
        if provider is not None:
            sweep_info["provider"] = str(getattr(provider, "name", "") or "")
            sweep_info["model"] = str(getattr(provider, "model", "") or "")
            spans, sweep_warnings, ran = sweep_spans(provider, working)
            # `ran` is False when every chunk failed: the UI must then say
            # "deterministic pass only", not advertise a two-pass scan.
            sweep_info["ran"] = ran
            report["warnings"].extend(sweep_warnings)
            already = {det.normalized for det in pass1}
            for det in pass1:
                already.add(normalize_text(det.code or ""))
            candidates, validation_warnings = _llm_detections(working, spans, already)
            report["warnings"].extend(validation_warnings)
            sweep_info["spans"] = len(spans)
            # Never overlap a code that pass 1 just wrote.
            pass2 = [
                det
                for det in _resolve_overlaps(candidates)
                if not CODE_RE.fullmatch(det.text.strip())
            ]
            for det in pass2:
                if swapping:
                    det.code = code_for(db, int(course_id), det.text, det.kind)
                else:
                    det.code = lookup_code(db, course_id, det.text, det.kind)
        else:
            report["warnings"].append(
                "The local model sweep is switched off — only the deterministic pass ran."
            )
    report["llm_sweep"] = sweep_info

    final = _apply(working, pass2) if swapping else working

    # -- report -------------------------------------------------------------
    findings: dict[tuple[str, str, str], dict[str, Any]] = {}
    for det in list(pass1) + list(pass2):
        key = (det.normalized, det.kind, det.source)
        entry = findings.get(key)
        if entry is None:
            findings[key] = {
                "text": det.text,
                "code": det.code,
                "kind": det.kind,
                "source": det.source,
                "occurrences": 1,
                "student_id": det.student_id,
            }
        else:
            entry["occurrences"] += 1
    by_kind: dict[str, int] = {}
    for entry in findings.values():
        by_kind[entry["kind"]] = by_kind.get(entry["kind"], 0) + 1

    report["findings"] = sorted(
        findings.values(), key=lambda e: (e["kind"], str(e["code"] or ""), e["text"])
    )
    report["codes"] = {
        entry["code"]: entry["text"] for entry in report["findings"] if entry["code"]
    }
    report["counts"] = {
        "total": len(report["findings"]),
        "pass1": len({(d.normalized, d.kind, d.source) for d in pass1}),
        "pass2": len({(d.normalized, d.kind, d.source) for d in pass2}),
        "by_kind": by_kind,
        "occurrences": sum(int(e["occurrences"]) for e in report["findings"]),
    }
    report["swapped"] = bool(swapping and report["counts"]["total"])
    return final, report


def protect_cloud_text(
    db: Session,
    text: str,
    *,
    course_id: int | None,
    settings: dict[str, Any] | None = None,
    subject: str = "submission",
    students: Sequence[Student] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Pseudonymize text for a swap-mode cloud request or fail closed."""
    settings = settings or config.load_privacy_settings()
    try:
        cleaned, report = pseudonymize_text(
            db,
            text,
            course_id=course_id,
            mode=MODE_SWAP,
            settings=settings,
            students=students,
        )
    except CloudSweepRefused:
        raise
    except Exception as exc:  # noqa: BLE001 - swap mode must fail closed
        log.exception("Privacy Guard failed for %s", subject)
        from app.ai.providers import ProviderError  # local import: cycle-safe

        raise ProviderError(
            f"The Privacy Guard could not pseudonymize this {subject}, so it was not "
            f"sent to the provider: {exc}"
        ) from exc

    if report.get("swap_unavailable"):
        from app.ai.providers import ProviderError  # local import: cycle-safe

        raise ProviderError(
            f"The Privacy Guard could not pseudonymize this {subject} because it is not "
            "linked to a course, so it was not sent to the provider. Link the assignment "
            "to a course, or switch Settings → Privacy to warn/off to send it as-is."
        )
    return cleaned, report


# --------------------------------------------------------------------------
# display layer — codes back to real names, for professor-facing text only
# --------------------------------------------------------------------------


def protect_unscoped_text(db: Session, text: str) -> str:
    """Protect a skill trial with no course using temporary, non-reversible codes.

    The detection pass sees every roster. It never creates a synthetic course
    or stores an identity map for a generic skill question.
    """
    settings = config.load_privacy_settings()
    students = list(db.scalars(select(Student)).all())
    _, report = pseudonymize_text(db, text, course_id=None, students=students,
                                 mode=MODE_WARN, settings=settings)
    cleaned = text
    findings = sorted(report["findings"], key=lambda entry: len(entry["text"]), reverse=True)
    for index, entry in enumerate(findings, 1):
        cleaned = _word_pattern(entry["text"]).sub(lambda _: f"[PRIVATE-{index}]", cleaned)
    return cleaned


def display_map(db: Session, course_id: int | None) -> dict[str, str]:
    """``{code: real text}`` for one course. Never sent anywhere."""
    mapping: dict[str, str] = {}
    if not course_id:
        return mapping
    for student in roster_students(db, course_id):
        mapping[student_code(student)] = student.name
        # The grading header identifies students as "Student #7"; the model
        # echoes that shape back in its feedback, so resolve it too.
        mapping[f"Student #{student.student_number}"] = student.name
    for row in db.scalars(_live_rows(course_id)).all():
        mapping[row.code] = row.original_text or row.original_text_normalized
    return mapping


def resubstitute(db: Session, text: str, course_id: int | None) -> str:
    """Swap codes back to real names for the professor-facing UI."""
    return render_for_display(db, text, course_id)["display"]


def render_for_display(
    db: Session, text: str, course_id: int | None
) -> dict[str, Any]:
    """Both forms of one piece of AI-written text.

    ``raw`` is exactly what the model wrote (codes intact — the professor can
    see what the AI saw), ``display`` has the real names back in. The frontend
    renders ``display`` with ``raw`` on the title attribute and a toggle.
    """
    raw = str(text or "")
    mapping = display_map(db, course_id)
    replacements: list[dict[str, str]] = []
    if not raw.strip() or not mapping:
        return {
            "raw": raw,
            "display": raw,
            "replacements": replacements,
            "course_id": course_id,
        }

    # Longest code first: "Person-AA" must win over "Person-A". The boundaries
    # stop a code that merely *starts* another token from being half-replaced
    # ("Person-Alpha" must not render as "Dave Kowalskilpha").
    codes = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![\w-])(?:" + "|".join(re.escape(code) for code in codes) + r")(?![\w-])"
    )
    used: dict[str, str] = {}

    def _swap(match: re.Match[str]) -> str:
        code = match.group(0)
        name = mapping.get(code, code)
        used[code] = name
        return name

    display = pattern.sub(_swap, raw)
    replacements = [{"code": code, "name": name} for code, name in sorted(used.items())]
    return {
        "raw": raw,
        "display": display,
        "replacements": replacements,
        "course_id": course_id,
    }


def display_result(db: Session, result: Any, course_id: int | None) -> dict[str, Any]:
    """The display-layer pass over one GradeResult's professor-facing text."""
    summary = render_for_display(db, getattr(result, "summary_feedback", "") or "", course_id)
    criteria: list[dict[str, Any]] = []
    for crit in getattr(result, "criteria", None) or []:
        if not isinstance(crit, dict):
            continue
        rendered = render_for_display(db, str(crit.get("comment") or ""), course_id)
        criteria.append(
            {
                "key": crit.get("key"),
                "raw": rendered["raw"],
                "display": rendered["display"],
                "replacements": rendered["replacements"],
            }
        )
    replacements = {r["code"]: r["name"] for r in summary["replacements"]}
    for crit in criteria:
        for item in crit["replacements"]:
            replacements[item["code"]] = item["name"]
    return {
        "summary_feedback": summary,
        "criteria": criteria,
        "replacements": [{"code": c, "name": n} for c, n in sorted(replacements.items())],
    }


# --------------------------------------------------------------------------
# persistence of scan reports
# --------------------------------------------------------------------------


def record_scan(db: Session, submission_id: int, report: dict[str, Any]) -> PrivacyScan:
    """One scan per submission — a regrade replaces the previous report."""
    for existing in db.scalars(
        select(PrivacyScan).where(PrivacyScan.submission_id == submission_id)
    ).all():
        db.delete(existing)
    scan = PrivacyScan(
        submission_id=submission_id,
        mode=str(report.get("mode") or MODE_SWAP),
        findings=report,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan


def latest_scan(db: Session, submission_id: int) -> Optional[PrivacyScan]:
    return db.scalars(
        select(PrivacyScan)
        .where(PrivacyScan.submission_id == submission_id)
        .order_by(PrivacyScan.id.desc())
    ).first()


def scan_dict(scan: PrivacyScan | None) -> Optional[dict[str, Any]]:
    if scan is None:
        return None
    report = scan.findings if isinstance(scan.findings, dict) else {}
    counts = report.get("counts") or {}
    total = int(counts.get("total") or 0)
    return {
        "id": scan.id,
        "submission_id": scan.submission_id,
        "mode": scan.mode,
        "created_at": scan.created_at.isoformat() if scan.created_at else None,
        "total": total,
        "swapped": bool(report.get("swapped")),
        # "3 identifiers swapped — view"
        "headline": _headline(scan.mode, total, bool(report.get("swapped"))),
        "report": report,
    }


def _headline(mode: str, total: int, swapped: bool) -> str:
    noun = "identifier" if total == 1 else "identifiers"
    if mode == MODE_OFF:
        return "Privacy Guard off — the submission was sent as-is."
    if swapped:
        return f"{total} {noun} swapped"
    if total:
        return f"{total} {noun} found, nothing swapped"
    return "No identifiers found"


# --------------------------------------------------------------------------
# grading-path integration
# --------------------------------------------------------------------------

PSEUDONYMIZED_HEADER = (
    "[Privacy Guard: this is the locally extracted text of the student's "
    "submission. Names, emails, phone numbers and id numbers have been replaced "
    "with stable codes (Student-07, Person-A, [EMAIL-1]). Treat each code as a "
    "consistent identity and reuse the codes in your feedback.]"
)


#: What ``extract_pdf_text`` appends when it hits its character cap.
TRUNCATION_MARKER = "[... truncated ...]"


def _looks_truncated(text: str) -> bool:
    return TRUNCATION_MARKER in str(text or "")[-200:]


def submission_text_for_scan(block: dict[str, Any], raw: bytes | None) -> str:
    """The text the guard can actually inspect for one submission block.

    In swap mode this is also exactly what the provider receives, so the guard
    can never be scanning less than it sends.
    """
    btype = block.get("type")
    if btype == "text":
        return str(block.get("text") or "")
    if btype == "document":
        existing = block.get("_text")
        if existing:
            return str(existing)
        if raw:
            from app.ai.grading import extract_pdf_text  # local import: cycle-safe

            return extract_pdf_text(raw)
    return ""


def protect_submission_block(
    db: Session,
    submission: Submission,
    block: dict[str, Any],
    *,
    provider: str,
    course_id: int | None,
    raw: bytes | None = None,
    notes: list[str] | None = None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """Apply the Privacy Guard to the block that carries the submission.

    Returns ``(block, report)``. In ``swap`` mode a PDF becomes pseudonymized
    **extracted text** — the fidelity tradeoff docs/INCREMENT_1.md accepts —
    while native files that cannot be pseudonymized are not sent. Local and
    mock providers are exempt: their calls never leave the machine.
    """
    notes = notes if notes is not None else []
    settings = config.load_privacy_settings()
    mode = settings["mode"]

    if not config.is_cloud_provider(provider):
        return block, None
    if mode == MODE_OFF:
        notes.append(
            "Privacy Guard is off — the submission was sent to the provider unmodified."
        )
        return block, None

    btype = block.get("type")
    if btype == "image":
        if mode == MODE_SWAP:
            from app.ai.providers import ProviderError  # local import: cycle-safe

            report = empty_report(MODE_SWAP)
            report["warnings"].append(
                "The image could not be pseudonymized, so it was not sent to the provider."
            )
            _safe_record(db, submission, report)
            raise ProviderError(report["warnings"][0])
        report = empty_report(MODE_WARN)
        report["warnings"].append(
            "Image submissions cannot be text-swapped; this one was sent as an image "
            "(warn behaviour). Grade image work with a local skill for full privacy."
        )
        notes.append(
            "Privacy Guard: image submission — identifiers in the image could not be "
            "swapped."
        )
        _safe_record(db, submission, report)
        return block, report

    text = submission_text_for_scan(block, raw)
    if not text.strip():
        if mode == MODE_SWAP:
            from app.ai.providers import ProviderError  # local import: cycle-safe

            report = empty_report(MODE_SWAP)
            report["warnings"].append(
                "No text layer was available, so the file could not be pseudonymized and "
                "was not sent to the provider."
            )
            _safe_record(db, submission, report)
            raise ProviderError(report["warnings"][0])
        report = empty_report(MODE_WARN)
        report["warnings"].append(
            "No text layer could be extracted from this submission, so nothing could be "
            "swapped; it was sent in its original form (warn behaviour)."
        )
        notes.append(
            "Privacy Guard: no extractable text (likely a scan) — the file was sent "
            "unmodified."
        )
        _safe_record(db, submission, report)
        return block, report

    if mode == MODE_SWAP:
        cleaned, report = protect_cloud_text(
            db,
            text,
            course_id=course_id,
            settings=settings,
            subject="submission",
        )
    else:
        try:
            cleaned, report = pseudonymize_text(
                db, text, course_id=course_id, mode=mode, settings=settings
            )
        except CloudSweepRefused:
            raise
        except Exception as exc:  # noqa: BLE001 - warn mode keeps current behavior
            log.exception(
                "Privacy Guard failed for submission %s", getattr(submission, "id", "?")
            )
            report = empty_report(mode)
            report["warnings"].append(f"Privacy scan failed: {exc}")
            _safe_record(db, submission, report)
            notes.append("Privacy Guard: the scan failed; the submission was sent unmodified.")
            return block, report

    _safe_record(db, submission, report)

    if mode == MODE_SWAP:
        total = int(report["counts"]["total"])
        # ALWAYS the extracted text, never the native file. A text pass that
        # found nothing has not cleared the file: PDF metadata, a letterhead or
        # signature image, form-field values and anything past the extraction
        # cap are invisible to the scanner and perfectly legible to the cloud
        # model. docs/INCREMENT_1.md: swap-mode cloud calls get extracted text
        # "not as the native PDF".
        if total:
            notes.append(
                f"Privacy Guard: {total} identifier(s) swapped for stable codes; the "
                "provider received extracted text, not the original file."
            )
        else:
            notes.append(
                "Privacy Guard: no identifiers were found in the extracted text; the "
                "provider still received that text rather than the original file, so "
                "nothing outside the text layer could leak."
            )
        if _looks_truncated(cleaned):
            notes.append(
                "Privacy Guard: the submission was longer than the extraction limit, so "
                "the provider received the truncated text that was actually scanned."
            )
        return {"type": "text", "text": f"{PSEUDONYMIZED_HEADER}\n\n{cleaned}"}, report

    notes.append(
        f"Privacy Guard (warn): {report['counts']['total']} identifier(s) found and left "
        "in place — switch to swap mode in Settings → Privacy to replace them."
    )
    return block, report


def _safe_record(db: Session, submission: Submission, report: dict[str, Any]) -> None:
    submission_id = getattr(submission, "id", None)
    if not submission_id:
        return
    try:
        record_scan(db, int(submission_id), report)
    except Exception:  # noqa: BLE001 - a report is never worth failing a grade over
        log.warning("Could not persist the privacy scan for submission %s", submission_id)
        db.rollback()


__all__ = [
    "MODE_OFF",
    "MODE_WARN",
    "MODE_SWAP",
    "KIND_PERSON",
    "KIND_EMAIL",
    "KIND_PHONE",
    "KIND_ID",
    "CODE_RE",
    "SPAN_SCHEMA",
    "SWEEP_SYSTEM_PROMPT",
    "CloudSweepRefused",
    "Detection",
    "student_code",
    "person_code",
    "typed_code",
    "normalize_text",
    "normalize_kind",
    "chunk_text",
    "code_for",
    "code_index",
    "lookup_code",
    "pseudonym_entries_by_course",
    "pseudonym_entries",
    "retire_pseudonym",
    "check_sweep_target",
    "roster_students",
    "sweep_spans",
    "get_sweep_provider",
    "pseudonymize_text",
    "protect_cloud_text",
    "display_map",
    "resubstitute",
    "render_for_display",
    "display_result",
    "record_scan",
    "latest_scan",
    "scan_dict",
    "protect_submission_block",
    "submission_text_for_scan",
    "empty_report",
]
