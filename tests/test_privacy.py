"""Privacy Guard tests — pseudonymization, the local provider, and the display layer.

The deterministic pass is unit-tested end to end with a **fake** local model, so
every assertion here is offline and repeatable (SPEC: tests never hit real
APIs). The handful of tests marked ``local_llm`` do talk to the real llama.cpp
server on 127.0.0.1:3782 and skip cleanly when it is not up.

The load-bearing assertions:
  * roster names, nicknames, emails, phones and id numbers are SWAPPED for
    stable codes — not blacked out;
  * the same identity gets the same code across submissions and sessions;
  * the payload that would reach a cloud provider contains ZERO of the
    original strings;
  * a span the model hallucinates can never rewrite the submission;
  * local/mock providers are exempt, because their calls never leave the box.
"""

from __future__ import annotations

import io
import json
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app import config
from app.ai import assistant as assistant_mod
from app.ai import grading as engine
from app.ai import privacy
from app.ai import providers as providers_mod
from app.ai.providers import LocalProvider, parse_json_object
from app.db import get_db
from app.main import app
from app.models import (
    Base,
    ChatMessage,
    ChatSession,
    Course,
    KnowledgeDoc,
    PrivacyScan,
    PseudonymMap,
    Skill,
    Student,
    Submission,
)
from app.routers import grading as grading_router

ROSTER = ["Amara Osei", "Ben Whitaker", "Claudia Moreno"]

RUBRIC_CRITERIA = [
    {"key": "thesis", "title": "Thesis", "description": "States a thesis.", "max_points": 10},
    {"key": "evidence", "title": "Evidence", "description": "Uses sources.", "max_points": 10},
]

#: One submission with every kind of identifier the guard must handle.
SUBMISSION_TEXT = (
    "Essay by Amara Osei for PHIL 210.\n\n"
    "I worked through the trolley cases with my roommate Dave Kowalski, and Whitaker "
    "pushed back on my second premise. Amara can be reached at amara.osei@uni.edu or "
    "on 555-214-8890, and my student id is 99241033.\n\n"
    "Kant is the subject of section two, not a classmate."
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def Session(tmp_path):
    engine_ = create_engine(
        f"sqlite:///{tmp_path / 'privacy.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine_)
    factory = sessionmaker(bind=engine_, autoflush=False, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine_.dispose()


@pytest.fixture()
def db(Session):
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Privacy settings live in DATA_DIR — never touch the developer's real one."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    return data_dir


@pytest.fixture()
def client(Session, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "submissions")
    monkeypatch.setattr(grading_router, "session_factory", Session)

    def override_get_db():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture()
def course(db):
    course = Course(name="PHIL 210", term="Fall 2026")
    db.add(course)
    db.commit()
    db.refresh(course)
    for number, name in enumerate(ROSTER, start=1):
        db.add(
            Student(
                course_id=course.id,
                name=name,
                student_number=number,
                email=f"{name.split()[0].lower()}.{name.split()[1].lower()}@uni.edu",
            )
        )
    db.commit()
    return course


class FakeLocalProvider:
    """A stand-in for the Gemma sweep: returns scripted spans, records calls.

    Only the spans that really occur in the chunk are returned by default, so
    the fake behaves like a well-behaved model; ``extra_spans`` lets a test
    script a hallucination.
    """

    name = config.LOCAL_PROVIDER
    model = "fake-gemma-3-4b-it"

    def __init__(self, spans=(), extra_spans=(), fail_times=0, bad_json_times=0):
        self.spans = list(spans)
        self.extra_spans = list(extra_spans)
        self.fail_times = fail_times
        self.bad_json_times = bad_json_times
        self.calls: list[dict] = []

    def grade(self, system_prompt, content_blocks, schema):
        text = "\n".join(b.get("text", "") for b in content_blocks)
        self.calls.append(
            {"system_prompt": system_prompt, "text": text, "schema": schema}
        )
        if self.fail_times:
            self.fail_times -= 1
            raise providers_mod.ProviderConnectionError("local model is down")
        if self.bad_json_times:
            self.bad_json_times -= 1
            raise ValueError("not JSON")
        found = [dict(span) for span in self.spans if span["text"] in text]
        return {"spans": found + [dict(s) for s in self.extra_spans]}


class CapturingCloudChatProvider:
    name = "anthropic"
    model = "capturing-cloud"

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls: list[dict] = []

    def chat(self, system_prompt, messages, tools=None):
        self.calls.append(
            {
                "system": system_prompt,
                "messages": json.loads(json.dumps(messages)),
                "tools": tools,
            }
        )
        turn = self.turns.pop(0)
        return {
            "text": turn.get("text", ""),
            "tool_calls": [
                {
                    "id": f"toolu_{index}",
                    "name": call["name"],
                    "arguments": call["input"],
                }
                for index, call in enumerate(turn.get("tool_calls", []))
            ],
            "stop_reason": "tool_use" if turn.get("tool_calls") else "end_turn",
        }


@pytest.fixture()
def fake_llm():
    return FakeLocalProvider(
        spans=[
            {"text": "Dave Kowalski", "kind": "person"},
            {"text": "Whitaker", "kind": "person"},
        ]
    )


def _text_pdf(text: str) -> bytes:
    """A minimal single-page PDF with a real text layer (one line per paragraph)."""
    lines = [line for line in text.splitlines() if line.strip()]
    parts = ["BT /F1 12 Tf 72 720 Td 14 TL"]
    for line in lines:
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        parts.append(f"({escaped}) Tj T*")
    parts.append("ET")
    content = " ".join(parts).encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources "
        b"<< /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += str(index).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    start = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += ("%010d 00000 n \n" % offset).encode()
    out += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\nstartxref\n"
        + str(start).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


# --------------------------------------------------------------------------
# codes
# --------------------------------------------------------------------------


def test_code_shapes_are_stable_and_ordered():
    assert privacy.student_code(7) == "Student-07"
    assert privacy.student_code(12) == "Student-12"
    assert privacy.person_code(0) == "Person-A"
    assert privacy.person_code(25) == "Person-Z"
    assert privacy.person_code(26) == "Person-AA"
    assert privacy.typed_code(privacy.KIND_EMAIL, 0) == "[EMAIL-1]"
    assert privacy.typed_code(privacy.KIND_PHONE, 2) == "[PHONE-3]"
    assert privacy.typed_code(privacy.KIND_ID, 0) == "[ID-1]"
    for code in ("Student-07", "Person-AA", "[EMAIL-1]", "[PHONE-3]", "[ID-1]"):
        assert privacy.looks_like_code(code)
    assert not privacy.looks_like_code("Amara Osei")


# --------------------------------------------------------------------------
# pass 1 — deterministic
# --------------------------------------------------------------------------


def test_pass_one_swaps_roster_names_emails_phones_and_ids(db, course):
    text, report = privacy.pseudonymize_text(
        db, SUBMISSION_TEXT, course_id=course.id, use_llm=False
    )

    # Swapped, not blacked out: the codes are all there...
    assert "Student-01" in text
    assert "[EMAIL-1]" in text
    assert "[PHONE-1]" in text
    assert "[ID-1]" in text
    # ...and every original is gone.
    for secret in ("Amara", "Osei", "amara.osei@uni.edu", "555-214-8890", "99241033"):
        assert secret not in text

    kinds = {f["kind"] for f in report["findings"]}
    assert kinds == {privacy.KIND_PERSON, privacy.KIND_EMAIL, privacy.KIND_PHONE, privacy.KIND_ID}
    assert report["swapped"] is True
    assert report["counts"]["pass2"] == 0
    assert report["mode"] == privacy.MODE_SWAP


def test_roster_first_and_last_names_alone_are_caught(db, course):
    text, _ = privacy.pseudonymize_text(
        db,
        "Osei argued with Claudia about Whitaker's draft.",
        course_id=course.id,
        use_llm=False,
    )
    assert text == "Student-01 argued with Student-03 about Student-02's draft."


def test_longest_match_wins_over_the_first_name(db, course):
    text, _ = privacy.pseudonymize_text(
        db, "Amara Osei and Amara agree.", course_id=course.id, use_llm=False
    )
    assert text == "Student-01 and Student-01 agree."


def test_an_email_is_not_shredded_into_name_fragments(db, course):
    text, report = privacy.pseudonymize_text(
        db, "Write to amara.osei@uni.edu today.", course_id=course.id, use_llm=False
    )
    assert text == "Write to [EMAIL-1] today."
    assert [f["kind"] for f in report["findings"]] == [privacy.KIND_EMAIL]


def test_phone_formats_and_labelled_ids(db, course):
    text, _ = privacy.pseudonymize_text(
        db,
        "Call (555) 214-8890 or +1 555 214 8891. Student ID: 4471. Section 12 met twice.",
        course_id=course.id,
        use_llm=False,
    )
    assert "[PHONE-1]" in text and "[PHONE-2]" in text
    assert "[ID-1]" in text
    # A small ordinary number is not an identifier.
    assert "Section 12 met twice." in text


def test_off_mode_changes_nothing(db, course):
    text, report = privacy.pseudonymize_text(
        db, SUBMISSION_TEXT, course_id=course.id, mode=privacy.MODE_OFF, use_llm=False
    )
    assert text == SUBMISSION_TEXT
    assert report["findings"] == []
    assert db.scalars(select(PseudonymMap)).all() == []


def test_warn_mode_reports_without_touching_the_text_or_the_map(db, course):
    text, report = privacy.pseudonymize_text(
        db, SUBMISSION_TEXT, course_id=course.id, mode=privacy.MODE_WARN, use_llm=False
    )
    assert text == SUBMISSION_TEXT
    assert report["counts"]["total"] >= 4
    assert report["swapped"] is False
    # warn allocates nothing — the professor has not decided yet.
    assert db.scalars(select(PseudonymMap)).all() == []


# --------------------------------------------------------------------------
# pass 2 — the local LLM sweep (fake model)
# --------------------------------------------------------------------------


def test_sweep_catches_the_third_party_and_the_nickname(db, course, fake_llm):
    fake_llm.spans.append({"text": "Bex", "kind": "person"})
    text, report = privacy.pseudonymize_text(
        db,
        SUBMISSION_TEXT + "\nBex lent me her notes.",
        course_id=course.id,
        provider=fake_llm,
    )
    assert "Dave Kowalski" not in text and "Bex" not in text
    assert "Person-A" in text and "Person-B" in text
    # "Whitaker" is on the roster: pass 1 already claimed it, so the sweep's
    # duplicate must not mint a second code for the same person.
    assert "Student-02" in text
    assert report["counts"]["pass2"] == 2
    assert report["llm_sweep"]["ran"] is True
    assert report["llm_sweep"]["provider"] == config.LOCAL_PROVIDER
    codes = {f["code"] for f in report["findings"] if f["source"] == privacy.SOURCE_LLM}
    assert codes == {"Person-A", "Person-B"}


def test_the_sweep_never_sees_pii_the_first_pass_already_removed(db, course, fake_llm):
    privacy.pseudonymize_text(db, SUBMISSION_TEXT, course_id=course.id, provider=fake_llm)
    swept = "\n".join(call["text"] for call in fake_llm.calls)
    for secret in ("Amara", "Osei", "amara.osei@uni.edu", "555-214-8890", "99241033"):
        assert secret not in swept
    assert "Student-01" in swept


def test_a_hallucinated_span_can_never_rewrite_the_text(db, course):
    liar = FakeLocalProvider(
        spans=[{"text": "Dave Kowalski", "kind": "person"}],
        extra_spans=[{"text": "Marguerite Delacroix-Fenwick", "kind": "person"}],
    )
    text, report = privacy.pseudonymize_text(
        db, SUBMISSION_TEXT, course_id=course.id, provider=liar
    )
    assert "Person-A" in text  # the real span was swapped
    assert "Person-B" not in text  # the invented one was not
    assert any("does not occur" in w for w in report["warnings"])
    assert not db.scalars(
        select(PseudonymMap).where(
            PseudonymMap.original_text_normalized == "marguerite delacroix-fenwick"
        )
    ).all()


def test_a_broken_sweep_degrades_to_pass_one_with_a_warning(db, course):
    broken = FakeLocalProvider(bad_json_times=10)
    text, report = privacy.pseudonymize_text(
        db, SUBMISSION_TEXT, course_id=course.id, provider=broken
    )
    # Pass 1 still protected everything it knows about.
    assert "Student-01" in text and "[EMAIL-1]" in text
    assert "Dave Kowalski" in text  # only the sweep could have caught this
    assert any("sweep failed" in w for w in report["warnings"])
    assert report["counts"]["pass2"] == 0
    # Two attempts per chunk: invalid JSON is retried exactly once.
    assert len(broken.calls) == 2


def test_the_sweep_refuses_to_run_on_a_cloud_provider(db, course):
    class PretendCloud:
        name = "anthropic"
        model = "claude-opus-5"

        def grade(self, *_args, **_kwargs):  # pragma: no cover - must never run
            raise AssertionError("the sweep must never call a cloud provider")

    with pytest.raises(privacy.CloudSweepRefused):
        privacy.sweep_spans(PretendCloud(), SUBMISSION_TEXT)


def test_chunking_covers_the_whole_text():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 80 for i in range(12))
    chunks = privacy.chunk_text(body)
    assert len(chunks) > 1
    assert all(len(c) <= privacy.SWEEP_CHUNK_CHARS + 40 for c in chunks)
    assert "".join(chunks).replace("\n", "").replace(" ", "") == body.replace(
        "\n", ""
    ).replace(" ", "")


# --------------------------------------------------------------------------
# stability of codes
# --------------------------------------------------------------------------


def test_the_same_person_gets_the_same_code_across_two_submissions(db, course, fake_llm):
    first, report_one = privacy.pseudonymize_text(
        db, "Dave Kowalski explained Rawls to me.", course_id=course.id, provider=fake_llm
    )
    second, report_two = privacy.pseudonymize_text(
        db,
        "Later Dave Kowalski disagreed, and so did Amara Osei.",
        course_id=course.id,
        provider=fake_llm,
    )
    assert "Person-A" in first and "Person-A" in second
    assert report_one["codes"]["Person-A"] == "Dave Kowalski"
    assert report_two["codes"]["Person-A"] == "Dave Kowalski"
    assert "Student-01" in second
    # One identity, one row — not one row per sighting.
    rows = db.scalars(select(PseudonymMap).where(PseudonymMap.kind == "person")).all()
    assert len(rows) == 1


def test_code_lookup_is_case_and_whitespace_insensitive(db, course):
    first = privacy.code_for(db, course.id, "Dave Kowalski", privacy.KIND_PERSON)
    again = privacy.code_for(db, course.id, "  dave   kowalski ", privacy.KIND_PERSON)
    assert first == again == "Person-A"
    assert len(db.scalars(select(PseudonymMap)).all()) == 1


def test_codes_are_scoped_per_course(db, course):
    other = Course(name="HIST 101")
    db.add(other)
    db.commit()
    db.refresh(other)
    assert privacy.code_for(db, course.id, "Dave Kowalski", privacy.KIND_PERSON) == "Person-A"
    assert privacy.code_for(db, other.id, "Nia Roberts", privacy.KIND_PERSON) == "Person-A"
    assert privacy.code_for(db, course.id, "Nia Roberts", privacy.KIND_PERSON) == "Person-B"


def test_a_forgotten_mapping_never_hands_its_code_to_someone_else(db, course):
    """Forgetting Dave must not make Frank the new "Person-A".

    Every stored GradeResult, scan report and card that says "Person-A" would
    otherwise silently start re-substituting to a different human being.
    """
    privacy.code_for(db, course.id, "Dave Kowalski", privacy.KIND_PERSON)  # Person-A
    second = privacy.code_for(db, course.id, "Erin Vaughn", privacy.KIND_PERSON)  # Person-B
    assert second == "Person-B"

    row = db.scalars(select(PseudonymMap).where(PseudonymMap.code == "Person-A")).first()
    privacy.retire_pseudonym(db, row)

    # The identity really is forgotten...
    assert privacy.lookup_code(db, course.id, "Dave Kowalski", privacy.KIND_PERSON) is None
    assert "Person-A" not in privacy.display_map(db, course.id)
    assert [e["code"] for e in privacy.pseudonym_entries(db, course.id) if e["id"]] == [
        "Person-B"
    ]
    # ...but its code stays spoken for.
    assert privacy.code_for(db, course.id, "Frank Mueller", privacy.KIND_PERSON) == "Person-C"
    # And Dave, sighted again, is a genuinely new identity.
    assert privacy.code_for(db, course.id, "Dave Kowalski", privacy.KIND_PERSON) == "Person-D"


def test_retiring_the_highest_code_still_does_not_recycle_it(db, course):
    privacy.code_for(db, course.id, "Dave Kowalski", privacy.KIND_PERSON)  # Person-A
    privacy.code_for(db, course.id, "Erin Vaughn", privacy.KIND_PERSON)  # Person-B
    row = db.scalars(select(PseudonymMap).where(PseudonymMap.code == "Person-B")).first()
    privacy.retire_pseudonym(db, row)
    assert privacy.code_for(db, course.id, "Frank Mueller", privacy.KIND_PERSON) == "Person-C"


def test_typed_codes_come_off_the_same_high_water_mark(db, course):
    assert privacy.code_for(db, course.id, "a@uni.edu", privacy.KIND_EMAIL) == "[EMAIL-1]"
    assert privacy.code_for(db, course.id, "b@uni.edu", privacy.KIND_EMAIL) == "[EMAIL-2]"
    row = db.scalars(select(PseudonymMap).where(PseudonymMap.code == "[EMAIL-1]")).first()
    privacy.retire_pseudonym(db, row)
    assert privacy.code_for(db, course.id, "c@uni.edu", privacy.KIND_EMAIL) == "[EMAIL-3]"


def test_code_index_inverts_every_code_shape():
    for index in (0, 1, 25, 26, 27, 701, 702):
        assert privacy.code_index(privacy.person_code(index)) == index
    for index in (0, 5, 41):
        assert privacy.code_index(privacy.typed_code(privacy.KIND_EMAIL, index)) == index
    assert privacy.code_index("Student-07") is None


# --------------------------------------------------------------------------
# the display layer — codes back to real names
# --------------------------------------------------------------------------


def test_display_layer_puts_the_real_names_back(db, course, fake_llm):
    privacy.pseudonymize_text(db, SUBMISSION_TEXT, course_id=course.id, provider=fake_llm)
    feedback = "Student-01 leans on Person-A's argument; see Student #2 for contrast."
    rendered = privacy.render_for_display(db, feedback, course.id)

    assert rendered["raw"] == feedback  # what the AI actually saw
    assert rendered["display"] == (
        "Amara Osei leans on Dave Kowalski's argument; see Ben Whitaker for contrast."
    )
    assert {r["code"] for r in rendered["replacements"]} == {
        "Student-01",
        "Person-A",
        "Student #2",
    }
    assert privacy.resubstitute(db, feedback, course.id) == rendered["display"]


def test_display_layer_is_a_no_op_without_a_course(db):
    rendered = privacy.render_for_display(db, "Person-A did well", None)
    assert rendered["display"] == rendered["raw"] == "Person-A did well"


def test_display_layer_prefers_the_longer_code(db, course):
    for index in range(27):
        privacy.code_for(db, course.id, f"Person Number {index}", privacy.KIND_PERSON)
    mapping = privacy.display_map(db, course.id)
    assert mapping["Person-AA"] == "Person Number 26"
    rendered = privacy.render_for_display(db, "Person-AA wrote it", course.id)
    assert rendered["display"] == "Person Number 26 wrote it"


def test_display_result_renders_summary_and_criterion_comments(db, course, fake_llm):
    privacy.pseudonymize_text(db, SUBMISSION_TEXT, course_id=course.id, provider=fake_llm)

    class FakeResult:
        summary_feedback = "Person-A pushed Student-01 to sharpen the thesis."
        criteria = [{"key": "thesis", "comment": "Student-01 answers Person-A directly."}]

    rendered = privacy.display_result(db, FakeResult(), course.id)
    assert "Dave Kowalski" in rendered["summary_feedback"]["display"]
    assert "Amara Osei" in rendered["criteria"][0]["display"]
    assert rendered["criteria"][0]["raw"].startswith("Student-01")
    assert {r["code"] for r in rendered["replacements"]} == {"Student-01", "Person-A"}


# --------------------------------------------------------------------------
# the grading path
# --------------------------------------------------------------------------


def _submission_with(db, course, *, text=SUBMISSION_TEXT, provider="openai", mime=None,
                     tmp_path=None, raw=None):
    """A stored submission for the first roster student, on a cloud skill."""
    from app.models import Assignment, Rubric

    skill = Skill(
        name="Cloud Grader",
        system_prompt="Grade PHIL 210 essays.",
        provider=provider,
        model=config.default_model_for(provider) if provider != "mock" else config.MOCK_MODEL,
        max_tokens=4000,
    )
    rubric = Rubric(name="Essay Rubric", criteria=RUBRIC_CRITERIA)
    db.add_all([skill, rubric])
    db.commit()
    assignment = Assignment(
        course_id=course.id, name="Essay 1", skill_id=skill.id, rubric_id=rubric.id
    )
    db.add(assignment)
    db.commit()

    payload = raw if raw is not None else _text_pdf(text)
    path = tmp_path / f"submission-{len(payload)}-{provider}.bin"
    path.write_bytes(payload)
    student = db.scalars(
        select(Student).where(Student.course_id == course.id).order_by(Student.student_number)
    ).first()
    submission = Submission(
        assignment_id=assignment.id,
        student_id=student.id,
        file_path=str(path),
        original_filename="essay.pdf",
        mime_type=mime or "application/pdf",
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)
    return submission


def test_swap_mode_sends_zero_original_strings_to_a_cloud_provider(
    db, course, tmp_path, monkeypatch, fake_llm
):
    """The definition-of-done check: nothing identifying reaches the wire."""
    config.save_privacy_settings({"mode": "swap"})
    monkeypatch.setattr(privacy, "get_sweep_provider", lambda *a, **k: fake_llm)
    submission = _submission_with(db, course, tmp_path=tmp_path)

    request = engine.build_grade_request(db, submission)
    payload = json.dumps(providers_mod.public_blocks(request.content_blocks))

    for secret in (
        "Amara",
        "Osei",
        "Kowalski",
        "Whitaker",
        "amara.osei@uni.edu",
        "555-214-8890",
        "99241033",
    ):
        assert secret not in payload, f"{secret!r} leaked into the provider payload"

    for code in ("Student-01", "Person-A", "[EMAIL-1]", "[PHONE-1]", "[ID-1]"):
        assert code in payload

    # Swap mode sends extracted text, not the native PDF.
    assert request.content_blocks[0]["type"] == "text"
    assert request.privacy["swapped"] is True
    assert any("Privacy Guard" in note for note in request.notes)

    # ...and the report is persisted for the grading UI.
    scan = privacy.latest_scan(db, submission.id)
    assert scan is not None and scan.mode == "swap"
    assert privacy.scan_dict(scan)["headline"].endswith("identifiers swapped")


def test_the_same_person_keeps_one_code_across_two_submissions_of_a_run(
    db, course, tmp_path, monkeypatch, fake_llm
):
    config.save_privacy_settings({"mode": "swap"})
    monkeypatch.setattr(privacy, "get_sweep_provider", lambda *a, **k: fake_llm)

    first = _submission_with(db, course, tmp_path=tmp_path)
    second = _submission_with(
        db,
        course,
        text="Dave Kowalski and I met again. Reach me on 555-214-8890.",
        tmp_path=tmp_path,
    )
    first_payload = json.dumps(
        providers_mod.public_blocks(engine.build_grade_request(db, first).content_blocks)
    )
    second_payload = json.dumps(
        providers_mod.public_blocks(engine.build_grade_request(db, second).content_blocks)
    )
    assert "Person-A" in first_payload and "Person-A" in second_payload
    assert "[PHONE-1]" in first_payload and "[PHONE-1]" in second_payload
    assert "Person-B" not in second_payload
    assert len(db.scalars(select(PseudonymMap).where(PseudonymMap.kind == "person")).all()) == 1


def test_local_and_mock_providers_are_exempt(db, course, tmp_path, monkeypatch):
    config.save_privacy_settings({"mode": "swap"})
    monkeypatch.setattr(
        privacy,
        "get_sweep_provider",
        lambda *a, **k: pytest.fail("an on-device provider must not be scanned"),
    )
    for provider in (config.MOCK_PROVIDER, config.LOCAL_PROVIDER):
        submission = _submission_with(db, course, provider=provider, tmp_path=tmp_path)
        request = engine.build_grade_request(db, submission)
        assert request.privacy is None
        assert request.content_blocks[0]["type"] == "document"
        assert privacy.latest_scan(db, submission.id) is None


def test_off_mode_keeps_the_native_pdf(db, course, tmp_path):
    config.save_privacy_settings({"mode": "off"})
    submission = _submission_with(db, course, tmp_path=tmp_path)
    request = engine.build_grade_request(db, submission)
    assert request.content_blocks[0]["type"] == "document"
    assert request.privacy is None
    assert any("Privacy Guard is off" in note for note in request.notes)


def test_warn_mode_reports_but_still_sends_the_original(db, course, tmp_path, fake_llm, monkeypatch):
    config.save_privacy_settings({"mode": "warn"})
    monkeypatch.setattr(privacy, "get_sweep_provider", lambda *a, **k: fake_llm)
    submission = _submission_with(db, course, tmp_path=tmp_path)
    request = engine.build_grade_request(db, submission)

    assert request.content_blocks[0]["type"] == "document"
    assert request.privacy["counts"]["total"] >= 4
    assert request.privacy["swapped"] is False
    assert any("warn" in note for note in request.notes)
    assert privacy.latest_scan(db, submission.id).mode == "warn"


def test_swap_mode_refuses_native_image_for_cloud_provider(db, course, tmp_path):
    config.save_privacy_settings({"mode": "swap"})
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    submission = _submission_with(
        db, course, raw=png, mime="image/png", tmp_path=tmp_path
    )
    with pytest.raises(providers_mod.ProviderError, match="not sent"):
        engine.build_grade_request(db, submission)
    scan = privacy.latest_scan(db, submission.id)
    assert scan is not None and scan.mode == "swap"
    assert any("not sent" in warning for warning in scan.findings["warnings"])


def test_privacy_status_copy_describes_native_file_hard_stop(client, db, course, tmp_path):
    config.save_privacy_settings({"mode": "swap"})
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    submission = _submission_with(
        db, course, raw=png, mime="image/png", tmp_path=tmp_path
    )

    scanned = client.post(f"/api/submissions/{submission.id}/privacy/scan")
    assert scanned.status_code == 200
    status = client.get(f"/api/submissions/{submission.id}/privacy")
    assert status.status_code == 200
    copy = json.dumps(status.json()).lower()
    assert "not sent" in copy
    assert "fall back" not in copy


def test_swap_mode_refuses_textless_pdf_for_cloud_provider(db, course, tmp_path):
    config.save_privacy_settings({"mode": "swap"})
    submission = _submission_with(
        db, course, raw=b"%PDF-1.4 not really a pdf", tmp_path=tmp_path
    )
    with pytest.raises(providers_mod.ProviderError, match="not sent"):
        engine.build_grade_request(db, submission)
    scan = privacy.latest_scan(db, submission.id)
    assert scan is not None and scan.mode == "swap"
    assert any("not sent" in warning for warning in scan.findings["warnings"])


def test_swap_mode_scan_crash_fails_closed(db, course, tmp_path, monkeypatch):
    config.save_privacy_settings({"mode": "swap"})
    submission = _submission_with(db, course, tmp_path=tmp_path)

    def crash_scan(*args, **kwargs):
        raise RuntimeError("scan crashed")

    monkeypatch.setattr(privacy, "pseudonymize_text", crash_scan)
    with pytest.raises(providers_mod.ProviderError, match="not sent"):
        engine.build_grade_request(db, submission)


def _attach_knowledge_doc(db, submission, path, *, title, mime_type):
    doc = KnowledgeDoc(
        skill_id=submission.assignment.skill_id,
        title=title,
        filename=path.name,
        file_path=str(path),
        mime_type=mime_type,
        size_bytes=path.stat().st_size,
    )
    db.add(doc)
    db.commit()
    db.refresh(submission.assignment.skill)
    return doc


def test_swap_mode_pseudonymizes_cloud_knowledge_documents(db, course, tmp_path):
    config.save_privacy_settings({"mode": "swap"})
    submission = _submission_with(db, course, tmp_path=tmp_path)
    text_path = tmp_path / "course-notes.txt"
    text_path.write_text("Amara Osei uses amara.osei@uni.edu.", encoding="utf-8")
    pdf_path = tmp_path / "course-reader.pdf"
    pdf_path.write_bytes(_text_pdf("Amara Osei uses amara.osei@uni.edu."))
    _attach_knowledge_doc(
        db, submission, text_path, title="Course notes", mime_type="text/plain"
    )
    _attach_knowledge_doc(
        db, submission, pdf_path, title="Course reader", mime_type="application/pdf"
    )

    request = engine.build_grade_request(db, submission)
    payload = request.system_prompt + json.dumps(
        providers_mod.public_blocks(request.content_blocks)
    )

    assert "Amara Osei" not in payload
    assert "amara.osei@uni.edu" not in payload
    assert "Student-01" in payload
    assert "[EMAIL-1]" in payload
    assert not {"document", "image"} & {
        block["type"] for block in providers_mod.public_blocks(request.content_blocks)
    }


def test_swap_mode_refuses_textless_cloud_knowledge_pdf(db, course, tmp_path):
    config.save_privacy_settings({"mode": "swap"})
    submission = _submission_with(db, course, tmp_path=tmp_path)
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 not really a pdf")
    _attach_knowledge_doc(
        db, submission, pdf_path, title="Scanned reader", mime_type="application/pdf"
    )

    with pytest.raises(providers_mod.ProviderError, match="not sent"):
        engine.build_grade_request(db, submission)


def test_swap_mode_refuses_missing_cloud_knowledge_document(db, course, tmp_path):
    config.save_privacy_settings({"mode": "swap"})
    submission = _submission_with(db, course, tmp_path=tmp_path)
    title = "Missing reader"
    doc = KnowledgeDoc(
        skill_id=submission.assignment.skill_id,
        title=title,
        filename="missing.txt",
        file_path=str(tmp_path / "missing.txt"),
        mime_type="text/plain",
        size_bytes=0,
    )
    db.add(doc)
    db.commit()
    db.refresh(submission.assignment.skill)

    with pytest.raises(providers_mod.ProviderError) as caught:
        engine.build_grade_request(db, submission)

    assert title in str(caught.value)
    assert "not sent" in str(caught.value)


def test_swap_mode_knowledge_scan_crash_fails_closed(
    db, course, tmp_path, monkeypatch
):
    config.save_privacy_settings({"mode": "swap"})
    submission = _submission_with(db, course, tmp_path=tmp_path)
    text_path = tmp_path / "course-notes.txt"
    text_path.write_text("Amara Osei", encoding="utf-8")
    _attach_knowledge_doc(
        db, submission, text_path, title="Course notes", mime_type="text/plain"
    )

    def crash_scan(*args, **kwargs):
        raise RuntimeError("knowledge scan crashed")

    monkeypatch.setattr(privacy, "pseudonymize_text", crash_scan)
    with pytest.raises(providers_mod.ProviderError, match="not sent"):
        engine.build_grade_request(db, submission)


def test_a_regrade_replaces_the_previous_scan(db, course, tmp_path, monkeypatch, fake_llm):
    config.save_privacy_settings({"mode": "swap"})
    monkeypatch.setattr(privacy, "get_sweep_provider", lambda *a, **k: fake_llm)
    submission = _submission_with(db, course, tmp_path=tmp_path)
    engine.build_grade_request(db, submission)
    engine.build_grade_request(db, submission)
    scans = db.scalars(
        select(PrivacyScan).where(PrivacyScan.submission_id == submission.id)
    ).all()
    assert len(scans) == 1


def test_the_sweep_is_off_under_pytest_unless_asked_for(monkeypatch):
    monkeypatch.delenv(privacy.SWEEP_ENV_VAR, raising=False)
    assert privacy.sweep_allowed() is False
    assert privacy.get_sweep_provider() is None
    monkeypatch.setenv(privacy.SWEEP_ENV_VAR, "1")
    assert privacy.sweep_allowed() is True
    assert isinstance(privacy.get_sweep_provider(), LocalProvider)


# --------------------------------------------------------------------------
# review fixes — the ways the guard used to leak
# --------------------------------------------------------------------------


def test_swap_mode_never_sends_the_native_file_even_with_nothing_to_swap(
    db, course, tmp_path, monkeypatch, fake_llm
):
    """A clean text layer does not clear the FILE.

    PDF /Info metadata, a scanned letterhead, a signature image and form-field
    values are all invisible to the text pass and perfectly legible to the
    cloud model, so swap mode always sends the extracted text.
    """
    config.save_privacy_settings({"mode": "swap"})
    monkeypatch.setattr(privacy, "get_sweep_provider", lambda *a, **k: fake_llm)
    submission = _submission_with(
        db, course, text="Kant is the subject of section two. Nobody else is named.",
        tmp_path=tmp_path,
    )
    request = engine.build_grade_request(db, submission)

    assert request.privacy["counts"]["total"] == 0
    assert request.content_blocks[0]["type"] == "text"
    assert request.content_blocks[0]["text"].startswith(privacy.PSEUDONYMIZED_HEADER)
    assert "Kant is the subject" in request.content_blocks[0]["text"]
    assert not any("original form" in note for note in request.notes)


def test_swap_mode_refuses_to_send_a_submission_it_cannot_pseudonymize(
    db, course, tmp_path, monkeypatch
):
    """No course → no stable codes → nothing goes out. Fail closed."""
    from app.models import Assignment

    config.save_privacy_settings({"mode": "swap"})
    monkeypatch.setattr(privacy, "get_sweep_provider", lambda *a, **k: None)
    submission = _submission_with(db, course, tmp_path=tmp_path)
    assignment = db.get(Assignment, submission.assignment_id)

    block = {"type": "text", "text": SUBMISSION_TEXT}
    notes: list[str] = []
    with pytest.raises(providers_mod.ProviderError):
        privacy.protect_submission_block(
            db, submission, block, provider="openai", course_id=None, notes=notes
        )
    assert assignment is not None  # the submission is otherwise perfectly gradeable


def test_a_recased_span_from_the_sweep_is_still_swapped(db, course):
    """Small models re-case constantly; that is not a hallucination."""
    sloppy = FakeLocalProvider()
    sloppy.extra_spans = [{"text": "dave  kowalski", "kind": "person"}]
    text, report = privacy.pseudonymize_text(
        db,
        "I worked with Dave Kowalski on the essay.",
        course_id=course.id,
        provider=sloppy,
    )
    assert "Dave Kowalski" not in text and "Person-A" in text
    assert report["counts"]["pass2"] == 1
    assert not any("does not occur" in w for w in report["warnings"])
    # The map records the name as it really appears in the text.
    assert report["codes"]["Person-A"] == "Dave Kowalski"


def test_two_students_sharing_a_surname_do_not_collide(db, course):
    """Ben Osei's surname must never come back out as Amara Osei."""
    db.add(
        Student(course_id=course.id, name="Ben Osei", student_number=4, email="b.osei@uni.edu")
    )
    db.commit()
    text, report = privacy.pseudonymize_text(
        db,
        "Osei wrote the essay. Amara Osei presented it and Ben Osei filmed it.",
        course_id=course.id,
        use_llm=False,
    )
    # The full names still resolve to the right students...
    assert "Student-01 presented it and Student-04 filmed it." in text
    # ...and the shared bare token is coded without being attributed to either.
    assert text.startswith("Person-A wrote the essay.")
    assert any("more than one student" in w for w in report["warnings"])
    assert privacy.display_map(db, course.id)["Person-A"] == "Osei"


def test_an_unshared_bare_name_still_maps_to_its_student(db, course):
    text, report = privacy.pseudonymize_text(
        db, "Moreno handed hers in early.", course_id=course.id, use_llm=False
    )
    assert text == "Student-03 handed hers in early."
    assert report["warnings"] == []


def test_a_sweep_that_never_parsed_anything_is_not_reported_as_a_sweep(db, course):
    broken = FakeLocalProvider(fail_times=99)
    _, report = privacy.pseudonymize_text(
        db, SUBMISSION_TEXT, course_id=course.id, provider=broken
    )
    assert report["llm_sweep"]["ran"] is False
    assert any("sweep failed" in w for w in report["warnings"])


def test_a_text_too_long_for_the_sweep_says_so(db, course):
    long_text = "\n\n".join(f"Paragraph {i} " + "word " * 120 for i in range(60))
    quiet = FakeLocalProvider()
    _, report = privacy.pseudonymize_text(
        db, long_text, course_id=course.id, provider=quiet
    )
    assert len(quiet.calls) == privacy.SWEEP_MAX_CHUNKS
    assert any("too long for the local sweep" in w for w in report["warnings"])


def test_local_provider_is_strictly_loopback_only():
    for url in (
        "http://127.0.0.1:3782/v1",
        "http://localhost:3782/v1",
        "http://[::1]:3782/v1",
    ):
        assert config.check_local_base_url(url)[0] is True, url
    for url in (
        "http://192.168.1.40:3782/v1",
        "http://10.0.0.5:8080/v1",
        "http://169.254.1.1:3782/v1",
        "http://8.8.8.8/v1",
        "http://gpu.example.com/v1",
    ):
        assert config.check_local_base_url(url)[0] is False, url
    with pytest.raises(providers_mod.ProviderConfigError):
        LocalProvider(base_url="http://192.168.1.40:3782/v1")


def test_persisted_remote_local_url_falls_back_without_a_request():
    config.privacy_settings_path().write_text(
        json.dumps(
            {
                "mode": "swap",
                "local_model": {
                    ("allow_" + "remote"): True,
                    "base_url": "https://api.openai.com/v1",
                },
            }
        ),
        encoding="utf-8",
    )
    assert config.local_model_settings()["base_url"] == config.LOCAL_DEFAULT_BASE_URL
    http = FakeHttpClient()
    provider = LocalProvider(client=http)
    assert provider.base_url == config.LOCAL_DEFAULT_BASE_URL
    assert http.requests == []


def test_saving_an_off_box_local_address_is_refused():
    with pytest.raises(ValueError):
        config.save_privacy_settings({"local_model": {"base_url": "https://api.openai.com/v1"}})
    assert config.local_model_settings()["base_url"] == config.LOCAL_DEFAULT_BASE_URL


def test_a_code_that_only_prefixes_another_token_is_left_alone(db, course):
    privacy.code_for(db, course.id, "Dave Kowalski", privacy.KIND_PERSON)  # Person-A
    rendered = privacy.render_for_display(
        db, "Person-A helped, Person-AB did not, and Person-Alpha is a typo.", course.id
    )
    assert rendered["display"] == (
        "Dave Kowalski helped, Person-AB did not, and Person-Alpha is a typo."
    )
    assert privacy.render_for_display(db, "Student-011 wrote it", course.id)["display"] == (
        "Student-011 wrote it"
    )


# --------------------------------------------------------------------------
# LocalProvider (fake transport)
# --------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if isinstance(payload, dict) else str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeHttpClient:
    """Records requests and replays scripted responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def _next(self):
        return self.responses.pop(0) if self.responses else FakeResponse({}, 500)

    def post(self, url, json=None, headers=None):
        self.requests.append({"url": url, "json": json, "headers": headers})
        return self._next()

    def get(self, url, headers=None):
        self.requests.append({"url": url, "json": None, "headers": headers})
        return self._next()


def _completion(content):
    return {
        "model": "gemma-3-4b-it",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    }


def test_local_provider_needs_no_key_and_targets_the_configured_server():
    provider = LocalProvider(client=FakeHttpClient())
    assert provider.name == "local"
    assert provider.base_url == config.LOCAL_DEFAULT_BASE_URL
    assert provider.api_key == config.LOCAL_DUMMY_API_KEY
    assert provider.supports_pdf() is False
    # It is a real Provider, not a look-alike.
    assert isinstance(provider, providers_mod.Provider)
    assert providers_mod.get_provider("local").name == "local"


def test_local_provider_grade_uses_json_schema_and_parses_the_object():
    http = FakeHttpClient(FakeResponse(_completion('{"spans": [{"text": "Dave", "kind": "person"}]}')))
    provider = LocalProvider(client=http)
    out = provider.grade("system", [{"type": "text", "text": "body"}], privacy.SPAN_SCHEMA)
    assert out["spans"][0]["text"] == "Dave"

    sent = http.requests[0]["json"]
    assert sent["response_format"]["json_schema"]["schema"] == privacy.SPAN_SCHEMA
    assert sent["messages"][0]["role"] == "system"
    assert sent["stream"] is False
    assert http.requests[0]["url"].endswith("/chat/completions")


def test_local_provider_retries_once_without_the_schema():
    http = FakeHttpClient(
        FakeResponse({"error": "grammar unsupported"}, status_code=400),
        FakeResponse(_completion("```json\n{\"spans\": []}\n```")),
    )
    provider = LocalProvider(client=http)
    assert provider.grade("system", [{"type": "text", "text": "b"}], privacy.SPAN_SCHEMA) == {
        "spans": []
    }
    assert len(http.requests) == 2
    assert "response_format" in http.requests[0]["json"]
    assert "response_format" not in http.requests[1]["json"]


def test_local_provider_gives_up_after_the_retry():
    http = FakeHttpClient(
        FakeResponse(_completion("sorry, no")), FakeResponse(_completion("still no"))
    )
    with pytest.raises(providers_mod.ProviderResponseError):
        LocalProvider(client=http).grade("s", [{"type": "text", "text": "b"}], {})


def test_local_provider_refuses_images_and_needs_extracted_text_for_pdfs():
    provider = LocalProvider(client=FakeHttpClient())
    with pytest.raises(providers_mod.ProviderUnsupportedError):
        provider.grade("s", [{"type": "image", "source": {"data": "x"}}], {})
    with pytest.raises(providers_mod.ProviderUnsupportedError):
        provider.grade("s", [{"type": "document", "source": {"data": "x"}}], {})


def test_local_provider_chat_parses_tool_calls_as_dicts():
    body = {
        "model": "gemma-3-4b-it",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "looking that up",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "get_course_summary",
                                "arguments": '{"course_id": 3}',
                            },
                        }
                    ],
                },
            }
        ],
    }
    turn = LocalProvider(client=FakeHttpClient(FakeResponse(body))).chat(
        "sys", [{"role": "user", "content": "how is PHIL 210?"}], tools=[
            {"name": "get_course_summary", "description": "d", "input_schema": {}}
        ]
    )
    assert turn.stop_reason == "tool_use"
    assert turn.tool_calls[0].arguments == {"course_id": 3}
    assert turn.text == "looking that up"


def test_local_provider_reports_an_unreachable_server_without_raising():
    class Dead:
        def get(self, *_a, **_k):
            raise OSError("connection refused")

        def post(self, *_a, **_k):
            raise OSError("connection refused")

    info = LocalProvider(client=Dead()).health()
    assert info["ok"] is False
    assert "Settings" in info["message"]
    with pytest.raises(providers_mod.ProviderConnectionError):
        LocalProvider(client=Dead()).grade("s", [{"type": "text", "text": "b"}], {})


def test_local_provider_health_names_the_served_model():
    http = FakeHttpClient(
        FakeResponse({"data": [{"id": "/home/me/models/gemma-3-4b-it-Q4_K_M.gguf"}]})
    )
    info = LocalProvider(client=http).health()
    assert info["ok"] is True
    assert info["message"] == "connected, gemma-3-4b-it-Q4_K_M.gguf"


def test_parse_json_object_survives_small_model_habits():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure!\n{"a": {"b": "}"}}\nHope that helps') == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        parse_json_object("no json at all")


def test_the_local_provider_is_registered_in_the_model_registry():
    registry = config.load_model_registry()
    assert registry["local"]["default"] == "gemma-3-4b-it"
    assert config.LOCAL_PROVIDER in config.PROVIDERS
    assert config.is_cloud_provider("local") is False
    assert config.is_cloud_provider("mock") is False
    assert config.is_cloud_provider("anthropic") is True


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------



def test_swap_mode_cloud_chat_pseudonymizes_all_provider_text(
    client, db, course, monkeypatch
):
    config.save_privacy_settings({"mode": "swap", "llm_sweep": False})
    raw_identity = "Amara Osei <amara.osei@uni.edu>"
    session = ChatSession(title="privacy", course_id=course.id)
    db.add(session)
    db.commit()
    db.add_all(
        [
            ChatMessage(
                session_id=session.id,
                course_id=course.id,
                role="user",
                content=f"Earlier question from {raw_identity}",
            ),
            ChatMessage(
                session_id=session.id,
                course_id=course.id,
                role="assistant",
                content=f"Earlier answer for {raw_identity}",
            ),
        ]
    )
    db.commit()

    provider = CapturingCloudChatProvider(
        [
            {
                "tool_calls": [
                    {"name": "get_course_summary", "input": {"course_id": course.id}}
                ]
            },
            {"text": "Amara Osei remains visible to the professor."},
        ]
    )
    monkeypatch.setattr(
        assistant_mod,
        "build_provider",
        lambda db, **kwargs: (provider, "anthropic", provider.model),
    )
    monkeypatch.setattr(
        assistant_mod,
        "dispatch_tool",
        lambda db, call: (
            {"student": "Amara Osei", "email": "amara.osei@uni.edu"},
            False,
            None,
        ),
    )

    response = client.post(
        "/api/chat",
        json={
            "message": f"Current question from {raw_identity}",
            "session_id": session.id,
            "context": {"page": "course_detail", "course_id": course.id},
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["reply"] == "Amara Osei remains visible to the professor."
    assert len(provider.calls) == 2
    captured_wire = json.dumps(provider.calls)
    assert "Amara Osei" not in captured_wire
    assert "amara.osei@uni.edu" not in captured_wire
    assert "Student-01" in captured_wire
    assert "[EMAIL-1]" in captured_wire

    db.expire_all()
    stored = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.id)
    ).all()
    assert [message.content for message in stored[:3]] == [
        f"Earlier question from {raw_identity}",
        f"Earlier answer for {raw_identity}",
        f"Current question from {raw_identity}",
    ]
    assert [message.course_id for message in stored] == [course.id] * 4


def test_swap_mode_cloud_chat_pseudonymizes_cross_course_history(
    client, db, course, monkeypatch
):
    config.save_privacy_settings({"mode": "swap", "llm_sweep": False})
    course_b = Course(name="PHIL 310", term="Fall 2026")
    db.add(course_b)
    db.commit()
    session = ChatSession(title="cross-course", course_id=course.id)
    db.add(session)
    db.commit()
    db.add_all(
        [
            ChatMessage(
                session_id=session.id,
                course_id=course.id,
                role="user",
                content="How is Amara Osei doing?",
            ),
            ChatMessage(
                session_id=session.id,
                course_id=course.id,
                role="assistant",
                content="Amara Osei needs more support.",
            ),
        ]
    )
    db.commit()

    provider = CapturingCloudChatProvider([{"text": "The current course is on track."}])
    monkeypatch.setattr(
        assistant_mod,
        "build_provider",
        lambda db, **kwargs: (provider, "anthropic", provider.model),
    )

    response = client.post(
        "/api/chat",
        json={
            "message": "How is this course doing?",
            "session_id": session.id,
            "context": {"page": "course_detail", "course_id": course_b.id},
        },
    )

    assert response.status_code == 200, response.text
    captured_wire = json.dumps(provider.calls)
    assert "Amara Osei" not in captured_wire
    assert "Student-01" in captured_wire


def test_swap_mode_cloud_chat_pseudonymizes_other_course_text_in_current_turn(
    client, db, course, monkeypatch
):
    config.save_privacy_settings({"mode": "swap", "llm_sweep": False})
    course_b = Course(name="PHIL 310", term="Fall 2026")
    db.add(course_b)
    db.commit()
    session = ChatSession(title="cross-course current turn", course_id=course.id)
    db.add(session)
    db.commit()

    provider = CapturingCloudChatProvider(
        [
            {
                "tool_calls": [
                    {"name": "get_course_summary", "input": {"course_id": course_b.id}}
                ]
            },
            {"text": "The current course is on track."},
        ]
    )
    monkeypatch.setattr(
        assistant_mod,
        "build_provider",
        lambda db, **kwargs: (provider, "anthropic", provider.model),
    )
    monkeypatch.setattr(
        assistant_mod,
        "dispatch_tool",
        lambda db, call: ({"student": "Amara Osei"}, False, None),
    )

    response = client.post(
        "/api/chat",
        json={
            "message": "What support does Amara Osei need?",
            "session_id": session.id,
            "context": {"page": "course_detail", "course_id": course_b.id},
        },
    )

    assert response.status_code == 200, response.text
    captured_wire = json.dumps(provider.calls)
    assert "Amara Osei" not in captured_wire
    assert "Student-01" in captured_wire
    db.expire_all()
    stored_user = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id, ChatMessage.role == "user")
        .order_by(ChatMessage.id.desc())
    ).first()
    assert stored_user is not None
    assert stored_user.content == "What support does Amara Osei need?"



def test_swap_mode_cloud_chat_legacy_unscoped_history_fails_closed(
    client, db, course, monkeypatch
):
    config.save_privacy_settings({"mode": "swap", "llm_sweep": False})
    course_b = Course(name="PHIL 310", term="Fall 2026")
    db.add(course_b)
    db.commit()
    session = ChatSession(title="legacy", course_id=course.id)
    db.add(session)
    db.commit()
    db.add_all(
        [
            ChatMessage(
                session_id=session.id,
                course_id=None,
                role="user",
                content="How is Amara Osei doing?",
            ),
            ChatMessage(
                session_id=session.id,
                course_id=None,
                role="assistant",
                content="Amara Osei needs more support.",
            ),
        ]
    )
    db.commit()

    provider = CapturingCloudChatProvider([{"text": "must not run"}])
    monkeypatch.setattr(
        assistant_mod,
        "build_provider",
        lambda db, **kwargs: (provider, "anthropic", provider.model),
    )

    response = client.post(
        "/api/chat",
        json={
            "message": "How is this course doing?",
            "session_id": session.id,
            "context": {"page": "course_detail", "course_id": course_b.id},
        },
    )

    assert response.status_code == 409
    assert "not sent" in response.json()["detail"]
    assert provider.calls == []


def test_swap_mode_cloud_chat_without_course_fails_before_provider_call(
    client, db, monkeypatch
):
    config.save_privacy_settings({"mode": "swap", "llm_sweep": False})
    provider = CapturingCloudChatProvider([{"text": "must not run"}])
    monkeypatch.setattr(
        assistant_mod,
        "build_provider",
        lambda db, **kwargs: (provider, "anthropic", provider.model),
    )

    before = (db.query(ChatSession).count(), db.query(ChatMessage).count())
    response = client.post(
        "/api/chat",
        json={"message": "Who needs help?", "context": {"page": "home"}},
    )

    assert response.status_code == 409
    assert "not sent" in response.json()["detail"]
    assert provider.calls == []
    assert (db.query(ChatSession).count(), db.query(ChatMessage).count()) == before


def test_swap_mode_cloud_chat_scan_crash_fails_closed(
    client, db, course, monkeypatch
):
    config.save_privacy_settings({"mode": "swap", "llm_sweep": False})
    provider = CapturingCloudChatProvider([{"text": "must not run"}])
    monkeypatch.setattr(
        assistant_mod,
        "build_provider",
        lambda db, **kwargs: (provider, "anthropic", provider.model),
    )

    def crash_scan(*args, **kwargs):
        raise RuntimeError("privacy scan crashed")

    monkeypatch.setattr(privacy, "pseudonymize_text", crash_scan)
    before = (db.query(ChatSession).count(), db.query(ChatMessage).count())
    response = client.post(
        "/api/chat",
        json={
            "message": "How is Amara Osei doing?",
            "context": {"page": "course_detail", "course_id": course.id},
        },
    )

    assert response.status_code == 409
    assert "not sent" in response.json()["detail"]
    assert provider.calls == []
    assert (db.query(ChatSession).count(), db.query(ChatMessage).count()) == before


def test_swap_mode_cloud_chat_rejection_does_not_move_existing_session(
    client, db, course, monkeypatch
):
    config.save_privacy_settings({"mode": "swap", "llm_sweep": False})
    course_b = Course(name="PHIL 310", term="Fall 2026")
    session = ChatSession(title="stays put", course_id=course.id)
    db.add_all([course_b, session])
    db.commit()
    provider = CapturingCloudChatProvider([{"text": "must not run"}])
    monkeypatch.setattr(
        assistant_mod,
        "build_provider",
        lambda db, **kwargs: (provider, "anthropic", provider.model),
    )

    def crash_scan(*args, **kwargs):
        raise RuntimeError("privacy scan crashed")

    monkeypatch.setattr(privacy, "pseudonymize_text", crash_scan)
    response = client.post(
        "/api/chat",
        json={
            "message": "How is Amara Osei doing?",
            "session_id": session.id,
            "context": {"page": "course_detail", "course_id": course_b.id},
        },
    )

    assert response.status_code == 409
    assert "not sent" in response.json()["detail"]
    assert provider.calls == []
    db.expire_all()
    assert db.get(ChatSession, session.id).course_id == course.id
    assert db.query(ChatMessage).count() == 0


def test_swap_mode_cloud_chat_tool_result_scan_crash_is_atomic(
    client, db, course, monkeypatch
):
    config.save_privacy_settings({"mode": "swap", "llm_sweep": False})
    course_b = Course(name="PHIL 310", term="Fall 2026")
    existing = ChatSession(title="stays atomic", course_id=course.id)
    db.add_all([course_b, existing])
    db.commit()
    db.add_all(
        [
            ChatMessage(
                session_id=existing.id,
                course_id=course.id,
                role="user",
                content="Earlier question",
            ),
            ChatMessage(
                session_id=existing.id,
                course_id=course.id,
                role="assistant",
                content="Earlier answer",
            ),
        ]
    )
    db.commit()

    provider = CapturingCloudChatProvider(
        [
            {
                "tool_calls": [
                    {"name": "get_course_summary", "input": {"course_id": course_b.id}}
                ]
            },
            {
                "tool_calls": [
                    {"name": "get_course_summary", "input": {"course_id": course_b.id}}
                ]
            },
        ]
    )
    monkeypatch.setattr(
        assistant_mod,
        "build_provider",
        lambda db, **kwargs: (provider, "anthropic", provider.model),
    )

    real_build_system_prompt = assistant_mod.build_system_prompt
    prompt_calls = []

    def tracked_build_system_prompt(db_, context=None):
        prompt_calls.append(context)
        return real_build_system_prompt(db_, context)

    monkeypatch.setattr(assistant_mod, "build_system_prompt", tracked_build_system_prompt)

    scanned = []

    def crash_on_tool_result(db_, text, **kwargs):
        scanned.append(text)
        if text.startswith('{"course_id":'):
            raise RuntimeError("tool-result privacy scan crashed")
        return text, privacy.empty_report()

    monkeypatch.setattr(privacy, "pseudonymize_text", crash_on_tool_result)

    before = (
        db.query(ChatSession).count(),
        db.query(ChatMessage).count(),
        db.query(ChatMessage).filter(ChatMessage.role == "assistant").count(),
    )
    response = client.post(
        "/api/chat",
        json={
            "message": "How is PHIL 310 doing?",
            "context": {"page": "course_detail", "course_id": course_b.id},
        },
    )

    assert response.status_code == 409
    assert "not sent" in response.json()["detail"]
    assert len(provider.calls) == 1
    assert len(prompt_calls) == 1
    assert any(text.startswith("You are the Agora assistant") for text in scanned)
    assert "How is PHIL 310 doing?" in scanned
    assert any(text.startswith('{"course_id":') for text in scanned)
    assert (
        db.query(ChatSession).count(),
        db.query(ChatMessage).count(),
        db.query(ChatMessage).filter(ChatMessage.role == "assistant").count(),
    ) == before

    existing_message_count = db.query(ChatMessage).filter(
        ChatMessage.session_id == existing.id
    ).count()
    response = client.post(
        "/api/chat",
        json={
            "message": "Move this session to PHIL 310",
            "session_id": existing.id,
            "context": {"page": "course_detail", "course_id": course_b.id},
        },
    )

    assert response.status_code == 409
    assert "not sent" in response.json()["detail"]
    assert len(provider.calls) == 2
    assert len(prompt_calls) == 2
    db.expire_all()
    assert db.get(ChatSession, existing.id).course_id == course.id
    assert (
        db.query(ChatMessage).filter(ChatMessage.session_id == existing.id).count()
        == existing_message_count
    )
    assert db.query(ChatMessage).filter(
        ChatMessage.session_id == existing.id, ChatMessage.role == "assistant"
    ).count() == 1


def test_privacy_settings_endpoints_round_trip(client):
    body = client.get("/api/settings/privacy").json()
    assert body["mode"] == "swap"
    assert [m["id"] for m in body["modes"]] == ["swap", "warn", "off"]

    updated = client.post("/api/settings/privacy", json={"mode": "warn"}).json()
    assert updated["mode"] == "warn"
    assert client.get("/api/settings/privacy").json()["mode"] == "warn"

    saved = client.post(
        "/api/settings/privacy",
        json={"mode": "swap", "local_model": {"base_url": "http://127.0.0.1:9/v1"}},
    ).json()
    assert saved["local_model"]["base_url"] == "http://127.0.0.1:9/v1"

    assert client.post("/api/settings/privacy", json={"mode": "nonsense"}).status_code == 400


def test_local_model_health_endpoint_answers_even_when_the_server_is_down(client, monkeypatch):
    monkeypatch.setattr(
        providers_mod,
        "local_model_health",
        lambda **_k: {"ok": False, "message": "refused", "base_url": "http://127.0.0.1:9/v1"},
    )
    body = client.get("/api/settings/local-model/health").json()
    assert body["ok"] is False
    assert body["label"] == "local model: not reachable"


def test_pseudonym_map_endpoint_answers_who_is_person_a(client, db, course, fake_llm):
    privacy.pseudonymize_text(db, SUBMISSION_TEXT, course_id=course.id, provider=fake_llm)
    body = client.get(f"/api/settings/privacy/pseudonyms?course_id={course.id}").json()
    entries = body["courses"][0]["entries"]
    lookup = {e["code"]: e["original_text"] for e in entries}
    assert lookup["Person-A"] == "Dave Kowalski"
    assert lookup["Student-01"] == "Amara Osei"

    person = next(e for e in entries if e["code"] == "Person-A")
    assert client.delete(f"/api/settings/privacy/pseudonyms/{person['id']}").status_code == 200
    after = client.get(f"/api/settings/privacy/pseudonyms?course_id={course.id}").json()
    assert "Person-A" not in {e["code"] for e in after["courses"][0]["entries"]}
    assert client.get("/api/settings/privacy/pseudonyms?course_id=9999").status_code == 404


def test_scan_report_and_display_endpoints(client, db, course, tmp_path, monkeypatch, fake_llm):
    config.save_privacy_settings({"mode": "swap"})
    monkeypatch.setattr(privacy, "get_sweep_provider", lambda *a, **k: fake_llm)
    submission = _submission_with(db, course, tmp_path=tmp_path)
    engine.build_grade_request(db, submission)

    body = client.get(f"/api/submissions/{submission.id}/privacy").json()
    assert body["scan"]["mode"] == "swap"
    assert body["scan"]["total"] >= 5
    assert "swapped" in body["headline"]
    codes = {f["code"] for f in body["scan"]["report"]["findings"]}
    assert {"Student-01", "Person-A", "[EMAIL-1]", "[PHONE-1]", "[ID-1]"} <= codes

    rescan = client.post(f"/api/submissions/{submission.id}/privacy/scan").json()
    assert rescan["scan"]["mode"] == "warn"

    display = client.post(
        "/api/privacy/display",
        json={"text": "Person-A helped Student-01", "course_id": course.id},
    ).json()
    assert display["display"] == "Dave Kowalski helped Amara Osei"
    assert display["raw"] == "Person-A helped Student-01"


def test_result_display_endpoint_reverses_the_codes(client, db, course, tmp_path, fake_llm):
    from app.models import GradeResult

    submission = _submission_with(db, course, tmp_path=tmp_path)
    privacy.pseudonymize_text(db, SUBMISSION_TEXT, course_id=course.id, provider=fake_llm)
    db.add(
        GradeResult(
            submission_id=submission.id,
            overall_score=16,
            max_score=20,
            summary_feedback="Person-A sharpened Student-01's thesis.",
            criteria=[{"key": "thesis", "score": 8, "max_points": 10, "comment": "Student-01 is clear."}],
            misconceptions=[],
            strengths=[],
            model="mock",
        )
    )
    db.commit()

    body = client.get(f"/api/submissions/{submission.id}/result/display").json()
    assert body["summary_feedback"]["display"] == "Dave Kowalski sharpened Amara Osei's thesis."
    assert body["summary_feedback"]["raw"].startswith("Person-A")
    assert body["criteria"][0]["display"] == "Amara Osei is clear."


# --------------------------------------------------------------------------
# integration with the REAL local model (skipped when it is not running)
# --------------------------------------------------------------------------


def _local_server_up() -> bool:
    try:
        return bool(LocalProvider().health().get("ok"))
    except Exception:  # noqa: BLE001
        return False


local_llm = pytest.mark.local_llm
needs_local = pytest.mark.skipif(
    not _local_server_up(), reason="local llama.cpp server on :3782 is not running"
)


@local_llm
@needs_local
def test_real_local_model_reports_health():
    info = LocalProvider().health()
    assert info["ok"] is True
    assert info["message"].startswith("connected, ")


@local_llm
@needs_local
def test_real_local_model_finds_the_names_the_regexes_cannot(db, course, monkeypatch):
    """Gemma 3 4B must catch the third party and the nickname the roster misses."""
    monkeypatch.setenv(privacy.SWEEP_ENV_VAR, "1")
    text = (
        "Essay by Amara Osei.\n\n"
        "My roommate Dave Kowalski walked me through the second case, and my lab "
        "partner (everyone calls him Beto) drew the diagram."
    )
    cleaned, report = privacy.pseudonymize_text(db, text, course_id=course.id)

    assert "Amara Osei" not in cleaned and "Student-01" in cleaned  # pass 1
    assert "Dave Kowalski" not in cleaned  # pass 2
    assert report["llm_sweep"]["ran"] is True
    assert report["counts"]["pass2"] >= 1
    assert {f["code"] for f in report["findings"] if f["source"] == "llm"} <= {
        "Person-A",
        "Person-B",
        "Person-C",
    }


@local_llm
@needs_local
def test_real_local_model_returns_a_structured_grade():
    provider = LocalProvider()
    payload = provider.grade(
        "You grade one criterion. " + engine.DEFAULT_GRADING_INSTRUCTIONS,
        [
            {
                "type": "text",
                "text": (
                    "RUBRIC\n- [thesis] Thesis (max 10 points)\n\n"
                    "Student-01 wrote: 'Recommendation systems narrow what we can want.'"
                ),
            }
        ],
        engine.GRADE_SCHEMA,
    )
    assert isinstance(payload, dict)
    assert isinstance(payload.get("criteria"), list) and payload["criteria"]
    assert isinstance(payload.get("summary_feedback"), str)


@local_llm
@needs_local
def test_real_local_model_never_receives_the_raw_names(db, course, monkeypatch):
    """Even the PII sweep itself only ever sees pass-1 output."""
    monkeypatch.setenv(privacy.SWEEP_ENV_VAR, "1")
    seen: list[str] = []
    real = LocalProvider()
    original_grade = real.grade

    def spy(system_prompt, blocks, schema):
        seen.append("\n".join(b.get("text", "") for b in blocks))
        return original_grade(system_prompt, blocks, schema)

    monkeypatch.setattr(real, "grade", spy)
    monkeypatch.setattr(privacy, "get_sweep_provider", lambda *a, **k: real)

    privacy.pseudonymize_text(db, SUBMISSION_TEXT, course_id=course.id)
    swept = "\n".join(seen)
    assert swept
    for secret in ("Amara", "Osei", "amara.osei@uni.edu", "555-214-8890", "99241033"):
        assert secret not in swept


def test_privacy_settings_query_count_is_constant(client, db):
    courses = [
        Course(name="Zulu 300", term="Fall 2026"),
        Course(name="Alpha 100", term="Fall 2026"),
        Course(name="Middle 200", term="Fall 2026"),
    ]
    db.add_all(courses)
    db.flush()
    zulu, alpha, middle = courses
    students_by_course = {
        alpha.id: [
            Student(course_id=alpha.id, name="Alpha One", student_number=1),
        ],
        middle.id: [
            Student(course_id=middle.id, name="Middle One", student_number=1),
            Student(course_id=middle.id, name="Middle Two", student_number=2),
        ],
        zulu.id: [
            Student(course_id=zulu.id, name="Zulu One", student_number=1),
            Student(course_id=zulu.id, name="Zulu Two", student_number=2),
            Student(course_id=zulu.id, name="Zulu Three", student_number=3),
        ],
    }
    db.add_all(
        [student for students in students_by_course.values() for student in students]
    )
    db.commit()

    privacy.code_for(db, middle.id, "Middle Visitor", privacy.KIND_PERSON)
    privacy.code_for(db, zulu.id, "zulu@example.edu", privacy.KIND_EMAIL)
    privacy.code_for(db, zulu.id, "Zulu Visitor", privacy.KIND_PERSON)
    privacy.code_for(db, alpha.id, "Retired Visitor", privacy.KIND_PERSON)
    retired = db.scalars(
        select(PseudonymMap).where(
            PseudonymMap.course_id == alpha.id,
            PseudonymMap.original_text == "Retired Visitor",
        )
    ).one()
    retired_code = privacy.retire_pseudonym(db, retired)

    db.expire_all()
    live_maps = db.scalars(
        select(PseudonymMap)
        .where(PseudonymMap.retired_at.is_(None))
        .order_by(PseudonymMap.course_id, PseudonymMap.kind, PseudonymMap.id)
    ).all()
    maps_by_course = {course.id: [] for course in courses}
    for row in live_maps:
        maps_by_course[row.course_id].append(row)

    def roster_entry(student):
        return {
            "id": None,
            "code": privacy.student_code(student),
            "original_text": student.name,
            "kind": privacy.KIND_PERSON,
            "source": privacy.SOURCE_ROSTER,
            "student_id": student.id,
            "created_at": student.created_at.isoformat(),
        }

    def map_entry(row):
        return {
            "id": row.id,
            "code": row.code,
            "original_text": row.original_text,
            "kind": row.kind,
            "source": "map",
            "student_id": None,
            "created_at": row.created_at.isoformat(),
        }

    expected_entries = {
        course.id: [
            *[roster_entry(student) for student in students_by_course[course.id]],
            *[map_entry(row) for row in maps_by_course[course.id]],
        ]
        for course in courses
    }

    def get_with_selects(path):
        selects = []

        def count_selects(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                selects.append(statement)

        engine_ = db.get_bind()
        event.listen(engine_, "before_cursor_execute", count_selects)
        try:
            response = client.get(path)
        finally:
            event.remove(engine_, "before_cursor_execute", count_selects)
        return response, selects

    status_response, status_selects = get_with_selects("/api/settings/privacy")
    assert status_response.status_code == 200
    assert status_response.json()["courses"] == [
        {"id": alpha.id, "name": "Alpha 100", "pseudonyms": 1},
        {"id": middle.id, "name": "Middle 200", "pseudonyms": 3},
        {"id": zulu.id, "name": "Zulu 300", "pseudonyms": 5},
    ]
    assert len(status_selects) == 3

    map_response, map_selects = get_with_selects(
        "/api/settings/privacy/pseudonyms"
    )
    assert map_response.status_code == 200
    map_courses = map_response.json()["courses"]
    assert [entry["course_name"] for entry in map_courses] == [
        "Alpha 100",
        "Middle 200",
        "Zulu 300",
    ]
    for entry in map_courses:
        assert entry["count"] == len(expected_entries[entry["course_id"]])
        assert entry["entries"] == expected_entries[entry["course_id"]]
    alpha_entries = next(
        entry["entries"] for entry in map_courses if entry["course_id"] == alpha.id
    )
    assert retired_code not in {item["code"] for item in alpha_entries}
    assert len(map_selects) == 3
