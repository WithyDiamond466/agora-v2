"""Student Insight tests — deriver, decay, nudges, cards, API.

Everything here is offline and deterministic (SPEC: tests use MockProvider and
never hit a network). The load-bearing assertions:

  * observations are derived, not authored: re-grading a submission REPLACES
    its observations instead of doubling them;
  * a misconception decays active → resolving → resolved as newer assignments
    stop mentioning it;
  * one nudge per course+assignment+tag, no matter how often detection runs;
  * a card can be written by the model (MockProvider, deterministic) or by the
    templates when the model is unavailable — both obey the voice rules, and
    neither can cite evidence that does not exist.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import config, insight
from app.ai import grading as engine
from app.ai import assistant as assistant_mod
from app.ai.providers import MockProvider, ProviderError
from app.db import get_db
from app.main import app
from app.models import (
    Assignment,
    Base,
    Course,
    CourseNudge,
    GradeResult,
    Observation,
    Rubric,
    Skill,
    Student,
    StudentCard,
    Submission,
)
from app.routers import grading as grading_router

RUBRIC_CRITERIA = [
    {"key": "thesis", "title": "Thesis", "description": "States a thesis.", "max_points": 10},
    {"key": "evidence", "title": "Evidence", "description": "Uses sources.", "max_points": 10},
    {"key": "mechanics", "title": "Mechanics", "description": "Prose quality.", "max_points": 5},
]
MAX_POINTS = {c["key"]: float(c["max_points"]) for c in RUBRIC_CRITERIA}


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def Session(tmp_path):
    engine_ = create_engine(
        f"sqlite:///{tmp_path / 'insight.db'}", connect_args={"check_same_thread": False}
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
    """Insight settings live in DATA_DIR — never touch the developer's real one."""
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
    """A course with three students, a rubric, a mock skill, three assignments."""
    course = Course(name="PHIL 210", term="Fall 2026")
    db.add(course)
    db.flush()
    for number, name in enumerate(["Amara Osei", "Ben Whitaker", "Claudia Moreno"], start=1):
        db.add(Student(course_id=course.id, name=name, student_number=number))
    rubric = Rubric(name="Essay rubric", criteria=RUBRIC_CRITERIA)
    skill = Skill(
        name="Grader",
        system_prompt="Grade the essay.",
        provider=config.MOCK_PROVIDER,
        model=config.MOCK_MODEL,
    )
    db.add_all([rubric, skill])
    db.flush()
    for index, name in enumerate(["Problem Set 1", "Essay 1", "Problem Set 2"], start=1):
        db.add(
            Assignment(
                course_id=course.id,
                name=name,
                description=f"Demo assignment {index}",
                skill_id=skill.id,
                rubric_id=rubric.id,
            )
        )
    db.commit()
    return course


def students(db, course) -> list[Student]:
    return list(
        db.scalars(
            select(Student).where(Student.course_id == course.id).order_by(Student.student_number)
        ).all()
    )


def assignments(db, course) -> list[Assignment]:
    return list(
        db.scalars(
            select(Assignment).where(Assignment.course_id == course.id).order_by(Assignment.id)
        ).all()
    )


def extra_assignment(db, course, name: str) -> Assignment:
    """A fourth+ assignment — `resolved` needs a longer history than 3."""
    template = assignments(db, course)[0]
    assignment = Assignment(
        course_id=course.id,
        name=name,
        description=name,
        skill_id=template.skill_id,
        rubric_id=template.rubric_id,
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return assignment


def grade(
    db,
    student: Student,
    assignment: Assignment,
    scores: dict[str, float],
    *,
    tags: list[str] | None = None,
    strengths: list[str] | None = None,
    derive: bool = True,
) -> tuple[Submission, GradeResult]:
    """Write a graded submission the way the engine would have persisted it."""
    submission = Submission(
        assignment_id=assignment.id,
        student_id=student.id,
        file_path=f"/tmp/{assignment.id}-{student.id}.pdf",
        original_filename=f"a{assignment.id}-s{student.id}.pdf",
        mime_type="application/pdf",
        status="graded",
    )
    db.add(submission)
    db.flush()
    criteria = [
        {
            "key": key,
            "score": float(score),
            "max_points": MAX_POINTS[key],
            "comment": "Comment.",
        }
        for key, score in scores.items()
    ]
    result = GradeResult(
        submission_id=submission.id,
        overall_score=sum(c["score"] for c in criteria),
        max_score=sum(c["max_points"] for c in criteria),
        summary_feedback="Summary feedback.",
        criteria=criteria,
        misconceptions=list(tags or []),
        strengths=list(strengths or []),
        model=config.MOCK_MODEL,
    )
    db.add(result)
    db.commit()
    db.refresh(submission)
    db.refresh(result)
    if derive:
        insight.derive_observations(db, submission, result)
    return submission, result


def kinds(rows) -> list[str]:
    return [row.kind for row in rows]


# --------------------------------------------------------------------------
# 1 · deriver
# --------------------------------------------------------------------------


def test_deriver_emits_one_observation_per_signal(db, course):
    student = students(db, course)[0]
    assignment = assignments(db, course)[0]
    grade(
        db,
        student,
        assignment,
        {"thesis": 10, "evidence": 3, "mechanics": 4},
        tags=["conflates legality with morality"],
        strengths=["clear thesis statement"],
    )

    rows = insight.observations_for(db, student.id, newest_first=False)
    assert kinds(rows).count(insight.KIND_CRITERION_HIGH) == 1  # thesis 10/10
    assert kinds(rows).count(insight.KIND_CRITERION_LOW) == 1  # evidence 3/10 (≤ 50%)
    assert kinds(rows).count(insight.KIND_MISCONCEPTION) == 1
    assert kinds(rows).count(insight.KIND_STRENGTH) == 1
    # mechanics 4/5 is neither low nor full marks: no observation.
    assert len(rows) == 4

    low = next(r for r in rows if r.kind == insight.KIND_CRITERION_LOW)
    assert low.data["criterion_key"] == "evidence"
    assert low.data["score"] == 3.0 and low.data["max_points"] == 10.0
    assert "3 of 10" in low.text
    assert all(r.submission_id and r.assignment_id and r.student_id for r in rows)


def test_criterion_low_boundary_is_half_marks(db, course):
    student = students(db, course)[0]
    assignment = assignments(db, course)[0]
    grade(db, student, assignment, {"thesis": 5, "evidence": 5.5, "mechanics": 5})

    rows = insight.observations_for(db, student.id)
    low = [r for r in rows if r.kind == insight.KIND_CRITERION_LOW]
    high = [r for r in rows if r.kind == insight.KIND_CRITERION_HIGH]
    assert [r.data["criterion_key"] for r in low] == ["thesis"]  # exactly 50% counts
    assert [r.data["criterion_key"] for r in high] == ["mechanics"]  # 5/5


def test_deriver_is_idempotent_when_a_submission_is_regraded(db, course):
    student = students(db, course)[0]
    assignment = assignments(db, course)[0]
    submission, result = grade(
        db,
        student,
        assignment,
        {"thesis": 2, "evidence": 2, "mechanics": 1},
        tags=["treats correlation as causation"],
    )
    first = insight.observations_for(db, student.id)
    assert len(first) == 4

    # Deriving again from the same result changes nothing.
    insight.derive_observations(db, submission, result)
    again = insight.observations_for(db, student.id)
    assert len(again) == len(first)
    assert [r.text for r in again] == [r.text for r in first]

    # A real re-grade replaces the evidence rather than accumulating it.
    result.criteria = [
        {"key": "thesis", "score": 10.0, "max_points": 10.0, "comment": "Better."},
        {"key": "evidence", "score": 9.0, "max_points": 10.0, "comment": "Better."},
        {"key": "mechanics", "score": 5.0, "max_points": 5.0, "comment": "Better."},
    ]
    result.misconceptions = []
    db.commit()
    insight.derive_observations(db, submission, result)

    rows = insight.observations_for(db, student.id)
    assert kinds(rows) == [insight.KIND_CRITERION_HIGH] * 2
    assert db.query(Observation).count() == 2


def test_every_observation_obeys_the_voice_rules(db, course):
    student = students(db, course)[0]
    grade(
        db,
        student,
        assignments(db, course)[0],
        {"thesis": 1, "evidence": 10, "mechanics": 2},
        tags=["conflates legality with morality because we recommend a standard deviation view"],
        strengths=["a very long strength phrase " * 12],
    )
    for row in insight.observations_for(db, student.id):
        assert insight.voice_violations(row.text) == [], row.text


def test_deriver_ignores_a_submission_with_no_grade(db, course):
    student = students(db, course)[0]
    submission = Submission(
        assignment_id=assignments(db, course)[0].id,
        student_id=student.id,
        status="pending",
    )
    db.add(submission)
    db.commit()
    assert insight.derive_observations(db, submission) == []


# --------------------------------------------------------------------------
# 2 · misconception decay
# --------------------------------------------------------------------------


def test_decay_moves_active_to_resolving_to_resolved_across_assignments(db, course):
    """INCREMENT_1 rule 1: a tag in EITHER of the last two assignments is active."""
    student = students(db, course)[0]
    a1, a2, a3 = assignments(db, course)
    a4 = extra_assignment(db, course, "Essay 2")
    scores = {"thesis": 6, "evidence": 6, "mechanics": 3}

    # Assignment 1 only.
    grade(db, student, a1, scores, tags=["fading tag", "sticky tag"])
    states = {s["tag"]: s["status"] for s in insight.misconception_states(db, student.id)}
    assert states == {"fading tag": "active", "sticky tag": "active"}

    # Assignment 2 drops "fading tag" — still inside the two-assignment window.
    grade(db, student, a2, scores, tags=["sticky tag"])
    states = {s["tag"]: s["status"] for s in insight.misconception_states(db, student.id)}
    assert states == {"fading tag": "active", "sticky tag": "active"}

    # Assignment 3 drops it too: now outside the window, and cooling.
    grade(db, student, a3, scores, tags=["sticky tag"])
    states = {s["tag"]: s["status"] for s in insight.misconception_states(db, student.id)}
    assert states == {"fading tag": "resolving", "sticky tag": "active"}

    # Assignment 4 drops it again: gone.
    grade(db, student, a4, scores, tags=["sticky tag"])
    states = {s["tag"]: s["status"] for s in insight.misconception_states(db, student.id)}
    assert states == {"fading tag": "resolved", "sticky tag": "active"}

    resolved = next(
        s for s in insight.misconception_states(db, student.id) if s["tag"] == "fading tag"
    )
    assert resolved["assignments_since_last_seen"] == 3
    assert resolved["graded_assignments"] == 4
    # Evidence survives the decay: the chip still links to the graded work.
    assert len(resolved["evidence"]) == 1
    assert db.get(Observation, resolved["evidence"][0]).kind == insight.KIND_MISCONCEPTION


def test_a_tag_that_comes_back_is_active_again(db, course):
    student = students(db, course)[0]
    a1, a2, a3 = assignments(db, course)
    scores = {"thesis": 6, "evidence": 6, "mechanics": 3}
    grade(db, student, a1, scores, tags=["relapse"])
    grade(db, student, a2, scores, tags=[])
    grade(db, student, a3, scores, tags=["relapse"])

    state = insight.misconception_states(db, student.id)[0]
    assert state["tag"] == "relapse"
    assert state["status"] == insight.STATUS_ACTIVE
    assert state["seen_count"] == 2
    assert len(state["evidence"]) == 2


def test_resolved_needs_more_than_the_two_assignment_window(db, course):
    student = students(db, course)[0]
    a1, a2, a3 = assignments(db, course)
    scores = {"thesis": 6, "evidence": 6, "mechanics": 3}
    grade(db, student, a1, scores, tags=["only once"])
    grade(db, student, a2, scores, tags=[])
    # Still one of the two most recent assignments: active, per rule 1.
    assert insight.misconception_states(db, student.id)[0]["status"] == insight.STATUS_ACTIVE
    grade(db, student, a3, scores, tags=[])
    assert insight.misconception_states(db, student.id)[0]["status"] == insight.STATUS_RESOLVING


def test_decay_status_mapping_is_explicit():
    # Rule 1 (the explicit one): in EITHER of the last two → active.
    assert insight.decay_status(0) == insight.STATUS_ACTIVE
    assert insight.decay_status(1) == insight.STATUS_ACTIVE
    assert insight.decay_status(2) == insight.STATUS_RESOLVING
    assert insight.decay_status(3) == insight.STATUS_RESOLVED
    assert insight.decay_status(7) == insight.STATUS_RESOLVED


def test_a_student_with_nothing_graded_has_no_decay_states(db, course):
    """No graded work means recency is undefined — not "everything is active"."""
    student = students(db, course)[0]
    a1 = assignments(db, course)[0]
    grade(db, student, a1, {"thesis": 6, "evidence": 6, "mechanics": 3}, tags=["ghost"])
    assert insight.misconception_states(db, student.id)

    # Ungrade it: the observation is orphaned from the graded order.
    for result in db.scalars(select(GradeResult)).all():
        db.delete(result)
    db.commit()
    assert insight.misconception_states(db, student.id) == []


# --------------------------------------------------------------------------
# 3 · nudges
# --------------------------------------------------------------------------


def test_nudge_appears_when_three_students_share_a_tag(db, course):
    roster = students(db, course)
    assignment = assignments(db, course)[0]
    scores = {"thesis": 6, "evidence": 4, "mechanics": 3}
    for student in roster[:2]:
        grade(db, student, assignment, scores, tags=["shared tag", "solo tag"])
    assert insight.detect_nudges(db, assignment.id) == []

    grade(db, roster[2], assignment, scores, tags=["shared tag"])
    created = insight.detect_nudges(db, assignment.id)
    assert len(created) == 1
    nudge = created[0]
    assert nudge.evidence["tag"] == "shared tag"
    assert sorted(nudge.evidence["student_ids"]) == sorted(s.id for s in roster)
    assert len(nudge.evidence["observation_ids"]) == 3
    assert nudge.course_id == course.id and nudge.assignment_id == assignment.id
    assert "3 students" in nudge.text
    assert insight.voice_violations(nudge.text) == []


def test_nudges_are_deduped_per_course_assignment_and_tag(db, course):
    roster = students(db, course)
    assignment = assignments(db, course)[0]
    for student in roster:
        grade(db, student, assignment, {"thesis": 6, "evidence": 4, "mechanics": 3}, tags=["dupe"])

    for _ in range(3):
        insight.detect_nudges(db, assignment.id)
    assert db.query(CourseNudge).count() == 1

    # Re-running after a dismissal refreshes evidence but does not resurrect it.
    nudge = db.query(CourseNudge).one()
    insight.dismiss_nudge(db, nudge.id)
    insight.detect_nudges(db, assignment.id)
    assert db.query(CourseNudge).count() == 1
    assert db.get(CourseNudge, nudge.id).dismissed_at is not None


def test_detect_nudges_removes_duplicate_rows_without_resurrecting_dismissal(db, course):
    roster = students(db, course)
    assignment = assignments(db, course)[0]
    tag = "repeated misconception"
    observations = [
        Observation(
            student_id=student.id,
            assignment_id=assignment.id,
            kind=insight.KIND_MISCONCEPTION,
            text=tag,
            data={"tag": tag},
        )
        for student in roster
    ]
    db.add_all(observations)
    db.flush()
    db.add_all(
        [
            CourseNudge(
                course_id=course.id,
                assignment_id=assignment.id,
                text="First copy",
                evidence={"tag": tag},
            ),
            CourseNudge(
                course_id=course.id,
                assignment_id=assignment.id,
                text="Dismissed copy",
                evidence={"tag": tag},
                dismissed_at=insight.utcnow(),
            ),
        ]
    )
    db.commit()
    assert db.query(CourseNudge).count() == 2

    insight.detect_nudges(db, assignment.id)

    remaining = db.query(CourseNudge).one()
    assert remaining.dismissed_at is not None
    assert remaining.evidence["student_ids"] == sorted(student.id for student in roster)
    assert remaining.evidence["observation_ids"] == sorted(obs.id for obs in observations)


def test_nudges_are_scoped_to_one_assignment(db, course):
    roster = students(db, course)
    a1, a2, _ = assignments(db, course)
    scores = {"thesis": 6, "evidence": 4, "mechanics": 3}
    for student in roster:
        grade(db, student, a1, scores, tags=["shared tag"])
        grade(db, student, a2, scores, tags=["shared tag"])
    insight.detect_nudges(db, a1.id)
    insight.detect_nudges(db, a2.id)

    nudges = insight.nudges_for_course(db, course.id)
    assert len(nudges) == 2
    assert {n.assignment_id for n in nudges} == {a1.id, a2.id}


def test_deleting_student_refreshes_affected_course_nudges(client, db, course):
    roster = students(db, course)
    assignment = assignments(db, course)[0]
    scores = {"thesis": 6, "evidence": 4, "mechanics": 3}
    deleted_submission = None
    deleted_result = None
    for student in roster:
        submission, result = grade(
            db,
            student,
            assignment,
            scores,
            tags=["shared misconception"],
        )
        if student.id == roster[0].id:
            deleted_submission = submission
            deleted_result = result

    insight.detect_nudges(db, assignment.id)
    assert db.query(CourseNudge).count() == 1

    deleted_student_id = roster[0].id
    submission_id = deleted_submission.id
    result_id = deleted_result.id
    response = client.delete(f"/api/students/{deleted_student_id}")
    assert response.status_code == 200

    db.expire_all()
    assert db.get(Student, deleted_student_id) is None
    assert db.scalars(
        select(Observation).where(Observation.student_id == deleted_student_id)
    ).all() == []
    preserved_submission = db.get(Submission, submission_id)
    assert preserved_submission is not None
    assert preserved_submission.student_id is None
    assert db.get(GradeResult, result_id) is not None
    nudges = client.get(f"/api/courses/{course.id}/nudges").json()
    assert nudges["count"] == 0
    assert nudges["nudges"] == []


# --------------------------------------------------------------------------
# 4 · card consolidation
# --------------------------------------------------------------------------


def seeded_student(db, course) -> Student:
    """One student with four graded assignments and every decay state.

    A tag is `active` while it is in either of the two most recent assignments
    (INCREMENT_1 rule 1), so reaching `resolved` takes four: "fading tag" was
    last seen on #1 (resolved), "middling tag" on #2 (resolving), "sticky tag"
    on #4 (active).
    """
    student = students(db, course)[0]
    a1, a2, a3 = assignments(db, course)
    a4 = extra_assignment(db, course, "Essay 2")
    grade(
        db,
        student,
        a1,
        {"thesis": 3, "evidence": 10, "mechanics": 2},
        tags=["fading tag", "sticky tag"],
        strengths=["clear thesis statement"],
    )
    grade(
        db,
        student,
        a2,
        {"thesis": 4, "evidence": 10, "mechanics": 3},
        tags=["sticky tag", "middling tag"],
        strengths=["clear thesis statement"],
    )
    grade(
        db,
        student,
        a3,
        {"thesis": 5, "evidence": 10, "mechanics": 4},
        tags=["sticky tag"],
        strengths=["clear thesis statement"],
    )
    grade(
        db,
        student,
        a4,
        {"thesis": 6, "evidence": 10, "mechanics": 4},
        tags=["sticky tag"],
        strengths=["clear thesis statement"],
    )
    return student


def assert_card_is_well_formed(db, card_payload: dict, student_id: int) -> None:
    allowed = {row.id for row in insight.observations_for(db, student_id)}
    statements = [card_payload["summary"], card_payload["trajectory"]]
    for entry in card_payload["strengths"] + card_payload["weaknesses"]:
        statements.append(entry["text"])
        assert entry["evidence"], "every claim carries evidence"
        assert set(entry["evidence"]) <= allowed
    for statement in statements:
        assert statement
        assert insight.voice_violations(statement) == [], statement
    for entry in card_payload["misconception_state"]:
        assert entry["status"] in insight.STATUSES
        assert set(entry["evidence"]) <= allowed


def test_card_is_written_by_the_mock_provider_deterministically(db, course):
    student = seeded_student(db, course)

    card = insight.refresh_card(db, student.id, provider=MockProvider())
    assert card.model == config.MOCK_MODEL
    payload = insight.card_dict(card)
    assert_card_is_well_formed(db, payload, student.id)

    assert "Student #1" in payload["summary"]
    assert "%" in payload["summary"]
    assert payload["weaknesses"], "thesis was at or below half marks three times"
    assert payload["weaknesses"][0]["evidence"]
    assert payload["strengths"], "evidence scored full marks three times"

    states = {m["tag"]: m["status"] for m in payload["misconception_state"]}
    assert states == {
        "sticky tag": insight.STATUS_ACTIVE,
        "middling tag": insight.STATUS_RESOLVING,
        "fading tag": insight.STATUS_RESOLVED,
    }

    # Deterministic: same input, same card.
    again = insight.refresh_card(db, student.id, provider=MockProvider())
    assert insight.card_dict(again) | {"updated_at": None} == payload | {"updated_at": None}
    assert db.query(StudentCard).count() == 1


def test_card_falls_back_to_templates_when_the_provider_fails(db, course):
    student = seeded_student(db, course)

    class DeadProvider:
        model = "dead-model"

        def grade(self, *args, **kwargs):
            raise ProviderError("no API key configured")

    card = insight.refresh_card(db, student.id, provider=DeadProvider())
    assert card.model == "template"
    payload = insight.card_dict(card)
    assert_card_is_well_formed(db, payload, student.id)
    assert payload["weaknesses"] and payload["strengths"]
    assert {m["tag"] for m in payload["misconception_state"]} == {
        "sticky tag",
        "middling tag",
        "fading tag",
    }


def test_llm_disabled_uses_the_templates(db, course):
    student = seeded_student(db, course)
    insight.save_insight_settings({"llm_enabled": False})
    payload, model, warning = insight.consolidate_card(db, student.id)
    assert model == "template" and warning is None
    assert_card_is_well_formed(db, payload, student.id)


def test_card_drops_hallucinated_evidence_and_keeps_computed_decay(db, course):
    student = seeded_student(db, course)
    real_id = insight.observations_for(db, student.id)[0].id

    class WildProvider:
        model = "wild-model"

        def grade(self, system_prompt, blocks, schema):
            return {
                "summary": "Student #1 is drifting, and we recommend a standard deviation review "
                "of the last three essays plus every problem set they have ever submitted here.",
                "trajectory": "Up 3 points across three assignments.",
                "strengths": [
                    {"text": "Cites the reading closely.", "evidence": [999999]},
                    {"text": "Holds full marks on Evidence.", "evidence": [real_id, 424242]},
                ],
                "weaknesses": [{"text": "No evidence at all.", "evidence": []}],
                "misconception_state": [
                    {"tag": "fading tag", "status": "active", "evidence": [999999]},
                    {"tag": "invented tag", "status": "active", "evidence": [1]},
                ],
            }

    card = insight.refresh_card(db, student.id, provider=WildProvider())
    payload = insight.card_dict(card)
    assert card.model == "wild-model"
    assert_card_is_well_formed(db, payload, student.id)

    # The unevidenced strength and weakness are gone; the survivor keeps only
    # the observation id that actually exists.
    assert [s["text"] for s in payload["strengths"]] == ["Holds full marks on Evidence."]
    assert payload["strengths"][0]["evidence"] == [real_id]
    assert payload["weaknesses"] == []

    # Jargon and preaching are cut, and the sentence is back under the limit.
    assert "we recommend" not in payload["summary"].lower()
    assert "standard deviation" not in payload["summary"].lower()
    assert len(payload["summary"].split()) <= insight.MAX_STATEMENT_WORDS

    # The model does not get to overrule the deterministic decay pass, and it
    # cannot invent a tag the student was never given.
    states = {m["tag"]: m["status"] for m in payload["misconception_state"]}
    assert states["fading tag"] == insight.STATUS_RESOLVED
    assert "invented tag" not in states


def test_card_is_rebuilt_only_when_observations_move_on(db, course):
    student = seeded_student(db, course)
    card = insight.refresh_card(db, student.id, provider=MockProvider())
    assert insight.card_is_stale(db, student.id, card) is False

    grade(
        db,
        student,
        assignments(db, course)[2],
        {"thesis": 1, "evidence": 1, "mechanics": 1},
        tags=["brand new tag"],
    )
    assert insight.card_is_stale(db, student.id) is True

    fresh = insight.get_card(db, student.id)  # auto-refresh on read
    assert insight.card_is_stale(db, student.id, fresh) is False
    assert "brand new tag" in {m["tag"] for m in fresh.misconception_state}


def test_no_card_for_a_student_with_no_observations(db, course):
    student = students(db, course)[1]
    assert insight.get_card(db, student.id) is None


def test_template_card_handles_a_student_with_nothing_graded(db, course):
    student = students(db, course)[1]
    payload = insight.template_card(insight.build_card_input(db, student.id))
    assert payload["summary"] and payload["trajectory"]
    assert payload["strengths"] == [] and payload["weaknesses"] == []
    assert insight.voice_violations(payload["summary"]) == []


def test_card_input_never_contains_the_student_name(db, course):
    student = seeded_student(db, course)
    card_input = insight.build_card_input(db, student.id)
    assert insight.card_user_text(card_input).find(student.name) == -1
    assert card_input["student"]["label"] == "Student #1"


# --------------------------------------------------------------------------
# 5 · the post-grading hook (engine integration)
# --------------------------------------------------------------------------


def test_persisting_a_grade_runs_the_insight_hook(db, course):
    roster = students(db, course)
    assignment = assignments(db, course)[0]
    for student in roster:
        submission = Submission(
            assignment_id=assignment.id,
            student_id=student.id,
            file_path=f"/tmp/hook-{student.id}.pdf",
            mime_type="application/pdf",
            status="grading",
        )
        db.add(submission)
        db.commit()
        db.refresh(submission)
        validated = engine.validate_grade_payload(
            {
                "criteria": [
                    {"key": "thesis", "score": 2, "comment": "Thin."},
                    {"key": "evidence", "score": 10, "comment": "Excellent."},
                    {"key": "mechanics", "score": 4, "comment": "Fine."},
                ],
                "summary_feedback": "Summary.",
                "misconceptions": ["Conflates Legality With Morality"],
                "strengths": ["clear thesis statement"],
            },
            RUBRIC_CRITERIA,
        )
        engine.persist_grade_result(db, submission, validated, config.MOCK_MODEL)

    first = roster[0]
    rows = insight.observations_for(db, first.id)
    assert kinds(rows).count(insight.KIND_CRITERION_LOW) == 1
    assert kinds(rows).count(insight.KIND_MISCONCEPTION) == 1
    # The hook consolidates the card once per graded submission…
    card = insight.get_stored_card(db, first.id)
    assert card is not None and card.summary
    # …and the third student's grade trips the class-wide nudge.
    nudges = insight.nudges_for_course(db, course.id)
    assert len(nudges) == 1
    assert nudges[0].evidence["tag"] == "conflates legality with morality"


def test_the_hook_never_breaks_grading(db, course, monkeypatch):
    student = students(db, course)[0]
    submission = Submission(
        assignment_id=assignments(db, course)[0].id,
        student_id=student.id,
        file_path="/tmp/boom.pdf",
        mime_type="application/pdf",
        status="grading",
    )
    db.add(submission)
    db.commit()

    def boom(*args, **kwargs):
        raise RuntimeError("insight exploded")

    monkeypatch.setattr(insight, "derive_observations", boom)
    validated = engine.validate_grade_payload(
        {
            "criteria": [{"key": "thesis", "score": 7, "comment": "Good."}],
            "summary_feedback": "Summary.",
            "misconceptions": [],
            "strengths": [],
        },
        RUBRIC_CRITERIA,
    )
    result = engine.persist_grade_result(db, submission, validated, config.MOCK_MODEL)
    assert result.id and db.get(Submission, submission.id).status == "graded"


# --------------------------------------------------------------------------
# 6 · API
# --------------------------------------------------------------------------


def test_router_is_mounted(client):
    assert "app.routers.insight" in client.get("/api/health").json()["routers"]


def test_card_and_observation_endpoints(client, db, course):
    student = seeded_student(db, course)

    body = client.get(f"/api/students/{student.id}/card").json()
    assert body["label"] == "Student #1"
    assert body["card"]["summary"]
    assert body["stale"] is False
    assert body["observation_count"] == len(insight.observations_for(db, student.id))
    assert_card_is_well_formed(db, body["card"], student.id)

    refreshed = client.post(f"/api/students/{student.id}/card/refresh").json()
    assert refreshed["card"]["summary"] == body["card"]["summary"]
    assert refreshed["stale"] is False

    feed = client.get(f"/api/students/{student.id}/observations").json()
    assert feed["count"] == body["observation_count"]
    assert {o["kind"] for o in feed["observations"]} <= set(insight.KINDS)
    assert {m["status"] for m in feed["misconception_state"]} == set(insight.STATUSES)

    limited = client.get(f"/api/students/{student.id}/observations?limit=2").json()
    assert limited["count"] == 2
    only_tags = client.get(
        f"/api/students/{student.id}/observations?kind=misconception"
    ).json()
    assert {o["kind"] for o in only_tags["observations"]} == {"misconception"}

    assert client.get("/api/students/9999/card").status_code == 404
    assert client.post("/api/students/9999/card/refresh").status_code == 404
    assert client.get("/api/students/9999/observations").status_code == 404
    assert (
        client.get(f"/api/students/{student.id}/observations?kind=nonsense").status_code == 422
    )


def test_nudge_endpoints(client, db, course):
    roster = students(db, course)
    assignment = assignments(db, course)[0]
    for student in roster:
        grade(db, student, assignment, {"thesis": 6, "evidence": 4, "mechanics": 3}, tags=["dupe"])
    insight.detect_nudges(db, assignment.id)

    body = client.get(f"/api/courses/{course.id}/nudges").json()
    assert body["count"] == 1
    nudge = body["nudges"][0]
    assert nudge["evidence"]["tag"] == "dupe"
    assert nudge["dismissed"] is False

    dismissed = client.post(f"/api/nudges/{nudge['id']}/dismiss").json()["nudge"]
    assert dismissed["dismissed"] is True and dismissed["dismissed_at"]

    assert client.get(f"/api/courses/{course.id}/nudges").json()["count"] == 0
    assert (
        client.get(f"/api/courses/{course.id}/nudges?include_dismissed=true").json()["count"] == 1
    )
    assert client.get("/api/courses/9999/nudges").status_code == 404
    assert client.post("/api/nudges/9999/dismiss").status_code == 404


# --------------------------------------------------------------------------
# 7 · chat assistant
# --------------------------------------------------------------------------


def test_get_student_summary_returns_the_card(db, course):
    student = seeded_student(db, course)
    insight.refresh_card(db, student.id, provider=MockProvider())

    payload = assistant_mod.tool_get_student_summary(
        db,
        {"course_id": student.course_id, "student_number": student.student_number},
    )
    assert payload["card"]["summary"]
    assert payload["card"]["misconceptions"]
    assert all(m["status"] in insight.STATUSES for m in payload["card"]["misconceptions"])
    # The old aggregate is still there as the fallback, and no name leaks.
    assert payload["graded_count"] == 4
    assert student.name not in str(payload)


def test_get_student_summary_without_a_card_falls_back_to_the_aggregate(db, course):
    student = students(db, course)[0]
    grade(db, student, assignments(db, course)[0], {"thesis": 6, "evidence": 6, "mechanics": 3})

    payload = assistant_mod.tool_get_student_summary(
        db,
        {"course_id": student.course_id, "student_number": student.student_number},
    )
    assert "card" not in payload
    assert payload["graded_count"] == 1
