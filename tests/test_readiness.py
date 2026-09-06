"""Professor handoff regressions, using synthetic records and no provider calls."""
import csv
import io
import json
import pytest
from sqlalchemy import select
from app import config, insight, review, terms
from app.ai import grading
from app.models import Observation, Rubric, Skill
from test_review import Session, db, client, _graded_assignment
from test_privacy import _text_pdf

@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path / 'data')
    config.save_privacy_settings({'mode': 'swap', 'llm_sweep': False, 'local_model': {'enabled': False}})
    monkeypatch.delenv('AGORA_AI_PROVIDER', raising=False)


def test_existing_scores_and_comments_are_editable(client, db):
    a, subs = _graded_assignment(db, n=1)
    html = client.get(f'/grading/{a.id}').text
    assert html.count('data-crit-score') == 2
    assert html.count('data-crit-comment') == 2


def test_complete_cloud_request_protects_professor_supplied_text(db, tmp_path):
    a, subs = _graded_assignment(db, provider='anthropic', n=1)
    s = subs[0]
    s.student.name = 'Alice Example'
    a.description = 'Alice Example must address the counterargument.'
    skill = db.get(Skill, a.skill_id)
    skill.system_prompt = 'Give Alice Example constructive feedback.'
    rubric = db.get(Rubric, a.rubric_id)
    rubric.criteria = [{**c, 'description': 'Assess Alice Example fairly.'} for c in rubric.criteria]
    p = tmp_path / 'essay.pdf'
    p.write_bytes(_text_pdf('Alice Example argues from evidence.'))
    s.file_path = str(p)
    db.commit()
    request = grading.build_grade_request(db, s)
    assert 'Alice Example' not in json.dumps([request.system_prompt, request.content_blocks])
    assert 'Student-01' in request.system_prompt


def test_incomplete_scores_cannot_be_released_or_exported(client, db):
    a, subs = _graded_assignment(db, n=1)
    s = subs[0]
    s.grade_result.mode = 'selective'
    s.grade_result.criteria = [{'key':'thesis','score':8,'max_points':10}, {'key':'evidence','score':None,'max_points':10,'manual':True}]
    db.commit()
    review.mark_seen(db, s)
    terms.accept(db, 'Synthetic professor')
    r = client.post(f'/api/assignments/{a.id}/release', json={'include_unseen': True}).json()
    assert r['released'] == []
    assert r['held'][0]['reason'] == 'incomplete scores'
    # A legacy released row must not bypass the export gate.
    db.refresh(s.grade_result)
    s.grade_result.released_at = review.utcnow()
    db.commit()
    assert client.get(f'/api/assignments/{a.id}/export').status_code == 409


def test_score_correction_rebuilds_insights(client, db):
    a, subs = _graded_assignment(db, n=1)
    s = subs[0]
    s.grade_result.criteria = [{'key':'thesis','score':1,'max_points':10,'comment':'weak'}]
    db.commit()
    insight.derive_observations(db, s)
    assert client.patch(f'/api/submissions/{s.id}/result', json={'criteria':[{'key':'thesis','score':10}]}).status_code == 200
    db.expire_all()
    observations = list(db.scalars(select(Observation).where(Observation.submission_id == s.id)))
    assert any(o.kind == 'criterion_high' for o in observations)
    assert not any(o.kind == 'criterion_low' for o in observations)


def test_nonfinite_score_rejected(client, db):
    a, subs = _graded_assignment(db, n=1)
    resp = client.patch(f'/api/submissions/{subs[0].id}/result', json={'criteria':[{'key':'thesis','score':'NaN'}]})
    assert resp.status_code == 400


def test_untrusted_origin_cannot_release(client, db):
    a, subs = _graded_assignment(db, n=1)
    review.mark_seen(db, subs[0])
    terms.accept(db, 'Synthetic professor')
    r = client.post(f'/api/assignments/{a.id}/release', headers={'Origin':'https://untrusted.example'})
    assert r.status_code == 403
    assert client.get('/api/health', headers={'Host':'untrusted.example'}).status_code == 400


def test_opening_is_not_approval_and_edits_revoke_release(client, db):
    a, subs = _graded_assignment(db, n=1)
    s = subs[0]
    terms.accept(db, 'Synthetic professor')
    assert client.post(f'/api/submissions/{s.id}/review/seen').status_code == 200
    assert not client.post(f'/api/assignments/{a.id}/release', json={'include_unseen': True}).json()['released']
    approved = client.post(f'/api/submissions/{s.id}/review/approve')
    assert approved.json()['review']['state'] == 'approved'
    assert len(client.post(f'/api/assignments/{a.id}/release', json={}).json()['released']) == 1
    assert client.get(f'/api/assignments/{a.id}/export').status_code == 200
    client.patch(f'/api/submissions/{s.id}/result', json={'summary_feedback':'Revised feedback'})
    assert client.get(f'/api/assignments/{a.id}/export').status_code == 409
    assert client.get(f'/api/submissions/{s.id}/review').json()['review']['approved_at'] is None


def test_generic_skill_text_uses_private_placeholders(db):
    from app.ai.privacy import protect_unscoped_text
    a, subs = _graded_assignment(db, n=1)
    subs[0].student.name = 'Alice Example'
    db.commit()
    text = protect_unscoped_text(db, 'Alice Example emailed alice@example.invalid.')
    assert 'Alice Example' not in text and 'alice@example.invalid' not in text
    assert '[PRIVATE-' in text


def test_csv_does_not_execute_formula_text(client, db):
    a, subs = _graded_assignment(db, n=1)
    a.name = '=1+1'
    db.commit()
    terms.accept(db, 'Synthetic professor')
    client.post(f'/api/submissions/{subs[0].id}/review/approve')
    client.post(f'/api/assignments/{a.id}/release', json={})
    content = client.get(f'/api/assignments/{a.id}/export').content.decode('utf-8-sig')
    assert list(csv.DictReader(io.StringIO(content)))[0]['assignment'] == "'=1+1"


def test_legacy_scores_without_maxima_use_the_assignment_rubric(client, db):
    a, subs = _graded_assignment(db, n=1)
    s = subs[0]
    s.grade_result.criteria = [{'key':'thesis','score':7,'comment':'old'}, {'key':'evidence','score':6,'comment':'old'}]
    db.commit()
    resp = client.patch(f'/api/submissions/{s.id}/result', json={'criteria':[{'key':'thesis','score':9}]}).json()
    assert resp['result']['overall_score'] == 15
    assert resp['result']['max_score'] == 20
    assert 'Score out of 10' in client.get(f'/grading/{a.id}').text


def test_fresh_assignment_can_be_configured_from_browser(client, db):
    from app.models import Assignment, Course
    course = Course(name="Synthetic setup", term="Fall 2026")
    skill = Skill(name="Synthetic skill", system_prompt="Assess evidence", provider="mock", model="mock-grader-1")
    db.add_all([course, skill]); db.commit()
    a = Assignment(course_id=course.id, name="First essay")
    db.add(a); db.commit()
    html = client.get(f"/grading/{a.id}").text
    assert 'id="assignment-setup"' in html
    assert 'Synthetic skill' in html
    response = client.post(f"/api/assignments/{a.id}/setup", json={
        "skill_id": skill.id, "rubric_name": "Evidence rubric",
        "criteria": [{"title": "Claim", "description": "State a position", "max_points": 10},
                     {"title": "Evidence", "description": "Support the position", "max_points": 5}]
    })
    assert response.status_code == 200
    db.expire_all()
    assert a.skill_id == skill.id
    rubric = db.get(Rubric, a.rubric_id)
    assert rubric.total_points == 15
    assert [c["title"] for c in rubric.criteria] == ["Claim", "Evidence"]


def test_setup_rejects_changes_to_graded_assignment(client, db):
    a, subs = _graded_assignment(db, n=1)
    old_rubric = a.rubric_id
    response = client.post(f"/api/assignments/{a.id}/setup", json={
        "skill_id": a.skill_id, "rubric_name": "Replacement",
        "criteria": [{"title": "Different", "max_points": 10}]
    })
    assert response.status_code == 409
    db.expire_all()
    assert a.rubric_id == old_rubric
