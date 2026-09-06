"""Model comparison on real submissions — the professor's own bake-off.

There is no fixed grading model, so the professor needs a way to decide: the
same submission graded by two to four candidates, side by side, nothing
persisted. Each candidate goes through the full request builder (privacy
guard included, for *that* provider), the provider, and validation, exactly as
a real grading run would.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import config, terms
from app.ai import grading as engine
from app.ai.providers import ProviderError, get_provider, resolve_provider
from app.db import get_db
from app.models import Assignment, Submission

router = APIRouter()


class CandidateIn(BaseModel):
    provider: str = Field(min_length=1, max_length=50)
    model: Optional[str] = Field(default=None, max_length=100)


class SubmissionCompareIn(BaseModel):
    candidates: list[CandidateIn] = Field(min_length=1, max_length=4)


def candidate_options(db: Session) -> list[dict[str, Any]]:
    """Every provider/model the UI can offer, with whether it can run right now."""
    options: list[dict[str, Any]] = []
    auto = resolve_provider(db, config.AUTO_PROVIDER)
    options.append(
        {
            "provider": config.AUTO_PROVIDER,
            "model": config.AUTO_MODEL,
            "label": "Automatic" + (f" → {auto.provider} · {auto.model}" if auto.available else ""),
            "available": auto.available,
        }
    )
    registry = config.load_model_registry()
    for provider, entry in registry.items():
        for m in entry.get("models", []):
            resolution = resolve_provider(db, provider, m.get("id"))
            options.append(
                {
                    "provider": provider,
                    "model": m.get("id"),
                    "label": f"{entry.get('label', provider.title())} · {m.get('label', m.get('id'))}",
                    "available": resolution.available,
                }
            )
    options.append(
        {
            "provider": config.MOCK_PROVIDER,
            "model": config.MOCK_MODEL,
            "label": "Mock grader (offline, deterministic)",
            "available": True,
        }
    )
    return options


@router.get("/api/compare/candidates")
def list_candidates(db: Session = Depends(get_db)) -> dict[str, Any]:
    return {"candidates": candidate_options(db), "terms": terms.status(db)}


@router.post("/api/submissions/{submission_id}/compare")
def compare_submission(
    submission_id: int, payload: SubmissionCompareIn, db: Session = Depends(get_db)
) -> dict[str, Any]:
    submission = db.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail=f"Submission {submission_id} not found")
    if submission.student_id is None:
        raise HTTPException(
            status_code=409, detail="Assign this file to a student before comparing models."
        )
    assignment = db.get(Assignment, submission.assignment_id)

    resolutions = [resolve_provider(db, c.provider, c.model) for c in payload.candidates]
    if any(config.is_cloud_provider(r.provider) for r in resolutions if r.available):
        try:
            terms.require_accepted(db, "comparing models on a cloud provider")
        except terms.TermsNotAccepted as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    results: list[dict[str, Any]] = []
    for cand, resolution in zip(payload.candidates, resolutions):
        started = time.perf_counter()
        entry: dict[str, Any] = {
            "requested_provider": cand.provider,
            "requested_model": cand.model,
            "provider": resolution.provider,
            "model": resolution.model,
            "note": resolution.note,
            "grade": None,
            "error": None,
            "notes": [],
        }
        if not resolution.available:
            entry["error"] = resolution.note
            entry["elapsed_ms"] = 0
            results.append(entry)
            continue
        try:
            request = engine.build_grade_request(
                db, submission, provider=resolution.provider, model=resolution.model
            )
            provider = get_provider(
                {
                    "provider": request.provider,
                    "model": request.model,
                    "max_tokens": request.max_tokens,
                },
                db=db,
            )
            raw = provider.grade(request.system_prompt, request.content_blocks, request.schema)
            validated = engine.validate_grade_payload(
                raw,
                request.criteria,
                label=request.anon_label or request.filename,
                scored=request.scored,
                manual_criteria=request.manual_criteria,
                mode=request.mode,
            )
            entry["grade"] = validated.to_dict()
            entry["model"] = getattr(provider, "served_model", None) or provider.model
            entry["notes"] = list(request.notes)
        except ProviderError as exc:
            entry["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not sink the comparison
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        results.append(entry)

    return {
        "submission_id": submission.id,
        "assignment_id": submission.assignment_id,
        "assignment": assignment.name if assignment else None,
        "persisted": False,
        "results": results,
    }


__all__ = ["router", "candidate_options"]
