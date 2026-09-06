"""Terms of use: the page, the acceptance, and the status the shell polls."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import config, terms
from app.db import get_db

router = APIRouter()
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


class AcceptIn(BaseModel):
    signed_by: Optional[str] = Field(default=None, max_length=200)
    #: The form's checkbox; the JSON API may omit it.
    agree: Optional[bool] = None


@router.get("/api/terms/status")
def terms_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    return terms.status(db)


@router.get("/api/terms")
def terms_text() -> dict[str, Any]:
    return {
        "version": terms.TERMS_VERSION,
        "title": terms.TERMS_TITLE,
        "points": [{"title": t, "body": b} for t, b in terms.TERMS_POINTS],
        "syllabus_statement": terms.SYLLABUS_STATEMENT,
        "disclosure": terms.AI_DISCLOSURE,
    }


@router.post("/api/terms/accept", status_code=201)
def accept_terms(payload: AcceptIn | None = None, db: Session = Depends(get_db)) -> dict[str, Any]:
    payload = payload or AcceptIn()
    row = terms.accept(db, payload.signed_by)
    data = terms.status(db)
    data["id"] = row.id
    data["message"] = f"Terms v{row.version} accepted."
    return data


@router.get("/terms", response_class=HTMLResponse)
def page_terms(request: Request, db: Session = Depends(get_db)) -> Any:
    return templates.TemplateResponse(
        request,
        "terms.html",
        {
            "terms_status": terms.status(db),
            "terms_version": terms.TERMS_VERSION,
            "terms_title": terms.TERMS_TITLE,
            "terms_points": terms.TERMS_POINTS,
            "syllabus_statement": terms.SYLLABUS_STATEMENT,
            "disclosure": terms.AI_DISCLOSURE,
            "nav_active": "settings",
        },
    )


__all__ = ["router"]
