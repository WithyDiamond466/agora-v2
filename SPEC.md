# Agora architecture

Agora serves a single professor through a local FastAPI server and browser UI. SQLite stores course records; local folders hold submissions and Skill documents. It does not provide accounts or a multi-user deployment mode.

## Domain

A Course owns its roster and Assignments. Students receive numbers within a course. An Assignment selects a Rubric and a Skill. A Rubric defines criterion keys, descriptions, and maximum points. A Skill combines instructions, provider/model preferences, reference documents, and a grading mode. A Submission links an uploaded file to an assignment and optionally a student.

A GradeResult stores criterion scores and comments, feedback, tags, and the model that answered. Missing scored criteria make the result incomplete. Feedback-only criteria have no score. Professor edits refresh deterministic observations and student cards.

Review events distinguish opening, editing, approval, withholding, and release. Export requires a complete graded result with explicit approval and release. Editing or regrading clears prior approval and release; historical events remain. Released CSV rows include an AI disclosure and escape text that could become a spreadsheet formula.

## Boundaries

- `app/routers/` handles page routes and APIs.
- `app/ai/grading.py` assembles grading requests and validates responses.
- `app/ai/providers.py` adapts Anthropic, OpenAI, local, and mock providers.
- `app/ai/privacy.py` detects identifiers and substitutes codes for cloud requests in swap mode. Detection has limits. Local endpoints receive content as configured.
- `app/review.py` and `app/release.py` enforce approval and export eligibility.
- `app/insight.py` and `app/analytics.py` derive observations and summaries from results.
- `app/backups.py` snapshots SQLite, packages files without credentials, and restores into a new directory.
- `app/local_requests.py` rejects unrecognized hosts and cross-origin mutations.
- `templates/` and `static/` provide server-rendered pages and browser interactions.

Database startup adds missing columns for existing installations. Back up before upgrading. The default paths and model registry live in `app/config.py`. `.agoraskill` archives transfer reusable Skills, not course records. Full coursework backup is a separate operation.

The tests use disposable databases and synthetic inputs. Mock-provider success verifies application integration, not real-provider availability or grading quality. Current evidence and remaining acceptance work live in [STATUS.md](STATUS.md).
