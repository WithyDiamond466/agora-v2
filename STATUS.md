# Verification and remaining work

September 6, 2026. Agora is a development preview for synthetic-course evaluation. The readiness pass corrected score editing, incomplete-score handling, explicit approval and release, stale insights after edits, outbound prompt privacy coverage, local request protections, and portable backups.

## Completed evidence

- Fresh Python 3.12 environment on Linux installed the constrained dependencies and passed `pip check`.
- The offline suite passed 334 tests, with 4 optional local-model tests excluded. Final published CI results are available in the repository Actions tab.
- The mock workflow exercised course/roster/rubric/Skill creation, uploads and mapping, grading, feedback edits, approval, release, one-row disclosed CSV export, comparison, and course-context chat.
- A full synthetic demo backup restored into a new directory with 1 course, 12 students, 2 assignments, 24 submissions, 24 grades, and all file references valid. API credentials were excluded and the source stayed intact.
- Browser checks covered criterion edits, persistence after reload, explicit approval, and a release dialog that held back 11 unapproved results. See the acceptance checklist below for remaining checks.
- Unit tests cover privacy request contents, approval invalidation, incomplete results, CSV formula protection, invalid numeric scores, failed deletions, archive traversal, and request origin/host validation.

## Remaining acceptance work

- Live Anthropic and OpenAI grading, comparison, and course chat have not been run with this build. No provider key was available for the readiness pass. Fake-client tests verify request shape only.
- A professor needs to compare a small synthetic set against their own rubric scores before relying on grading quality.
- Windows and macOS interactive launch/browser checks remain unverified. Automated CI covers installation and offline workflows on these systems, not native usability.
- The optional local model is not bundled. Tests marked `local_llm` are excluded from the offline gate.
- Privacy detection can miss identifiers. Text-based cloud grading loses diagrams and layout; default protection rejects images and PDFs without extractable text. Provider failures and unsupported input must be resolved before approval.

## Recipient acceptance checklist

1. Follow [setup](docs/GETTING_STARTED.md) on the recipient's computer and explore the demo.
2. Complete the [synthetic course](examples/README.md) with the intended provider. Confirm errors and cost expectations before a batch.
3. Correct a criterion score and comment, reload, approve one result, release, and inspect the exported CSV. Leave a second result unapproved and confirm it is absent.
4. Inspect course analytics and the student's evidence after the correction. Try course chat and compare two candidate models.
5. Export/import a Skill. Make a backup, restore into a new directory, and verify the roster and uploaded work.
6. Record OS, Python version, provider/model, timings, and any differences from instructor scores. Never include real student work or credentials in an issue.

[Architecture](SPEC.md) · [Original review](docs/HANDOFF_REVIEW.md) · [Release plan](docs/RELEASE_PLAN.md)
