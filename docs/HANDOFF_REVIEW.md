> Historical baseline review from September 6, 2026, before the readiness fixes. Confirmed defects below have since been addressed in code unless listed in [current status](../STATUS.md). This document preserves the original findings.

# Professor handoff review

Reviewed September 6, 2026, on `ready-fall` at `a627902`. Target handoff is Tuesday, September 8. The user confirmed both professor use and repository branching across a mix of operating systems. Plan for Windows, macOS, and Linux, with GitHub as the handoff entry point.

The application has a useful foundation, but I would not hand this build over for real student grading yet. A small pilot is a reasonable target after the blockers below are fixed and the actual professor workflow passes on a clean machine. This review changes no application code.

## Must fix before real coursework

### 1. Professors cannot correct scores in the grading screen

Source: `templates/grading.html:327`, `static/js/grading.js:657`.

AI-assigned criterion scores render as text. The only score inputs appear for manual criteria that have no score. After the professor saves a manual score, that criterion also renders as text. Criterion comments are likewise displayed without an editor. The API already supports score and comment edits, but the screen only exposes editing the summary feedback and entering previously blank manual scores.

Verified by rendering a graded assignment with two criteria: zero `data-crit-score` inputs. Saving a previously blank manual score through the API and rendering again also produces zero inputs.

Fix: expose editing for every scored criterion and its comment, including previously saved manual scores. Preserve feedback-only mode without numeric grades. Verify save, reload, edit again, recalculated totals, and the exported values through the browser.

### 2. The privacy guard does not cover the complete grading request

Source: `app/ai/grading.py:369`, `app/ai/grading.py:389`, `app/ai/grading.py:475`, `app/ai/grading.py:527`.

The uploaded submission and supported knowledge documents are pseudonymized in cloud swap mode. Skill instructions, rubric text, custom mode instructions, course/assignment names, and the assignment description enter other parts of the request without that protection.

Verified with a synthetic roster member, Alice Example. Her name was removed from the extracted essay but remained in the outgoing system prompt and assignment header when included in the professor's instructions. This check assembled the request locally; no data was sent to a provider.

Fix: apply protection to all outgoing free text, including the complete system prompt and assignment header. Extend the captured-request tests to check both system and content payloads. Review equivalent paths for chat, skill trials, comparisons, and student cards. Replace the absolute promise that student names never leave the machine with an accurate explanation of protection and its limits.

### 3. Incomplete selective grades can be released as final scores

Source: `app/review.py:298`, `app/ai/grading.py:780`, `app/release.py:85`.

The release screen warns that manual scores are missing, but the server still releases the result. Totals include the maximum for the unscored criteria while treating their missing scores as zero.

Verified with an AI criterion scored 8/10 and a manual criterion still blank out of 10. Release returned HTTP 200 and CSV export contained 8/20, or 40 percent.

Fix: hold incomplete scored results at the server boundary, including when the caller requests release of unseen results. Export must also reject or omit incomplete scored results. Feedback-only results should remain releasable without numbers. Show partial grading as incomplete rather than presenting it as a final percentage.

### 4. Localhost requests lack origin and host checks

Source: `app/main.py:47`, `app/routers/release.py:125`.

Binding to loopback limits network exposure, but the app has no origin validation, trusted-host middleware, or request token for mutations. Some mutating routes accept empty POST bodies.

Verified against the isolated app with accepted terms and an opened result: an empty form POST to the release endpoint with an unrelated Origin and Host returned HTTP 200 and released the result. This proves missing server checks; an actual cross-site browser exploit was not exercised.

Fix: restrict accepted hosts and reject cross-origin mutations. Use an appropriate local session/request token where needed, covering shutdown, grading, review, release, uploads, and settings. Test legitimate local requests and rejected unrelated origins separately. Preserve the loopback-only launcher.

### 5. Correcting a grade leaves student insights based on the old grade

Source: `app/routers/grading.py:1104`, `app/routers/grading.py:1169`, `app/ai/grading.py:858`.

The grading engine updates observations and student cards when it saves an AI result. The professor edit endpoint saves the grade without rebuilding or invalidating that derived information.

Verified by deriving an observation for a 1/10 criterion, then changing the score to 10/10 through the edit API. The saved score was 10, but the observation still said "Scored 1 of 10 on Thesis in Essay 1."

Fix: rederive observations and invalidate or refresh the affected student card and course nudges after relevant edits. Verify that correcting scores, misconceptions, and strengths changes the corresponding insights. If this cannot be completed before the pilot, disable the affected insights until they are refreshed from corrected records.

### 6. Qualify the actual AI workflow before calling the release ready

Source: `STATUS.md` fall-readiness section, `scripts/live_smoke.py:158`, `app/routers/chat.py:51`.

The recorded readiness work has no real-provider end-to-end result. This review also made no live provider calls. Passing mock tests does not establish model availability, accepted request schemas, latency, cost, or grading usefulness.

The supplied smoke script has a concrete false-success case. It sends `course_id` at the top level of the chat request, but the API expects `context.course_id`. My mock run replied "Open a course before asking for a course summary." The script still printed `OK`. Comparison errors are printed rather than asserted as failures, and export checks also need explicit expected outcomes.

Fix: correct the chat request and make the script fail on unsuccessful required outcomes. Then run synthetic coursework through each provider advertised for the pilot: upload, grade, correct, review, release, export, and ask a course question. Record the model actually used, time, approximate spend, and useful feedback. Include invalid credentials, an unavailable model, retry behavior, and an interrupted batch. Have a professor compare the output with their own rubric judgments before using it for grades.

## Scope and handoff work for Tuesday

1. **State which submissions work.** The default cloud swap mode deliberately rejects images and PDFs without a text layer, in `app/ai/privacy.py:1246`. Text extraction also drops visual content from otherwise readable PDFs. For Tuesday, support text-based assignments explicitly and reject unsupported files before a batch starts. Do not promise scanned or handwritten work without a separately verified path. The README's general PDF/image workflow currently overstates the default behavior.
2. **Provide setup for the recipients' machines.** The quickstart assumes a developer checkout at `~/repos/agora-v2`, Python 3.12, and Unix virtualenv paths. Add clone/download instructions, OS-specific launch commands, a reproducible dependency lock, and a tested clean installation. Include a sample roster, rubric, and submission, plus a short first-course walkthrough. Explain the difference between mock demonstration grades and real AI grading.
3. **Provide backup and recovery instructions.** There is no documented full-course backup/restore procedure. Verify recovery of a synthetic course, uploaded files, grades, mappings, and settings into a fresh installation. Separate demo data from real coursework and document safe upgrades and interrupted grading recovery.
4. **Prepare the repository for branching.** There is no tracked license, contribution guide, dependency lock, or CI workflow. Choose the sharing license, document test and extension workflows, and add CI. Remove or clearly separate old build-run logs and machine-specific tooling from the handoff material. Reconcile stale spec/status statements with the current implementation.
5. **Define review explicitly.** Opening a result currently marks it seen, and release can optionally include unopened results. Decide the pilot's intended review contract. An explicit approval action would make the professor's decision clearer than treating opening a pane as review. This is a product behavior recommendation, not a determination about provider policy compliance.

## Suggested order

- Sunday: complete score editing, protect all outgoing grading text, and hold incomplete grades. Add focused regressions for each confirmed defect.
- Monday: add local request protections, refresh insights after corrections, strengthen the smoke script, and exercise the intended cloud provider with synthetic coursework. Complete clean installation and restore verification on a recipient's operating system.
- Tuesday: have one professor complete a new course through export without developer assistance. Fix any blocking step, tag the pilot release, and hand over the same tested build with its instructions and known limits. If a blocker remains, offer a demonstration or a development branch with synthetic data and an explicit defect list.

## Verification performed

- 321 offline tests passed, with the 4 local-model integration tests excluded, in the existing virtualenv. A first run with a global mock override interfered with provider-selection tests; the reported passing run removed that override and cloud key environment variables.
- All 12 templates compiled; all 7 routes in `scripts/render_smoke.py` rendered successfully; all 8 JavaScript files passed syntax checks.
- `pip check` found no broken installed requirements. A fresh dependency installation was not performed.
- The supplied mock smoke flow graded two submissions, recorded review/edit/release events, exported one released row with disclosure, and ran comparison. Its incorrect chat response demonstrates why the current final `OK` is insufficient.
- Isolated synthetic reproductions confirmed findings 1 through 5. Existing privacy tests and source verify the default image/textless-PDF rejection.
- No interactive browser acceptance test, recipient OS test, local-model integration test, or live cloud-provider test was completed. Existing user data was not used for the reproductions.

The green suite is useful evidence for the behavior it covers. The confirmed defects above are gaps in that coverage and in the professor workflow.
