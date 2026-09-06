# Agora professor release plan

Target: a public GitHub repository that professors can download and run on Windows, macOS, or Linux, and colleagues can fork and develop. The intended handoff date is September 8, 2026. Public GitHub publication is authorized. Readiness claims depend on completed verification, not the date.

The September 6 baseline is 321 passing offline tests and a set of confirmed workflow, privacy, and distribution defects in [HANDOFF_REVIEW.md](HANDOFF_REVIEW.md). Setup and contribution guides are drafted. Implementation and publication evidence is tracked in [STATUS.md](../STATUS.md). The plan below preserves the acceptance criteria.

## Working brief

Make Agora understandable and dependable for a professor using it without the developer present. Inspect every advertised workflow. Fix correctness and data protection before presentation. Preserve local records and development history. Every advertised feature must have a verified path, or a clear restriction before a user depends on it. Leave a repository another developer can install, test, and extend.

## Pass 1: complete the product audit

- Walk a fresh installation through first launch, demo, settings, provider configuration, course creation, roster import, rubrics, skills, assignment creation, file mapping, grading, correction, review, release, export, analytics, chat, and skill sharing.
- Inspect empty, loading, failure, cancellation, and recovery states as well as successful paths.
- Use synthetic coursework and a repeatable acceptance checklist. Capture screenshots and exact reproduction steps where useful.
- Review data deletion, filesystem cleanup, upgrades, local storage, and the boundary of every outbound AI request.
- Consolidate findings into one prioritized backlog with acceptance criteria. Existing findings are the starting point, not the full scope.

Exit: every advertised workflow has an explicit acceptance check; newly discovered blockers join the backlog.

## Pass 2: fix correctness and privacy

- Allow editing all criterion scores and comments, including previously saved manual scores. Recompute totals and preserve the professor's edits on reload.
- Keep incomplete selective grades out of release and export. Label partial results accurately throughout grading and analytics.
- Define an explicit professor approval action and make release behavior consistent with that decision. Preserve an understandable audit trail through regrading and later edits.
- Rebuild or invalidate observations, cards, and nudges after relevant grade corrections.
- Protect all outgoing content in grading, chat, skill trials, comparisons, and insights. Check system prompts, metadata, filenames, history, and attachments, including degradation when the local privacy model is unavailable.
- Add appropriate host and cross-origin request protections without breaking normal local operation.
- Review input validation, uploaded files, imported skill bundles, CSV exports, failures, and interrupted batches. Correct confirmed issues using focused reproductions.

Exit: the confirmed defects have regression coverage, and corrected workflows pass in the real browser and exported files.

## Pass 3: polish the professor experience

- Make the next action clear on first launch and every empty screen.
- Explain mock demonstration results, provider selection, local versus cloud processing, and supported document types in plain language.
- Reject unsupported submissions before starting an expensive batch. Make text-extraction limits visible when they affect grading fidelity.
- Improve score editing, save feedback, grading progress, retries, release summaries, and export naming.
- Check keyboard navigation, focus, labels, contrast, light/dark themes, laptop widths, dialogs, tables, and long content.
- Remove misleading or stale copy and resolve visual inconsistencies within the existing design.

Exit: a synthetic first-course walkthrough can be completed without hidden API calls or developer instructions.

## Pass 4: installation and data durability

- Provide simple setup and launch paths for Windows, macOS, and Linux. Evaluate small launch helpers that remove unnecessary terminal steps.
- Produce reproducible dependency installation and an explicit supported Python version.
- Keep demo data distinct from real coursework and avoid destructive surprises when reopening the app.
- Implement or document a complete backup/restore path and prove it with a synthetic course in a fresh installation. Check records, uploads, mappings, grades, and preferences.
- Verify upgrades from an earlier database and recovery after an interrupted grading run. Explain what recovery preserves.
- Exercise installation and offline checks on the operating systems through CI. Record any native interactive checks that remain outstanding.

Exit: a clean checkout installs reproducibly, local data survives restart, and a restored synthetic course matches the original.

## Pass 5: real AI and acceptance testing

- Correct the smoke script's chat context and replace printed status with assertions for required outcomes, including comparison and exported rows.
- Verify the selected provider adapters against current official documentation and installed SDKs before live calls.
- With an available authorized provider, run synthetic work through grading, correction, approval, release, export, comparison, and course chat.
- Capture the actual model used, latency, approximate spend, and whether the result follows the supplied rubric. Test invalid configuration and provider failure paths.
- Check grading usefulness against a small instructor-scored synthetic set. Technical success does not establish grading quality.
- Rerun the complete offline suite and browser acceptance checks after integration. Keep live provider testing separate from ordinary CI.

Exit: advertised AI paths have real-provider evidence, and a professor can complete the acceptance walkthrough. Missing provider access or native OS access must be recorded as unverified, not inferred from mock tests.

## Pass 6: publish and hand off through GitHub

- Publish under the confirmed owner after checking repository availability and the publication contents. Public visibility is already approved.
- Inspect both current files and any history to be published for student data, secrets, local configuration, and internal build artifacts. Preserve existing local history. Use a separate clean publication snapshot if needed.
- Choose the project's sharing license with the owner before making an open-source licensing claim.
- Provide a front page with two clear paths: run Agora and fork/develop Agora.
- Include sample materials, setup instructions, contributor guidance, architecture orientation, troubleshooting, limitations, and issue/PR templates.
- Add automated checks across supported operating systems. Confirm that the published workflows actually run and report their results.
- Verify a fresh clone from GitHub rather than relying only on the original workspace.
- Tag a release only after its acceptance gates pass. Otherwise publish an honestly labeled development preview with the outstanding blockers.

Exit: one working GitHub URL, a verified download/clone path, clear release status, and sufficient instructions for both audiences.

## How execution will be managed

Use small commits grouped by completed behavior. For each substantive defect, reproduce it, implement the smallest suitable correction, run the focused check, then verify the full user flow. Keep one current backlog and record evidence against its acceptance checks. Share concise progress updates with completed work, discoveries, and remaining uncertainty.

Routine implementation choices do not need repeated permission. Public publication and the project work are authorized. Ask only when progress requires an unavailable external input or a material owner decision, such as the sharing license. Do not widen the work into fleet or infrastructure changes.

The date orders the work; it does not waive a failed safety, correctness, or installation gate. Any deferred feature must be restricted or identified clearly at the point of use and in the release notes.
