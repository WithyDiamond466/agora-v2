# Supplemental local-model check

September 6, 2026. An installed Gemma 3 4B instruction model, Q4_K_M, ran through Agora's local provider on a temporary loopback server. All coursework was synthetic. No cloud-provider request was used for this check.

## Findings

- Real inference produced structured criterion scores and feedback. The demo-placeholder run completed grading, professor feedback editing, explicit approval, release, and a CSV with exactly one disclosed row. Local-versus-mock comparison completed.
- The shipped `Alex_Example.pdf` also produced a stored real-model result, 22/35 against the demonstration course rubric. It also completed approval and one-row disclosed export. Student-card synthesis timed out and correctly fell back to a template while preserving the grade. This was an integration test, not an instructor-calibrated grade.
- Grading quality failed a basic negative check: the demo placeholder contains no essay, yet Gemma assigned 25/35 and praised its grasp of the concepts. Do not treat successful structured output as evidence of useful assessment.
- Course chat did not pass. Gemma printed a code block resembling a tool call, but the saved record contained zero executed course tools. The initial script printed success, but inspection of its saved chat record invalidated that result. The smoke script now requires a successful tool execution, as well as the expected provider and a nonempty answer.
- Preparing this check found that chat ignored the saved provider preference and started from Anthropic. That application defect is fixed and has a regression test.

The application changes passed 338 offline tests. This evidence supplements the existing browser, backup, and three-platform checks. Anthropic/OpenAI acceptance and instructor calibration remain outstanding.

## Reproduce with your own local server

Start a compatible model server separately. Then run from the Agora folder, using its actual loopback endpoint:

```bash
.venv/bin/python scripts/live_smoke.py \
  --provider local --model gemma-3-4b-it \
  --local-base-url http://127.0.0.1:3782/v1 \
  --max-tokens 2048 --count 1 \
  --sample-essay examples/Alex_Example.pdf --compare --keep
```

On Windows use `.\.venv\Scripts\python.exe` and place the arguments on one line. The script creates disposable course records and applies the supplied local endpoint only there. `--keep` preserves those synthetic records for inspection. A model that does not execute course tools should fail the final chat check.
