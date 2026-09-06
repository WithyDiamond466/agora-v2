# Provider implementation notes

Checked September 6, 2026. Installed SDK behavior, the provider tests, and official documentation should be checked again when changing dependencies or models.

## Anthropic

Grading uses JSON-schema structured output through `output_config.format`. Opus 5 and Sonnet 5 receive the configured effort. Haiku 4.5 omits effort because that model does not support it. Sampling parameters and a manual thinking budget are omitted. Refusals and truncated responses become explicit failures.

Optional server-side fallback sends `fallbacks: "default"` and the `server-side-fallback-2026-07-01` beta header. Set `AGORA_ANTHROPIC_FALLBACKS=0` to disable it. The result records the model that actually answered.

References: [model overview](https://platform.claude.com/docs/en/models/overview), [effort](https://platform.claude.com/docs/en/build-with-claude/effort), and [refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback).

## OpenAI

The adapter uses the installed SDK's chat completions API with a JSON-schema response format for grading. Text extraction provides PDF content; native PDF rendering is not used by this adapter. Models remain selectable through the registry and Skill configuration. See the [official model documentation](https://developers.openai.com/api/docs/models).

## Testing and privacy

Provider unit tests inspect requests using fake clients. They do not establish live account access, service availability, or assessment quality. `scripts/live_smoke.py --provider mock --compare` verifies the application flow without a paid request. A real-provider run uses synthetic work and requires separately configured API credentials.

Cloud swap mode protects the assembled grading prompt, assignment header, extracted submission text, reference text, course chat, Skill trials/comparisons, and student-card content. Generic Skill trials use temporary identifier codes across known rosters. Images and PDFs without extracted text cannot be sent through the protected text path. Identifier detection can miss context; never describe it as guaranteed anonymity.
