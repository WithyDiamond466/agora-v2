# Develop Agora

Read [current verification](STATUS.md) for remaining acceptance work and the [product spec](SPEC.md) for domain concepts. Current code and tests take precedence over older implementation notes.

## Fork and install

Choose **Fork** on GitHub, clone your fork using its **Code** URL, and create a feature branch:

```bash
git switch -c fix/describe-the-change
```

Follow the [platform setup guide](docs/GETTING_STARTED.md) for Python 3.12 and dependency installation. Use `requirements-lock.txt` as the constraints file for reproducible dependency versions.

## Verify a change

Use synthetic data. Do not globally set `AGORA_AI_PROVIDER=mock` for the full suite; provider-resolution tests need to select their own providers.

```bash
# macOS / Linux
.venv/bin/python -m pytest -m "not local_llm"
.venv/bin/python -m pip check
```

```powershell
# Windows
.\.venv\Scripts\python.exe -m pytest -m "not local_llm"
.\.venv\Scripts\python.exe -m pip check
```

For template/render checks, set `AGORA_DATA_DIR` and `AGORA_CONFIG_DIR` to disposable directories, then run `scripts/render_smoke.py` with the virtualenv's Python. This avoids demonstration seeding touching normal local files. JavaScript syntax checks need Node.js.

The `local_llm` tests require a separately configured local model. Cloud smoke tests require local provider configuration and may cost money. Do not use student data for either. The smoke script asserts course chat context, comparison success, and exactly one approved export row with disclosure.

## Code map

- `app/routers/`: HTTP endpoints and page routes.
- `app/ai/`: providers, grading requests, privacy, and assistant behavior.
- `app/models.py`, `app/db.py`: persistence and database initialization.
- `app/review.py`, `app/release.py`: review, release, and export.
- `templates/`, `static/`: Jinja templates, JavaScript, and CSS.
- `tests/`: offline regression tests.

Keep changes focused. Reproduce a bug before fixing it and cover the corrected behavior. UI changes also need a browser walkthrough. Document schema changes and upgrade effects.

## Pull requests

Push a branch to your fork and open a pull request against the shared repository's default branch. Describe the problem, resulting behavior, and verification. Screenshots and examples must use synthetic coursework.

Never commit student work, real rosters, local databases, API keys, environment files, or machine-specific configuration. Open issues with reproduction steps instead of private data.

Agora is available under the [MIT license](LICENSE). Retain its copyright and license notice when sharing copies or modifications.
