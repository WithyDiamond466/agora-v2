# Agora

Agora is a grading assistant that runs on a professor's computer. Organize courses and rosters, grade against rubrics, correct feedback, approve results, and export them. Share reusable grading instructions and reference documents as `.agoraskill` bundles.

**Development preview.** The offline workflow is tested with synthetic coursework. Live cloud grading and instructor calibration remain unverified. See [verification and limitations](STATUS.md) before using real coursework.

- [Download and run on Windows, macOS, or Linux](docs/GETTING_STARTED.md)
- [Fork and contribute](CONTRIBUTING.md)
- [Back up, restore, and update](docs/BACKUP_AND_RESTORE.md)

## Start

Install Python 3.12, download and extract this repository, then run the launcher from its folder:

```powershell
# Windows PowerShell
.\start-agora.bat --demo
```

```bash
# macOS or Linux
bash start-agora.sh --demo
```

The launcher installs dependencies on first use and opens `http://127.0.0.1:8811`. The demonstration contains synthetic students and mock grades. Reopening it preserves your changes. Omit `--demo` to start without adding the demonstration course. Stop Agora with Ctrl+C in its terminal or Quit in Settings.

## Grade a course

1. Create a course and import a CSV roster.
2. Create a Skill, then create and open an assignment. In Assignment setup, select the Skill and create a rubric or choose an existing one.
3. Upload submissions and check each student mapping. Configure a provider in Settings and accept the application's terms before cloud grading.
4. Grade the submissions. Open each result, correct criterion scores or comments, save, and choose **Approve and next**.
5. Release approved results, then export CSV feedback. Every exported row includes an AI disclosure.

Opening a result records that it was seen. Approval is a separate action. Incomplete, withheld, and unapproved results cannot be exported. Editing or regrading removes earlier approval and release. Selective grading leaves some scores for the professor; feedback mode produces comments without scores.

[Sample materials](examples/README.md) provide a small synthetic course to try. The course assistant, analytics, and student insight pages use local course records. Review any generated conclusions against the underlying work.

## Data and providers

Records and uploads stay in the local `data/` folder until you choose an action that sends content to a configured AI provider. Anthropic and OpenAI require API access and may charge for requests. Subscription access to a chat application is not an API key. Settings stores API credentials encrypted using a key in the local configuration directory. Backups exclude credentials.

The default cloud privacy mode substitutes detected names and identifiers before sending extracted text. Detection can miss identifying details; it is not a guarantee of anonymity. Without the optional local privacy model, deterministic matching still runs and records a warning. Review the scan report and your institution's requirements before sending coursework. `warn` and `off` modes allow identifying content to leave the computer.

Default cloud mode requires PDFs with extractable text and rejects images or scanned PDFs without it. Text extraction loses diagrams and layout. The local provider can use a separately installed OpenAI-compatible model server; its setup is an advanced option, not part of the standard installation. Any server you configure receives the request content.

Agora is a single-user localhost application. It has no user accounts or shared-server access controls. GitHub distributes the code; it does not store your courses.

## Development

Python, FastAPI, SQLite, Jinja templates, and vanilla JavaScript. Install with `requirements.txt` and `requirements-lock.txt`; Python 3.12 is the tested version.

```bash
.venv/bin/python -m pytest -m "not local_llm"
.venv/bin/python -m pip check
.venv/bin/python scripts/live_smoke.py --provider mock --compare
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for Windows commands, isolated render checks, and pull requests. [SPEC.md](SPEC.md) describes the architecture. Open issues with synthetic reproduction steps. Never attach student records or API keys.

Licensed under [MIT](LICENSE).
