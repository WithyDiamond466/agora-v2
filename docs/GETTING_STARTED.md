# Get started with Agora

Agora runs on your computer and opens in a browser. GitHub distributes the source code; it does not host your courses or receive uploaded coursework.

This is a development preview. Start with synthetic data and read [verification and limitations](../STATUS.md). Live cloud grading and instructor calibration remain unverified.

## Download

On the Agora GitHub repository, choose **Code → Download ZIP**, extract it into a folder you own, and open a terminal in that folder. Developers can instead fork and clone the repository using the URL under **Code**.

Install Python 3.12 first. Calling the virtualenv's Python directly avoids needing to activate it.

You can use `start-agora.bat --demo` on Windows or `bash start-agora.sh --demo` on macOS/Linux. These install dependencies on first use. The equivalent manual steps follow.

## Windows

Run these commands in PowerShell in the extracted Agora folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c requirements-lock.txt
.\.venv\Scripts\python.exe run.py --demo
```

If `py` is unavailable, install Python 3.12 with the Windows launcher and reopen PowerShell.

## macOS and Linux

Run these commands in Terminal in the extracted Agora folder:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -c requirements-lock.txt
.venv/bin/python run.py --demo
```

If `python3.12` is unavailable, install Python 3.12 using the installer or package manager for your operating system. Some Linux distributions package the `venv` module separately.

## Explore the demonstration

Open `http://127.0.0.1:8811` if the browser does not open automatically.

1. Open the PHIL 210 course and inspect its roster and assignments.
2. Open an assignment to see criterion scores and feedback.
3. Open a student page and course analytics.
4. Explore Skills and export a demonstration skill bundle.

Demonstration grades come from a deterministic mock provider. They are not assessments of essay quality. No paid AI provider is needed to explore these seeded results.

The `--demo` flag creates the demonstration course if missing and preserves an existing course. For normal use, omit the flag on later launches:

```powershell
# Windows
.\.venv\Scripts\python.exe run.py
```

```bash
# macOS / Linux
.venv/bin/python run.py
```

Stop Agora with Ctrl+C in its terminal or Quit in Settings. Closing a browser tab does not stop the server. If the port is occupied, check for an existing Agora terminal.

## Try a synthetic course

Create a course and import [sample-roster.csv](../examples/sample-roster.csv). Use the [sample rubric and essays](../examples/README.md) to create a rubric, Skill, and assignment, then upload the sample PDFs. Map it to a synthetic student before grading.

Cloud grading needs an available provider configured in Settings and accepted terms. Provider use may cost money. Default cloud privacy mode rejects images and scanned PDFs without selectable text. Extracted text does not preserve diagrams or layout. Do not disable privacy protection as a workaround for unsupported coursework.

After grading, edit scores and comments, save, then choose **Approve and next**. Release approved results and export the CSV. Incomplete results require all scored criteria before approval.

## Local records and updates

Course records and uploaded files live under `data/` by default. Git ignores this directory. Never upload it to GitHub or attach it to an issue. Keep coursework outside shared or automatically cloud-synced folders.

Follow [backup, restore, and updates](BACKUP_AND_RESTORE.md) before moving computers or installing an update. The backup includes records and uploaded files, but excludes API credentials. A restore into a new directory has been verified with a synthetic course. Downloading a ZIP does not transfer your local records.
