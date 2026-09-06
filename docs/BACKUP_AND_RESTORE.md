# Back up and restore coursework

Stop Agora before backing up, so file uploads and deletions cannot race the database snapshot. Backups contain student records and coursework. Keep them private and outside the repository.

Run these commands from the Agora folder. On Windows, replace `.venv/bin/python` with `.\.venv\Scripts\python.exe`.

```bash
.venv/bin/python scripts/data_backup.py backup --output ../agora-coursework.zip
```

The tool backs up `data/` by default. Use `--data-dir /path/to/data` if you changed `AGORA_DATA_DIR`. Choose a new filename each time; existing backups are never overwritten. Never save the archive inside the active data directory.

The archive includes the SQLite snapshot, submissions, knowledge documents, custom modes, and preferences. Provider credentials are excluded. An interrupted grading job becomes pending in the restored copy.

To restore, choose a new directory that does not exist yet:

```bash
.venv/bin/python scripts/data_backup.py restore ../agora-coursework.zip --destination ../agora-restored-data
```

Launch Agora with that directory:

```bash
# macOS / Linux
AGORA_DATA_DIR="../agora-restored-data" .venv/bin/python run.py
```

```powershell
# Windows PowerShell
$env:AGORA_DATA_DIR = "../agora-restored-data"
.\.venv\Scripts\python.exe run.py
```

Remove any separately configured `AGORA_DATABASE_URL` when restoring; otherwise it overrides the database inside the restored directory. Enter provider API keys again in Settings. Do not copy encryption keys between machines.

Verify the roster, assignments, uploaded files, and a released export before retiring the old installation. The source course stays untouched by backup and restore. Upgrades add missing database columns on startup; make a backup before installing a new version. Legacy released results require explicit approval before they can be exported with this build.
