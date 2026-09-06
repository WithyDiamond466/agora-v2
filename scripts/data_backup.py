#!/usr/bin/env python3
"""Back up coursework or restore it into a new local directory."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.backups import create_backup, restore_backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    backup = sub.add_parser('backup', help='Stop Agora first; create a private coursework archive')
    backup.add_argument('--data-dir', type=Path, default=Path('data'))
    backup.add_argument('--output', type=Path, required=True)
    restore = sub.add_parser('restore', help='Restore into a directory that does not exist yet')
    restore.add_argument('archive', type=Path)
    restore.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'backup':
            result = create_backup(args.data_dir, args.output)
        else:
            result = restore_backup(args.archive, args.destination)
        print(f'{args.command.capitalize()} complete: {result}')
        print('Keep this coursework private. Restored installations require provider keys to be entered again.')
        return 0
    except Exception as exc:
        print(f'{args.command.capitalize()} failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
