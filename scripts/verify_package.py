"""Check file integrity, Python syntax and accidental private text markers."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Match absolute host paths, not repository-relative paths or /path/to placeholders.
PRIVATE = re.compile(r'[A-Za-z]:[\\/]Users[\\/][^\\/]+[\\/]|(?<![A-Za-z0-9_./-])/(?:data|home)/(?!path/)[A-Za-z][^/\s]+/', re.I)
SECRETS = re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}|\bsk-[A-Za-z0-9]{25,}')


def main():
    errors, python_count = [], 0
    with (ROOT / 'metadata/FILE_MANIFEST.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = (ROOT / row['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            errors.append({'path': row['path'], 'kind': 'missing_or_outside_root'}); continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            errors.append({'path': row['path'], 'kind': 'sha256_mismatch'})
        if path.suffix.lower() in {'.py', '.sh', '.csv', '.json', '.md', '.txt', '.tex'}:
            text = path.read_text(encoding='utf-8-sig')
            # This checker necessarily contains the marker expressions.
            if path.name != 'verify_package.py' and (PRIVATE.search(text) or SECRETS.search(text)):
                errors.append({'path': row['path'], 'kind': 'private_marker_or_token'})
            if path.suffix == '.py':
                try:
                    ast.parse(text)
                    python_count += 1
                except SyntaxError:
                    errors.append({'path': row['path'], 'kind': 'python_syntax'})
    result = {'status': 'PASS' if not errors else 'FAIL', 'manifest_files': len(rows),
              'python_files': python_count, 'errors': errors,
              'scope': 'integrity_and_static_privacy_not_upstream_reproduction'}
    output = ROOT / 'outputs/package_verification.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
