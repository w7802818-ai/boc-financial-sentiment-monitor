#!/usr/bin/env python
"""Pre-publish safety scan for the BOC deployment repository.

Scans text files for secrets, credentials, and machine-specific absolute paths
before the repository is pushed to GitHub or the model is published.

Exit code 0 = clean, 1 = findings present.

Usage:
    python scan_for_secrets.py [directory]
"""

from __future__ import annotations

import pathlib
import re
import sys

PATTERNS = {
    "hugging_face_token": re.compile(r"hf_[A-Za-z0-9]{25,}"),
    "openai_key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "windows_user_path": re.compile(r"[A-Za-z]:\\+Users\\+[A-Za-z0-9._-]*[A-Za-z0-9]"),
    "posix_user_path": re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]*[A-Za-z0-9]/"),
    "email_address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".csv", ".json", ".example",
    ".sh", ".ps1", ".cfg", ".ini", ".yaml", ".yml", ".html", ".js",
}

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules"}


def scan(root: pathlib.Path) -> int:
    findings = 0
    scanned = 0

    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        scanned += 1
        for name, rx in PATTERNS.items():
            for m in rx.finditer(text):
                line = text[: m.start()].count("\n") + 1
                rel = path.relative_to(root)
                print(f"  !! {rel}:{line}  [{name}]  {m.group(0)!r}")
                findings += 1

    print()
    print(f"  scanned {scanned} text file(s)")
    if findings:
        print(f"  RESULT: {findings} finding(s) - review before publishing")
        return 1
    print("  RESULT: CLEAN - no secrets or machine-specific paths detected")
    return 0


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    print("=" * 68)
    print("Pre-publish safety scan")
    print(f"  directory: {root}")
    print("=" * 68)
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}")
        return 2
    return scan(root)


if __name__ == "__main__":
    sys.exit(main())
