"""Fail if any tracked file matches a term from the local denylist.

The denylist lives outside the repository (default ~/.portfolio/denylist.txt,
override with --denylist or DENYLIST_PATH). One term per line; blank lines and
lines starting with # are ignored. Each term is tried as a case-insensitive
regular expression and falls back to a literal match if it does not compile.

Matches are reported by file, line and term number only, never by the term
itself, so the output is safe to show in logs.

Generic patterns (long digit runs, ticket-like keys, internal-looking URLs) are
reported as warnings and do not fail the scan.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

GENERIC = {
    "digit run of 8+ outside synthetic id formats": re.compile(r"(?<![\w.])\d{8,}(?![\w.])"),
    "ticket-like key": re.compile(r"\b[A-Z]{2,5}-\d{2,6}\b"),
    "internal-looking url": re.compile(r"https?://[^\s\"')]*(\.corp|jira|confluence)", re.IGNORECASE),
}
SYNTHETIC_ID = re.compile(r"\b(acc|ent)_\d+\b")


def load_terms(path: Path) -> list[re.Pattern]:
    terms = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            terms.append(re.compile(line, re.IGNORECASE))
        except re.error:
            terms.append(re.compile(re.escape(line), re.IGNORECASE))
    return terms


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    return [root / p for p in out.splitlines() if p]


def main() -> int:
    parser = argparse.ArgumentParser()
    default = os.environ.get("DENYLIST_PATH", str(Path.home() / ".portfolio" / "denylist.txt"))
    parser.add_argument("--denylist", type=Path, default=Path(default))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    if not args.denylist.is_file():
        print("denylist not found at the configured path; refusing to pass without it", file=sys.stderr)
        return 2
    terms = load_terms(args.denylist)
    hits = warnings = 0
    for path in tracked_files(args.root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        rel = path.relative_to(args.root).as_posix()
        names = [rel] + text.splitlines()
        for lineno, line in enumerate(names):
            where = f"{rel} (file name)" if lineno == 0 else f"{rel}:{lineno}"
            for i, term in enumerate(terms, start=1):
                if term.search(line):
                    print(f"DENYLIST {where}: matches term #{i}")
                    hits += 1
            if lineno == 0:
                continue
            cleaned = SYNTHETIC_ID.sub("", line)
            for label, pattern in GENERIC.items():
                if pattern.search(cleaned):
                    print(f"warning {where}: {label}")
                    warnings += 1
    print(f"denylist scan: {len(terms)} terms, {hits} hit(s), {warnings} warning(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
