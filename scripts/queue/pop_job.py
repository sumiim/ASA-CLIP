#!/usr/bin/env python3
"""Atomically pop the next job id from pending.txt."""
from __future__ import annotations

import fcntl
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PENDING = os.path.join(ROOT, "pending.txt")


def main():
    if not os.path.exists(PENDING):
        sys.exit(1)
    with open(PENDING, "r+", encoding="utf-8") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        lines = file.readlines()
        kept = []
        job = None
        for line in lines:
            if job is None:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    job = stripped
                    continue
            kept.append(line)
        file.seek(0)
        file.writelines(kept)
        file.truncate()
    if not job:
        sys.exit(1)
    print(job)


if __name__ == "__main__":
    main()
