#!/usr/bin/env python3
"""Standalone entrypoint for the Preview Ladder CLI prototype.

Run from the isolated workdir:
    PYTHONPATH=. python prototypes/preview_ladder_cli_prototype.py generate-fixtures --out /tmp/pld-fixtures
"""

from __future__ import annotations

from preview_ladder_dit.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
