"""Prepare time-ordered research tables after raw downloads.

This entry point is intentionally conservative: raw schemas differ across
source releases, so transformations should be implemented after the downloaded
files and feature definitions have been inspected.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print("Inspect raw files in", RAW)
    print("Write time-ordered modeling tables to", PROCESSED)
    print("Required outputs: Taiwan controlled-validation table and LendingClub temporal table.")


if __name__ == "__main__":
    main()
