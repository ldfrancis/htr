"""Writing result CSVs."""

import csv


def write_csv(path, rows, fieldnames, append=False):
    """Write dict rows to a CSV, creating parent dirs. When appending, the header is written only for a new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (append and path.exists())
    with path.open("a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
