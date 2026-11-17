"""Build the browser snapshot directly from the checked backtest results."""

import csv
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "results"
DIST = HERE / "dist"


def rows(name):
    with (SOURCE / f"{name}.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def value(raw):
    if raw == "":
        return None
    if raw in ("True", "False"):
        return raw == "True"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def main():
    coverage = json.loads((SOURCE / "coverage.json").read_text(encoding="utf-8"))
    signals = rows("signals")
    summary = rows("summary")
    assert len(signals) == coverage["signals"]
    assert len({(row["code"], row["signal_date"]) for row in signals}) == len(signals)
    assert len(summary) == 32
    headers = list(signals[0])
    snapshot = {
        "coverage": coverage,
        "fields": headers,
        "records": [[value(row[key]) for key in headers] for row in signals],
        "summary": [{key: value(raw) for key, raw in row.items()} for row in summary],
        "bars": [[int(row["date"]), float(row["adj_close"]), int(row["volume"])]
                 for row in rows("bear_daily")],
        "quote": {key: value(raw) for key, raw in rows("bear_next_day_quote")[0].items()},
    }
    DIST.mkdir(exist_ok=True)
    payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (DIST / "data.js").write_text(f"window.snapshot = {payload};\n", encoding="utf-8")
    for name in ("signals.csv", "summary.csv"):
        shutil.copyfile(SOURCE / name, DIST / name)


if __name__ == "__main__":
    main()
