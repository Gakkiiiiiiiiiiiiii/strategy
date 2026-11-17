"""Export the OHLCV windows used by the per-signal charts."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import numpy as np

from export_data import DIST, HERE, SOURCE, rows

sys.path.insert(0, str(SOURCE.parent))
import backtest as bt


SHARDS = 32
PRICE_FIELDS = ("open", "high", "low", "close")


def window(bars, factors, calendar, signal, followup):
    dates = bars["date"]
    day = int(signal["signal_date"])
    index = np.searchsorted(dates, day)
    assert index < len(dates) and dates[index] == day, (signal["code"], day)
    calendar_index = int(np.searchsorted(calendar, day))
    limit = calendar[min(calendar_index + 30, len(calendar) - 1)]
    stop = np.searchsorted(dates, limit, side="right")
    pit = np.searchsorted(dates, int(signal["pit_high_date"]))
    start = min(int(pit), max(0, int(index) - 60))
    segment = bars[start:stop]
    scale = factors[start:stop] / factors[index]
    prices = [np.rint(segment[field] * scale).astype(int) for field in PRICE_FIELDS]
    payload = np.column_stack([segment["date"], *prices, segment["volume"]]).tolist()
    if signal["code"] == bt.BEAR and day == 20261116 and followup["date"] > payload[-1][0]:
        payload.append([followup["date"], *[round(followup[field] * 100) for field in PRICE_FIELDS],
                        followup["volume_shares"]])
    assert all(bar[2] >= max(bar[1], bar[4]) and bar[3] <= min(bar[1], bar[4]) for bar in payload)
    assert payload[index - start][1] == round(float(signal["signal_open"]) * 100)
    assert payload[index - start][4] == round(float(signal["signal_close"]) * 100)
    return payload


def export(tdx):
    coverage = json.loads((SOURCE / "coverage.json").read_text(encoding="utf-8"))
    followup = json.loads((HERE / "bear-followup.json").read_text(encoding="utf-8"))
    matches = defaultdict(list)
    for signal in rows("signals"):
        matches[signal["code"]].append(signal)
    actions = bt.load_actions()
    shards = [dict() for _ in range(SHARDS)]
    with ExitStack() as stack:
        _, read = bt.market_files(tdx / "vipdoc", stack)
        base_end = coverage["tdx_end"]
        calendar = bt.end_of_market(np.frombuffer(read("sh000001"), bt.BAR_DTYPE), base_end)
        new_dates, updates = bt.updated_bars(base_end, coverage["calendar_end"], {})
        assert new_dates.tolist() == coverage["update_dates"]
        calendar = np.r_[calendar, new_dates]
        for code, signals in matches.items():
            bars = np.frombuffer(read(code.replace(".", "")), bt.BAR_DTYPE)
            dates = bars["date"]
            bars = bars[(dates >= 20210101) & (dates <= base_end)
                        & (bars["volume"] > 0) & (bars["low"] > 0)]
            addition, preclose = updates.get(code, (None, {}))
            if addition is not None:
                bars = np.r_[bars, addition]
            bars = bars[np.isin(bars["date"], calendar)]
            _, _, factors = bt.adjusted_prices(bars, actions.get(code, ()), preclose)
            shard = shards[int(code.split(".")[1]) % SHARDS]
            for signal in signals:
                key = f"{code}-{signal['signal_date']}"
                shard[key] = window(bars, factors, calendar, signal, followup)
    directory = DIST / "chart-data"
    directory.mkdir(parents=True, exist_ok=True)
    assert sum(map(len, shards)) == coverage["signals"]
    for index, shard in enumerate(shards):
        packed = json.dumps(shard, separators=(",", ":"), ensure_ascii=False)
        (directory / f"{index}.js").write_text(
            f"window.candleRanges[{index}] = {packed};\n", encoding="utf-8"
        )
    print(f"Exported {coverage['signals']} OHLCV windows across {SHARDS} shards")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tdx", type=Path, default=Path("D:/tdx"))
    export(parser.parse_args().tdx)
