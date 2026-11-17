"""Screen TDX day bars for a selloff, volume rally, dry pullback, and breakout."""

import argparse
import re
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HORIZONS = (5, 10, 20, 30)
BEAR = "sz.002959"
ST_LIMIT_CHANGE = 20260706
BAR_DTYPE = np.dtype([
    ("date", "<i4"), ("open", "<i4"), ("high", "<i4"), ("low", "<i4"),
    ("close", "<i4"), ("amount", "<f4"), ("volume", "<u4"), ("reserved", "<i4"),
])
STOCK = re.compile(r"(?:sh(?:60[0135]|688)|sz(?:00[0123]|30[01]))\d{3}")


@dataclass(frozen=True)
class Rules:
    drawdown: float = 0.20
    rally_return: float = 0.14
    wash_min_drop: float = 0.05
    wash_max_drop: float = 0.25
    wash_min_retrace: float = 0.25
    wash_max_retrace: float = 0.95
    wash_volume: float = 0.75
    breakout_return: float = 0.06
    wick_to_body: float = 1 / 3


def end_of_market(index, cutoff):
    dates = index["date"]
    dates = dates[dates <= cutoff]
    if len(dates) == 0 or not np.all(np.diff(dates) > 0):
        raise ValueError("Invalid SSE daily trading calendar")
    return dates


def market_files(source, stack):
    if source.is_file():
        archive = stack.enter_context(ZipFile(source))
        members = {Path(name.replace("\\", "/")).stem: name for name in archive.namelist()
                   if name.lower().endswith(".day")}
        return members, lambda stem: archive.read(members[stem])
    members = {file.stem: file for file in source.glob("*/lday/*.day")}
    return members, lambda stem: members[stem].read_bytes()


def current_names(tdx_root):
    names = {}
    for market in ("sh", "sz"):
        path = tdx_root / "T0002" / "hq_cache" / f"{market}s.tnf"
        if not path.exists():
            continue
        data = path.read_bytes()
        for offset in range(50, len(data) - 359, 360):
            code = data[offset:offset + 6].decode("ascii", errors="ignore")
            name = data[offset + 31:offset + 70].split(b"\0", 1)[0].decode("gb18030", errors="replace")
            if code.isdigit() and name:
                names[f"{market}.{code}"] = name.strip()
    return names


def adjusted_prices(bars, actions, supplied_preclose=None):
    """Apply each ex-rights factor forward, using only the previous raw close."""
    close = bars["close"].astype(float)
    reference = np.r_[close[0], close[:-1]]
    for day, dividend, rights_price, bonus, rights_ratio in actions:
        row = np.searchsorted(bars["date"], int(day))
        if 0 < row < len(bars):
            reference[row] = (
                reference[row] - 10 * dividend + 10 * rights_ratio * rights_price
            ) / (1 + (bonus + rights_ratio) / 10)
    for day, value in (supplied_preclose or {}).items():
        row = np.searchsorted(bars["date"], day)
        if row < len(bars) and bars["date"][row] == day:
            reference[row] = value
    if np.any(reference <= 0):
        raise ValueError("Nonpositive ex-rights reference price")
    # Prices live in integer cents. A reference quoted to cents determines the limits.
    preclose = np.floor(reference + 0.50000001).astype(int)
    step = np.r_[1.0, close[:-1] / reference[1:]]
    factors = np.cumprod(step)
    return preclose, close * factors, factors


def detect(bars, preclose, close, rules, start, *, deduplicate=True):
    """Yield signal indices and metrics. No bar after the signal enters detection."""
    dates, opens, highs = (bars[key] for key in ("date", "open", "high"))
    volume = bars["volume"].astype(float)
    body = bars["close"] - opens
    upper_wick = highs - bars["close"]
    day_return = (bars["close"] - preclose) / preclose
    candidates = np.flatnonzero(
        (dates >= start) & (day_return > rules.breakout_return)
        & (body > 0) & (upper_wick <= body * rules.wick_to_body + 1e-9)
    )
    volumes = np.r_[0.0, volume.cumsum()]
    used_peaks = set()
    for signal in candidates:
        if signal < 119:
            continue
        recent_volume_5 = (volumes[signal] - volumes[signal - 5]) / 5
        recent_volume_10 = (volumes[signal] - volumes[signal - 10]) / 10
        if volume[signal] <= max(recent_volume_5, recent_volume_10):
            continue
        # The highest close in the 25 preceding bars must precede a 5–24 bar pullback.
        peak = signal - 25 + np.argmax(close[signal - 25:signal])
        wash_days = signal - peak - 1
        if wash_days < 5:
            continue
        wash_drop = 1 - close[signal - 1] / close[peak]
        if not rules.wash_min_drop <= wash_drop <= rules.wash_max_drop:
            continue
        floor = peak - 22 + np.argmin(close[peak - 22:peak - 2])
        rally_return = close[peak] / close[floor] - 1
        if rally_return < rules.rally_return:
            continue
        retracement = (close[peak] - close[signal - 1]) / (close[peak] - close[floor])
        if not rules.wash_min_retrace <= retracement <= rules.wash_max_retrace:
            continue
        if close[peak + 1:signal].min() < close[floor]:
            continue
        rally_avg = (volumes[peak + 1] - volumes[floor + 1]) / (peak - floor)
        base_avg_5 = (volumes[floor + 1] - volumes[floor - 4]) / 5
        base_avg_10 = (volumes[floor + 1] - volumes[floor - 9]) / 10
        wash_avg = (volumes[signal] - volumes[peak + 1]) / wash_days
        wash_volume = wash_avg / rally_avg
        if rally_avg <= max(base_avg_5, base_avg_10) or wash_volume > rules.wash_volume:
            continue
        left = max(0, floor - 150)
        pit_high = left + np.argmax(close[left:floor - 9])
        drawdown = 1 - close[floor] / close[pit_high]
        if drawdown < rules.drawdown or close[floor] > 1.12 * close[floor - 59:floor + 1].min():
            continue
        if deduplicate:
            if peak in used_peaks:
                continue
            used_peaks.add(peak)
        yield signal, {
            "pit_high_date": int(dates[pit_high]), "floor_date": int(dates[floor]),
            "rally_peak_date": int(dates[peak]), "wash_end_date": int(dates[signal - 1]),
            "pit_drawdown": drawdown, "rally_return": rally_return,
            "rally_volume_ratio": rally_avg / ((volumes[floor + 1] - volumes[floor - 19]) / 20),
            "rally_volume_ratio_5": rally_avg / base_avg_5,
            "rally_volume_ratio_10": rally_avg / base_avg_10,
            "rally_days": int(peak - floor),
            "wash_drop": wash_drop, "wash_retracement": retracement,
            "wash_volume_ratio": wash_volume, "wash_days": int(wash_days),
            "signal_return": day_return[signal],
            "signal_volume_ratio_5": volume[signal] / recent_volume_5,
            "signal_volume_ratio": volume[signal] / recent_volume_10,
            "upper_wick_body_ratio": upper_wick[signal] / body[signal],
            "signal_open": opens[signal] / 100, "signal_close": bars["close"][signal] / 100,
        }


def limit_price(preclose, rate, up=True):
    change = (preclose * (100 + rate if up else 100 - rate) + 50) // 100
    return max(change, preclose + 1) if up else min(change, preclose - 1)


def price_limits(code, bar, preclose, st_status, st_needed, for_exit=False):
    day = int(bar["date"])
    if code[3:6] in {"300", "301", "688"}:
        rate = 20
    else:
        rate = 10
        if day < ST_LIMIT_CHANGE:
            cap = limit_price(preclose, 5)
            floor = limit_price(preclose, 5, up=False)
            possible_st = (
                bar["high"] == bar["low"] == floor if for_exit else
                bar["high"] <= cap and bar["low"] >= floor
                and cap in (bar["open"], bar["high"], bar["close"])
            )
            if possible_st:
                key = code, day
                if key not in st_status:
                    st_needed.add(key)
                    return None, None
                if st_status[key]:
                    rate = 5
    return limit_price(preclose, rate), limit_price(preclose, rate, up=False)


def trade(code, bars, preclose, close, factors, signal, calendar, st_status, st_needed):
    dates = bars["date"]
    t = np.searchsorted(calendar, dates[signal])
    if t + 1 >= len(calendar):
        return {"entry_group": "pending"}
    entry_date = calendar[t + 1]
    i = np.searchsorted(dates, entry_date)
    record = {"entry_date": int(entry_date), "entry_group": "no_bar"}
    if i == len(dates) or dates[i] != entry_date:
        return record
    bar = bars[i]
    cap, _ = price_limits(code, bar, preclose[i], st_status, st_needed)
    group = "unknown_limit"
    if cap is not None:
        group = "open_not_at_limit"
        if bar["open"] == cap:
            group = "one_price_limit" if bar["low"] == cap else "opens_at_limit"
    entry_price = bar["open"] * factors[i]
    record.update(
        entry_group=group, entry_open=bar["open"] / 100, entry_close=bar["close"] / 100,
        entry_gap=bar["open"] / preclose[i] - 1, entry_preclose=preclose[i] / 100,
        entry_limit_price=cap / 100 if cap is not None else np.nan,
        entry_hits_limit=bool(bar["high"] == cap) if cap is not None else None,
        entry_closes_at_limit=bool(bar["close"] == cap) if cap is not None else None,
    )
    for horizon in HORIZONS:
        if t + horizon >= len(calendar):
            continue
        day = calendar[t + horizon]
        record[f"exit_date_{horizon}"] = int(day)
        j = np.searchsorted(dates, day)
        if j == len(dates) or dates[j] != day:
            continue
        _, floor = price_limits(code, bars[j], preclose[j], st_status, st_needed, for_exit=True)
        record[f"exit_close_{horizon}"] = bars[j]["close"] / 100
        record[f"return_{horizon}"] = close[j] / entry_price - 1
        record[f"exit_one_price_down_{horizon}"] = (
            bool(bars["high"][j] == bars["low"][j] == floor) if floor is not None else None
        )
    return record


def load_actions():
    frame = pd.read_csv(ROOT / "data" / "tdx_actions.csv", dtype={"code": str})
    return {
        f"{'sz' if market == 0 else 'sh'}.{code}":
        group.sort_values("datetime").iloc[:, 2:].drop(columns="category").to_numpy()
        for (market, code), group in frame[frame.market.isin([0, 1])].groupby(["market", "code"])
    }


def updated_bars(after, end, st_status):
    days, pieces = [], []
    for path in sorted((ROOT / "data" / "daily").glob("*.csv")):
        date = int(path.stem.replace("-", ""))
        if after < date <= end:
            frame = pd.read_csv(path, usecols=[
                "code", "open", "high", "low", "close", "volume", "amount",
                "preclose", "isST", "tradestatus",
            ])
            frame = frame[frame.tradestatus.eq(1) & frame.volume.gt(0)].dropna(subset=["preclose"])
            frame = frame.loc[frame.code.str.replace(".", "", regex=False).str.fullmatch(STOCK, na=False)].copy()
            if len(frame) < 4000 or frame.code.duplicated().any():
                raise ValueError(f"Incomplete or duplicated daily update {path}")
            frame["date"] = date
            days.append(date)
            pieces.append(frame)
    by_code = {}
    if pieces:
        for code, frame in pd.concat(pieces, ignore_index=True).groupby("code"):
            bars = np.zeros(len(frame), dtype=BAR_DTYPE)
            for field in ("date", "amount", "volume"):
                bars[field] = frame[field]
            for field in ("open", "high", "low", "close"):
                bars[field] = np.rint(frame[field] * 100)
            by_code[code] = (bars, dict(zip(frame.date, frame.preclose * 100)))
            st_status.update({(code, int(row.date)): int(row.isST)
                              for row in frame.dropna(subset=["isST"]).itertuples(index=False)})
    return np.asarray(days, dtype=int), by_code


def summary(frame):
    open_limit = frame.entry_group.isin(["opens_at_limit", "one_price_limit"])
    normal = frame.entry_group == "open_not_at_limit"
    groups = {
        "开盘未涨停": normal,
        "开盘涨停（理论）": open_limit,
        "开盘涨停、盘中开板（理论）": frame.entry_group == "opens_at_limit",
        "开盘涨停、全天一字（理论）": frame.entry_group == "one_price_limit",
        "次日收盘涨停（含假设成交）": frame.entry_closes_at_limit.eq(True),
        "次日收盘未涨停（含假设成交）": frame.entry_closes_at_limit.eq(False),
        "开盘未涨停、收盘涨停": normal & frame.entry_closes_at_limit.eq(True),
        "开盘未涨停、收盘未涨停": normal & frame.entry_closes_at_limit.eq(False),
    }
    rows = []
    for label, mask in groups.items():
        group = frame[mask]
        for horizon in HORIZONS:
            returns = group[f"return_{horizon}"].dropna()
            rows.append({
                "组别": label, "持有交易日": horizon, "有开盘价信号数": len(group),
                "收益样本数": len(returns), "平均收益": returns.mean(), "中位收益": returns.median(),
                "胜率": (returns > 0).mean(), "25分位": returns.quantile(.25),
                "75分位": returns.quantile(.75), "最大收益": returns.max(), "最小收益": returns.min(),
                "到期一字跌停数": group[f"exit_one_price_down_{horizon}"].eq(True).sum(),
            })
    return pd.DataFrame(rows)


def run(source, tdx_root, start, end, out):
    out.mkdir(parents=True, exist_ok=True)
    actions = load_actions()
    names = current_names(tdx_root)
    st_file = ROOT / "data" / "st_checks.csv"
    st_status = {}
    if st_file.exists():
        for row in pd.read_csv(st_file).dropna().itertuples(index=False):
            st_status[(row.code, int(row.date))] = int(row.isST)

    rows, st_needed, latest_counts = [], set(), Counter()
    elapsed = Counter()
    with ExitStack() as stack:
        files, read = market_files(source, stack)
        calendar = end_of_market(np.frombuffer(read("sh000001"), BAR_DTYPE), end)
        base_end = int(calendar[-1])
        update_dates, updates = updated_bars(base_end, end, st_status)
        calendar = np.r_[calendar, update_dates]
        for stem in sorted(files):
            if not STOCK.fullmatch(stem):
                continue
            code = f"{stem[:2]}.{stem[2:]}"
            bars = np.frombuffer(read(stem), BAR_DTYPE)
            dates = bars["date"]
            bars = bars[(dates >= 20210101) & (dates <= base_end)
                        & (bars["volume"] > 0) & (bars["low"] > 0)]
            addition, preclose_update = updates.get(code, (None, {}))
            if addition is not None:
                bars = np.r_[bars, addition]
            if not len(bars):
                continue
            # A few proprietary series have non-market dates. Only exchange sessions count.
            bars = bars[np.isin(bars["date"], calendar)]
            if not len(bars) or not np.all(np.diff(bars["date"]) > 0):
                elapsed["bad_or_missing_series"] += 1
                continue
            elapsed["securities"] += 1
            elapsed["bars"] += len(bars)
            latest_counts[int(bars["date"][-1])] += 1
            preclose, close, factors = adjusted_prices(bars, actions.get(code, ()), preclose_update)
            if code == BEAR:
                chart = pd.DataFrame(bars[bars["date"] >= 20260301])
                chart["adj_close"] = close[-len(chart):] / factors[-1] / 100
                chart.to_csv(out / "bear_daily.csv", index=False)
            for signal, metrics in detect(bars, preclose, close, Rules(), start):
                rows.append({
                    "code": code, "name": names.get(code, ""), "signal_date": int(bars["date"][signal]),
                    **metrics, **trade(code, bars, preclose, close, factors, signal, calendar, st_status, st_needed),
                })

    if not rows:
        raise ValueError("No matched signals in the supplied date range")
    frame = pd.DataFrame(rows).sort_values(["signal_date", "code"], ascending=[False, True])
    for horizon in HORIZONS:
        for field in ("exit_date", "exit_close", "return", "exit_one_price_down"):
            column = f"{field}_{horizon}"
            if column not in frame:
                frame[column] = np.nan
    for key in ("entry_date", *(f"exit_date_{h}" for h in HORIZONS)):
        frame[key] = frame[key].astype("Int64")
    frame.to_csv(out / "signals.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    summary(frame).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    frame[frame.signal_date.isin(calendar[-5:])].to_csv(
        out / "latest_5_sessions.csv", index=False, encoding="utf-8-sig", float_format="%.8f"
    )
    frame[frame.code == BEAR].to_csv(out / "bear_signals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sorted(st_needed), columns=["code", "date"]).to_csv(out / "st_checks_needed.csv", index=False)
    state = {
        **dict(elapsed), "requested_start": start, "calendar_end": int(calendar[-1]),
        "tdx_end": base_end, "update_dates": update_dates.tolist(),
        "securities_with_bar_at_end": latest_counts[int(calendar[-1])],
        "first_signal": int(frame.signal_date.min()), "last_signal": int(frame.signal_date.max()),
        "signals": len(frame), "signal_codes": int(frame.code.nunique()),
        "entry_groups": {key: int(value) for key, value in frame.entry_group.value_counts().items()},
        "st_checks_needed": len(st_needed),
    }
    pd.Series(state).to_json(out / "coverage.json", indent=2, force_ascii=False)
    print(state, flush=True)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="TDX vipdoc directory or daily archive ZIP")
    parser.add_argument("--tdx", type=Path, default=Path("D:/tdx"))
    parser.add_argument("--start", type=int, default=20220101)
    parser.add_argument("--end", type=int, default=20261116)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run(args.source or args.tdx / "vipdoc", args.tdx, args.start, args.end, args.output)


if __name__ == "__main__":
    main()
