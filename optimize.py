"""Search price/volume setup and exits, then evaluate locked choices on 2026."""

import argparse
from contextlib import ExitStack
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import backtest as bt


PERIODS = ("2022–2024", "2025", "2026")
COST = 0.002
STRESS_COST = 0.0035
HOLDS = (5, 10, 20, 30)
EXITS = [
    (days, take, stop)
    for days in HOLDS
    for take, stop in product((0., .02, .035, .05, .08, .12, .18, .25), (0., .035, .06, .10))
    if (take, stop) not in {(.18, .10), (.25, .10)}
]
BASE = (0.20, 0.14, 10.0, 0.25, 0.75, 0, 0.06, 10.0, 10.0)
ENTRY_FIELDS = ("pit_min", "rally_min", "rally_max", "wash_max", "wash_volume_max",
                "peak_rule", "signal_min", "signal_max", "entry_gap_max")
ENTRIES = [
    (pit, rally_min, rally_max, wash, volume, peak, move_min, move_max, gap)
    for pit, (rally_min, rally_max), wash, volume, peak, (move_min, move_max), gap in product(
        (.20, .28, .36), ((.14, .32), (.14, .50), (.14, 10.), (.20, 10.)),
        (.12, .18, .25), (.55, .65, .75), (0, 1, 2),
        ((.06, 10.), (.06, .09), (.08, 10.)), (0., .025, 10.)
    )
]


@njit
def simulate(bars, offsets, locked, entry, exits):
    returns = np.full(len(exits), np.nan)
    realized = np.zeros(len(exits), np.int32)
    for k, (hold, take, stop) in enumerate(exits):
        target = entry * (1 + take)
        stop_level = entry * (1 - stop)
        for i, offset in enumerate(offsets):
            if offset > hold + 15:
                break
            if locked[i]:
                continue
            o, h, low, c, factor = bars[i]
            if offset > hold:
                price = o
            else:
                take_raw = np.ceil(target / factor - 1e-9)
                stop_raw = np.floor(stop_level / factor + 1e-9)
                # A gap clears the trigger at the open. On an ambiguous intraday
                # bar with both thresholds touched, assume the stop executed first.
                if stop > 0 and o <= stop_raw:
                    price = o
                elif take > 0 and o >= take_raw:
                    price = take_raw
                elif stop > 0 and low <= stop_raw:
                    price = stop_raw
                elif take > 0 and h >= take_raw:
                    price = take_raw
                elif offset == hold:
                    price = c
                else:
                    continue
            returns[k] = price * factor / entry - 1
            realized[k] = offset
            break
    return returns, realized


def period(date):
    return 0 if date < 20250101 else 1 if date < 20260101 else 2


def test_events(tdx, end, *, isolate_years=False):
    actions = bt.load_actions()
    st_path = bt.ROOT / "data" / "st_checks.csv"
    st = {(r.code, r.date): r.isST for r in pd.read_csv(st_path).itertuples(index=False)}
    # Enclose the whole grid in a single looser scan, preserving the same anchors.
    envelope = replace(bt.Rules(), wash_min_drop=.04)
    out, returns, date_rows, planned_rows, names, peak_ids = [], [], [], [], [], {}
    uncertain_entries = 0
    exit_matrix = np.asarray(EXITS)
    holds = exit_matrix[:, 0].astype(int)
    with ExitStack() as stack:
        files, read = bt.market_files(tdx / "vipdoc", stack)
        calendar = bt.end_of_market(np.frombuffer(read("sh000001"), bt.BAR_DTYPE), end)
        base_end = int(calendar[-1])
        extra_dates, additions = bt.updated_bars(base_end, end, st)
        calendar = np.r_[calendar, extra_dates]
        for stem in sorted(files):
            if not bt.STOCK.fullmatch(stem):
                continue
            code = stem[:2] + "." + stem[2:]
            bars = np.frombuffer(read(stem), bt.BAR_DTYPE)
            dates = bars["date"]
            bars = bars[(dates >= 20210101) & (dates <= base_end)
                        & (bars["volume"] > 0) & (bars["low"] > 0)]
            extra, previous = additions.get(code, (None, {}))
            if extra is not None:
                bars = np.r_[bars, extra]
            bars = bars[np.isin(bars["date"], calendar)]
            if len(bars) < 120 or np.any(np.diff(bars["date"]) <= 0):
                continue
            prec, close, factor = bt.adjusted_prices(bars, actions.get(code, ()), previous)
            dates = bars["date"]
            market_indices = np.searchsorted(calendar, dates)
            growth = code[3:6] in {"300", "301", "688"}
            for i, metrics in bt.detect(bars, prec, close, envelope, 20220101, deduplicate=False):
                signal_index = int(market_indices[i])
                peak = int(np.searchsorted(dates, metrics["rally_peak_date"]))
                floor = int(np.searchsorted(dates, metrics["floor_date"]))
                is_true_peak = bool(close[floor + 1:i].max() <= close[peak])
                near_peak = is_true_peak and close[i] >= .95 * close[peak]
                peak_rule = 2 if near_peak else int(is_true_peak)
                tags = (metrics["pit_drawdown"], metrics["rally_return"], metrics["wash_drop"],
                        metrics["wash_volume_ratio"], peak_rule, metrics["signal_return"])
                key = code, int(dates[peak])
                group = peak_ids.setdefault(key, len(peak_ids))
                result = np.full(len(EXITS), np.nan)
                result_days = np.zeros(len(EXITS), dtype=np.int32)
                entry_gap, entry_day, entry_open = 100., 0, np.nan
                if signal_index + 1 < len(calendar):
                    entry_day = int(calendar[signal_index + 1])
                    j = int(np.searchsorted(dates, entry_day))
                    if j < len(bars) and dates[j] == entry_day:
                        o = int(bars["open"][j])
                        limit = bt.limit_price(int(prec[j]), 20 if growth else 10)
                        if not growth and entry_day < bt.ST_LIMIT_CHANGE:
                            st_limit = bt.limit_price(int(prec[j]), 5)
                            if st.get((code, entry_day), 0):
                                limit = st_limit
                            elif (code, entry_day) not in st and o == st_limit:
                                limit = st_limit
                                uncertain_entries += 1
                        if o < limit:
                            entry_gap = o / prec[j] - 1
                            entry_open = o / 100
                            last = np.searchsorted(market_indices, signal_index + 45, side="right")
                            future = bars[j + 1:last]
                            if len(future):
                                local_close = np.column_stack(
                                    [future[name] for name in ("open", "high", "low", "close")]
                                    + [factor[j + 1:last]]
                                )
                                prev = prec[j + 1:last].astype(np.int64)
                                rate = 20 if growth else 10
                                down = np.minimum((prev * (100 - rate) + 50) // 100, prev - 1)
                                locked = (future["high"] == future["low"]) & (future["low"] <= down)
                                # Ambiguous one-price -5% rows are treated as blocked unless
                                # the reference status is known to be ordinary.
                                if not growth:
                                    five = np.minimum((prev * 95 + 50) // 100, prev - 1)
                                    possible = np.flatnonzero(
                                        (future["high"] == future["low"]) & (future["low"] == five)
                                        & (future["date"] < bt.ST_LIMIT_CHANGE)
                                    )
                                    for k in possible:
                                        locked[k] = st.get((code, int(future["date"][k])), 1) == 1
                                off = market_indices[j + 1:last] - signal_index
                                result, result_days = simulate(local_close, off, locked, o * factor[j], exit_matrix)
                sell_dates = np.zeros(len(EXITS), np.int32)
                available = result_days > 0
                sell_dates[available] = calendar[signal_index + result_days[available]]
                planned_dates = np.zeros(len(EXITS), np.int32)
                in_range = signal_index + holds < len(calendar)
                planned_dates[in_range] = calendar[signal_index + holds[in_range]]
                out.append((*tags, entry_gap, group))
                returns.append(result)
                date_rows.append(sell_dates)
                planned_rows.append(planned_dates)
                names.append((code, int(dates[i]), entry_day, entry_open, int(dates[peak])))
    gross = np.asarray(returns)
    exit_dates = np.asarray(date_rows)
    planned = np.asarray(planned_rows)
    meta = pd.DataFrame(names, columns=["code", "signal_date", "entry_date", "entry_open", "recorded_peak_date"])
    entry_segments = np.asarray([period(d) for d in meta.entry_date])
    exit_segments = (exit_dates >= 20250101).astype(int) + (exit_dates >= 20260101)
    planned_segments = (planned >= 20250101).astype(int) + (planned >= 20260101)
    complete = ((entry_segments[:, None] == exit_segments) & (exit_segments == planned_segments)
                & (exit_dates > 0) & (planned > 0))
    if isolate_years:
        entry_year = meta.entry_date.to_numpy() // 10000
        complete &= ((exit_dates // 10000 == entry_year[:, None])
                     & (planned // 10000 == entry_year[:, None]))
    segments = np.where(complete, exit_segments, -1).astype(np.int8)
    net = (1 + gross) * (1 - COST / 2) / (1 + COST / 2) - 1
    print(f"{len(out):,} candidate events; {uncertain_entries} ambiguous opening prices", flush=True)
    return np.asarray(out), net, gross, segments, exit_dates, meta, len(peak_ids)


@njit
def matching(features, cfg, n_peaks):
    pit, rally_min, rally_max, wash, vol, peak, move_min, move_max, max_gap = cfg
    seen = np.zeros(n_peaks, np.bool_)
    chosen = np.zeros(len(features), np.bool_)
    for i, (drop, rally, depth, dry, peak_grade, move, gap, group) in enumerate(features):
        key = int(group)
        if seen[key] or drop < pit or rally < rally_min or rally > rally_max:
            continue
        if depth < .05 or depth > wash or dry > vol or peak_grade < peak or move <= move_min or move > move_max:
            continue
        seen[key] = True
        chosen[i] = gap <= max_gap + 1e-12
    return chosen


@njit
def search(features, returns, segments, years, entries, n_peaks):
    # Keep outcomes isolated by entry year. The selection functions decide
    # which years they may use.
    n_exits = returns.shape[1]
    table = np.zeros((len(entries), n_exits, 5, 6))
    for idx, cfg in enumerate(entries):
        selected = matching(features, cfg, n_peaks)
        for i in range(len(features)):
            if not selected[i]:
                continue
            year = years[i]
            if year < 0 or year >= 5:
                continue
            for j in range(n_exits):
                p = segments[i, j]
                if p < 0:
                    continue
                v = returns[i, j]
                table[idx, j, year, 0] += 1
                table[idx, j, year, 1] += v
                if v > 0:
                    table[idx, j, year, 2] += 1
                    table[idx, j, year, 3] += v
                elif v < 0:
                    table[idx, j, year, 4] -= v
                    table[idx, j, year, 5] += 1
    return table


def statistics(values):
    wins, losses = values[values > 0], values[values < 0]
    gain = wins.mean() if len(wins) else np.nan
    loss = -losses.mean() if len(losses) else np.nan
    return dict(n=len(values), net_mean=values.mean() if len(values) else np.nan,
                median=np.median(values) if len(values) else np.nan,
                win_rate=len(wins) / len(values) if len(values) else np.nan,
                payoff=gain / loss, avg_win=gain, avg_loss=loss)


def choose(summary, label):
    n, result, win, gain, loss, loss_count = np.moveaxis(summary, -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_win = win / n
        payoff = (gain / win) / (loss / loss_count)
        avg = result / n
    eligible = (n[:, :, 0] >= 100) & (n[:, :, 1] >= 35)
    eligible &= (np.min(avg, axis=-1) > 0) & np.isfinite(payoff).all(axis=-1)
    counts = int(eligible.sum())
    if not counts:
        raise ValueError("No eligible fits in development and validation")
    points = np.argwhere(eligible)
    target = {"win": p_win, "payoff": payoff, "mean": avg}[label]
    worst = np.min(target, axis=-1)
    # Tie-breaking prefers a more robust mean, then more validation observations.
    a, b = points.T
    idx = np.lexsort((-n[a,b,1], -avg[a,b,:].min(axis=1), -worst[a,b]))[0]
    return tuple(int(x) for x in points[idx]), counts


def choose_robust(yearly, label, max_hold):
    annual_n = yearly[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        annual_mean = yearly[..., 1] / annual_n
        combined = np.concatenate((yearly[:, :, :3].sum(axis=2, keepdims=True), yearly[:, :, 3:]), axis=2)
        n, result, win, gain, loss, loss_count = np.moveaxis(combined, -1, 0)
        win_rate = win / n
        payoff = (gain / win) / (loss / loss_count)
    worst = annual_mean.min(axis=-1)
    eligible = (n[:, :, 0] >= 100) & (n[:, :, 1] >= 35)
    if n.shape[-1] == 3:
        eligible &= n[:, :, 2] >= 40
    eligible &= (annual_n.min(axis=-1) >= 20) & (worst > 0)
    eligible &= np.asarray([days <= max_hold for days, _, _ in EXITS])[None, :]
    if label == "win":
        eligible &= (win_rate >= .60).all(axis=-1)
    else:
        eligible &= ((payoff >= 2.0) & np.isfinite(payoff) & (win_rate >= .25)).all(axis=-1)
    points = np.argwhere(eligible)
    if len(points) == 0:
        return None, 0
    a, b = points.T
    paired_mean = (result[a, b] / n[a, b]).min(axis=-1)
    i = np.lexsort((-n[a, b, 1], -paired_mean, -worst[a, b]))[0]
    return tuple(int(x) for x in points[i]), len(points)


def diagnostics(selected, exits, gross, net, segments, exit_dates, meta, label, entry_cfg, out):
    j = EXITS.index(exits)
    rows = []
    trades = meta.loc[selected].copy()
    trades["exit_date"] = exit_dates[selected, j]
    trades["gross_return"] = gross[selected, j]
    trades["net_return"] = net[selected, j]
    trades["segment"] = segments[selected, j]
    for p, name in enumerate(PERIODS):
        group = trades.loc[trades.segment == p]
        numbers = statistics(group.net_return.to_numpy())
        stress = (1 + group.gross_return) * (1 - STRESS_COST / 2) / (1 + STRESS_COST / 2) - 1
        rows.append({"candidate": label, "period": name, **numbers,
                     "stress_mean": stress.mean(), "stress_win_rate": stress.gt(0).mean(),
                     "months": (group.entry_date // 100).nunique(), **dict(zip(ENTRY_FIELDS, entry_cfg)),
                     "hold": exits[0], "take": exits[1], "stop": exits[2]})
    if label != "baseline":
        trades.loc[trades.segment >= 0].to_csv(out / f"trades_{label}.csv", index=False, encoding="utf-8-sig")
    return rows


def walk_forward(features, net, gross, segments, exit_dates, meta, n_peaks, out):
    """Choose by the weakest prior annual expectancy under stressed fees."""
    years = meta.entry_date.to_numpy() // 10000
    stressed = (1 + gross) * (1 - STRESS_COST / 2) / (1 + STRESS_COST / 2) - 1
    configs = np.asarray(ENTRIES)
    evidence = search(features, stressed, segments, years - 2022, configs, n_peaks)
    baseline = matching(features, BASE, n_peaks)
    rows, orders = [], []

    for year in (2024, 2025, 2026):
        previous = evidence[:, :, :year - 2022]
        count = previous[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            by_year = previous[..., 1] / count
        worst = by_year.min(axis=-1)
        eligible = (count.min(axis=-1) >= 30) & (count.sum(axis=-1) >= 120) & (worst > 0)
        choices = np.argwhere(eligible)
        strategies = [("baseline_5", BASE, EXITS.index((5, 0., 0.)), baseline, np.nan)]
        if len(choices):
            i, j = choices.T
            order = np.lexsort((-count[i, j, :].sum(axis=-1), -by_year[i, j, -1], -worst[i, j]))[0]
            ix, k = (int(value) for value in choices[order])
            cfg = ENTRIES[ix]
            strategies.append(("rolling", cfg, k, matching(features, cfg, n_peaks), worst[ix, k]))
        else:
            rows.append({"strategy": "rolling", "year": year, "n": 0,
                         "eligible_before_year": 0, "prior_worst_stress_mean": np.nan})
        for name, params, k, selected, prior_worst in strategies:
            executed = selected & (years == year) & (segments[:, k] >= 0)
            stats = statistics(net[executed, k])
            rows.append({"strategy": name, "year": year, **stats,
                         "stress_mean": stressed[executed, k].mean() if stats["n"] else np.nan,
                         "eligible_before_year": len(choices) if name == "rolling" else np.nan,
                         "prior_worst_stress_mean": prior_worst,
                         **dict(zip(ENTRY_FIELDS, params)), **dict(zip(("hold", "take", "stop"), EXITS[k]))})
            traded = meta.loc[executed].copy()
            traded["strategy"] = name
            traded["exit_date"] = exit_dates[executed, k]
            traded["gross_return"] = gross[executed, k]
            traded["net_return"] = net[executed, k]
            traded["stress_return"] = stressed[executed, k]
            orders.append(traded)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "walkforward.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    pd.concat(orders, ignore_index=True).to_csv(out / "walkforward_trades.csv", index=False, encoding="utf-8-sig")
    print(comparison[["strategy", "year", "n", "hold", "take", "stop", "win_rate", "payoff",
                      "net_mean", "stress_mean", "eligible_before_year"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tdx", type=Path, default=Path("D:/tdx"))
    parser.add_argument("--end", type=int, default=20261116)
    parser.add_argument("--output", type=Path, default=bt.ROOT / "results" / "optimization")
    parser.add_argument("--walk-forward", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    features, net, gross, segments, exit_dates, meta, n_peaks = test_events(
        args.tdx, args.end, isolate_years=args.walk_forward
    )
    print(f"Searching {len(ENTRIES):,} entry sets × {len(EXITS)} exits", flush=True)
    if args.walk_forward:
        walk_forward(features, net, gross, segments, exit_dates, meta, n_peaks, args.output)
        return
    yearly = search(features, net, segments, meta.entry_date.to_numpy() // 10000 - 2022,
                    np.asarray(ENTRIES), n_peaks)
    table = np.stack((yearly[:, :, :3].sum(axis=2), yearly[:, :, 3]), axis=2)
    rows, candidates = [], {}
    finalists = [(label, choose(table, label)) for label in ("win", "payoff", "mean")]
    finalists += [(f"robust_{label}_{limit}", choose_robust(yearly[:, :, :4], label, limit))
                  for label, limit in product(("win", "payoff"), (10, 30))]
    # These exploratory fits use all available years. They are deliberately
    # identified separately from candidates selected before the 2026 check.
    finalists += [(f"historical_{label}", choose_robust(yearly, label, 30))
                  for label in ("win", "payoff")]
    search_counts = {}
    for label, (point, count) in finalists:
        search_counts[label] = count
        if point is None:
            continue
        a, b = point
        config, exits = ENTRIES[a], EXITS[b]
        candidates[label] = {**dict(zip(ENTRY_FIELDS, config)), "exit": exits}
        rows.extend(diagnostics(matching(features, config, n_peaks), exits, gross, net, segments,
                                exit_dates, meta, label, config, args.output))
    for hold in HOLDS:
        exits = (hold, 0., 0.)
        rows.extend(diagnostics(matching(features, BASE, n_peaks), exits, gross, net, segments,
                                exit_dates, meta, "baseline", BASE, args.output))
    results = pd.DataFrame(rows)
    results.to_csv(args.output / "comparison.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    manifest = {"entry_sets": len(ENTRIES), "exits": len(EXITS),
                "eligible_development_and_validation": search_counts, "round_trip_cost": COST,
                "stress_cost": STRESS_COST, "selected": candidates}
    (args.output / "selected.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(results[["candidate", "period", "hold", "n", "net_mean", "win_rate", "payoff", "stress_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
