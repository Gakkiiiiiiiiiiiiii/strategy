"""Cache BaoStock daily data and historical ST checks for the TDX backtest."""

import argparse
import socket
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".deps"))
import baostock as bs
from baostock.common import context as session


def frame(result):
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(result.error_msg)
    return pd.DataFrame(rows, columns=result.fields)


def save(frame, path):
    partial = path.with_suffix(".tmp")
    frame.to_csv(partial, index=False)
    partial.replace(path)


def daily(start, end):
    directory = ROOT / "data" / "daily"
    directory.mkdir(parents=True, exist_ok=True)
    calendar = frame(bs.query_trade_dates(start, end))
    for day in calendar.loc[calendar.is_trading_day == "1", "calendar_date"]:
        path = directory / f"{day}.csv"
        if path.exists():
            continue
        for attempt in range(3):
            try:
                bars = frame(bs.query_daily_history_k_AStock(day))
                break
            except RuntimeError:
                if attempt == 2:
                    raise
                session.default_socket.close()
                if bs.login().error_code != "0":
                    raise RuntimeError("BaoStock reconnect failed")
        if bars.empty:
            print(f"{day} no published daily data", flush=True)
            continue
        save(bars, path)
        print(f"{day} {len(bars)}", flush=True)


def st_checks():
    requests_path = ROOT / "results" / "st_checks_needed.csv"
    status_path = ROOT / "data" / "st_checks.csv"
    needed = pd.read_csv(requests_path)
    known = (
        pd.read_csv(status_path)
        if status_path.exists()
        else pd.DataFrame(columns=["code", "date", "isST"])
    )
    needed = needed.merge(known, on=["code", "date"], how="left")
    unresolved = needed.loc[needed.isST.isna()]
    for code, days in unresolved.groupby("code"):
        start, end = [
            str(pd.to_datetime(int(day), format="%Y%m%d").date())
            for day in (days.date.min(), days.date.max())
        ]
        bars = frame(bs.query_history_k_data_plus(code, "date,isST", start, end))
        if not bars.empty:
            bars["date"] = bars.date.str.replace("-", "").astype(int)
            bars["code"] = code
            known = pd.concat([known, bars.loc[bars.date.isin(days.date)]], ignore_index=True)
        save(known, status_path)
    print(f"Cached {len(known)} dated risk-warning checks", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", default="2026-11-05")
    parser.add_argument("--through", default="2026-11-17")
    parser.add_argument("--st-checks", action="store_true")
    args = parser.parse_args()
    socket.setdefaulttimeout(45)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    try:
        if args.st_checks:
            st_checks()
        else:
            daily(args.start, args.through)
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
