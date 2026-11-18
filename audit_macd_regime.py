"""Diagnose the published v3 MACD model without selecting new parameters.

All results remain retrospective on the previously searched frozen snapshot.
Random rankings and deleted winning episodes are diagnostics, not valid
selection-adjusted significance tests or deployable parameter recommendations.
"""
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import finalize_v2 as f
import portfolio_search as ps
import research_v2 as r
import research_v3 as v
from review_v2 import stats


OUT = bt.ROOT / 'results' / 'macd_regime_audit'


def main():
    OUT.mkdir(exist_ok=True)
    e, d, common, ranks = v.prepare()
    cal = common['calendar']
    mask = np.asarray(v.base_rules(e)['original_market0.02'], bool)
    exit_id = r.EXITS.index((30, 0., 0., 0.))

    def run(m, ranking=None, mode=2, cost=r.COST):
        z = ps.simulate_port(m, mode, exit_id, 1., 10,
                             **{**common, **(ranks['volume_macd'] if ranking is None else ranking)}, cost=cost)
        s = f.summarize(z, cal)
        s['entry_dates'] = int(e.iloc[z[12][:, 0].astype(int)].entry_date.nunique())
        return s, z

    baseline, z = run(mask)
    original = pd.read_csv(v.OUT / 'macd_trades.csv')
    np.testing.assert_array_equal(z[12][:, 0].astype(int), original.event_id.to_numpy())
    np.testing.assert_allclose(z[12][:, 1], original.net, atol=1e-14)
    np.testing.assert_allclose(z[12][:, 3], original.capital_fraction_of_initial, atol=1e-14)
    trades = original.copy()
    trades['pnl_initial_capital'] = trades.net * trades.capital_fraction_of_initial
    assert abs(trades.pnl_initial_capital.sum() - baseline['total']) < 1e-12

    market = pd.read_csv(r.OUT / 'market.csv').set_index('date')
    cumulative = (1 + market.daily).cumprod()
    np.testing.assert_allclose(trades.daily, market.loc[trades.signal_date, 'daily'])
    np.testing.assert_allclose(trades.market20, market.loc[trades.signal_date, 'market20'])
    # Signal close -> actual stock exit close is contextual, not executable
    # equal-weight index replication; it includes the following opening gap.
    trades['equal_weight_context'] = [cumulative.loc[x] / cumulative.loc[s] - 1
                                     for s, x in zip(trades.signal_date, trades.exit_date)]
    indices = {}
    with ExitStack() as stack:
        files, read = bt.market_files(Path('D:/tdx/vipdoc'), stack)
        for stem in ('sh000001', 'sh000300', 'sz399006'):
            if stem not in files:
                continue
            b = np.frombuffer(read(stem), bt.BAR_DTYPE)
            frame = pd.DataFrame({'date': b['date'], 'open': b['open'], 'close': b['close']}).set_index('date')
            if not set(trades.entry_date).issubset(frame.index) or not set(trades.exit_date).issubset(frame.index):
                continue
            # Same actual stock entry/exit dates; index bought at daily open.
            returns = frame.loc[trades.exit_date, 'close'].to_numpy() / frame.loc[trades.entry_date, 'open'].to_numpy() - 1
            net = (1 + returns) * (1-r.COST/2)/(1+r.COST/2)-1
            trades[stem + '_net'] = net
            indices[stem] = dict(stats=stats(net), stock_minus_index_mean=float((trades.net-net).mean()),
                                 stock_beats_index=float((trades.net>net).mean()))

    daily = trades.groupby('entry_date').agg(
        trades=('net','size'), mean=('net','mean'), pnl=('pnl_initial_capital','sum'),
        signal_date=('signal_date','first'), market_day=('daily','first'), market20=('market20','first'),
        equal_weight_context=('equal_weight_context','mean'))
    daily['pnl_share'] = daily.pnl / baseline['total']
    ordered = trades.sort_values(['entry_date','exit_date']).copy()
    cluster = -1
    last_exit = 0
    labels = []
    for x in ordered.itertuples():
        if x.entry_date > last_exit:
            cluster += 1
        last_exit = max(last_exit, x.exit_date)
        labels.append(cluster)
    ordered['episode'] = labels
    trades['episode'] = ordered.episode.reindex(trades.index)
    episodes = trades.groupby('episode').agg(start=('entry_date','min'), end=('exit_date','max'),
        trades=('net','size'), entry_dates=('entry_date','nunique'), mean=('net','mean'),
        total_returns=('net','sum'), pnl=('pnl_initial_capital','sum'))
    episodes['pnl_share'] = episodes.pnl / baseline['total']

    # Prespecified nearby/removed market conditions; every result is retained.
    shape = (e.price>=3)&(e.amount20>=3e7)&(e.signal_return<.095)&(e.drawdown>=.30)&(e.floor_days<=8)&(e.prior_from_floor<=.08)
    variations = []
    for threshold in (0., .01, .015, .02, .025, .03):
        m = np.asarray(shape & (e.market20<0) & (e.daily>=threshold), bool)
        s, _ = run(m)
        variations.append(dict(change='market_day_min', value=threshold, **s))
    for label, m in [('remove_day_filter',shape&(e.market20<0)),
                     ('remove_prior20_filter',shape&(e.daily>=.02)),
                     ('remove_both_market_filters',shape),
                     ('prior20_nonnegative_day2',shape&(e.market20>=0)&(e.daily>=.02))]:
        s, _ = run(np.asarray(m,bool)); variations.append(dict(change=label,**s))
    for mode in (0, 1, 3):
        s, _ = run(mask, mode=mode)
        variations.append(dict(change='entry',value=r.BUYS[mode],**s))
    for cost in (.007, .01):
        s, _ = run(mask, cost=cost)
        variations.append(dict(change='round_trip_cost',value=cost,**s))

    removal = []
    for n in (1, 3):
        dates = daily.nlargest(n,'pnl').index.to_numpy()
        s, _ = run(mask & ~e.entry_date.isin(dates).to_numpy())
        removal.append(dict(remove_top_entry_dates=n,dates=dates.tolist(),**s))
    for n in (1, 2):
        ids = episodes.nlargest(n,'pnl').index
        ranges = episodes.loc[ids,['start','end']].to_numpy()
        m = mask.copy()
        for start, end in ranges:
            m &= ~e.entry_date.between(start,end).to_numpy()
        s, _ = run(m)
        removal.append(dict(remove_top_exposure_episodes=n,date_ranges=ranges.tolist(),**s))

    # Same eligible stocks and causal fills. Random scores affect only order
    # priority on the same signal day, before learning whether orders fill.
    rng = np.random.default_rng(20261118)
    ids = np.flatnonzero(e.complete30.eq(1))
    days = e.signal_index.to_numpy()+1
    random_rows = []
    for iteration in range(200):
        score = rng.random(len(ids))
        rows = ids[np.lexsort((e.code.to_numpy()[ids],score,days[ids]))]
        ranking = dict(rows=rows,offsets=np.searchsorted(days[rows],np.arange(len(cal)+1)))
        s, _ = run(mask,ranking=ranking)
        random_rows.append(dict(iteration=iteration,cagr=s['cagr'],max_drawdown=s['max_drawdown'],fills=s['fills']))
    random_table = pd.DataFrame(random_rows)
    random_summary = dict(iterations=len(random_table), seed=20261118,
        cagr_quantiles=random_table.cagr.quantile([.05,.25,.5,.75,.95]).to_dict(),
        fraction_cagr_at_least_selected=float((random_table.cagr>=baseline['cagr']).mean()),
        note='Conditional retrospective ranking diagnostic; not a multiple-testing-adjusted p-value.')

    boot_rng = np.random.default_rng(81126)
    draws = boot_rng.integers(0,len(episodes),size=(10000,len(episodes)))
    means = episodes.total_returns.to_numpy()[draws].sum(1)/episodes.trades.to_numpy()[draws].sum(1)
    report = dict(snapshot_end=r.END, baseline=baseline,
        annual_entry_dates=trades.groupby(trades.entry_date//10000).entry_date.nunique().to_dict(),
        market_rule='Signal-close equal-weight daily return >=2%; trailing20 INCLUDING signal day <0; next-day order.',
        unique_market_observations=int(len(daily)), exposure_episodes=int(len(episodes)),
        episode_bootstrap_mean_ci95=np.quantile(means,[.025,.975]).tolist(),
        top_entry_dates=daily.sort_values('pnl',ascending=False).reset_index().to_dict('records'),
        episodes=episodes.reset_index().to_dict('records'),
        same_dates_indices=indices,
        equal_weight_context=stats(trades.equal_weight_context),
        by_future_equal_weight_sign=[dict(future_market_positive=bool(g),**stats(t.net),
                                         entry_dates=int(t.entry_date.nunique()))
                                    for g,t in trades.groupby(trades.equal_weight_context>0)],
        variations=variations, winning_period_removal=removal, random_rankings=random_summary,
        limitations=['All diagnostics use previously inspected data and inherit surviving-stock universe bias.',
                     'Index controls are conditional on actual stock fill dates; they are not an investable portfolio replay.',
                     'Random-ranking fraction and bootstrap interval do not correct 7,360+ historical searches.',
                     'Overlapping exposure windows are grouped; separated windows are not guaranteed independent.',
                     'No new strategy parameters are selected by this diagnostic.'])
    trades.to_csv(OUT/'matched_trades.csv',index=False,encoding='utf-8-sig')
    random_table.to_csv(OUT/'random_rankings.csv',index=False)
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:report[k] for k in ('snapshot_end','unique_market_observations','exposure_episodes','annual_entry_dates',
                                         'same_dates_indices','equal_weight_context','random_rankings')},ensure_ascii=True),flush=True)
    print('VARIATIONS',[(s['change'],s.get('value'),round(s['cagr']*100,3),round(s['max_drawdown']*100,3),s['fills']) for s in variations],flush=True)
    print('REMOVALS',[(s.get('remove_top_entry_dates',s.get('remove_top_exposure_episodes')),s.get('dates',s.get('date_ranges')),round(s['cagr']*100,3)) for s in removal],flush=True)
    print('TOP_DATES',daily.nlargest(5,'pnl').to_json(orient='index'),flush=True)


if __name__ == '__main__':
    main()
