"""Build two review pages from audited multi-timeframe research outputs."""
import copy
import json
import re
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;DIST=HERE/'dist'
sys.path.insert(0,str(ROOT))
from finalize_v2 import PARAMETERS

def export():
    src=ROOT/'results/research_v3';report=json.loads((src/'report.json').read_text(encoding='utf-8'))
    report['audits']=json.loads((src/'audit.json').read_text(encoding='utf-8'))
    report['ranking_comparison']=json.loads((src/'ranking_comparison.json').read_text(encoding='utf-8'))
    template=(DIST/'research.html').read_text(encoding='utf-8')
    calendar=np.load(ROOT/'results/research_v2/outcomes.npz')['calendar']
    for slug,actual_name,all_name,equity_name,index in [('multiframe','selected_trades.csv','all_signal_trades.csv','equity.csv',1),('macd','macd_trades.csv','macd_all_signals.csv','macd_equity.csv',2)]:
        aggressive=slug=='macd';hold=30 if aggressive else 20;current=copy.deepcopy(report)
        if aggressive:
            current['selected']=copy.deepcopy(report['comparisons'][2]);current['sensitivity']=[];current['horizons']=[]
            current['parameters']={**PARAMETERS,'hold_market_sessions':30,'rank':'clip(log(bull_bear_mean_volume40),-1,1)+near_divergence+0.5*hist_turn'}
        actual=pd.read_csv(src/actual_name);allrows=pd.read_csv(src/all_name);executed=set(actual.event_id);independent=set(allrows.event_id)
        data=pd.concat([allrows,actual.loc[~actual.event_id.isin(independent)]],ignore_index=True).fillna({'name':'缓存无名称'})
        rows=[]
        for x in data.itertuples():
            row={k:getattr(x,k) for k in ('code','name','signal_date','entry_date','exit_date','net','gross','buy_price','drawdown','floor_days','prior_from_floor','signal_return','volume5','volume10','daily','market20','amount20','event_id','vol_mean_40','vol_share_40','near_divergence','divergence','hist_turn','w_bar_date','m_bar_date','w_mom_improving')}
            if not aggressive:row['drawdown']=x.drop_60_250
            row.update(executed=x.event_id in executed,independent=x.event_id in independent,pit_high_date=0,candle_source='full')
            row.update({f'exit_date_{h}':int(calendar[int(x.signal_index)+h]) for h in (5,10,20,30)})
            rows.append(row)
        assert sum(x['executed'] for x in rows)==current['selected']['fills']
        payload=dict(report=current,rows=sorted(rows,key=lambda x:(-x['signal_date'],x['code'])),equity=pd.read_csv(src/equity_name).to_numpy().tolist())
        (DIST/f'{slug}-data.js').write_text('window.researchData='+json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)+';\n',encoding='utf-8')
        shutil.copyfile(src/actual_name,DIST/f'{slug}-trades.csv');shutil.copyfile(src/all_name,DIST/f'{slug}-all.csv')
        (DIST/f'{slug}-report.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
        title='量能与 MACD 排序' if aggressive else '把底部量能放进周线背景'
        html=template.replace('research-data.js',f'{slug}-data.js').replace('research.js','multiframe.js')
        html=html.replace('优化策略研究 · 量价研究',title+' · 量价研究').replace('从第二波，前移到首波反转',title)
        html=html.replace('final_portfolio_trades.csv',f'{slug}-trades.csv').replace('final_all_signal_trades.csv',f'{slug}-all.csv').replace('final_report.json',f'{slug}-report.json')
        html=re.sub(r'<nav aria-label="页面导航">.*?</nav>',f'<nav aria-label="页面导航"><a href="multiframe.html" {"aria-current=page" if not aggressive else ""}>↗ 量能与周线</a><a href="macd.html" {"aria-current=page" if aggressive else ""}>◇ MACD排序版</a><a href="research.html">▤ 上一版策略</a><a href="index.html">◎ 原形态策略</a><a href="#trades">▦ 信号与K线</a><a href="#evidence">≋ 因子与验证</a></nav>',html,count=1)
        html=html.replace('深跌后的放量阳线 <span>→</span> 市场普遍反弹 <span>→</span> 次日回落挂单 <span>→</span> 持有 20 日',
             '较长区间的深跌 <span>→</span> 阳量与动能验证 <span>→</span> 次日回落挂单 <span>→</span> 持有 '+str(hold)+' 日')
        summary=current['selected'];html=html.replace('98 笔成交集中在 25 个买入日',f'{summary["fills"]} 笔成交集中在 {summary["active_entry_dates"]} 个买入日').replace('不能按 98 次独立试验理解','同日股票的收益相关，不能视为独立试验')
        html=html.replace('<th>成交笔数</th>','<th>入场笔数</th>').replace('历史组合净值和回撤','历史组合净值曲线')
        common='信号日涨幅大于6%、小于9.5%；阳线实体为正，上影线不超过实体1/3；成交量大于此前5日和10日均量。股价至少3元、近20日均成交额至少3000万元。'
        if not aggressive:
            rules=[('01 / 扩大历史背景','在信号前60个交易日寻找最低收盘价L，前250日起至L形成日的最高价到L跌幅至少30%。L距信号最多15日，前一日收盘不超过1.12×L。需要完整250日历史。'),
                   ('02 / 阳量大，阴量小','信号前40日，阳线（收盘高于开盘）的日均成交量 ÷ 阴线（收盘低于开盘）的日均成交量至少为1。各至少3根，十字线剔除，不含信号日。'+common),
                   ('03 / 已完成周线放缓','最近一根已完成周K的收盘至收盘涨幅，大于前一周的涨幅。允许两周均下跌但跌势收缓。全市场等权当日涨幅至少1.5%，近20日复合收益仍小于0。'),
                   ('04 / 资金和退出','次日按前收盘价98%挂单，涨停开盘跳过，未成交当日撤单。最多10只、每笔最多10%权益；同日按近20日均成交额排序。买入日为第1日，第20日收盘退出，无固定止盈止损。')]
        else:
            rules=[('01 / 保留首波反转筛选','最低价L取前60日，前120日起至L的跌幅至少30%，距L最多8日，前一日不超过1.08×L。市场当日等权涨幅至少2%，近20日复合收益小于0。'+common),
                   ('02 / 阳阴量用于排序','计算信号前40日阳线平均量与阴线平均量之比R。量能得分为ln(R)，限于-1至1之间，不含信号日。'),
                   ('03 / MACD底背离候选','MACD采用12、26、9。最近15个交易日附近的低点需左右各2根K线确认，确认完成于信号前；向前120日寻找相隔至少5根的已确认低点。后低点价格不高于前低点102%、DIF更高且两处DIF均为负时，加1分。'),
                   ('04 / 排序、挂单、退出','MACD柱最近连续两次回升再加0.5分。按量能分＋背离分＋回升分由高到低安排仓位，同分按成交额排序。次日回落2%挂单，最多10只，每笔最多10%，第30个交易日收盘退出，无固定止盈止损。')]
        rulehtml='<section class="panel research-panel" id="parameters"><div class="section-head"><div><h2>本页采用的完整规则</h2><p>日线特征与周月线均按信号时可见的信息计算。</p></div><a class="button secondary" href="'+slug+'-report.json" download>下载完整研究</a></div><div class="rule-grid">'+''.join('<article><b>'+a+'</b><p>'+b+'</p></article>' for a,b in rules)+'</div><p class="research-note">周线只使用截至信号日已完成的周；月线只使用已完成的月。月MACD需26根完整月K，历史不足时记为缺失，不视为满足条件。月线硬过滤未进入最终候选。K线图按完整历史聚合，信号所在周月可含其后行情，仅供复盘。年度单笔统计按入场年分组，组合收益按自然年权益计算。</p></section>'
        html=re.sub(r'<section class="panel research-panel" id="parameters">.*?</section>',rulehtml,html,count=1,flags=re.S)
        comparisons='<section class="panel research-panel"><div class="section-head"><div><h2>三组方案放在一起比较</h2><p>相同资金、费用和成交约束；均为反复看过历史后的候选。</p></div></div><div class="table-scroll"><table><thead><tr><th>方案</th><th>组合年化</th><th>最大回撤</th><th>胜率</th><th>盈亏比</th><th>成交数</th></tr></thead><tbody id="comparisons"></tbody></table></div></section>'
        html=html.replace('<section class="panel research-panel" id="parameters">',comparisons+'<section class="panel research-panel" id="parameters">',1)
        ablation='<h3>同一基础策略逐项增加条件</h3><p class="research-note">前12行保持上一版的基础形态与持有20日不变。最后4行使用250日高点、15日底部位置及1.5%市场反弹门槛，观察量能和周线的共同作用。</p><div class="table-scroll"><table><thead><tr><th>增加的条件</th><th>年化收益</th><th>最大回撤</th><th>成交数</th></tr></thead><tbody id="ablations"></tbody></table></div>'
        html=html.replace('<h3>5 / 10 / 20 / 30 日的比较</h3>',ablation+'<h3>5 / 10 / 20 / 30 日的比较</h3>')
        rank_table='<h3>区分持有期与排序的贡献</h3><p class="research-note">同一首波反转基础筛选、挂单和资金规则，只改变排序或持有期。30日的改善同时依赖持有期与排序，不能全部归因于MACD。</p><div class="table-scroll"><table><thead><tr><th>持有期</th><th>同日候选排序</th><th>组合年化</th><th>最大回撤</th><th>胜率</th><th>盈亏比</th><th>成交数</th></tr></thead><tbody id="ranking-comparison"></tbody></table></div>'
        html=html.replace('<h3>5 / 10 / 20 / 30 日的比较</h3>',rank_table+'<h3>5 / 10 / 20 / 30 日的比较</h3>')
        html=html.replace('前250日起至L形成日的最高价到L','前250日起至L形成日的最高收盘价到L')
        html=html.replace('最低价L取前60日，前120日起至L的跌幅至少30%','最低收盘价L取前60日，前120日起至L的最高收盘价到L跌幅至少30%')
        html=html.replace('周线只使用截至信号日已完成的周；','低开低于挂单价时按开盘价买；盘中需向下穿过挂单价才认定成交，除权日使用参考前收盘价。先预留订单资金，未成交资金当天不补选；考虑T+1、整手、停牌和跌停卖出顺延。复权总回报采用现金分红再投资近似。周线只使用截至信号日已完成的周；')
        html=html.replace('按最弱年度收益与年化收益评分','按年化、最弱年度收益与回撤评分').replace('锁定参数后检验下一年。','锁定参数后检验下一年，保留跨年持仓。')
        html=html.replace('累计搜索 138,897 个事件参数组合与 17,568 个有限资金组合',f'本轮另检验 {report["search_count"]:,} 个多周期组合，包含此前搜索的选择影响')
        html=html.replace('20 日净收益',f'{hold} 日净收益')
        if aggressive:
            html=html.replace('一次只改变一个条件，其他规则保持相同。完整数值可下载。','本页保留量能与MACD排序的30日候选。单项敏感性和四种持有期的完整对照见量能与周线页面。')
            html=html.replace('<tbody id="sensitivity"></tbody>','<tbody id="sensitivity"><tr><td colspan="4"><a href="multiframe.html#evidence">查看20日平衡版的参数敏感性</a></td></tr></tbody>')
        (DIST/f'{slug}.html').write_text(html,encoding='utf-8')
        print(f'{slug}: {len(rows)} signal rows; {len(executed)} portfolio fills; hold {hold}')
    js=(DIST/'research.js').read_text(encoding='utf-8')
    js=js.replace('S=R.selected,$=', 'S=R.selected,H=R.parameters.hold_market_sessions||20,$=')
    js=js.replace('最长持有 20 个交易日','计划持有 ${H} 个交易日')
    js=js.replace("['20 日净收益',pct(row.net)]","[H+' 日净收益',pct(row.net)]")
    js=js.replace('t.payoff.toFixed(2)',"(t.payoff==null?'—':t.payoff.toFixed(2))")
    js=js.replace("const labels={pit:","const labels={volume:'阳/阴日均量门槛',window:'量能观察日数',pit:")
    js=js.replace("['age','slots','hold'].includes(v.change)","['age','slots','hold','window'].includes(v.change)")
    js=js.replace("vals[v.value]||(","vals[v.value]||(v.change==='volume'?v.value.toFixed(2)+'倍':")
    js=js.replace("$('#sensitivity').innerHTML=", "if(R.sensitivity.length)$('#sensitivity').innerHTML=")
    js+='''\n(()=>{const D=window.researchData,R=D.report,$=s=>document.querySelector(s),p=x=>(x*100).toFixed(2)+'%';
const names={none:'不加条件','bullavg40_ge1.0':'40日阳/阴均量≥1','bullavg60_ge1.2':'60日阳/阴均量≥1.2','bullavg120_ge1.0':'120日阳/阴均量≥1',down_volume_dry:'阴量缩至此前80%以下',divergence:'严格MACD底背离',near_divergence:'近似双底的MACD背离',macd_hist_turn:'MACD柱连续回升',weekly_momentum:'已完成周线动能改善',monthly_momentum:'已完成月线动能改善',monthly_hist_up:'已完成月线MACD柱上升',volume_weekly:'40日阳量优势＋周线改善',long250_none:'250日背景，不加量能周线','long250_bullavg40_ge1.0':'250日背景＋阳量优势',long250_weekly_momentum:'250日背景＋周线改善',long250_volume_weekly:'250日背景＋阳量优势＋周线改善'};
$('#comparisons').innerHTML=R.comparisons.map((x,i)=>'<tr><td><a href="'+['research.html','multiframe.html','macd.html'][i]+'">'+x.label+'</a></td><td>'+p(x.cagr)+'</td><td>'+p(x.max_drawdown)+'</td><td>'+p(x.trade_stats.win)+'</td><td>'+x.trade_stats.payoff.toFixed(2)+'</td><td>'+x.fills+'</td></tr>').join('');
$('#ablations').innerHTML=R.ablations.map(x=>'<tr><td>'+names[x.condition]+'</td><td>'+p(x.cagr)+'</td><td>'+p(x.max_drawdown)+'</td><td>'+x.fills+'</td></tr>').join('');
const rankNames={liquidity:'成交额',volume:'阳阴量比',macd:'MACD',volume_macd:'阳阴量比＋MACD'};
$('#ranking-comparison').innerHTML=R.ranking_comparison.map(x=>'<tr><td>'+x.hold+'日</td><td>'+rankNames[x.rank]+'</td><td>'+p(x.cagr)+'</td><td>'+p(x.max_drawdown)+'</td><td>'+p(x.trade_stats.win)+'</td><td>'+x.trade_stats.payoff.toFixed(2)+'</td><td>'+x.fills+'</td></tr>').join('');
if(!R.horizons.length)$('#holds').innerHTML='<tr><td colspan="7"><a href="multiframe.html#evidence">查看20日平衡版的四种持有期对照</a></td></tr>';
$('#trade-rows').addEventListener('click',ev=>{const b=ev.target.closest('[data-event]');if(!b)return;const x=D.rows.find(x=>x.event_id===Number(b.dataset.event));$('#detail-rule').textContent+=' 近40日阳/阴日均量比 '+x.vol_mean_40.toFixed(2)+'，阳量占比 '+p(x.vol_share_40)+'。已完成周截止 '+x.w_bar_date+'，周动能改善 '+(x.w_mom_improving?'是':'否')+'。日MACD底背离候选 '+(x.near_divergence?'是':'否')+'。';});
})();\n'''
    (DIST/'multiframe.js').write_text(js,encoding='utf-8')
    for filename in ('index.html','research.html'):
        path=DIST/filename;text=path.read_text(encoding='utf-8')
        if 'href="multiframe.html"' not in text:
            text=text.replace('<nav aria-label="页面导航">','<nav aria-label="页面导航"><a href="multiframe.html"><span aria-hidden="true">↗</span> 多周期优化</a>',1)
            path.write_text(text,encoding='utf-8')

if __name__=='__main__':export()
