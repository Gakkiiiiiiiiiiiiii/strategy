"""Display the full preceding decline for a signal-time-liquidity-picked case."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import macd_leaders_v9 as n


def main():
    e=pd.read_csv(n.OUT/'primary_signals.csv');x=e.sort_values(['amount20','code'],ascending=[False,True]).iloc[0]
    panel=pd.read_pickle(n.v.d.r.OUT/'panel.pkl');vol=n.v.d.volume_data(panel)[x.code][:,0]
    dates,raw,close,factor,pre,mi,*_=panel[x.code];s=int(x.impulse_index);t=int(x.stock_index);p=int(x.old_peak_index);f=int(x.floor_index)
    a=max(0,p-20);b=min(len(dates),t+41);ids=np.arange(a,b);axis=np.arange(len(ids));price=raw[ids]*factor[ids,None]/factor[t]/100
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(4,1,figsize=(17,12),sharex=True,gridspec_kw={'height_ratios':[3.5,1,1.2,1.5]})
    colors=np.where(price[:,3]>=price[:,0],'#d74759','#17816e')
    for k,(o,h,lo,c) in enumerate(price):
        axs[0].vlines(k,lo,h,color=colors[k],lw=.8)
        axs[0].add_patch(Rectangle((k-.33,min(o,c)),.66,max(abs(c-o),.01),facecolor=colors[k],edgecolor=colors[k],lw=.5))
    ma=pd.Series(close/factor[t]/100).rolling(60).mean().to_numpy()
    axs[0].plot(axis,ma[ids],color='#b18b25',lw=1.3,label='60日均线')
    axs[0].axvspan(p-a,f-a,color='#5366b9',alpha=.07)
    axs[0].text(.02,.94,f'前高至底部：−{x.decline*100:.2f}% / {int(x.decline_days)} 个交易日\n首次放量前收盘距底部：{x.prior_from_floor*100:.2f}%',
                transform=axs[0].transAxes,va='top',fontsize=11,bbox=dict(facecolor='white',edgecolor='#dfe3eb',alpha=.94,pad=7))
    axs[0].set_ylabel('按确认日复权 / 元');axs[0].legend(loc='upper right',frameon=False)
    axs[1].bar(axis,vol[ids]/1e6,color=colors,width=.66,alpha=.7);axs[1].set_ylabel('成交量 / 百万股')
    dif,dea,hist=n.m3.macd(close/factor[t]/100)
    axs[2].bar(axis,hist[ids],color=np.where(hist[ids]>=0,'#d74759','#17816e'),width=.66,alpha=.55)
    axs[2].plot(axis,dif[ids],label='DIF',color='#bf8b24',lw=1.1);axs[2].plot(axis,dea[ids],label='DEA',color='#7255bd',lw=1.1)
    axs[2].axhline(0,color='#aeb6c4',lw=.6);axs[2].set_ylabel('MACD 12/26/9');axs[2].legend(loc='upper right',frameon=False,ncol=2)
    m=pd.read_csv(n.v.d.r.OUT/'market.csv');market=(1+m.daily).cumprod().to_numpy()
    axs[3].plot(axis,(close[ids]/close[p]-1)*100,color='#5e4ccc',label='个股：以前高日收盘为基准')
    axs[3].plot(axis,(market[mi[ids]]/market[mi[p]]-1)*100,color='#647487',label='市场等权：同日起点')
    axs[3].set_ylabel('累计涨跌 / %');axs[3].legend(loc='upper left',fontsize=9,frameon=False)
    for ax in axs:
        ax.axvline(s-a,color='#287aa7',ls='--',lw=1)
        ax.axvline(t-a,color='#cb861a',ls='--',lw=1)
        ax.grid(axis='y',color='#e8ecf2',lw=.6);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axs[0].text(s-a-1,1.015,'首次逆市放量',transform=axs[0].get_xaxis_transform(),ha='right',fontsize=10,color='#287aa7')
    axs[0].text(t-a+1,1.015,'再次确认',transform=axs[0].get_xaxis_transform(),ha='left',fontsize=10,color='#b07417')
    ticks=np.arange(0,len(ids),max(1,len(ids)//8));axs[3].set_xticks(ticks,[f'{str(int(dates[ids[k]]))[:4]}-{str(int(dates[ids[k]]))[4:6]}-{str(int(dates[ids[k]]))[6:]}' for k in ticks],fontsize=9)
    fig.suptitle(f'先确认长期下跌和底部，再研究后续形态\n{x["name"]} {x.code} · 首次放量 {x.impulse_date} · 确认 {x.signal_date}',x=.07,ha='left',fontsize=17,y=.98)
    fig.text(.07,.025,'案例按主组信号时已知的20日平均成交额选择，未按后续收益选取。图中包含确认后的复盘行情；MACD只使用当时已知数据进行排序。',fontsize=10,color='#596477')
    fig.subplots_adjust(top=.88,bottom=.09,left=.07,right=.98,hspace=.13)
    fig.savefig(n.OUT/'long-decline-case.png',dpi=130,facecolor='white');plt.close(fig)
    n.dump('chart_case.json',dict(code=x.code,name=x['name'],signal_date=int(x.signal_date),impulse_date=int(x.impulse_date),
        decline=float(x.decline),decline_days=int(x.decline_days),selection='Largest signal-time amount20 in long-decline primary signals; no return selection.'))
    print(json.dumps(dict(code=x.code,signal_date=int(x.signal_date),decline=float(x.decline),decline_days=int(x.decline_days))),flush=True)


if __name__=='__main__':main()
