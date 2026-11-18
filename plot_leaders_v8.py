"""Show two cases chosen by signal-time liquidity, without selecting winners."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
import leaders_v8 as v


def main():
    e=pd.read_pickle(v.OUT/'events.pkl');p=pd.read_pickle(v.d.r.OUT/'panel.pkl');vols=v.d.volume_data(p)
    market=pd.read_csv(v.d.r.OUT/'market.csv');level=(1+market.daily).cumprod().to_numpy()
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(3,2,figsize=(17,10),gridspec_kw={'height_ratios':[2.2,1.4,1]},sharex='col')
    records=[]
    for col,(kind,label) in enumerate([('pullback','缩量回调型'),('resistant','横盘抗跌型')]):
        g=e[e[kind]&(e.signal_date//10000==2024)].sort_values('signal_date').drop_duplicates(['code','impulse_date'])
        x=g.sort_values(['amount20','code'],ascending=[False,True]).iloc[0]
        dates,raw,close,factor,pre,mi,*_=p[x.code];s=int(x.impulse_index);t=int(x.stock_index)
        a=max(0,s-25);b=min(len(dates),t+41);inds=np.arange(a,b);idx=np.arange(len(inds));adjusted=raw[inds]*factor[inds,None]/factor[t]/100
        colors=np.where(adjusted[:,3]>=adjusted[:,0],'#d84559','#16846d')
        ax=axes[0,col]
        for n,(o,h,lo,c) in enumerate(adjusted):
            ax.vlines(n,lo,h,color=colors[n],lw=.9)
            ax.add_patch(Rectangle((n-.32,min(o,c)),.64,max(abs(c-o),.005),facecolor=colors[n],edgecolor=colors[n],lw=.6))
        ax.set_title(f'{label}  |  {x["name"]} {x.code}\n信号 {x.signal_date} · 行业 {x.industry}',loc='left',fontsize=12,pad=12)
        ax.set_ylabel('按信号日复权 / 元')
        relative_stock=close[inds]/close[s-1]-1;relative_market=level[mi[inds]]/level[mi[s-1]]-1
        axes[1,col].plot(idx,relative_stock*100,label='个股累计涨跌',color='#5645ca',lw=1.8)
        axes[1,col].plot(idx,relative_market*100,label='市场等权累计涨跌',color='#6e7688',lw=1.6)
        axes[1,col].axhline(0,color='#c6ccd4',lw=.8);axes[1,col].set_ylabel('从逆市放量前收起 / %')
        axes[1,col].legend(loc='upper left',fontsize=8,frameon=False)
        volume=vols[x.code][inds,0]/1e6;axes[2,col].bar(idx,volume,color=colors,width=.65,alpha=.7)
        axes[2,col].set_ylabel('成交量 / 百万股')
        for row in range(3):
            q=axes[row,col];q.axvline(s-a,color='#2473a5',linestyle='--',lw=1);q.axvline(t-a,color='#ce8b18',linestyle='--',lw=1)
            q.grid(axis='y',color='#e9ecf1',lw=.7);q.set_axisbelow(True)
            q.spines[['top','right']].set_visible(False)
        tight=t-s<12
        ax.text(s-a-(1 if tight else 0),1.01,'逆市放量',transform=ax.get_xaxis_transform(),color='#2473a5',fontsize=9,ha='right' if tight else 'center')
        ax.text(t-a+(1 if tight else 0),1.01,'市场转强确认',transform=ax.get_xaxis_transform(),color='#a26b10',fontsize=9,ha='left' if tight else 'center')
        ticks=np.arange(0,len(idx),max(1,len(idx)//6));axes[2,col].set_xticks(ticks,[str(int(dates[inds[k]]))[4:6]+'-'+str(int(dates[inds[k]]))[6:] for k in ticks],fontsize=9)
        records.append(dict(type=kind,code=x.code,name=x['name'],signal_date=int(x.signal_date),impulse_date=int(x.impulse_date),
            selection='Highest signal-time 20-day average turnover among 2024 first qualifying signals; no outcome selection.',
            pullback=float(x.draw_from_peak),dry_ratio=float(x.dry_ratio),phase_market=float(x.phase_market),phase_stock=float(x.phase_stock)))
    fig.suptitle('先逆市放量，再等待市场转强：两种路径',fontsize=18,x=.06,ha='left',y=.98)
    fig.text(.06,.025,'案例按信号日已知成交额选择，不按后续收益选取。虚线左侧用于识别，右侧仅用于复盘；量价不能直接证明吸筹或板块因果关系。',fontsize=10,color='#5c6475')
    fig.subplots_adjust(top=.86,bottom=.09,hspace=.12,wspace=.22,left=.06,right=.98)
    fig.savefig(v.OUT/'leader-patterns.png',dpi=135,facecolor='white');plt.close(fig)
    v.dump('chart_cases.json',records);print(json.dumps(records,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
