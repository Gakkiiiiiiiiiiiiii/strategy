"use strict";
(() => {
  window.fullCandleData=Object.create(null);
  const requests=new Map(),up='#c24c65',down='#258b78',purple='#6655c9';
  const day=d=>String(d).replace(/(\d{4})(\d{2})(\d{2})/,'$1-$2-$3');
  const money=v=>(v/100).toFixed(2),pct=v=>v==null?'—':`${v>0?'+':''}${(v*100).toFixed(2)}%`;
  const dateInt=d=>Number(d.toISOString().slice(0,10).replaceAll('-',''));
  function load(row){
    if(!requests.has(row.code)) requests.set(row.code,new Promise((resolve,reject)=>{
      const s=document.createElement('script');s.src='ohlcv/'+row.code.replace('.','')+'.js';
      s.onload=()=>{s.remove();const data=window.fullCandleData[row.code];data?resolve(data):reject(Error('Missing candles'))};
      s.onerror=()=>{s.remove();requests.delete(row.code);reject(Error('Chart load failed'))};document.head.append(s);
    }));return requests.get(row.code);
  }
  function periods(daily,type,asof){
    let bars=[];
    if(type==='day') bars=daily.map(b=>({date:b[0],start:b[0],end:b[0],o:b[1],h:b[2],l:b[3],c:b[4],v:b[5],pct:b[6],complete:true}));
    else{
      for(const b of daily){const d=new Date(day(b[0])+'T00:00:00Z');let end;
        if(type==='week'){const wd=d.getUTCDay();d.setUTCDate(d.getUTCDate()+(5-wd+7)%7);end=dateInt(d)}
        else{end=dateInt(new Date(Date.UTC(d.getUTCFullYear(),d.getUTCMonth()+1,0)))}
        let last=bars[bars.length-1];
        if(!last||last.end!==end){last={date:b[0],start:b[0],end,o:b[1],h:b[2],l:b[3],c:b[4],v:b[5],pct:null,complete:end<=asof};bars.push(last)}
        else{last.date=b[0];last.h=Math.max(last.h,b[2]);last.l=Math.min(last.l,b[3]);last.c=b[4];last.v+=b[5]}
      }
      bars.forEach((b,i)=>{if(i)b.pct=b.c/bars[i-1].c-1});
    }
    let fast=0,slow=0,signal=0;
    bars.forEach((b,i)=>{fast=i?fast+2/13*(b.c-fast):b.c;slow=i?slow+2/27*(b.c-slow):b.c;
      const dif=fast-slow;signal=i?signal+2/10*(dif-signal):dif;b.dif=i>=25?dif:null;b.dea=i>=25?signal:null;b.hist=i>=25?2*(dif-signal):null;
      b.ma5=i>=4?bars.slice(i-4,i+1).reduce((s,x)=>s+x.c,0)/5:null;
      b.ma20=i>=19?bars.slice(i-19,i+1).reduce((s,x)=>s+x.c,0)/20:null;
    });return bars;
  }
  function draw(canvas,bars,selected,row,type){
    const w=canvas.clientWidth,h=canvas.clientHeight;if(w<100||!bars.length)return null;
    const ratio=Math.min(devicePixelRatio||1,2.5);canvas.width=w*ratio;canvas.height=h*ratio;const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
    const left=9,right=w-57,top=45,priceEnd=h*.59,volumeTop=h*.63,volumeEnd=h*.76,macdTop=h*.81,macdEnd=h-26;
    const step=(right-left)/bars.length,cw=Math.max(.5,Math.min(10,step*.62));
    const max=Math.max(...bars.map(b=>b.h)),min=Math.min(...bars.map(b=>b.l)),pad=Math.max((max-min)*.07,2),vmax=Math.max(...bars.map(b=>b.v),1);
    const x=i=>left+step*(i+.5),y=v=>top+(max+pad-v)/(max-min+2*pad)*(priceEnd-top);
    const signalAt=bars.findIndex(b=>row.signal_date>=b.start&&row.signal_date<=b.end);
    if(signalAt>=0){ctx.fillStyle='#f0edfa';ctx.fillRect(x(signalAt)-step/2,top,right-x(signalAt)+step/2,macdEnd-top)}
    ctx.font='11px "Microsoft YaHei",sans-serif';ctx.textBaseline='middle';
    for(let i=0;i<5;i++){const py=top+(priceEnd-top)*i/4;ctx.strokeStyle='#e1e4f0';ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(right,py);ctx.stroke();ctx.fillStyle='#65708a';ctx.textAlign='left';ctx.fillText(money(max+pad-(max-min+2*pad)*i/4),right+5,py)}
    bars.forEach((b,i)=>{const px=x(i);ctx.strokeStyle=b.c>=b.o?up:down;ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.moveTo(px,y(b.h));ctx.lineTo(px,y(b.l));ctx.stroke();ctx.fillRect(px-cw/2,Math.min(y(b.o),y(b.c)),cw,Math.max(1,Math.abs(y(b.o)-y(b.c))));ctx.globalAlpha=.55;ctx.fillRect(px-cw/2,volumeEnd-(volumeEnd-volumeTop)*b.v/vmax,cw,Math.max(.6,(volumeEnd-volumeTop)*b.v/vmax));ctx.globalAlpha=1});
    for(const [field,color] of [['ma5','#c9a025'],['ma20','#4387bc']]){ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath();let begun=false;bars.forEach((b,i)=>{if(b[field]!=null){begun?ctx.lineTo(x(i),y(b[field])):ctx.moveTo(x(i),y(b[field]));begun=true}});ctx.stroke()}
    const macdValues=bars.flatMap(b=>[b.dif,b.dea,b.hist].filter(v=>v!=null));
    const amp=Math.max(1,...macdValues.map(Math.abs))*1.08,my=v=>(macdTop+macdEnd)/2-v/amp*(macdEnd-macdTop)/2;
    ctx.strokeStyle='#d8dce9';ctx.beginPath();ctx.moveTo(left,my(0));ctx.lineTo(right,my(0));ctx.stroke();
    bars.forEach((b,i)=>{if(b.hist!=null){ctx.fillStyle=b.hist>=0?up:down;ctx.fillRect(x(i)-cw/2,Math.min(my(0),my(b.hist)),cw,Math.max(.5,Math.abs(my(b.hist)-my(0))))}});
    for(const [field,color] of [['dif','#cb9e24'],['dea','#7868c3']]){ctx.strokeStyle=color;ctx.beginPath();let begun=false;bars.forEach((b,i)=>{if(b[field]!=null){begun?ctx.lineTo(x(i),my(b[field])):ctx.moveTo(x(i),my(b[field]));begun=true}});ctx.stroke()}
    ctx.fillStyle='#717c95';ctx.textAlign='right';ctx.fillText('成交量',w-3,volumeTop+8);ctx.fillText('MACD',w-3,macdTop+8);ctx.fillText('0',right+5,my(0));
    const marks=[[row.signal_date,'信号',purple],[row.entry_date,'买','#987032'],...[5,10,20,30].map(n=>[row['exit_date_'+n],n+'日','#8b83a9'])];
    const grouped=new Map();for(const [d,label,color] of marks){if(!d)continue;const i=bars.findIndex(b=>d>=b.start&&d<=b.end);if(i<0)continue;if(!grouped.has(i))grouped.set(i,{labels:[],color});grouped.get(i).labels.push(label)}
    const tracks=[[],[],[]];for(const [i,m] of grouped){const px=x(i);ctx.setLineDash([3,3]);ctx.strokeStyle=m.color;ctx.beginPath();ctx.moveTo(px,top);ctx.lineTo(px,macdEnd);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=m.color;ctx.textAlign='left';const labels=m.labels.join('/'),width=ctx.measureText(labels).width,begin=Math.max(left,Math.min(right-width,px-width/2));const track=tracks.findIndex(items=>items.every(([a,b])=>begin>b+4||begin+width<a-4));if(track>=0){tracks[track].push([begin,begin+width]);ctx.fillText(labels,begin,9+track*13)}}
    if(selected>=0){ctx.strokeStyle='#575c75';ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(x(selected),top);ctx.lineTo(x(selected),macdEnd);ctx.stroke();ctx.setLineDash([])}
    const ticks=w<500?3:5;for(let i=0;i<ticks;i++){const j=Math.round((bars.length-1)*i/(ticks-1));ctx.textAlign=i===0?'left':i===ticks-1?'right':'center';ctx.fillStyle='#6d7791';ctx.fillText(day(bars[j].date).slice(0,7),x(j),h-8)}
    return {left,step};
  }
  window.mountSignalCandles=(root,row,supplementary)=>{
    if(supplementary?.quote_date)row={...row,entry_date:Number(supplementary.quote_date.replaceAll('-',''))};
    let toolbar=root.querySelector('.candle-toolbar');if(!toolbar){toolbar=document.createElement('div');toolbar.className='candle-toolbar';const old=root.querySelector('select');old.closest('label').replaceWith(toolbar)}
    toolbar.innerHTML='<label>周期 <select class="candle-period" aria-label="K线周期"><option value="day">日K</option><option value="week">周K</option><option value="month">月K</option></select></label><label>信号前 <select class="candle-lookback" aria-label="信号前区间"></select></label><label>信号后 <select class="candle-forward" aria-label="信号后区间"><option value="30">30日</option><option value="60">60日</option><option value="120" selected>120日</option><option value="all">全部行情</option></select></label>';
    const period=toolbar.querySelector('.candle-period'),range=toolbar.querySelector('.candle-lookback'),forward=toolbar.querySelector('.candle-forward');
    const info=root.querySelector('.candle-info'),canvas=root.querySelector('canvas'),caption=root.querySelector('.candle-caption');
    let daily=[],all=[],shown=[],selection=0,geom=null,alive=true,asof=0;
    function options(){const type=period.value,values=type==='day'?[60,120,250,500]:type==='week'?[26,52,104,260]:[12,24,36,60],unit={day:'日',week:'周',month:'月'}[type];range.innerHTML=values.map(n=>`<option value="${n}">${n}${unit}</option>`).join('')+'<option value="all">全部历史</option>';range.value=type==='day'?'250':type==='week'?'104':'60'}
    function render(){if(!alive||!shown.length)return;const b=shown[selection],type=period.value;info.replaceChildren();const description=document.createElement('span');description.textContent=(type==='day'?day(b.date):day(b.start)+'～'+day(b.date))+'　开 '+money(b.o)+'　高 '+money(b.h)+'　低 '+money(b.l)+'　收 '+money(b.c)+'　';const change=document.createElement('strong');change.className='candle-change '+(b.pct>0?'gain':b.pct<0?'loss':'');change.textContent='涨跌幅 '+pct(b.pct);const rest=document.createElement('span');rest.textContent='　量 '+Math.round(b.v).toLocaleString('zh-CN')+' 股'+(b.complete?'':'　周期未完结')+'　'+(b.dif==null?'MACD暖机不足26根':'DIF '+money(b.dif)+' / DEA '+money(b.dea)+' / MACD '+money(b.hist));info.append(description,change,rest);geom=draw(canvas,shown,selection,row,type)}
    function slice(){if(!daily.length)return;const type=period.value;all=periods(daily,type,asof);const at=all.findIndex(b=>row.signal_date>=b.start&&row.signal_date<=b.end);if(at<0)return;const signalDaily=daily.findIndex(b=>b[0]===row.signal_date);const endDate=forward.value==='all'?asof:daily[Math.min(daily.length-1,signalDaily+Number(forward.value))][0];const start=range.value==='all'?0:Math.max(0,at-Number(range.value));shown=all.slice(start).filter(b=>b.start<=endDate);selection=at-start;const unit={day:'日',week:'周',month:'月'}[type];caption.textContent=`${day(shown[0].start)} 至 ${day(shown[shown.length-1].date)} · 信号前 ${selection} 根 / 信号后 ${shown.length-selection-1} 根${unit}K · MA5（金）/ MA20（蓝） · MACD(12,26,9)`+(type==='day'?' · 涨跌幅按除权参考前收盘价计算':' · 周月涨跌幅按复权前周期收盘价计算。信号所在周期可能含之后行情，仅供复盘；选股只使用已完成周月。')+(supplementary?' · 含11月17日单独核对行情':'');render()}
    function typeChange(){options();slice()}
    function pointer(e){if(!geom)return;const at=Math.floor((e.clientX-canvas.getBoundingClientRect().left-geom.left)/geom.step);if(at>=0&&at<shown.length&&at!==selection){selection=at;render()}}
    function key(e){const step=e.key==='ArrowLeft'?-1:e.key==='ArrowRight'?1:0;if(!step||!shown.length)return;e.preventDefault();selection=Math.max(0,Math.min(shown.length-1,selection+step));render()}
    options();period.addEventListener('change',typeChange);range.addEventListener('change',slice);forward.addEventListener('change',slice);canvas.addEventListener('pointermove',pointer);canvas.addEventListener('pointerdown',pointer);canvas.addEventListener('keydown',key);const observer=new ResizeObserver(render);observer.observe(canvas);
    load(row).then(data=>{if(!alive)return;asof=data.asof;const anchor=data.rows.find(b=>b[0]===row.signal_date);if(!anchor)throw Error('Missing signal');daily=data.rows.map(b=>[b[0],...b.slice(1,5).map(p=>p*b[7]/anchor[7]),b[5],b[6]]);slice()}).catch(()=>{if(alive){info.textContent='K线加载失败，请关闭后重试。';canvas.hidden=true}});
    return ()=>{alive=false;observer.disconnect();period.removeEventListener('change',typeChange);range.removeEventListener('change',slice);forward.removeEventListener('change',slice);canvas.removeEventListener('pointermove',pointer);canvas.removeEventListener('pointerdown',pointer);canvas.removeEventListener('keydown',key)};
  };
})();
