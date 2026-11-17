"use strict";

(() => {
  window.candleRanges = Object.create(null);
  window.researchCandleRanges = Object.create(null);
  const loading = new Map();
  const gain = "#c24c65";
  const loss = "#258b78";
  const signalColor = "#6655c9";
  const entryColor = "#876128";
  const horizons = [5, 10, 20, 30];
  const number = new Intl.NumberFormat("zh-CN");

  function loadCandles(row) {
    const key = row.code + "-" + row.signal_date;
    const shard = Number(row.code.split(".")[1]) % 32;
    const research = row.candle_source === "research";
    const bucket = research ? window.researchCandleRanges : window.candleRanges;
    const directory = research ? "research-chart-data" : "chart-data";
    const loadingKey = directory + "/" + shard;
    if (!loading.has(loadingKey)) {
      const request = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = directory + "/" + shard + ".js";
        script.onload = () => {
          script.remove();
          if (bucket[shard]) resolve(bucket[shard]);
          else reject(new Error("Invalid OHLCV data"));
        };
        script.onerror = () => {
          script.remove();
          reject(new Error("Could not load OHLCV data"));
        };
        document.head.append(script);
      }).catch((error) => {
        loading.delete(loadingKey);
        throw error;
      });
      loading.set(loadingKey, request);
    }
    return loading.get(loadingKey).then((values) => {
      if (!values[key]) throw new Error("Missing OHLCV range");
      return values[key];
    });
  }

  function day(value) {
    const raw = String(value);
    return raw.slice(0, 4) + "-" + raw.slice(4, 6) + "-" + raw.slice(6, 8);
  }

  function amounts(cents) {
    return (cents / 100).toFixed(2);
  }

  function drawChart(canvas, bars, selected, signal, entry, exits) {
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (width <= 90 || height <= 90) return null;
    const ratio = Math.min(window.devicePixelRatio || 1, 2.5);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    const left = 9, right = width - 46, firstY = 21;
    const priceEnd = height - 78, volumeTop = height - 59, volumeEnd = height - 21;
    const xStep = (right - left) / bars.length;
    const candleWidth = Math.max(1, Math.min(xStep * 0.59, 10));
    const max = Math.max(...bars.map((bar) => bar[2]));
    const min = Math.min(...bars.map((bar) => bar[3]));
    const extra = Math.max((max - min) * 0.08, 2);
    const peakVolume = Math.max(...bars.map((bar) => bar[5]), 1);
    const x = (i) => left + xStep * (i + 0.5);
    const y = (cents) => firstY + (max + extra - cents) / (max - min + extra * 2) * (priceEnd - firstY);

    const eventIndex = bars.findIndex(([value]) => value === signal);
    if (eventIndex + 1 < bars.length) {
      const after = x(eventIndex) + xStep / 2;
      ctx.fillStyle = "#f0edfa";
      ctx.fillRect(after, firstY, right - after, volumeEnd - firstY);
    }
    ctx.font = '10px Inter, "Microsoft YaHei UI", sans-serif';
    ctx.textBaseline = "middle";
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const py = firstY + (priceEnd - firstY) * i / 4;
      const cents = max + extra - (max - min + extra * 2) * i / 4;
      ctx.strokeStyle = "#e5e7f1";
      ctx.beginPath();
      ctx.moveTo(left, py + 0.5);
      ctx.lineTo(right, py + 0.5);
      ctx.stroke();
      ctx.fillStyle = "#626c87";
      ctx.fillText(amounts(cents), right + 5, py);
    }

    const markers = [
      [signal, "信号", signalColor],
      [entry, "买", entryColor],
      ...exits.map(([value, horizon]) => [value, horizon + "日", "#827da3"]),
    ].filter(([value]) => value);
    for (const [value, label, color] of markers) {
      const i = bars.findIndex(([actual]) => actual === value);
      if (i < 0) continue;
      const px = x(i);
      ctx.strokeStyle = color;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(px, firstY - 6);
      ctx.lineTo(px, volumeEnd);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.textAlign = label === "信号" ? "right" : "left";
      ctx.fillText(label, label === "信号" ? px - 3 : px + 3, label === "买" ? 15 : 8);
    }

    bars.forEach((bar, i) => {
      const [, open, high, low, close, volume] = bar;
      const px = x(i);
      ctx.strokeStyle = close >= open ? gain : loss;
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = i === selected ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(px, y(high));
      ctx.lineTo(px, y(low));
      ctx.stroke();
      ctx.fillRect(px - candleWidth / 2, Math.min(y(open), y(close)), candleWidth,
        Math.max(1, Math.abs(y(open) - y(close))));
      const size = (volumeEnd - volumeTop) * volume / peakVolume;
      ctx.globalAlpha = i === selected ? 1 : 0.5;
      ctx.fillRect(px - candleWidth / 2, volumeEnd - size, candleWidth, Math.max(0.5, size));
      ctx.globalAlpha = 1;
    });

    if (selected >= 0) {
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#4f4d68";
      ctx.setLineDash([1, 3]);
      ctx.beginPath();
      ctx.moveTo(x(selected), firstY);
      ctx.lineTo(x(selected), volumeEnd);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = "#626c87";
    ctx.textAlign = "right";
    ctx.fillText("成交量", width - 5, volumeEnd - 10);
    const ticks = width < 440 ? 3 : 5;
    for (let i = 0; i < ticks; i++) {
      const index = Math.round((bars.length - 1) * i / (ticks - 1));
      ctx.textAlign = i === 0 ? "left" : i === ticks - 1 ? "right" : "center";
      ctx.fillText(day(bars[index][0]).slice(5), x(index), height - 9);
    }
    return { left, xStep };
  }

  window.mountSignalCandles = (root, row, supplementary) => {
    const range = root.querySelector("select");
    const info = root.querySelector(".candle-info");
    const canvas = root.querySelector("canvas");
    const caption = root.querySelector(".candle-caption");
    const entry = supplementary ? Number(supplementary.quote_date.replaceAll("-", "")) : row.entry_date;
    const exits = horizons.map((h) => [row["exit_date_" + h], h]);
    let allBars = [];
    let shown = [];
    let selected = 0;
    let geometry = null;
    let alive = true;

    function render() {
      if (!shown.length) return;
      const [d, open, high, low, close, volume] = shown[selected];
      info.textContent = day(d) + "　开 " + amounts(open) + "　高 " + amounts(high) + "　低 " +
        amounts(low) + "　收 " + amounts(close) + "　量 " + number.format(volume) + " 股";
      geometry = drawChart(canvas, shown, selected, row.signal_date, entry, exits);
    }

    function setRange() {
      if (!allBars.length) return;
      const at = allBars.findIndex(([value]) => value === row.signal_date);
      const start = range.value === "full" ? allBars.findIndex(([value]) => value >= row.pit_high_date)
        : Math.max(0, at - Number(range.value));
      shown = allBars.slice(Math.max(0, start));
      selected = shown.findIndex(([value]) => value === row.signal_date);
      const after = shown.length - selected - 1;
      caption.textContent = "信号前 " + selected + " 根 · 信号后 " + after + " 根日 K" +
        (supplementary ? " · 含 11 月 17 日单独核对行情" : "");
      render();
    }

    function pick(event) {
      if (!geometry || !shown.length) return;
      const at = Math.floor((event.clientX - canvas.getBoundingClientRect().left - geometry.left) / geometry.xStep);
      if (at < 0 || at >= shown.length || at === selected) return;
      selected = at;
      render();
    }

    range.addEventListener("change", setRange);
    canvas.addEventListener("pointermove", pick);
    canvas.addEventListener("pointerdown", pick);
    function keydown(event) {
      if (!shown.length) return;
      const step = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
      if (!step) return;
      event.preventDefault();
      selected = Math.max(0, Math.min(shown.length - 1, selected + step));
      render();
    }
    canvas.addEventListener("keydown", keydown);
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    loadCandles(row).then((bars) => {
      if (!alive) return;
      allBars = bars;
      range.disabled = false;
      setRange();
    }).catch(() => {
      if (!alive) return;
      info.textContent = "K 线加载失败。关闭详情后可重新打开。";
      canvas.hidden = true;
    });
    return () => {
      alive = false;
      observer.disconnect();
      range.removeEventListener("change", setRange);
      canvas.removeEventListener("pointermove", pick);
      canvas.removeEventListener("pointerdown", pick);
      canvas.removeEventListener("keydown", keydown);
    };
  };
})();
