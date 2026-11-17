"use strict";

const { coverage, fields, records, summary, bars, quote } = window.snapshot;
const $ = (id) => document.getElementById(id);
const horizons = [5, 10, 20, 30];
const signals = records.map((row) => Object.fromEntries(fields.map((field, index) => [field, row[index]])));
const groups = {
  open_not_at_limit: ["开盘未涨停", "normal"],
  opens_at_limit: ["涨停开盘 · 开板", "limit"],
  one_price_limit: ["一字涨停", "limit"],
  pending: ["次日超出快照", "pending"],
  no_bar: ["次日无日线", "pending"],
  unknown_limit: ["涨停档位待核实", "pending"],
};
const chartGroups = {
  open: [
    { label: "开盘未涨停", key: "开盘未涨停", color: "#6257c6" },
    { label: "开盘涨停（理论）", key: "开盘涨停（理论）", color: "#73bcab" },
  ],
  close: [
    { label: "收盘未涨停", key: "次日收盘未涨停（含假设成交）", color: "#6257c6" },
    { label: "收盘涨停", key: "次日收盘涨停（含假设成交）", color: "#73bcab" },
  ],
};
const bear = signals.find((r) => r.code === quote.code && date(r.signal_date) === quote.signal_date);
const count = new Intl.NumberFormat("zh-CN");
const pageSize = 12;
const state = { page: 1, sort: "signal_date", descending: true, basis: "open" };
const controls = ["search", "year", "entry", "close"];
let chartBars = [];
let closeCandles = null;

function date(value, divider = "-") {
  if (!value) return "—";
  return String(value).replaceAll("-", "").replace(/^(\d{4})(\d{2})(\d{2})$/, "$1" + divider + "$2" + divider + "$3");
}

function pct(value, sign = true) {
  if (value == null) return "—";
  return (sign && value > 0 ? "+" : "") + (value * 100).toFixed(2) + "%";
}

function price(value) {
  return value == null ? "—" : value.toFixed(2) + " 元";
}

function multiple(value) {
  return value == null ? "—" : value.toFixed(2) + "×";
}

function safe(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
}

function ticker(code) {
  const [market, number] = code.split(".");
  return number + "." + market.toUpperCase();
}

function tone(value) {
  return value == null ? "missing" : value > 0 ? "positive" : value < 0 ? "negative" : "";
}

function rowMetric(key, horizon) {
  return summary.find((row) => row["组别"] === key && row["持有交易日"] === horizon);
}

function updateStats() {
  const c = coverage.entry_groups;
  for (const [id, value] of Object.entries({
    signals: coverage.signals, codes: coverage.signal_codes, universe: coverage.securities,
    normal: c.open_not_at_limit, limit: c.opens_at_limit + c.one_price_limit,
    opened: c.opens_at_limit, locked: c.one_price_limit,
  })) $("stat-" + id).textContent = count.format(value);
  $("stat-win").textContent = pct(rowMetric("开盘未涨停", 5)["胜率"], false);
  $("case-return").textContent = pct(bear.signal_return);
  $("case-volume-5").textContent = multiple(bear.signal_volume_ratio_5);
  $("case-volume").textContent = multiple(bear.signal_volume_ratio);
  const years = [...new Set(signals.map((row) => Math.trunc(row.signal_date / 10000)))];
  $("year").append(...years.map((year) => new Option(year + " 年", String(year))));
}

function canvasContext(canvas) {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return null;
  const scale = Math.min(window.devicePixelRatio || 1, 2.5);
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  return { ctx, w, h };
}

function paintReturns() {
  const canvas = $("return-chart");
  const dimensions = canvasContext(canvas);
  if (!dimensions) return;
  const { ctx, w, h } = dimensions;
  const metric = $("metric").value;
  const active = chartGroups[state.basis];
  const series = active.map(({ key }) => horizons.map((horizon) => rowMetric(key, horizon)));
  const percentages = series.flatMap((rows) => rows.map((row) => row[metric] * 100));
  const step = metric === "胜率" ? 20 : 2;
  const cap = Math.max(step, Math.ceil((Math.max(...percentages) + step * 0.14) / step) * step);
  const low = Math.min(0, Math.floor((Math.min(0, ...percentages) - step * 0.04) / step) * step);
  const bottom = metric === "胜率" ? 0 : low;
  const left = 43, right = 8, top = 12, height = h - top - 29;
  const slot = (w - left - right) / horizons.length;
  const bar = Math.min(slot / 3.7, 30);
  const y = (value) => top + (cap - value) * height / (cap - bottom);
  ctx.font = '10px Inter, "Microsoft YaHei UI", sans-serif';
  ctx.textBaseline = "middle";
  for (let tick = bottom; tick <= cap; tick += step) {
    const at = y(tick);
    ctx.strokeStyle = tick === 0 ? "#c5c8d9" : "#eceef5";
    ctx.beginPath();
    ctx.moveTo(left, at + 0.5);
    ctx.lineTo(w - right, at + 0.5);
    ctx.stroke();
    ctx.fillStyle = "#69728b";
    ctx.textAlign = "right";
    ctx.fillText(tick + "%", left - 7, at);
  }
  chartBars = [];
  horizons.forEach((horizon, index) => {
    const center = left + slot * (index + 0.5);
    ctx.fillStyle = "#545d75";
    ctx.textAlign = "center";
    ctx.fillText(horizon + " 日", center, h - 11);
    active.forEach((group, i) => {
      const row = series[i][index];
      const value = row[metric];
      const end = y(value * 100);
      const zero = y(0);
      const startX = center + (i === 0 ? -bar - 2 : 2);
      ctx.fillStyle = group.color;
      ctx.fillRect(startX, Math.min(end, zero), bar, Math.max(1, Math.abs(end - zero)));
      chartBars.push({ left: startX, width: bar, top: Math.min(end, zero), bottom: Math.max(end, zero), horizon, row, group });
      ctx.fillStyle = i === 0 ? "#4d41a3" : "#236d5c";
      const label = pct(value, metric !== "胜率");
      const yLabel = value >= 0 ? end - 7 : end + 7;
      if (w >= 365) ctx.fillText(label, startX + bar / 2, yLabel);
    });
  });
  canvas.setAttribute("aria-label", metric + "对比。" + active.map(
    (group, i) => group.label + "，" + series[i].map((row, j) => horizons[j] + "日 " + pct(row[metric], metric !== "胜率") + "，" + row["收益样本数"] + "笔").join("，")
  ).join("。"));
  $("chart-legend").innerHTML = active.map((group) => '<span><i style="background:' + group.color + '"></i>' + group.label + "</span>").join("");
  $("chart-note").textContent = state.basis === "open"
    ? "涨停开盘组按开盘价估值，含 " + coverage.entry_groups.one_price_limit + " 个全天一字涨停样本。"
    : "收盘状态用于复盘分组，两组均含按开盘价假设成交的样本。";
  $("chart-table-head").innerHTML = "<tr><th scope='col'>持有期</th>" + active.map(
    (group) => "<th scope='col'>" + group.label + "<br>" + metric + " / 样本数</th>"
  ).join("") + "</tr>";
  $("chart-table-body").innerHTML = horizons.map((horizon, i) => "<tr><th scope='row'>" + horizon + " 日</th>" +
    series.map((rows) => "<td>" + pct(rows[i][metric], metric !== "胜率") + " / " + count.format(rows[i]["收益样本数"]) + "</td>").join("") + "</tr>"
  ).join("");
}

function paintBear(canvas) {
  const dimensions = canvasContext(canvas);
  if (!dimensions) return;
  const { ctx, w, h } = dimensions;
  const left = 4;
  const right = 11;
  const last = bars.length - 1;
  const values = bars.map((bar) => bar[1]);
  const lo = Math.min(...values) - 1;
  const hi = Math.max(...values) + 1;
  const priceHeight = h - 29;
  const x = (index) => left + index * (w - left - right) / last;
  const y = (value) => 8 + (hi - value) / (hi - lo) * priceHeight;
  for (const fraction of [0.25, 0.50, 0.75, 1]) {
    const at = 8 + priceHeight * fraction;
    ctx.strokeStyle = "#3f4261";
    ctx.beginPath();
    ctx.moveTo(left, at);
    ctx.lineTo(w - right, at);
    ctx.stroke();
  }
  const shade = ctx.createLinearGradient(0, 8, 0, priceHeight + 8);
  shade.addColorStop(0, "#a098f952");
  shade.addColorStop(1, "#a098f909");
  ctx.beginPath();
  ctx.moveTo(x(0), priceHeight + 8);
  bars.forEach((bar, i) => ctx.lineTo(x(i), y(bar[1])));
  ctx.lineTo(x(last), priceHeight + 8);
  ctx.fillStyle = shade;
  ctx.fill();
  ctx.beginPath();
  bars.forEach((bar, i) => { if (i === 0) ctx.moveTo(x(i), y(bar[1])); else ctx.lineTo(x(i), y(bar[1])); });
  ctx.lineWidth = 1.8;
  ctx.strokeStyle = "#b9aeff";
  ctx.stroke();
  const milestones = [bear.pit_high_date, bear.floor_date, bear.rally_peak_date, bear.signal_date];
  milestones.forEach((day, i) => {
    const index = bars.findIndex((bar) => bar[0] === day);
    if (index < 0) return;
    ctx.beginPath();
    ctx.arc(x(index), y(values[index]), i === 3 ? 4.3 : 3.4, 0, 2 * Math.PI);
    ctx.fillStyle = i === 3 ? "#ff9caa" : "#d1cbff";
    ctx.fill();
  });
  ctx.fillStyle = "#c4c9e7";
  ctx.font = '9px Inter, "Microsoft YaHei UI", sans-serif';
  for (const month of [3, 6, 9, 11]) {
    const i = bars.findIndex(([day]) => Math.trunc(day / 100) % 100 === month);
    if (i >= 0) ctx.fillText(String(month).padStart(2, "0") + "月", Math.min(x(i), w - 23), h - 4);
  }
}

function matches(row) {
  const term = $("search").value.trim().toLowerCase();
  const year = $("year").value;
  const entry = $("entry").value;
  const close = $("close").value;
  if (term && ![row.name, row.code, ticker(row.code)].some((item) => item?.toLowerCase().includes(term))) return false;
  if (year && String(row.signal_date).slice(0, 4) !== year) return false;
  if (entry === "any_limit" && !["opens_at_limit", "one_price_limit"].includes(row.entry_group)) return false;
  if (entry && entry !== "any_limit" && entry !== row.entry_group) return false;
  return !close || row.entry_closes_at_limit === (close === "unknown" ? null : close === "yes");
}

function compare(a, b) {
  const x = a[state.sort], y = b[state.sort];
  if (x == null && y == null) return b.signal_date - a.signal_date;
  if (x == null) return 1;
  if (y == null) return -1;
  return (state.descending ? y - x : x - y) || b.signal_date - a.signal_date || a.code.localeCompare(b.code);
}

function pill(label, css) {
  return '<span class="pill ' + css + '">' + label + "</span>";
}

function closeStatus(value) {
  if (value == null) return '<span class="missing">—</span>';
  return value ? pill("收盘涨停", "limit") : pill("未涨停", "");
}

function renderTable() {
  const rows = signals.filter(matches).sort(compare);
  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  state.page = Math.min(state.page, pages);
  const start = (state.page - 1) * pageSize;
  $("signal-rows").innerHTML = rows.slice(start, start + pageSize).map((row) => {
    const [label, css] = groups[row.entry_group] || groups.unknown_limit;
    const returns = horizons.map((h) => {
      const value = row["return_" + h];
      return '<td class="number ' + tone(value) + '">' + pct(value) + "</td>";
    }).join("");
    return `<tr>
      <td><button class="symbol" type="button" data-signal="${safe(row.code)}-${row.signal_date}">${safe(row.name ?? ticker(row.code))}${row.name ? '<span class="ticker">' + safe(ticker(row.code)) + '</span>' : ''}</button></td>
      <td>${date(row.signal_date)}</td><td class="number ${tone(row.signal_return)}">${pct(row.signal_return)}</td>
      <td>${pill(label, css)}</td><td>${closeStatus(row.entry_closes_at_limit)}</td>${returns}
    </tr>`;
  }).join("") || '<tr class="empty"><td colspan="9">当前筛选条件下没有信号</td></tr>';
  $("match-count").textContent = rows.length
    ? "共 " + count.format(rows.length) + " 条 · 显示 " + count.format(start + 1) + "–" + count.format(Math.min(start + pageSize, rows.length))
    : "共 0 条";
  $("page-number").textContent = state.page + " / " + pages;
  $("previous").disabled = state.page === 1;
  $("next").disabled = state.page === pages;
  document.querySelectorAll("[data-sort]").forEach((button) => {
    const th = button.closest("th");
    if (button.dataset.sort === state.sort) th.setAttribute("aria-sort", state.descending ? "descending" : "ascending");
    else th.removeAttribute("aria-sort");
    button.lastElementChild.textContent = button.dataset.sort === state.sort ? (state.descending ? "↓" : "↑") : "↕";
  });
}

function cell(label, value, note, css = "") {
  return `<article class="detail-cell"><h4>${label}</h4><strong class="${css}">${value}</strong><small>${note}</small></article>`;
}

function showDetail(row) {
  const isBear = row === bear;
  const title = safe(row.name || "历史信号");
  $("detail-heading").innerHTML = `<h2 id="detail-title">${title}</h2><p>${safe(ticker(row.code))} · 信号日 ${date(row.signal_date)}</p>`;
  const phases = [
    cell("挖坑下跌", pct(-row.pit_drawdown), date(row.pit_high_date) + " 至 " + date(row.floor_date), "negative"),
    cell("放量拉升", pct(row.rally_return), row.rally_days + " 根日 K · " + multiple(row.rally_volume_ratio_5) + " 坑底 5 日均量 · " + multiple(row.rally_volume_ratio_10) + " 坑底 10 日均量", "positive"),
    cell("缩量回落", pct(-row.wash_drop), row.wash_days + " 根日 K · " + pct(row.wash_volume_ratio, false) + " 拉升量", "negative"),
    cell("放量突破", pct(row.signal_return), multiple(row.signal_volume_ratio_5) + " 前 5 日均量 · " + multiple(row.signal_volume_ratio) + " 前 10 日均量 · 上影 / 实体 " + pct(row.upper_wick_body_ratio, false), "positive"),
  ];
  const open = isBear ? quote.open : row.entry_open;
  const close = isBear ? quote.close : row.entry_close;
  const limit = isBear ? quote.upper_limit : row.entry_limit_price;
  const entryDate = isBear ? quote.quote_date : row.entry_date;
  const group = isBear ? "开盘未涨停 · 收盘涨停" : (groups[row.entry_group] || groups.unknown_limit)[0];
  const nextDay = [
    cell("买入日", date(entryDate), safe(group)),
    cell("开盘", price(open), "未复权"),
    cell("收盘", price(close), "未复权"),
    cell("涨停价", price(limit), "未复权"),
  ];
  const returns = horizons.map((h) => {
    const value = row["return_" + h];
    const day = row["exit_date_" + h];
    const note = value != null ? date(day) + (row["exit_one_price_down_" + h] ? " · 一字跌停估值" : " · 收盘估值")
      : isBear ? "尚未到期" : day ? date(day) + " · 缺少到期日线" : "暂无可用收益";
    return cell(h + " 日", pct(value), note, tone(value));
  });
  $("detail-body").innerHTML = `
    <div class="phases">${phases.join("")}</div>
    <section class="detail-section candles" id="signal-candles">
      <div class="candle-header"><div><h3>信号前后 K 线</h3><p>按信号日复权 · 成交量以股计</p></div>
        <label class="control-label">区间<select id="candle-range" disabled><option value="30">信号前 30 日</option><option value="60">信号前 60 日</option><option value="full">完整形态</option></select></label>
      </div>
      <div class="candle-info" role="status">正在加载历史 K 线…</div>
      <canvas role="img" tabindex="0" aria-label="信号前后日 K 线。可用左右方向键查看各交易日的开盘、最高、最低、收盘与成交量。"></canvas>
      <p class="candle-caption"></p>
      <p class="candle-key">浅紫底为信号后区间 · 图上「买」为次日开盘，「5日」等为到期估值日。</p>
    </section>
    <section class="detail-section"><h3>次日行情</h3><div class="quote-grid">${nextDay.join("")}</div>
      ${isBear ? '<p>11 月 17 日为单独核对的行情。<a href="' + safe(quote.source) + '" target="_blank" rel="noopener">腾讯行情</a></p>' : ''}
    </section>
    <section class="detail-section"><h3>持有期收益</h3><div class="returns">${returns.join("")}</div></section>`;
  $("detail").showModal();
  closeCandles = window.mountSignalCandles($("signal-candles"), row, isBear ? quote : null);
}

updateStats();
renderTable();
const chartObserver = new ResizeObserver(() => {
  paintReturns();
  paintBear($("bear-chart"));
});
chartObserver.observe($("return-chart"));
chartObserver.observe($("bear-chart"));
$("basis").addEventListener("click", (event) => {
  const button = event.target.closest("[data-value]");
  if (!button) return;
  state.basis = button.dataset.value;
  $("basis").querySelectorAll("button").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
  paintReturns();
});
$("metric").addEventListener("change", paintReturns);
$("return-chart").addEventListener("pointermove", (event) => {
  const bounds = $("return-chart").getBoundingClientRect();
  const x = event.clientX - bounds.left, y = event.clientY - bounds.top;
  const hit = chartBars.find((bar) => x >= bar.left && x <= bar.left + bar.width && y >= bar.top - 5 && y <= bar.bottom + 5);
  $("return-chart").title = hit ? hit.group.label + " · " + hit.horizon + " 日 · " + hit.row["收益样本数"] + " 笔" : "";
});
for (const id of controls) {
  $(id).addEventListener(id === "search" ? "input" : "change", () => {
    state.page = 1;
    renderTable();
  });
}
$("reset").addEventListener("click", () => {
  controls.forEach((id) => { $(id).value = ""; });
  Object.assign(state, { page: 1, sort: "signal_date", descending: true });
  renderTable();
});
document.querySelectorAll("[data-sort]").forEach((button) => button.addEventListener("click", () => {
  const sort = button.dataset.sort;
  state.descending = sort === state.sort ? !state.descending : true;
  state.sort = sort;
  state.page = 1;
  renderTable();
}));
$("previous").addEventListener("click", () => { state.page -= 1; renderTable(); });
$("next").addEventListener("click", () => { state.page += 1; renderTable(); });
$("signal-rows").addEventListener("click", (event) => {
  const button = event.target.closest("[data-signal]");
  if (button) showDetail(signals.find((row) => row.code + "-" + row.signal_date === button.dataset.signal));
});
$("case-detail").addEventListener("click", () => showDetail(bear));
$("detail").addEventListener("close", () => {
  closeCandles?.();
  closeCandles = null;
});
const navigation = [...document.querySelectorAll(".rail nav a")];
function highlightNavigation() {
  const current = location.hash || "#overview";
  navigation.forEach((link) => {
    if (link.hash === current) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  });
}
addEventListener("hashchange", highlightNavigation);
highlightNavigation();
