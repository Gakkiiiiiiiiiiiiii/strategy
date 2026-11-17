# A 股量价策略与多周期回测

从“挖坑下跌 → 放量拉升 → 缩量回落 → 再次放量上涨”出发，逐步扩展到首波反转、底部阳阴量、MACD 底背离、周线和月线的策略研究。

包含信号识别、5 / 10 / 20 / 30 日收益统计、有限资金组合模拟、参数搜索、逐笔核对，以及可以切换日 K / 周 K / 月 K 的静态网页。

## 在线查看

- [量能与周线平衡版](https://volume-signal-review.sharp-coin-9055.chatgpt.site/multiframe.html)
- [量能与 MACD 排序版](https://volume-signal-review.sharp-coin-9055.chatgpt.site/macd.html)
- [上一版首波反转策略](https://volume-signal-review.sharp-coin-9055.chatgpt.site/research.html)
- [原始第二波形态与小熊电器案例](https://volume-signal-review.sharp-coin-9055.chatgpt.site/)

本仓库保存代码和少量研究结果。完整行情缓存、预计算矩阵、网页行情分片、安装包和部署凭据没有入库。克隆后须准备数据并运行导出步骤，才能在本地加载完整网页；也可以直接访问上述在线页面。

## 目录

| 路径 | 用途 |
| --- | --- |
| `backtest.py` | TDX 日线解析、复权、第二波形态及次日涨停分组回测 |
| `update_market.py` | BaoStock 日线补充与历史 ST 状态缓存 |
| `optimize.py` | 原始形态参数搜索 |
| `research_v2.py` | 首波反转事件、交易结果缓存与参数搜索 |
| `portfolio_search.py` | 有限现金、整手、持仓约束下的组合模拟 |
| `review_v2.py`、`finalize_v2.py` | 上一版结果汇总及核对 |
| `research_v3.py` | 阳阴量、确认后的 MACD 背离、长历史窗口、已完成周月线特征 |
| `finalize_v3.py` | 多周期候选选择、敏感性、逐项加条件对照及滚动时间检验 |
| `compare_rank_v3.py` | 分离持有期与排序规则的贡献 |
| `audit_v3.py` | 从原始价格重算成交，并截断历史重算指标 |
| `sites/volume-signals/export_*.py` | 将研究结果导出到静态网页 |
| `sites/volume-signals/dist/` | HTML、CSS、JavaScript 和 K 线交互组件 |
| `results/research_v2/final_report.json` | 上一版回测结果快照 |
| `results/research_v3/` | 最新结果、参数对照、核对记录和两组模拟成交明细 |

## 环境与数据

研究环境为 Windows、Python 3.12.7。核心依赖版本保存在 `requirements.txt`：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

早期下载的 `tdxquant==0.3.0` 是可选依赖，安装列表见 `requirements-tdxquant.txt`。当前回测直接解析通达信 `.day` 文件，不需要安装该包才能运行。

需要自行准备以下数据：

1. 通达信日线，例如 `D:/tdx/vipdoc/sh/lday/*.day`、`D:/tdx/vipdoc/sz/lday/*.day`，以及用于交易日历的上证指数日线。
2. 股票名称缓存：`D:/tdx/T0002/hq_cache/shs.tnf`、`szs.tnf`。缺失时名称可能为空。
3. 除权除息数据：`data/tdx_actions.csv`，当前读取器使用以下列顺序：

   ```text
   market,code,datetime,category,hongli_panqianliutong,peigujia_qianzongguben,songgu_qianzongguben,peigu_houzongguben
   ```

   `market` 为 0（深圳）或 1（上海），股票代码须保留前导零，日期为 YYYYMMDD。此文件须在回测前准备，不应以空表替代真实公司行动数据。
4. `data/daily/YYYY-MM-DD.csv`：用于补足 TDX 快照之后的日线，可由 `update_market.py` 缓存。
5. `data/st_checks.csv`：历史风险警示状态。原策略运行生成待核对清单后，可用 `update_market.py --st-checks` 补充，再重跑回测。

这是截至 **2026-11-16** 的研究快照。v2 / v3 和部分导出脚本保留了 `D:/tdx` 路径、日期及历史交易规则常量；换机器、换数据截止日时应统一检查这些配置。仅改变某一个脚本的日期不足以构成一次新的完整研究。

## 运行顺序

以下命令从仓库根目录执行。行情数据须先准备完毕。全市场特征构建和参数搜索需要较多时间与内存，Numba 首次执行还会编译。

```powershell
# 补充原 TDX 快照之后的行情（取决于数据服务可用性）
python update_market.py --from 2026-11-05 --through 2026-11-16

# 原始形态；若出现待核对 ST 状态，补充后重跑
python backtest.py --tdx D:/tdx --start 20220101 --end 20261116
python update_market.py --st-checks
python backtest.py --tdx D:/tdx --start 20220101 --end 20261116

# 首波反转研究及有限资金组合
python research_v2.py build
python research_v2.py search
python portfolio_search.py
python finalize_v2.py

# 多周期、阳阴量与 MACD 研究
python research_v3.py build
python research_v3.py search
python finalize_v3.py
python compare_rank_v3.py

# 导出基础页面数据与完整日线，再核对和生成最新页面
python sites/volume-signals/export_data.py
python sites/volume-signals/export_research.py
python sites/volume-signals/export_chart_store.py
python audit_v3.py
python sites/volume-signals/export_multiframe.py

# 本地预览
python -m http.server 8765 --bind 127.0.0.1 --directory sites/volume-signals/dist
```

浏览器访问 `http://127.0.0.1:8765/multiframe.html`。`export_data.py` 还需要原案例的 `results/bear_next_day_quote.csv`；该文件属于 11 月 17 日单独核对的小熊电器行情快照，已随本仓库保存。不要把它自动视作新交易日的全市场数据。

`export_candles.py` 与 `candles.js` 保留了上一版分片实现；当前页面使用 `export_chart_store.py` 和 `candles-v3.js`。部署服务的身份文件没有入库，本仓库不会自动修改现有网站的发布设置。

## 当前研究结果

下表是反复检验 2022—2026 年历史数据之后选出的候选，**没有独立的前瞻样本**。100 万元初始资金、最多 10 个持仓、整手交易，往返总成本按 0.35% 模拟。

| 方案 | 组合年化 | 最大回撤 | 胜率 | 盈亏比 | 单笔平均净收益 | 成交数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 上一版，持有 20 日 | 25.82% | -8.40% | 76.53% | 2.43 | 12.64% | 98 |
| 量能与周线，持有 20 日 | 28.31% | -8.31% | 73.74% | 2.33 | 14.11% | 99 |
| 量能与 MACD 排序，持有 30 日 | 34.85% | -10.89% | 77.11% | 3.33 | 19.62% | 83 |

完整参数以结果 JSON 和实现为准。优化版次日按参考前收盘价的 98% 挂限价单；开盘低于限价按开盘价计，盘中需向下穿过限价才认定成交，未成交当日撤单，涨停开盘跳过。原始形态页面的“次日开盘估值”口径与优化版不同。

日线放量定义始终为：成交量严格高于此前 5 日和 10 日均量，均量不含信号日。阳阴量按收盘与开盘的关系划分。周月指标只使用信号当时已经完成的周期；需要右侧 K 线确认的低点，也必须在信号前完成确认。

网页支持日 / 周 / 月 K、信号前 60 / 120 / 250 / 500 日或全部历史、信号后范围选择、成交量与 MACD，并逐根显示开高低收和涨跌幅。日涨跌幅使用除权参考前收，周月涨跌幅使用上一周期复权收盘价。

## 结果的限制

- 参数选择已使用 2026 年数据，逐年重新选参也是回顾性检验。该检验在 2025 年的组合收益仅约 0.49%。
- 99 笔、83 笔成交分别集中于 39 个、24 个买入日，同日股票收益相关。
- 当前股票池来自现有 TDX 目录，未完整恢复历年退市股票。
- 日线无法验证逐笔成交与盘口排队。复权总回报采用现金分红再投资近似。
- 多次搜索的最优历史结果不代表未来收益保证。

