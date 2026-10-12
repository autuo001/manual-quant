# manual_quant 接口契约（SPEC）

基于《经营未来》v5.0 投资手册重写的量化回测框架。**所有子模块必须严格按本文件实现**，
不得修改本文件、不得改动他人负责的文件。

## 0. 设计原则（来自手册）

| 原则 | 出处 | 对实现的要求 |
|---|---|---|
| 证据分级闸门 | 附录五元数据：Ⅱ 区（C/D 级）**不得单独构成买入依据** | 任何买入信号必须记账其依据的规则编号；含 C/D 级依据时必须由 `rules.gate()` 拦截 |
| 观测口径优先级 | 附录六：最大回撤、波动、换手率、资金占用、策略失效期 **＞ 收益率（排最后）** | 所有报告按此顺序输出指标 |
| 样本门槛 | 附录六：同一规则独立决策样本 **≥20 笔**，<20 笔视为噪声 | 判定器必须检查样本数并标注 |
| 过拟合判定 | 附录六：只在某个精确参数下有效 → **过拟合嫌疑，降级或不采用** | 协议执行器必须做参数敏感性并给出该判定 |
| 六维协议 | 附录六 | 协议执行器固定六维：历史回测／基准比较／交易成本／参数敏感性／样本外／市场环境 |
| 状态链 | 附录六：未验证→验证中→部分验证→已验证→已退役 | 记录表带 `状态` 字段，判定器可写回 |
| 记录表 7 字段 | 附录六 | 编号／规则／状态／数据区间／基准／结论／更新日期 |

## 1. 目录与文件归属（严禁跨文件修改）

```
manual_quant/
├── SPEC.md                    主控所有（子代理只读）
├── spec/                      【已完成】手册章节提取（只读参考）
├── config.py                  ① 所有者：data-loader
├── data/
│   ├── loader.py              ① 所有者：data-loader
│   └── extra.py               ② 所有者：extra-data
├── rules.py                   ③ 所有者：rules
├── core/
│   ├── engine.py              ④ 所有者：engine
│   ├── risk_budget.py         ⑤ 所有者：risk
│   ├── allocation.py          ⑥ 所有者：allocation
│   └── seven_step.py          ⑨ 所有者：integration
├── strategies/
│   ├── value_card.py          ⑦ 所有者：strategies
│   └── trend_card.py          ⑦ 所有者：strategies
├── validate/
│   ├── protocol.py            ⑧ 所有者：protocol
│   └── judge.py               ⑩ 所有者：judge
├── run.py                     ⑨ 所有者：integration
├── README.md                  ⑨ 所有者：integration
└── output/                    运行产物
```

## 2. 数据契约（只读，勿重复下载）

| 数据 | 路径 | 列 | 说明 |
|---|---|---|---|
| 个股日线 | `../six_layer_backtest/data/parquet/daily_all.parquet` | code(str6), date, open, close, volume | 4935 只，2016-2026，前复权，float32 |
| 指数日线 | `../six_layer_backtest/data/parquet/index.parquet` | date, open, high, low, close, volume | 沪深300 |
| 业绩报表（富字段，用这个） | `../backtest/data/parquet/yjbb.parquet` | 见下 | 44 期快照 |
| 业绩报表（瘦字段，勿用） | `../six_layer_backtest/data/parquet/yjbb.parquet` | 仅 5 列 | 缺财务字段 |

富字段 yjbb 列：`股票代码, 股票简称, 每股收益, 营业总收入-营业总收入, 营业总收入-同比增长, 净利润-净利润, 净利润-同比增长, 每股净资产, 净资产收益率, 每股经营现金流量, 销售毛利率, 所处行业, 最新公告日期, report_period`

**缺失数据（不得臆造，只能标注「数据不足」）**：指数 PE/盈利收益率、股息率、10 年期国债收益率、债券基金指数、资产负债表科目、美股数据。

路径统一由 `config.py` 提供（支持环境变量覆盖 `MQ_DATA_DIR` / `MQ_OUT_DIR`），模块内**禁止硬编码绝对路径**。

## 3. 模块接口（签名固定，实现可自由）

### ① config.py
```python
DATA_DIR: str            # 数据根目录（默认 ../six_layer_backtest/data/parquet）
YJBB_RICH: str           # 富字段业绩报表路径
OUT_DIR: str
INITIAL_CAPITAL = 100_000.0
COMMISSION_RATE = 0.00025; COMMISSION_MIN = 5.0
STAMP_TAX = 0.0005; SLIPPAGE = 0.001
LIMIT_PCT_MAIN = 0.10; LIMIT_PCT_GEM = 0.20      # 主板 / 创业板科创(300/688)
START_DATE = "2016-01-01"; END_DATE = "2026-09-30"
IS_END = "2023-12-31"; OOS_START = "2024-01-01"  # 样本外切分（附录六⑤）
RISK_ZERO_LOSS = 0.05       # R-04 归零损失上限（3%–10% 可调）
RISK_PER_TRADE = 0.015      # R-38 单笔风险（1%–2% 取中值）
MAX_POSITIONS = 10; MIN_POSITIONS = 3            # R-16
FIRST_TRANCHE = 0.20        # R-35 首笔 ≤ 计划额度 1/5
STOP_LOSS_PCT = 0.01        # R-36 最后买入价下方 1%
EQUITY_MIN = 0.25; EQUITY_MAX = 0.75; REBAL_BAND = 0.05   # R-09/R-13
```

### ① data/loader.py
```python
def load_daily(code: str) -> pd.DataFrame      # index=date, cols=[open, close, volume]
def load_daily_all() -> dict[str, pd.DataFrame] # 一次性载入全部（内存较大，提供即可）
def load_index() -> pd.DataFrame                # index=date, cols=[open, close]
def load_fundamentals() -> pd.DataFrame         # 富字段 → 派生列见下
def trading_calendar() -> list[pd.Timestamp]    # 以指数交易日为准
def universe(min_list_days: int = 250) -> list[str]
```
`load_fundamentals()` 派生列（**必须**）：
`code, name, report_period, ann_date(datetime), eps, revenue, net_profit, net_margin(净利率=净利润/营收), gross_margin(销售毛利率), roe, ocfps, rev_yoy, profit_yoy, industry`
另提供：
```python
def profit_growth_streak(code: str, periods: int = 10) -> int  # 连续正增长期数(R-04/财务门槛"EPS连续10年增长"的近似)
def financial_gate_row(code, report_period) -> dict           # 巴菲特财务门槛逐项判定
```
财务门槛（手册原文阈值，逐项布尔）：毛利率 ≥40%／净利率 >20%／ROE >0／营收同比 >0／净利润同比 >0；
返回 `{"pass": bool, "items": {名称: (阈值, 实际值, 是否通过)}, "missing": [缺字段]}`。

### ② data/extra.py
尝试获取并登记「数据不足」外部数据，**失败不得中断流程**：
```python
def try_fetch_index_pe() -> pd.DataFrame | None       # 沪深300 PE（用于盈利收益率 E/P）
def try_fetch_bond_yield() -> pd.DataFrame | None     # 10 年期国债收益率
def try_fetch_bond_index() -> pd.DataFrame | None     # 债券基金/国债指数
def availability() -> dict[str, dict]                 # 名称 → {"available":bool,"source":str,"rows":int,"note":str}
def earnings_yield() -> pd.Series | None              # E/P 序列（1/PE），无数据返回 None
```
产出写入 `output/data_availability.md`（供 README 引用）。网络请求必须带 45 秒硬超时 + 3 次重试；
现有可用源顺序：腾讯 → 东财 → 新浪（akshare）。

### ③ rules.py
```python
@dataclass
class Rule:
    id: str; name: str; level: str      # A/B/C/D
    nature: str                          # 安全底线/决策原则/经验参考/行为建议/待验证规则
    zone: int                            # 1=Ⅰ区可执行, 2=Ⅱ区仅参考
    applies: str; action: str; fail: str; verify: str
    needs_data: str; line: int
    kind: str                            # filter/sizing/exit/timing/allocation/behavior
    executable: bool                     # 能否编码为回测逻辑
    params: dict                         # 可调参数（用于参数敏感性）

RULES: dict[str, Rule]
def gate(rule_ids: list[str], only_trigger: bool = True) -> tuple[bool, str]
    # 附录五硬约束：若独立买入依据含 zone==2 或 level in ("C","D") → 返回 (False, 原因)
    # 若含 A/B 级 Ⅰ 区 → (True, "ok")
def encodable() -> list[str]             # executable=True 的编号
def by_kind(kind: str) -> list[Rule]
def sensitivity_grid(rule_id: str, rel: float = 0.20) -> list[dict]
    # 附录六④：阈值上下浮动（默认 ±20% 相对；V-01 用 ±2 个百分点绝对）
def dump_table() -> pd.DataFrame         # 记录表格式：编号/规则/状态/数据区间/基准/结论/更新日期
```
规则清单以 `spec/08-规则总表.md` 为准（R-01~R-41），并把 `spec/09-验证协议.md` 的 V-01~V-10
作为 `kind="timing"/"allocation"` 的待验证条目登记（level 按 spec 文件所标，D 级标记）。

### ④ core/engine.py
```python
@dataclass
class EngineConfig:
    initial_capital: float = None; commission_rate: float = None; ...
def run_backtest(signals: pd.DataFrame, calendar: list | None = None,
                 cfg: EngineConfig = None, liquidate_at_end: bool = True) -> dict
```
- `signals` 列：`date, code, side(buy/sell), reason, rule_id, weight(目标权重,可选), stop_pct(可选)`
- 规则：T 日收盘出信号 → **T+1 开盘成交**；开盘一字涨停不买、跌停不卖（按 code 前缀判 ±10%/±20%）；
  佣金万2.5（最低 5 元）+ 卖出印花税 0.05% + 双边滑点千1；停牌（当日无 bar）顺延或放弃
- 返回：`{"equity": DataFrame[date,cash,market_value,equity,exposure,n_positions], "trades": DataFrame[date,code,side,price,shares,amount,fee,pnl,reason,rule_id], "stats": dict}`
- `stats` 必须含：`总收益/年化/最大回撤/波动(年化)/换手率/资金占用/策略失效期(最长未创新高交易日)/交易笔数/胜率/盈亏比`

### ⑤ core/risk_budget.py
```python
def max_weight_zero_loss(equity: float, price: float) -> tuple[float, str]   # R-04 归零口径权重上限
def size_by_risk(equity: float, price: float, stop_pct: float, risk_pct: float) -> tuple[int, float]
def plan_tranches(total_value: float, n: int = 5) -> list[float]             # R-35 首笔 1/5
def can_add_position(n_current: int) -> tuple[bool, str]                      # R-16 3–10 只
def can_average_down(position) -> tuple[bool, str]                            # R-37 禁止摊平
def stress_test(equity: float, weights: dict[str, float], shocks: list[float]) -> pd.DataFrame
    # 手册压力线：权益 75% 遇 -40% → 组合 -30%；遇 -50% → -37.5%；容忍线 30% → 权益上限 60%
def fisher_class_cap(cls: str) -> float    # R-14 费雪 A≤20% / B 8–10% / C≤5%（与 R-04 取更严）
```
所有函数返回中必须能指出**绑定的规则编号**（用于记录表与闸门检查）。

### ⑥ core/allocation.py
```python
def equity_ratio_by_holding_years(years: float) -> float   # R-11：1–2 年 0%、5 年 25%、10 年 1/3 强、20 年 >50%、30 年 68%，线性插值并在端点外截断
def equity_ratio_by_age(age: float) -> float               # R-12：(100−年龄)%，仅作起点
def clamp_equity(x: float) -> float                        # R-09：25%–75%
def rebalance_orders(cur: dict[str, float], target: dict[str, float], band: float = 0.05,
                     prefer_new_cash: bool = True) -> list[dict]   # R-13 偏离 ±5pp 触发，优先用新增资金
def dca_plan(monthly_income: float, monthly_expense: float) -> float    # (收入−支出)×50%
def n_amplifier(earnings_yield: float | None, base_amount: float) -> tuple[float, str]
    # 盈利收益率 >10% → n=1（资金充裕 2）；10%~6.4% → 定额；<6.4% → 分 10 份卖出（无 E/P 数据时返回 数据不足）
def run_allocation_backtest(monthly_amount: float, calendar, index, bond: pd.DataFrame | None) -> dict
```

### ⑦ strategies/value_card.py, strategies/trend_card.py
每个 `run(...)` 返回统一信号格式（供 engine 直接消费）：
```python
def run(calendar=None, cfg=None, **kw) -> pd.DataFrame   # 列同 engine.signals
```
- **value_card**：七步法式筛选 → 巴菲特财务门槛（`financial_gate_row`）+ 安全边际三档（R-27，无内在价值数据时用可得替代并**明确标注降级为定性**）+ 持股数 3–10（R-16）+ 归零口径仓位（R-04）；
  卖出：三个卖出理由（R-28，可编码部分 = 基本面恶化 threshold）+ 三年守则（R-32）；禁止摊平（R-37）。
  每条信号必须填 `rule_id`（多个用 `+` 连接）。
- **trend_card**：最小阻力路线（区间上下沿定义：`上限 = 过去 N 日最高收盘×(1−ε)`、`下限 = 过去 N 日最低收盘`，突破上限才动手）+ 首笔 1/5（R-35，无账面利润不加仓）+ 止损最后买入价下方 1%（R-36）+ 单笔风险 1%–2%（R-38）+ 绝不平摊（R-37）。N 与 ε 必须进 `params` 以便敏感性测试。

### ⑧ validate/protocol.py
```python
def run_six_dim(strategy_fn, name: str, cfg=None) -> dict
```
六维（固定顺序，缺数据必须标注不可完成并给原因）：
1. 历史回测：全区间跑一次
2. 基准比较：①简单持有沪深300 ②按月定投 ③60/40 再平衡（债券腿无数据时标注）
3. 交易成本：三档（①含全部成本 ②零成本 ③成本×2）
4. 参数敏感性：调用 `rules.sensitivity_grid` 对策略可调参数扫 3 档
5. 样本外：`IS_END/OOS_START` 切两段（须在记录表登记区间）
6. 市场环境：牛／熊／震荡三段（分段标准必须显式登记在结果里，默认：指数 250 日涨幅 >+15% 牛、<−15% 熊、其余震荡）
返回：
```python
{"name":..., "dims": {dim: {...指标}}, "obs_order": ["最大回撤","波动","换手率","资金占用","策略失效期","收益率"],
 "sensitivity": DataFrame, "overfit_suspect": bool, "notes": [...]}
```

### ⑩ validate/judge.py
```python
def judge(report: dict) -> dict
    # 判定四选一：通过 / 部分通过 / 失败 / 过拟合嫌疑（依附录六）
    # 样本 ≥20 笔检查：不足则标注"样本不足，不得据此改规则"
def next_state(cur_state: str, verdict: str) -> str       # 状态链流转
def record_row(rule_id, rule_name, state, data_range, benchmark, verdict, date) -> dict  # 7 字段
def write_records(rows: list[dict], path: str = None) -> str  # output/verification_records.csv
def change_ticket(rule_id, old, new, reason, evidence) -> str  # 规则变更单模板（附录六）
def diagnosis(reason_code: str) -> dict    # 五归因 → 六处理映射
```

### ⑨ core/seven_step.py + run.py + README.md
- `seven_step.py`：把手册第〇章七步法实现为**逐层否决的流水线骨架**（每步返回 `(passed, reason, evidence)`，任一步否决即终止并记账）；步骤定义见 `spec/01-七步法与买入清单.md`。
- `run.py`：CLI 编排 `python3 run.py {check|value|trend|allocation|protocol|all}`；
  `check` 打印数据可用性与规则闸门自检；`protocol` 对指定策略跑六维协议并输出记录表。
- `README.md`：框架说明、规则来源（引用 spec 文件与手册行号）、如何运行、**已知数据缺口与不可验证项**（必须与 `output/data_availability.md` 一致）。


## 5. 实现期确认的口径（主控裁定，2026-10-12）

以下 6 项是并行实现+独立验证过程中确认的最终口径，**优先级高于本文件前文的字面描述**：

1. **闸门语义收窄**（③ rules）：附录五「Ⅱ 区不得单独构成买入依据」针对的是**买入触发依据**，
   不是仓位/卖出/行为类约束。故 `gate(rule_ids, only_trigger=True)` 默认只让 `kind ∈ TRIGGER_KINDS
   （timing/filter）` 的规则参与否决；`only_trigger=False` 保留严格旧语义。第二个返回值是人类可读
   原因（不保证字面 "ok"）。
2. **信号记账拆列**（⑦ strategies）：信号表新增 `constraint_ids` 列——`rule_id` 只放触发依据，
   `constraint_ids` 放约束类规则（R-04/R-16/R-32/R-35/R-36/R-37/R-38）。故信号列 = SPEC §3④ 的
   7 列 + `constraint_ids`（引擎按列名取用，多余列被忽略）。
3. **排序契约**（④ engine）：同一信号集合必须给出逐位相同的结果。引擎先按
   `(date↑, code↑, sell 优先, rule_id↑, weight↑, stop_pct↑, reason↑, kind="stable")` 重排并去重，
   再赋 `seq`；成交阶段按 `(到期日↑, 卖出优先, seq↑)` 执行。`_determinism_selfcheck()` 为回归自检。
4. **日历唯一口径**：所有命令与六维协议一律显式传 `data.loader.trading_calendar()` 全区间
   （禁止 `calendar=None`，否则引擎会裁成「首条信号日→末条信号日+5 日」）。
5. **最大回撤符号**：统一输出**负值**，字段名 `最大回撤(负值)`。
6. **样本口径**：样本门槛按**同规则**样本数（trades 的 rule_id 分组计数），不是总成交笔数。

## 4. 通用要求

- Python 3.9（系统 python3），只用 `pandas / numpy / akshare`（已有）+ 标准库；**不得新增第三方依赖**。
- 全部模块支持 `MQ_OFFLINE=1` 时不联网（缺数据即标注"数据不足"，不得中断）。
- 内存：`daily_all.parquet` 有 989 万行，**必须按需切片**，禁止无脑 `concat` 全表。
- 每个模块自带 `if __name__ == "__main__":` 冒烟测试，能在 60 秒内自证可跑。
- 代码注释只写"为什么"（尤其规则编号与手册行号的对应关系），不写废话。
- 输出文件统一写 `output/`，文件名用 ASCII。
