"""⑧ validate/protocol.py —— 六维验证协议执行器（附录六，spec/09-验证协议.md 行 2131）。

SPEC 未写明处的取舍（一律按最贴近字面实现）：
1. dims 的键用六维原名，dict 插入顺序即 SPEC 固定顺序；每个 dim 内先放附录六观测口径
   （最大回撤/波动/换手率/资金占用/策略失效期，收益率排最后），再放 完成/原因 等标注。
2. 基准腿（①简单持有 ②按月定投 ③60/40）不是 engine 信号，故在协议内用同为净值的等价口径计算；
   换手率/资金占用 留 None 并登记口径，避免与 engine 的指标定义混用。
3. 市场环境按「指数 250 日涨幅」逐日打标（+15%/−15% 为 SPEC 给出的默认标准，先在完整指数历史上
   算涨幅、再裁到本次回测区间），同标签连续区间合并后按各标签的交易日集合统计（分段标准与分段表
   显式写入结果）。
4. 过拟合嫌疑 = 基准参数下有效（收益率 > 0）且全部浮动档无效。原文「只在某个精确参数下有效」
   只用零界比较，不引入自造阈值（SPEC 第 5 条：不得发明数字）。
5. cfg 允许传 dict 作协议覆盖项（SPEC 只写 cfg=None，未定义其类型）；非 dict 一律作为 engine 的
   EngineConfig 透传给 strategy_fn / run_backtest。注意 core.engine 只接受 EngineConfig 对象，
   故 {"engine_config": {...}} 这种 dict 形式仅供测试用的兼容 backtest 函数，正式路径必须传对象。
6. config.py / core.engine / rules / data.* 由其他模块并行实现，此处全部惰性 import：
   缺失只标注「不可完成/数据不足」，绝不中断，也绝不在 import 时读 parquet。
7. 日历口径（验证报告 §6.2）：六维一律**显式传全区间日历**，来源优先 data.loader.trading_calendar()，
   并把「来源 + 天数 + 起止」登记进结果（report["日历"] 与用到日历的各维）。为什么不靠 engine 的默认日历：
   engine 在 calendar=None 时会把日历裁成「首条信号日 → 末条信号日 + 5 个交易日」，同一策略在
   run.py value 与 run.py protocol 两个入口的期末清仓日与头条数字会不同（21.77 vs 13.75、1.2354 vs 1.4373），
   两处数字无法互相引用。
8. 回撤符号（验证报告 §6.3）：协议内统一「负值表示回撤」，字段名写 (负值)（DD_KEY）；engine 的正数幅度口径
   在 _result_parts 里一次性取负，使同一份报告里的策略腿与基准腿可比。
9. 参数敏感性证据（验证报告 §6.4）：某参数三档取值完全相同、全部档位没跑出结果、或全部档位结果与基准逐位
   相同 → 该参数标「该参数未生效/不可扫」，既不计入「完成」，也不作为过拟合判定的证据。
10. 样本数口径（验证报告 §6.6、附录六行2158）：按 trades.rule_id 分组计数（同一规则独立决策样本），
   报告级「样本笔数」取各规则中的最小值（绑定约束），不再用总成交笔数顶替。
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pandas as pd

# 附录六观测口径（spec/09-验证协议.md 行 2131）：重点观测顺序，收益率只排最后
# 回撤字段名带 (负值) 以标明符号口径（验证报告 §6.3）：−0.1996 = 回撤 19.96%
DD_KEY = "最大回撤(负值)"
OBS_ORDER = [DD_KEY, "波动", "换手率", "资金占用", "策略失效期", "收益率"]
DIM_ORDER = ["历史回测", "基准比较", "交易成本", "参数敏感性", "样本外", "市场环境"]
METRIC_KEYS = OBS_ORDER + ["年化", "总收益", "交易笔数", "胜率", "盈亏比"]
SENS_COLS = [
    "规则编号",
    "参数",
    "值",
    "档",
    "收益率",
    "年化",
    DD_KEY,
    "波动",
    "换手率",
    "资金占用",
    "策略失效期",
    "交易笔数",
    "备注",
]

# 六维④ 参数生效判定（验证报告 §6.4）：证据只用报告自己的数字，不新造阈值
SENS_DEAD = "该参数未生效/不可扫"
SENS_RTOL, SENS_ATOL = 1e-9, 1e-12

# 样本口径（附录六 行2158/2183）：同一规则独立决策样本 ≥20 笔 → 按 trades.rule_id 分组计数
SAMPLE_KEY = "样本笔数"
SAMPLE_RULES_KEY = "规则样本数"
SAMPLE_NOTE_KEY = "样本口径"
# engine 的正数幅度回撤字段（含英文别名），在 _result_parts 里统一取负
DD_POS_KEYS = ("最大回撤", "max_drawdown")

# 六维⑥ 市场环境默认分段标准（原文未给，SPEC 第 3-⑧ 节给：指数 250 日涨幅 >+15% 牛、<−15% 熊、其余震荡）
ENV_LOOKBACK = 250
ENV_BULL = 0.15
ENV_BEAR = -0.15

# 六维③ 交易成本三档（SPEC 第 3-⑧ 节）：①含全部成本 ②零成本 ③成本×2
COST_FIELDS = ("commission_rate", "commission_min", "stamp_tax", "slippage")
# 成本字段 → config.py 常量名（EngineConfig 里这些字段可为 None，表示取 config 默认值）
_COST_CONST = {
    "commission_rate": "COMMISSION_RATE",
    "commission_min": "COMMISSION_MIN",
    "stamp_tax": "STAMP_TAX",
    "slippage": "SLIPPAGE",
}
COST_TIERS = (("含全部成本", "full"), ("零成本", "zero"), ("成本×2", "double"))

_ANN = 252.0
_SIGNAL_COLS = ["date", "code", "side", "reason", "rule_id"]

# config.py 未就绪时的兜底常量，取值与 SPEC 第 3-① 节逐字一致
_FALLBACK = {
    "START_DATE": "2016-01-01",
    "END_DATE": "2026-09-30",
    "IS_END": "2023-12-31",
    "OOS_START": "2024-01-01",
    "INITIAL_CAPITAL": 100_000.0,
}

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_path():
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)


def _config():
    _ensure_path()
    try:
        import config

        return config
    except ImportError:
        return None


def _cget(key, default=None):
    cfg = _config()
    if cfg is not None and hasattr(cfg, key):
        return getattr(cfg, key)
    if key in _FALLBACK:
        return _FALLBACK[key]
    return default


def _load_index():
    """惰性取沪深300（六维②⑥ 都需要）；loader 不可用时按 config.DATA_DIR 兜底读文件。"""
    _ensure_path()
    try:
        from data.loader import load_index

        df = load_index()
        if isinstance(df, pd.DataFrame) and not df.empty:
            return _norm_index(df)
    except ImportError:
        pass
    cfg = _config()
    data_dir = getattr(cfg, "DATA_DIR", None) if cfg is not None else None
    if not data_dir:
        return None
    path = os.path.join(data_dir, "index.parquet")
    if not os.path.exists(path):
        return None
    try:
        return _norm_index(pd.read_parquet(path))
    except Exception:
        return None


def _norm_index(df):
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")
    out.index = pd.to_datetime(out.index)
    if "close" not in out.columns:
        return None
    return out.sort_index()


def _load_bond():
    """六维②60/40 的债券腿。

    口径说明：债券腿用「上证国债指数」代表（≠ 债券基金指数，见 output/data_availability.md 的降级标注）。
    离线时**不要早退**——data.extra 已把抓到的数据缓存到 data/extra/*.parquet，早退会让 60/40 基准
    在离线态永远判「不可完成」（复核报告 §6.2 的连带问题）；取不到才按「数据不足」处理。
    """
    _ensure_path()
    try:
        from data.extra import try_fetch_bond_index
    except ImportError:
        return None
    try:
        df = try_fetch_bond_index()
    except Exception:
        return None
    if not isinstance(df, pd.DataFrame) or df.empty or "close" not in df.columns:
        return None
    return _norm_index(df)


def _lazy_grid(rule_id, rel=0.20):
    """六维④ 走 rules.sensitivity_grid（附录六④ ±20% 相对；V-01 由 rules 内改 ±2pp 绝对）。"""
    _ensure_path()
    try:
        import rules
    except ImportError:
        return None
    fn = getattr(rules, "sensitivity_grid", None)
    if fn is None:
        return None
    try:
        return fn(rule_id, rel=rel)
    except TypeError:
        try:
            return fn(rule_id)
        except Exception:
            return None
    except Exception:
        # rules.sensitivity_grid 对未登记编号抛 KeyError（信号里的 rule_id 可能是 V-xx/组合编号）
        return None


def _norm_variant(d):
    """rules.sensitivity_grid 的 dict 结构 SPEC 未固定，这里兼容常见键名。

    额外带上该参数的三档全量 values / varies —— 六维④ 判定「三档取值是否完全相同」要靠它，
    只留当前档的 value 看不出「档位根本没分开」。
    """
    if not isinstance(d, dict):
        return None
    param = d.get("参数") or d.get("参数名") or d.get("param") or d.get("name")
    if param is None:
        return None
    value = d.get("值", d.get("value"))
    if value is None:
        return None
    tier = d.get("档") or d.get("tier") or d.get("相对") or d.get("rel")
    rid = d.get("规则编号") or d.get("rule_id") or d.get("id")
    if tier in ("基准", "base"):
        return None  # rules.sensitivity_grid 自带基准档，dim① 已有基准行，不重复跑
    values = d.get("values")
    if isinstance(values, (list, tuple, set)):
        values = list(values)
    elif values is None:
        values = [value]
    else:
        values = [values]
    return {"规则编号": rid, "参数": param, "值": value, "档": tier, "values": values}


def _as_dt_index(seq):
    if seq is None:
        return None
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(list(seq)))
    except Exception:
        return None
    return idx.dropna().sort_values().unique()


def _slice_cal(cal, lo, hi):
    if cal is None:
        return None
    left = pd.Timestamp(lo) if lo else None
    right = pd.Timestamp(hi) if hi else None
    out = cal
    if left is not None:
        out = out[out >= left]
    if right is not None:
        out = out[out <= right]
    return out


CAL_FULL = "data.loader.trading_calendar()（全区间）"
CAL_NOTE = (
    "全区间＝config.START_DATE~config.END_DATE 的全部交易日；engine 必须显式收到该日历，"
    "不得走 calendar=None（否则 engine 会把日历裁成「首条信号日 → 末条信号日 + 5 个交易日」，"
    "期末清仓日与头条数字随入口变化）"
)


def _cal_meta(cal, source):
    """登记所用日历：来源 + 交易日天数 + 起止（验证报告 §6.2 要求显式可查，不做事后解释）。"""
    if cal is None or len(cal) == 0:
        return {"来源": source or "不可用", "天数": None, "起": None, "止": None, "口径": CAL_NOTE}
    return {
        "来源": source or CAL_FULL,
        "天数": int(len(cal)),
        "起": pd.Timestamp(cal[0]).strftime("%Y-%m-%d"),
        "止": pd.Timestamp(cal[-1]).strftime("%Y-%m-%d"),
        "口径": CAL_NOTE,
    }


def _resolve_calendar(ctx):
    """六维统一日历口径（SPEC 第 3-⑧ 节六维①「全区间跑一次」）→ (日历, 来源)。

    为什么不交给 engine 自动裁日历：见 CAL_NOTE 与验证报告 §6.2。降级链只在 loader 不可用时启用，
    且降级口径会写进结果，避免「区间悄悄变短」这种事后再解释。
    """
    loader_cal = _as_dt_index(ctx.calendar())
    if loader_cal is not None and len(loader_cal):
        return _slice_cal(loader_cal, ctx.start, ctx.end), CAL_FULL
    index = ctx.index()
    if index is not None:
        return (
            _slice_cal(_as_dt_index(index.index), ctx.start, ctx.end),
            "指数日期（data.loader.trading_calendar() 不可用时的降级口径）",
        )
    return None, "不可用"


def _longest_no_new_high(eq):
    """策略失效期（附录六观测口径）= 最长未创新高交易日数。"""
    if eq is None or len(eq) == 0:
        return 0
    peak = eq.cummax()
    worst = streak = 0
    for flag in (eq >= peak).to_numpy():
        streak = 0 if flag else streak + 1
        worst = max(worst, streak)
    return int(worst)


def _equity_stats(eq, n_trades=None):
    """协议内净值口径（仅用于基准腿与 engine 未给 stats 的兜底；换手率/资金占用不臆造）。

    回撤键统一用 DD_KEY 且为负值，与 engine 腿（_result_parts 取负后）同符号、可直接跨维比较。
    """
    if eq is None or len(eq.dropna()) < 2:
        return {}
    eq = eq.dropna()
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max(len(eq) / _ANN, 1e-9)
    annual = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1 else -1.0
    ret = eq.pct_change().dropna()
    return {
        DD_KEY: float((eq / eq.cummax() - 1.0).min()),
        "波动": float(ret.std() * np.sqrt(_ANN)),
        "换手率": None,
        "资金占用": None,
        "策略失效期": _longest_no_new_high(eq),
        "收益率": total,
        "年化": annual,
        "交易笔数": n_trades,
    }


def _period_stats(eq, dates):
    """环境分段口径：只取该段交易日的日收益，再复利成段内净值曲线。"""
    if eq is None or dates is None or len(dates) < 2:
        return {}
    ret = eq.pct_change().reindex(pd.DatetimeIndex(dates)).dropna()
    if len(ret) < 2:
        return {}
    curve = (1.0 + ret).cumprod()
    return {
        DD_KEY: float((curve / curve.cummax() - 1.0).min()),
        "波动": float(ret.std() * np.sqrt(_ANN)),
        "策略失效期": _longest_no_new_high(curve),
        "收益率": float(curve.iloc[-1] - 1.0),
    }


def _ordered(stats):
    """附录六：观测顺序固定，收益率排最后（缺失键保持缺失，不补 0）。"""
    out = {}
    for key in METRIC_KEYS:
        if key in stats:
            out[key] = stats[key]
    for key, val in stats.items():
        if key not in out:
            out[key] = val
    return out


def _dim(metrics=None, done=True, reason="ok", **extra):
    out = _ordered(metrics or {})
    out["完成"] = bool(done)
    out["原因"] = reason
    out.update(extra)
    return out


def _result_parts(res):
    """engine 返回 {equity, trades, stats} → (净值序列, stats, 交易笔数)。"""
    if not isinstance(res, dict):
        return None, {}, None
    stats = dict(res.get("stats") or {})
    # engine 的统计键是「总收益」「波动(年化)」（SPEC 第 3-④ 节），附录六观测口径用词是
    # 「收益率」「波动」，同值对齐，保证 dims 里六个观测键齐全
    if "收益率" not in stats and "总收益" in stats:
        stats["收益率"] = stats["总收益"]
    if "波动" not in stats and "波动(年化)" in stats:
        stats["波动"] = stats["波动(年化)"]
    equity_df = res.get("equity")
    eq = None
    if isinstance(equity_df, pd.DataFrame) and "equity" in equity_df.columns and not equity_df.empty:
        if "date" in equity_df.columns:
            idx = pd.to_datetime(equity_df["date"])
        else:
            idx = pd.to_datetime(equity_df.index)
        eq = pd.Series(np.asarray(equity_df["equity"], dtype=float), index=pd.DatetimeIndex(idx)).sort_index()
    trades = res.get("trades")
    n_trades = int(len(trades)) if isinstance(trades, pd.DataFrame) else None
    if "交易笔数" not in stats and n_trades is not None:
        stats["交易笔数"] = n_trades
    if not stats and eq is not None:
        stats = _equity_stats(eq, n_trades)
    return eq, stats, n_trades


def _call_strategy(strategy_fn, calendar, cfg, **kw):
    """策略 run(calendar=None, cfg=None, **kw) 契约；退回链只为兼容并行开发中的签名差异。"""
    attempts = [
        dict(calendar=calendar, cfg=cfg, **kw),
        dict(calendar=calendar, **kw),
        dict(**kw),
    ]
    last = None
    for args in attempts:
        try:
            return strategy_fn(**args), None
        except TypeError as exc:
            last = exc
    return None, f"strategy_fn 调用失败：{last!r}"


def _norm_signals(sig):
    if not isinstance(sig, pd.DataFrame) or sig.empty:
        return None
    out = sig.copy()
    if "date" not in out.columns:
        out = out.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"])
    if "rule_id" not in out.columns:
        out["rule_id"] = None
    return out.sort_values("date")


def _codes_from_signals(sig):
    """参数敏感性要按信号里出现过的规则编号去 rules 取可调参数。"""
    if sig is None or sig.empty or "rule_id" not in sig.columns:
        return []
    seen = []
    for cell in sig["rule_id"].dropna().astype(str):
        for rid in cell.replace("，", "+").split("+"):
            rid = rid.strip()
            if rid and rid not in seen:
                seen.append(rid)
    return seen


def _with_costs(base, mode):
    """成本三档：同一批信号只改 engine 配置，避免信号层引入额外差异。

    EngineConfig 的成本字段默认 None（表示取 config.py 默认值），必须先 resolved() 成具体数值，
    否则乘 0 / 乘 2 都落空，三档退化为同一档。
    """
    if base is None:
        return None
    resolved = getattr(base, "resolved", None)
    if callable(resolved):
        try:
            base = resolved()
        except Exception:
            pass
    else:
        base = copy.copy(base)  # 无 resolved() 时也不能就地改调用方对象
    if isinstance(base, dict):
        out = dict(base)
        for field in COST_FIELDS:
            val = out.get(field)
            if val is None:
                val = _cget(_COST_CONST[field])
            if isinstance(val, (int, float)):
                out[field] = 0.0 if mode == "zero" else (val * 2.0 if mode == "double" else val)
        return out
    for field in COST_FIELDS:
        val = getattr(base, field, None)
        if val is None:
            val = _cget(_COST_CONST[field])
        if isinstance(val, (int, float)):
            try:
                setattr(base, field, 0.0 if mode == "zero" else (val * 2.0 if mode == "double" else val))
            except Exception:
                pass
    return base


def _env_labels(index):
    """六维⑥ 分段标准：指数 250 日涨幅 >+15% 牛、<−15% 熊、其余震荡。

    窗口未满的前 250 个交易日标为「预热期」——数据在，只是 250 日涨幅尚不可算，不写成「数据不足」。
    """
    close = index["close"].astype(float)
    lr = close / close.shift(ENV_LOOKBACK) - 1.0
    label = pd.Series("震荡", index=close.index, dtype=object)
    label[lr > ENV_BULL] = "牛"
    label[lr < ENV_BEAR] = "熊"
    label[lr.isna()] = "预热期"
    return label


def _env_runs(label):
    grp = (label != label.shift(1)).cumsum()
    runs = []
    for _, block in label.groupby(grp, sort=True):
        runs.append(
            {
                "环境": str(block.iloc[0]),
                "起": block.index[0].strftime("%Y-%m-%d"),
                "止": block.index[-1].strftime("%Y-%m-%d"),
                "交易日": int(len(block)),
            }
        )
    return runs


class _Ctx:
    """把 cfg / 惰性依赖收在一处，run_six_dim 只做编排。"""

    def __init__(self, cfg):
        self.overrides = cfg if isinstance(cfg, dict) else {}
        self.engine_cfg = self.overrides.get("engine_config", None if isinstance(cfg, dict) else cfg)
        self._run_backtest = self.overrides.get("run_backtest")
        self._calendar = self.overrides.get("calendar")
        self._index = self.overrides.get("index")
        self._bond = self.overrides.get("bond")
        self._loaded = False
        self.start = self.overrides.get("start") or _cget("START_DATE")
        self.end = self.overrides.get("end") or _cget("END_DATE")
        self.is_end = self.overrides.get("is_end") or _cget("IS_END")
        self.oos_start = self.overrides.get("oos_start") or _cget("OOS_START")
        self.initial_capital = self.overrides.get("initial_capital") or _cget("INITIAL_CAPITAL")

    def _ensure_engine(self):
        if self._run_backtest is not None or self._loaded:
            return
        self._loaded = True
        _ensure_path()
        try:
            from core.engine import run_backtest
        except ImportError:
            return
        self._run_backtest = run_backtest

    def run_backtest(self, signals, calendar, engine_cfg=None):
        self._ensure_engine()
        if self._run_backtest is None:
            return None, "core.engine 不可用（模块未就绪或导入失败）"
        if signals is None or signals.empty:
            return None, "无信号，无法回测"
        cfg = self.engine_cfg if engine_cfg is None else engine_cfg
        try:
            return self._run_backtest(signals, calendar=calendar, cfg=cfg), "ok"
        except Exception as exc:
            return None, f"engine 执行失败：{exc!r}"

    def calendar(self):
        if self._calendar is None:
            _ensure_path()
            try:
                from data.loader import trading_calendar

                self._calendar = _as_dt_index(trading_calendar())
            except ImportError:
                self._calendar = None
        return _as_dt_index(self._calendar) if self._calendar is not None else None

    def index(self):
        if self._index is None:
            self._index = _load_index()
        return self._index

    def bond(self):
        if self._bond is None:
            self._bond = _load_bond()
        return self._bond

    def grid(self, rule_id):
        fn = self.overrides.get("sensitivity_grid")
        if fn is not None:
            if callable(fn):
                try:
                    return fn(rule_id)
                except TypeError:
                    return fn()
            return fn
        return _lazy_grid(rule_id)


def _bench_hold(index, cal, capital):
    close = index["close"].astype(float).reindex(cal).dropna()
    if len(close) < 2:
        return None
    return capital * close / close.iloc[0]


def _bench_dca(index, cal, capital):
    """按月定投：净值 = 当期市值 / 累计投入（value multiple，起点 1.0），与策略净值同口径可比。

    不能直接用「市值/首期投入」——首期只有一期的钱，分母错位；也不能用 TWR 单位净值——纯价格
    指数下 TWR 恒等于简单持有的收益（现金流时点被时间加权抹平），基准腿就失去区分度。
    """
    close = index["close"].astype(float).reindex(cal).dropna()
    if len(close) < 2:
        return None
    months = pd.PeriodIndex(close.index, freq="M")
    n_months = len(pd.unique(months))
    if n_months < 1:
        return None
    monthly = capital / float(n_months)
    buy_days = {close.index[i] for i, m in enumerate(months) if i == 0 or m != months[i - 1]}
    shares = 0.0
    invested = 0.0
    curve = []
    for day, price in close.items():
        price = float(price)
        if day in buy_days:
            invested += monthly
            shares += monthly / price
        curve.append(shares * price / invested if invested > 0 else 0.0)
    return pd.Series(curve, index=close.index)


def _bench_6040(index, cal, bond, capital):
    """60/40 再平衡：股票腿用沪深300，债券腿无数据（SPEC 第 2 节）则整腿标注不可完成。"""
    if bond is None:
        return None, "债券腿无数据（数据不足）"
    eq_close = index["close"].astype(float).reindex(cal).dropna()
    bd_close = bond["close"].astype(float).reindex(cal).dropna()
    common = eq_close.index.intersection(bd_close.index)
    eq_close, bd_close = eq_close.reindex(common), bd_close.reindex(common)
    if len(common) < 2:
        return None, "股债共同交易日不足 2 天（数据不足）"
    months = pd.PeriodIndex(common, freq="M")
    eq_units, bd_units = 0.0, 0.0
    curve = []
    for i, day in enumerate(common):
        if i == 0 or months[i] != months[i - 1]:
            total = eq_units * float(eq_close.loc[day]) + bd_units * float(bd_close.loc[day])
            if i == 0:
                total = capital
            eq_units = total * 0.6 / float(eq_close.loc[day])
            bd_units = total * 0.4 / float(bd_close.loc[day])
        curve.append(eq_units * float(eq_close.loc[day]) + bd_units * float(bd_close.loc[day]))
    return pd.Series(curve, index=common), "ok"


def _overfit_suspect(base_ret, rows):
    """附录六过拟合判定：只在某个精确参数下有效 → 过拟合嫌疑（阈值浮动后失效即命中）。

    只用「收益率 > 0」这一零界比较，不引入自造阈值；返回 (是否嫌疑, 浮动档收益率列表)。
    """
    floats = [r["收益率"] for r in rows if r.get("档") != "基准" and r.get("收益率") is not None]
    if base_ret is None or not (base_ret > 0) or not floats:
        return False, floats
    return bool(all(v <= 0 for v in floats)), floats


def run_six_dim(strategy_fn, name: str, cfg=None) -> dict:
    """按附录六固定六维顺序跑一遍策略，缺数据只标注不可完成并给原因（不中断、不臆造）。"""
    ctx = _Ctx(cfg)
    notes = []
    dims = {dim: _dim(done=False, reason="未执行") for dim in DIM_ORDER}
    sens_df = pd.DataFrame(columns=SENS_COLS)

    index = ctx.index()
    cal = _slice_cal(ctx.calendar(), ctx.start, ctx.end)
    if cal is None and index is not None:
        cal = _slice_cal(_as_dt_index(index.index), ctx.start, ctx.end)
        notes.append("交易日历不可用，改用指数日期作为日历口径。")
    if index is None:
        notes.append("沪深300 指数不可用 → 基准比较与市场环境无法完成（数据不足）。")

    # ① 历史回测：全区间跑一次（附录六①）
    sig_full, err = _call_strategy(strategy_fn, cal, ctx.engine_cfg)
    if sig_full is None:
        notes.append(err or "策略未返回信号")
        sig_full = pd.DataFrame(columns=_SIGNAL_COLS)
    sig_full = _norm_signals(sig_full)
    if sig_full is None:
        notes.append("策略返回空信号 → 六维均按无信号处理。")
        sig_full = pd.DataFrame(columns=_SIGNAL_COLS)
    if cal is None and not sig_full.empty:
        cal = _as_dt_index(sig_full["date"])
        notes.append("日历与指数均不可用，改用信号日期作为日历口径。")
    span = f"{ctx.start}~{ctx.end}"
    res_full, why = ctx.run_backtest(sig_full, cal)
    eq_full, stats_full, _ = _result_parts(res_full)
    trades_full = res_full.get("trades") if isinstance(res_full, dict) else None
    dims["历史回测"] = _dim(
        stats_full, done=(why == "ok"), reason=why if why == "ok" else f"不可完成：{why}", 数据区间=span, 信号数=int(len(sig_full))
    )

    # ② 基准比较：①简单持有沪深300 ②按月定投 ③60/40 再平衡（附录六②）
    if cal is None or index is None:
        dims["基准比较"] = _dim(done=False, reason="指数或日历不可用（数据不足）", 数据区间=span)
    else:
        capital = float(ctx.initial_capital or 100_000.0)
        bench = {}
        eq_hold = _bench_hold(index, cal, capital)
        if eq_hold is None:
            dims["基准比较"] = _dim(done=False, reason="指数在该区间不足 2 个交易日（数据不足）", 数据区间=span)
        else:
            bench["①简单持有沪深300"] = _equity_stats(eq_hold)
            eq_dca = _bench_dca(index, cal, capital)
            bench["②按月定投"] = _equity_stats(eq_dca) if eq_dca is not None else {"完成": False, "原因": "数据不足"}
            eq_6040, why_6040 = _bench_6040(index, cal, ctx.bond(), capital)
            bench["③60/40再平衡"] = (
                _equity_stats(eq_6040) if eq_6040 is not None else {"完成": False, "原因": why_6040}
            )
            if eq_6040 is None:
                notes.append(f"六维②60/40 不可完成：{why_6040}（债券资产序列缺失，SPEC 第 2 节已知缺口）。")
            notes.append(
                "基准腿口径：协议内净值口径（最大回撤/波动/策略失效期与 engine 定义一致），"
                "换手率/资金占用对基准腿不适用故留 None；①简单持有=期末市值/初始资金；"
                "②按月定投=期末市值/累计投入（区间月数等额分摊初始资金，起点 1.0）；"
                "③60/40=每月再平衡、股票腿沪深300；三条腿均为零成本口径（成本见六维③）。"
            )
            dims["基准比较"] = _dim(
                done=True, reason="ok", 数据区间=span, 基准=bench, 基准数=len(bench)
            )

    # ③ 交易成本：①含全部成本 ②零成本 ③成本×2（附录六③）
    if ctx.engine_cfg is None:
        dims["交易成本"] = _dim(
            done=False, reason="不可完成：无 EngineConfig 基线（cfg 未给且 core.engine 不可用），成本三档无法区分"
        )
        notes.append("六维③ 不可完成：缺少 engine 配置基线，成本三档无法区分。")
    else:
        base_cfg = ctx.overrides.get("engine_config", ctx.engine_cfg)
        tiers = {}
        ok_any = False
        last_why = ""
        for label, mode in COST_TIERS:
            res, why_t = ctx.run_backtest(sig_full, cal, engine_cfg=_with_costs(base_cfg, mode))
            last_why = why_t
            _, tstats, _ = _result_parts(res)
            tiers[label] = _ordered(tstats) if tstats else {"完成": False, "原因": why_t}
            ok_any = ok_any or bool(tstats)
        dims["交易成本"] = _dim(
            done=ok_any,
            reason="ok" if ok_any else f"不可完成：成本三档均未跑出结果（{last_why}）",
            数据区间=span,
            三档=tiers,
        )

    # ④ 参数敏感性：rules.sensitivity_grid 扫 3 档（附录六④）
    sens_rows = []
    overfit = False
    base_ret = stats_full.get("收益率") if stats_full else None
    rule_ids = ctx.overrides.get("sensitivity_rules") or _codes_from_signals(sig_full)
    variants = []
    for rid in rule_ids:
        grid = ctx.grid(rid)
        if not grid:
            continue
        for item in grid:
            norm = _norm_variant(item)
            if norm:
                norm["规则编号"] = norm["规则编号"] or rid
                variants.append(norm)
    if not variants:
        dims["参数敏感性"] = _dim(
            done=False,
            reason="不可完成：无浮动档可用（rules.sensitivity_grid 不可用、仅返回基准档，或信号未引用含可调参数的规则）",
        )
        notes.append("六维④ 不可完成 → 过拟合嫌疑无法判定（不臆造参数）。")
    else:
        if base_ret is not None:
            sens_rows.append({"规则编号": "基准", "参数": "基准", "值": None, "档": "基准", "收益率": base_ret})
        ran = 0
        for variant in variants:
            kw = {str(variant["参数"]): variant["值"]}
            sig_v, verr = _call_strategy(strategy_fn, cal, ctx.engine_cfg, **kw)
            sig_v = _norm_signals(sig_v)
            if sig_v is None:
                sens_rows.append(
                    {
                        "规则编号": variant["规则编号"],
                        "参数": variant["参数"],
                        "值": variant["值"],
                        "档": variant["档"],
                        "收益率": None,
                        "备注": verr or "策略不接受该参数注入",
                    }
                )
                continue
            res_v, why_v = ctx.run_backtest(sig_v, cal)
            _, stats_v, _ = _result_parts(res_v)
            row = {"规则编号": variant["规则编号"], "参数": variant["参数"], "值": variant["值"], "档": variant["档"]}
            if stats_v:
                ran += 1
                row.update({k: stats_v.get(k) for k in ("收益率", "年化", "最大回撤", "波动", "换手率", "资金占用", "策略失效期", "交易笔数")})
            else:
                row["收益率"] = None
                row["备注"] = why_v
            sens_rows.append(row)
        sens_df = pd.DataFrame(sens_rows).reindex(columns=SENS_COLS)
        overfit, floats = _overfit_suspect(base_ret, sens_rows)
        if overfit:
            notes.append("过拟合嫌疑：基准参数有效（收益率>0）而全部浮动档失效，命中附录六「只在某个精确参数下有效」。")
        elif base_ret is not None and base_ret > 0 and floats:
            notes.append("浮动档至少一档仍有效（收益率>0），未命中过拟合嫌疑。")
        if floats and ran == 0:
            notes.append("六维④ 变体均未跑出结果（engine 不可用或策略不接受参数注入），过拟合嫌疑未判定。")
        dims["参数敏感性"] = _dim(
            done=bool(ran),
            reason="ok" if ran else "不可完成：浮动档均未跑出结果",
            档数=len([r for r in sens_rows if r.get("档") != "基准"]),
            基准收益率=base_ret,
            可执行档数=ran,
        )

    # ⑤ 样本外：IS/OOS 按 config 切分（附录六⑤，区间必须登记）
    segs = {}
    for label, lo, hi in (("样本内IS", ctx.start, ctx.is_end), ("样本外OOS", ctx.oos_start, ctx.end)):
        cal_seg = _slice_cal(cal, lo, hi)
        sig_seg = _slice_signals(sig_full, lo, hi)
        res_seg, why_seg = ctx.run_backtest(sig_seg, cal_seg)
        _, stats_seg, _ = _result_parts(res_seg)
        entry = _ordered(stats_seg) if stats_seg else {"完成": False, "原因": why_seg}
        entry["数据区间"] = f"{lo}~{hi}"
        entry["信号数"] = int(len(sig_seg))
        segs[label] = entry
    dims["样本外"] = _dim(
        done=any("收益率" in v for v in segs.values()),
        reason="ok" if any("收益率" in v for v in segs.values()) else f"不可完成：两段均未产出结果（{why}）",
        切分=f"IS {ctx.start}~{ctx.is_end} / OOS {ctx.oos_start}~{ctx.end}",
        分段=segs,
    )
    notes.append(f"六维⑤ 样本区间登记：样本内 {ctx.start}~{ctx.is_end}，样本外 {ctx.oos_start}~{ctx.end}。")

    # ⑥ 市场环境：牛/熊/震荡（附录六⑥，分段标准必须显式登记）
    env_meta = {
        "分段标准": f"指数 {ENV_LOOKBACK} 日涨幅 >+{ENV_BULL:.0%} 牛、<{ENV_BEAR:.0%} 熊、其余震荡",
    }
    if index is None:
        dims["市场环境"] = _dim(done=False, reason="指数不可用，无法判定市场环境（数据不足）", **env_meta)
    else:
        # 先在完整指数历史上算 250 日涨幅（区间起点要借用区间前的历史），再裁到本次回测区间
        label = _env_labels(index)
        if cal is not None:
            label = label.reindex(pd.DatetimeIndex(cal)).dropna()
        if label.empty:
            dims["市场环境"] = _dim(
                done=False, reason="不可完成：回测区间与指数无共同交易日（数据不足）", **env_meta
            )
        else:
            env_meta["分段"] = _env_runs(label)
            if eq_full is None:
                dims["市场环境"] = _dim(
                    done=False, reason=f"不可完成：历史回测未产出净值（{why}），无法按段统计策略表现", **env_meta
                )
            else:
                env_stats = {}
                for env_name in ("牛", "熊", "震荡"):
                    dates = label.index[label == env_name]
                    if len(dates) == 0:
                        env_stats[env_name] = {"完成": False, "原因": "该环境在本区间未出现"}
                        continue
                    s = _period_stats(eq_full, dates)
                    if not s:
                        env_stats[env_name] = {"完成": False, "原因": "段内交易日不足（数据不足）"}
                        continue
                    close_idx = index["close"].astype(float).reindex(dates).dropna()
                    if len(close_idx) >= 2:
                        s["指数收益"] = float(close_idx.iloc[-1] / close_idx.iloc[0] - 1.0)
                    s["交易日"] = int(len(dates))
                    env_stats[env_name] = _ordered(s)
                dims["市场环境"] = _dim(done=True, reason="ok", **env_meta, **env_stats)

    notes.append("六维③ 成本三档口径：同一批信号仅改 engine 的佣金/印花税/滑点配置，不重算信号。")

    # 记录表需要「规则编号」与「各规则样本」：从历史回测维的成交明细里按 rule_id 统计（附录六记录表 7 字段、
    # 样本门槛按同规则样本数）。缺失时留空，让 judge 走兜底而非编造编号。
    rule_counts: dict = {}
    trades = trades_full
    if isinstance(trades, pd.DataFrame) and len(trades) and "rule_id" in trades.columns:
        for rid, cnt in trades["rule_id"].astype(str).value_counts().items():
            for one in str(rid).split("+"):
                one = one.strip()
                if one:
                    rule_counts[one] = rule_counts.get(one, 0) + int(cnt)
    return {
        "name": name,
        "benchmark": "沪深300",     # 记录表「基准」字段（附录六 行2153）
        "rule_id": "、".join(sorted(rule_counts)) if rule_counts else "",
        "各规则样本": rule_counts,
        "dims": dims,
        "obs_order": list(OBS_ORDER),
        "sensitivity": sens_df,
        "overfit_suspect": bool(overfit),
        "notes": notes,
    }


def _slice_signals(sig, lo, hi):
    if sig is None or sig.empty:
        return sig if sig is not None else pd.DataFrame(columns=_SIGNAL_COLS)
    out = sig
    if lo:
        out = out[out["date"] >= pd.Timestamp(lo)]
    if hi:
        out = out[out["date"] <= pd.Timestamp(hi)]
    return out


if __name__ == "__main__":
    # 冒烟测试：合成指数 + 假 engine，验证六维编排、观测口径排序、分段与过拟合判定
    import dataclasses

    dates = pd.bdate_range("2020-01-02", periods=420)
    wave = np.sin(np.arange(len(dates)) / 30.0)
    close = pd.Series(3000.0 * (1.0 + 0.25 * wave + np.arange(len(dates)) * 0.0002), index=dates)
    fake_index = pd.DataFrame({"close": close})

    def fake_strategy(calendar=None, cfg=None, **kw):
        cal = list(calendar) if calendar is not None else list(dates)
        n = int(kw.get("N", 30))
        step = max(1, int(round(300.0 / n)))  # 参数档不同 → 信号数不同，用于验证敏感性表能跑
        rows = [{"date": cal[i], "code": "000001", "side": "buy", "reason": "t", "rule_id": "R-35"} for i in range(0, len(cal), step)]
        return pd.DataFrame(rows)

    @dataclasses.dataclass
    class FakeCfg:
        commission_rate: float = 0.00025
        commission_min: float = 5.0
        stamp_tax: float = 0.0005
        slippage: float = 0.001

    def fake_run_backtest(signals, calendar=None, cfg=None):
        d = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(signals["date"])))) if signals is not None and len(signals) else dates
        rate = 0.0035
        if cfg is not None:
            drag = sum(float(getattr(cfg, f, 0.0) or 0.0) for f in ("commission_rate", "stamp_tax", "slippage"))
            rate -= drag
        eq = 100000.0 * np.cumprod(np.full(len(d), 1.0 + rate))
        equity = pd.DataFrame({"date": d, "cash": 0.0, "market_value": eq, "equity": eq, "exposure": 1.0, "n_positions": 1})
        return {"equity": equity, "trades": pd.DataFrame({"code": ["000001"] * 21}), "stats": {"总收益": eq[-1] / eq[0] - 1, "年化": 0.1, "最大回撤": -0.05, "波动": 0.12, "换手率": 0.3, "资金占用": 0.9, "策略失效期": 12, "交易笔数": 21, "胜率": 0.5, "盈亏比": 1.2}}

    report = run_six_dim(
        fake_strategy,
        "smoke",
        cfg={
            "calendar": list(dates),
            "index": fake_index,
            "bond": None,
            "run_backtest": fake_run_backtest,
            "engine_config": FakeCfg(),
            "sensitivity_grid": [{"规则编号": "R-35", "参数": "N", "值": 24, "档": "-20%"}, {"规则编号": "R-35", "参数": "N", "值": 36, "档": "+20%"}],
        },
    )
    print("dims keys:", list(report["dims"].keys()))
    print("obs_order:", report["obs_order"])
    print("历史回测 键序:", list(report["dims"]["历史回测"].keys()))
    print("收益率/总收益 同值:", report["dims"]["历史回测"]["收益率"] == report["dims"]["历史回测"]["总收益"])
    print("交易成本 三档收益率:", {k: round(v.get("收益率", float("nan")), 4) for k, v in report["dims"]["交易成本"]["三档"].items()})
    print("市场环境 标准:", report["dims"]["市场环境"]["分段标准"])
    print("市场环境 牛/熊/震荡 完成:", {k: ("收益率" in report["dims"]["市场环境"].get(k, {})) for k in ("牛", "熊", "震荡")})
    print("样本外 完成:", report["dims"]["样本外"]["完成"], "| 切分:", report["dims"]["样本外"]["切分"])
    print("样本外 分段:", {k: (round(v["收益率"], 4) if "收益率" in v else v.get("原因")) for k, v in report["dims"]["样本外"]["分段"].items()})
    print("sensitivity 行数:", len(report["sensitivity"]), "| overfit_suspect:", report["overfit_suspect"])
    print("sensitivity 参数档:", list(report["sensitivity"]["档"].values))
    print("notes:", len(report["notes"]))
    print("_overfit_suspect 全失效:", _overfit_suspect(0.05, [{"档": "-20%", "收益率": -0.01}, {"档": "+20%", "收益率": -0.02}]))
    print("_overfit_suspect 有档有效:", _overfit_suspect(0.05, [{"档": "-20%", "收益率": -0.01}, {"档": "+20%", "收益率": 0.03}]))
    print("_overfit_suspect 基准无效:", _overfit_suspect(-0.05, [{"档": "-20%", "收益率": -0.01}]))
