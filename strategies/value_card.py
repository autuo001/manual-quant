"""strategies/value_card.py — 价值投资流信号生成（手册第二章 + 第〇章七步法；规则见附录五）

SPEC 歧义处的取舍（逐条记账，便于主控核对）：
1) R-27 安全边际需要内在价值，而 SPEC §2 已声明 DCF/现金流/无风险利率数据不可得，故按 SPEC §3⑦
   的授权「用可得替代并明确标注降级为定性」：内在价值替代 = 格雷厄姆成长股公式 EPS×(8.5+2g)
   （spec/04 行 1071–1072），g = clip(min(营收同比, 净利润同比)/100, 0, 0.20)（增长率"必须偏保守"
   取自 spec/04 行 1080 的对应市盈率表，其上限档为 20% → 48.5 倍）。信号 reason 固定带该降级标注。
2) R-27 三档（≥30%/≥33%/≥50%）的适用档位依赖企业分类（二类企业/净流动资产），本项目无该分类
   数据，故默认取一般性底线 30%（margin_tier 可切到 0.33/0.50）。
3) 价值仓不设止损（spec/01 行 137–145：价值投资流"不用技术指标、不做跌破止损"），故 stop_pct 恒 NaN。
4) 信号 weight 按 SPEC §3④ 字面理解为「目标权重」（占组合权益比例）；卖出信号 weight=0。
5) R-01/R-02/R-03/R-08 属家庭财务层，回测中恒为满足（杠杆恒为 1、只用长期资金、留现金缓冲），
   不生成信号，仅在此记账。
6) 加仓口径：R-37 禁止摊平 → 已持有标的不因为跌破成本而补仓；本模块只在空仓时按目标权重建仓
   （entry_tranches 参数可改为分批，分批时每笔给递增的目标权重，且只在账面利润>0 时补足）。
7) 单位口径：与 data/loader.py 一致，毛利率/净利率/ROE/两条同比一律「百分数」（净利率 = 净利润 ÷ 营收 × 100），
   故财务门槛阈值为 40 / 20 / 0 / 0 / 0；EPS 取最近一期年报（避免累计期口径混用）。
8) rule_id 与 constraint_ids 分离（附录五元数据 2066「Ⅱ 区/C·D 级不得单独构成买入依据」）：
   - rule_id 只填**真正构成该信号依据**的规则 → 买入恒为 R-27（B 级·Ⅰ 区，安全边际）；
     卖出恒为 R-28（B 级·Ⅰ 区，三个卖出理由）。
   - constraint_ids 填**约束/限定了该信号执行方式的规则** → R-04（归零口径仓位上限）、R-16（价值仓 3—10 只）、
     R-37（禁止摊平）、R-32（三年守则）。这些编号一律不进 rule_id：R-16/R-32 属 C 级·Ⅱ 区，
     记进 rule_id 违反元数据 2066 的记账要求，且 rules.gate() 会给出「未记账任何触发类依据」的告警。
   - 卖出信号的 R-32 是**时间约束**（最短评估期三年）而非卖出依据本身，故记 constraint_ids；
     「公司变了」卖出按 R-32 自身失效条件（公司永久沉沦优先）豁免最短持有期，reason 里写明该豁免。
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pandas as pd

_SIGNAL_COLS = ["date", "code", "side", "reason", "rule_id", "constraint_ids", "weight", "stop_pct"]

# 依据类规则编号（rule_id 只装这些）：买入依据 = R-27 安全边际（B 级·Ⅰ 区）；
# 卖出依据 = R-28 三个卖出理由（B 级·Ⅰ 区）。附录五元数据 2066：C/D 级或 Ⅱ 区规则不得构成
# 买入依据，故 R-16(C)/R-32(C) 一律不进 rule_id，改记 constraint_ids。
BUY_BASIS = "R-27"
SELL_BASIS = "R-28"
# 约束类规则编号（约束/限定信号的执行方式，不构成依据本身）
C_POSITION = "R-04"   # 归零口径：单标的最大潜在损失 ≤ 可投资金融资产 5%（3%—10% 可自定）
C_COUNT = "R-16"      # 价值仓 3—10 只（20 只警戒）
C_NO_AVG_DOWN = "R-37"  # 亏损头寸不加码、绝不摊平
C_HOLD = "R-32"       # 三年守则：最短评估期 3 年（失效条件＝公司永久沉沦，此时不适用）

# R-27 安全边际三档（spec/08 R-27：≥30% 一般／≥33% 门槛／≥50% 经典，"同一标的适用多档时取更严"）
MARGIN_TIERS = (0.30, 0.33, 0.50)
# spec/04 行 1071–1072：价值 = 当期（正常）利润 × (8.5 + 2 × 预期年增长率)
GRAHAM_BASE, GRAHAM_GROWTH_MULT = 8.5, 2.0
# spec/04 行 1080：增长率上限档 20% → 对应市盈率 48.5 倍
GROWTH_CAP_PP = 20.0        # 增长率上限，百分点（手册对照表最大档 20% → 48.5 倍，保守口径）
# 财务门槛（手册原文阈值；与 data/loader.py 口径一致：毛利率/净利率/ROE/同比一律「百分数」，
# 净利率 = 净利润 ÷ 营收 × 100）
GATE_ITEMS = (
    ("毛利率", "gross_margin", ">=", 40.0),
    ("净利率", "net_margin", ">", 20.0),
    ("ROE", "roe", ">", 0.0),
    ("营收同比", "rev_yoy", ">", 0.0),
    ("净利润同比", "profit_yoy", ">", 0.0),
)
# A 股主板/创业板/科创板代码前缀（剔除 8xx/9xx/4xx 北交所与新三板：R-04 归零口径下的流动性/退市差异）
MAIN_PREFIXES = ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688")

# ① config.py 由他人并行实现；未就绪时用与 SPEC §① 完全一致的常量兜底（数值不得自创）
_DEFAULTS = {
    "START_DATE": "2016-01-01",
    "END_DATE": "2026-09-30",
    "RISK_ZERO_LOSS": 0.05,
    "RISK_PER_TRADE": 0.015,
    "MAX_POSITIONS": 10,
    "MIN_POSITIONS": 3,
    "FIRST_TRANCHE": 0.20,
    "STOP_LOSS_PCT": 0.01,
    "INITIAL_CAPITAL": 100_000.0,
}
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # manual_quant/

# 可调参数（供 rules.sensitivity_grid / validate.protocol 的参数敏感性维度调用）
PARAMS = {
    "margin_tier": 0.30,      # R-27 三档之一
    "max_positions": _DEFAULTS["MAX_POSITIONS"],   # R-16
    "min_positions": _DEFAULTS["MIN_POSITIONS"],   # R-16
    "max_weight": _DEFAULTS["RISK_ZERO_LOSS"],     # R-04 归零口径
    "entry_tranches": 1,      # 建仓笔数（1 = 一次建满目标权重）
    "min_hold_years": 3.0,    # R-32 三年守则
    "review": "Q",            # R-40 季度复盘
    "min_list_days": 250,     # 上市满 250 交易日（data/loader.py 的 universe 默认口径）
}

# rules.sensitivity_grid 输出的参数名 → 本模块参数名（否则协议④注入的档位会被静默忽略）
_ALIASES = {
    "margin_general": "margin_tier",   # R-27
    "eval_years": "min_hold_years",    # R-32
    "max_loss_pct": "max_weight",      # R-04
}


# ---------------------------------------------------------------- 上下文/路径

def _opt(modname: str):
    """惰性导入他人模块（并行实现中可能尚不存在，不得让本模块 import 失败）。"""
    if _ROOT not in sys.path:      # 支持 python3 strategies/xxx.py 直接运行时也能解析包内模块
        sys.path.insert(0, _ROOT)
    for cand in (modname, "manual_quant." + modname):
        try:
            return importlib.import_module(cand)
        except Exception:
            continue
    return None


def _const(name: str):
    cfg = _opt("config")
    if cfg is not None:
        val = getattr(cfg, name, None)
        if val is not None:
            return val
    return _DEFAULTS[name]


def _resolve(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for base in (os.getcwd(), _ROOT):
        cand = os.path.abspath(os.path.join(base, path))
        if os.path.exists(cand):
            return cand
    return path


def _data_dir() -> str:
    cfg = _opt("config")
    if cfg is not None and getattr(cfg, "DATA_DIR", None):
        return _resolve(str(cfg.DATA_DIR))
    env = os.environ.get("MQ_DATA_DIR")
    if env:
        return _resolve(env)
    return os.path.abspath(os.path.join(_ROOT, os.pardir, "six_layer_backtest", "data", "parquet"))


def _yjbb_rich() -> str:
    cfg = _opt("config")
    if cfg is not None and getattr(cfg, "YJBB_RICH", None):
        return _resolve(str(cfg.YJBB_RICH))
    return os.path.abspath(os.path.join(_ROOT, os.pardir, "backtest", "data", "parquet", "yjbb.parquet"))


# ---------------------------------------------------------------- 数据读取（惰性、按需切片）

def _calendar(start, end) -> pd.DatetimeIndex:
    loader = _opt("data.loader")
    if loader is not None and hasattr(loader, "trading_calendar"):
        try:
            cal = pd.DatetimeIndex(pd.to_datetime(loader.trading_calendar()))
            return cal[(cal >= start) & (cal <= end)]
        except Exception:
            pass
    idx = pd.read_parquet(os.path.join(_data_dir(), "index.parquet"), columns=["date"])
    cal = pd.DatetimeIndex(pd.to_datetime(idx["date"]).unique()).sort_values()
    return cal[(cal >= start) & (cal <= end)]


def _close_panel(codes, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """按 code 分块读日线闭市价 → 宽表（index=交易日, columns=code）。分块避免 989 万行整表驻留。"""
    codes = list(codes)
    arr = np.full((len(cal), len(codes)), np.nan, dtype=np.float32)
    pos = pd.Series(np.arange(len(codes)), index=codes)
    path = os.path.join(_data_dir(), "daily_all.parquet")
    lo, hi = cal[0], cal[-1]
    for i in range(0, len(codes), 400):
        chunk = codes[i:i + 400]
        df = pd.read_parquet(
            path, columns=["code", "date", "close"],
            filters=[("date", ">=", lo), ("date", "<=", hi), ("code", "in", chunk)],
        )
        if df.empty:
            continue
        ri = cal.get_indexer(pd.to_datetime(df["date"]).values)
        ci = pos.reindex(df["code"].astype(str)).to_numpy()
        ok = (ri >= 0) & pd.notna(ci)
        arr[ri[ok], ci[ok].astype(int)] = df["close"].to_numpy(dtype=np.float32)[ok]
    return pd.DataFrame(arr, index=cal, columns=codes)


_RICH_MAP = {
    "股票代码": "code", "股票简称": "name", "每股收益": "eps",
    "营业总收入-营业总收入": "revenue", "营业总收入-同比增长": "rev_yoy",
    "净利润-净利润": "net_profit", "净利润-同比增长": "profit_yoy",
    "每股净资产": "bps", "净资产收益率": "roe", "每股经营现金流量": "ocfps",
    "销售毛利率": "gross_margin", "所处行业": "industry",
    "最新公告日期": "ann_date", "report_period": "report_period",
}
_FUND_COLS = ["code", "name", "report_period", "ann_date", "eps", "revenue", "net_profit",
              "net_margin", "gross_margin", "roe", "ocfps", "rev_yoy", "profit_yoy", "industry",
              "eps_annual"]


def _as_percent(s: pd.Series) -> pd.Series:
    """单位护栏：yjbb 原始毛利率/ROE/同比为百分数；若上游 loader 已归一为比值则还原为百分数。"""
    s = pd.to_numeric(s, errors="coerce")
    m = s.abs().max(skipna=True)
    return s * 100.0 if (pd.notna(m) and m <= 1.5) else s


def _load_fundamentals() -> pd.DataFrame:
    """优先用 data/loader.load_fundamentals()，缺失时按 SPEC §2 富字段自行派生（列名同契约）。"""
    loader = _opt("data.loader")
    if loader is not None and hasattr(loader, "load_fundamentals"):
        try:
            f = loader.load_fundamentals()
            if f is not None and set(["code", "ann_date", "roe"]).issubset(f.columns):
                return _finalize_fundamentals(pd.DataFrame(f))
        except Exception:
            pass
    raw = pd.read_parquet(_yjbb_rich())
    f = raw.rename(columns=_RICH_MAP)
    f["code"] = f["code"].astype(str)
    f["report_period"] = f["report_period"].astype(str)
    f["ann_date"] = pd.to_datetime(f["ann_date"])
    return _finalize_fundamentals(f)


def _finalize_fundamentals(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f["code"] = f["code"].astype(str)
    f["ann_date"] = pd.to_datetime(f["ann_date"])
    f["report_period"] = f["report_period"].astype(str)
    for col in ("eps", "revenue", "net_profit", "roe", "gross_margin", "rev_yoy", "profit_yoy",
                "ocfps", "bps"):
        if col in f.columns:
            f[col] = pd.to_numeric(f[col], errors="coerce")
    for col in ("gross_margin", "roe", "rev_yoy", "profit_yoy"):
        if col in f.columns:
            f[col] = _as_percent(f[col])
    rev = f.get("revenue")
    if rev is not None:
        derived = pd.to_numeric(f["net_profit"], errors="coerce") / rev.where(rev != 0) * 100.0
    else:
        derived = pd.Series(np.nan, index=f.index)
    if "net_margin" in f.columns:
        nm = pd.to_numeric(f["net_margin"], errors="coerce")
        # 单位护栏：上游若按比值（净利润/营收）给出则折算为百分数，保证与阈值 20.0 同口径
        if pd.notna(nm.abs().median()) and nm.abs().median() <= 1.5:
            nm = nm * 100.0
        f["net_margin"] = nm.where(nm.notna(), derived)
    else:
        f["net_margin"] = derived
    # 格雷厄姆公式用年报刊口径（避免累计期口径混用）：只在 1231 报告期上取 EPS
    f["eps_annual"] = np.where(f["report_period"].str.endswith("1231"), f.get("eps"), np.nan)
    return f.sort_values("ann_date", kind="mergesort").reset_index(drop=True)


def _universe(min_list_days: int, fund: pd.DataFrame) -> list:
    loader = _opt("data.loader")
    if loader is not None and hasattr(loader, "universe"):
        try:
            codes = [str(c) for c in loader.universe(min_list_days=min_list_days)]
            if codes:
                return sorted(codes)
        except Exception:
            pass
    codes = sorted({c for c in fund["code"].unique() if str(c)[:3] in MAIN_PREFIXES})
    return codes


# ---------------------------------------------------------------- 筛选逻辑（七步法 ③④⑤ 层）

def _pit_slice(fund: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """point-in-time 切片：只用公告日 ≤ asof 的记录，逐列取"最后一个非空值"（无未来函数）。"""
    f = fund[fund["ann_date"] <= asof]
    if f.empty:
        return f
    return f.groupby("code", sort=False).last()


def _fair_value(eps_annual, rev_yoy, profit_yoy):
    """R-27 内在价值的可得替代：格雷厄姆成长股公式 价值 = EPS × (8.5 + 2×g)。

    单位口径（手册附录对照表 spec/04 序13，行 1080/1082）：g 必须按**百分点**代入——
    "5% → 18.5 倍、10% → 28.5 倍、20% → 48.5 倍" 对应 8.5+2×5 / 8.5+2×10 / 8.5+2×20。
    早期实现误把同比除以 100（g=0.10）得到 8.7 倍，内在价值被低估约 3.3 倍，
    连带把 R-27 的安全边际门槛变成"PE ≤ 6"，候选票几乎枯竭（2025 年仅 3 个买入信号）。
    """
    g_pp = np.minimum(rev_yoy, profit_yoy)                      # 增长率，百分点
    g_pp = np.clip(np.nan_to_num(g_pp, nan=0.0), 0.0, GROWTH_CAP_PP)
    return pd.to_numeric(eps_annual, errors="coerce") * (GRAHAM_BASE + GRAHAM_GROWTH_MULT * g_pp)


def _gate_mask(f: pd.DataFrame):
    """财务门槛逐项布尔（与 data/loader.financial_gate_row 同阈值），向量化以支撑全市场筛选。"""
    ok = pd.Series(True, index=f.index)
    detail = {}
    for label, col, op, thr in GATE_ITEMS:
        v = pd.to_numeric(f.get(col, pd.Series(np.nan, index=f.index)), errors="coerce")
        cond = (v >= thr) if op == ">=" else (v > thr)
        detail[label] = cond
        ok &= cond
    return ok, detail


def _signal_rows(rows):
    df = pd.DataFrame(rows, columns=_SIGNAL_COLS)
    if df.empty:
        return _empty_signals()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str)
    # 无约束的信号（理论上本策略不产生）也保持列存在且为字符串，避免下游 concat 出 NaN
    df["constraint_ids"] = df["constraint_ids"].fillna("").astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df["stop_pct"] = pd.to_numeric(df["stop_pct"], errors="coerce")
    return df.sort_values(["date", "side", "code"], ascending=[True, False, True]).reset_index(drop=True)


def _empty_signals():
    return pd.DataFrame({
        "date": pd.Series([], dtype="datetime64[ns]"),
        "code": pd.Series([], dtype="object"),
        "side": pd.Series([], dtype="object"),
        "reason": pd.Series([], dtype="object"),
        "rule_id": pd.Series([], dtype="object"),
        "constraint_ids": pd.Series([], dtype="object"),
        "weight": pd.Series([], dtype="float64"),
        "stop_pct": pd.Series([], dtype="float64"),
    })


def _risk_cap_weight() -> float:
    """R-04 归零口径单标的权重上限；core/risk_budget 就绪时以它为准。"""
    rb = _opt("core.risk_budget")
    if rb is not None and hasattr(rb, "max_weight_zero_loss"):
        try:
            w, _ = rb.max_weight_zero_loss(1.0, 1.0)
            return float(w)
        except Exception:
            pass
    return float(_const("RISK_ZERO_LOSS"))


# ---------------------------------------------------------------- 主入口

def _alias_kw(kw: dict) -> dict:
    """接受 rules.sensitivity_grid 输出的参数名（R-04 max_loss_pct / R-27 margin_* 等）作别名。"""
    out = {}
    for k, v in kw.items():
        if v is None:
            continue
        out[_ALIASES.get(k, k)] = v
    return out


def _cfg_overrides(cfg) -> dict:
    """cfg 允许提供本模块策略参数（config 模块 / EngineConfig 均可）；max_weight 只由 kw 指定。"""
    if cfg is None:
        return {}
    out = {}
    for k in list(PARAMS) + ["start", "end"]:
        if k == "max_weight":
            continue
        v = cfg.get(k) if isinstance(cfg, dict) else getattr(cfg, k, None)
        if v is not None:
            out[k] = v
    return out


def run(calendar=None, cfg=None, **kw) -> pd.DataFrame:
    """价值投资流信号：七步法式逐层否决 → 财务门槛 → R-27 安全边际 → R-16 持股数 → R-04 仓位。

    kw: start/end/review/codes/margin_tier/max_positions/min_positions/max_weight/
        entry_tranches/min_hold_years/min_list_days/verbose
    cfg: config 模块或 dict/对象，仅用于取 initial_capital 等标量（缺省时用 SPEC §① 常量）。
    """
    p = dict(PARAMS)
    p.update(_cfg_overrides(cfg))
    p.update(_alias_kw(kw))
    start = pd.Timestamp(p.get("start") or _const("START_DATE"))
    end = pd.Timestamp(p.get("end") or _const("END_DATE"))
    tier = float(p["margin_tier"])
    if tier not in MARGIN_TIERS:
        # rules.sensitivity_grid 的 R-27 档位（margin_general/gate/classic）可能给出档外值 → 就近取档
        tier = min(MARGIN_TIERS, key=lambda t: abs(t - tier))

    fund = _load_fundamentals()
    codes = p.get("codes") or _universe(int(p["min_list_days"]), fund)
    codes = [str(c) for c in codes]
    fund = fund[fund["code"].isin(set(codes))]

    if calendar is not None:
        cal = pd.DatetimeIndex(pd.to_datetime(list(calendar)))
        cal = cal[(cal >= start) & (cal <= end)]
    else:
        cal = _calendar(start, end)
    if len(cal) == 0:
        return _empty_signals()

    # 多取 400 个交易日用于上市天数与面板对齐
    pre = cal[0] - pd.Timedelta(days=400)
    panel = _close_panel(codes, _calendar(pre, end))
    review_dates = _review_dates(cal, p.get("review", "Q"))

    cap = float(p["max_weight"]) if p.get("max_weight") else _risk_cap_weight()
    max_pos, min_pos = int(p["max_positions"]), int(p["min_positions"])
    tranches = max(1, int(p["entry_tranches"]))
    min_hold_years = float(p["min_hold_years"])
    min_list_days = int(p["min_list_days"])

    holdings = {}   # code -> {"entry": Timestamp, "weight": 累计目标权重, "px": 最后买入价, "left": 剩余笔数, "plan": 计划权重}
    rows = []
    listed = panel.notna()
    for d in review_dates:
        if d not in panel.index:
            continue
        f = _pit_slice(fund, d)
        if f.empty:
            continue
        gate_ok, _ = _gate_mask(f)
        px = panel.loc[d]
        f = f.assign(price=px.reindex(f.index), gate=gate_ok.reindex(f.index).fillna(False))
        # 第三步 逻辑成立：财务门槛五项全过；第四步 价格合理：R-27 安全边际
        f["fair_value"] = _fair_value(f.get("eps_annual"), f.get("rev_yoy"), f.get("profit_yoy"))
        f["margin"] = 1.0 - f["price"] / f["fair_value"].where(f["fair_value"] > 0)
        # 上市满 250 交易日（data/loader.universe 口径）
        i_row = panel.index.get_loc(d)
        n_bars = listed.iloc[max(0, i_row - min_list_days + 1): i_row + 1].sum()
        cand = f[(f["gate"]) & (f["margin"] >= tier) & (f["price"] > 0)
                 & (n_bars.reindex(f.index).fillna(0) >= min_list_days)]
        cand = cand.sort_values("margin", ascending=False)
        n_target = int(min(max_pos, len(cand)))
        target = list(cand.index[:n_target])
        if p.get("verbose"):
            print("[value_card] %s 候选 %d → 目标持仓 %d" % (d.date(), len(cand), n_target))
            if n_target < min_pos:
                # R-16 下限 3 只：合格标的不足时标注「样本/数据不足」，不虚增持仓
                print("    注意：合格标的 %d < R-16 下限 %d 只 → 标注数据不足" % (n_target, min_pos))

        better_available = any(c not in holdings for c in target)
        # 第七步 退出条件：R-28 三个卖出理由的可编码部分 + R-32 三年守则 + R-37 禁止摊平
        for code in list(holdings):
            if code not in f.index:
                continue   # 数据不足（无公告日≤评估日的业绩记录）→ 凡说不清就不动手（spec/01 三层使用法）
            row = f.loc[code]
            held_years = (d - holdings[code]["entry"]).days / 365.25
            if not bool(row["gate"]):
                rows.append(_sell(d, code, holdings[code]["weight"],
                                  "R-28 公司变了：财务门槛不再全项通过(毛利率≥40%/净利率>20%/ROE>0/"
                                  "营收同比>0/净利润同比>0)；R-28 与盈亏无关故立即卖出；"
                                  "约束 R-32 三年守则被其失效条件豁免（公司永久沉沦优先），不设最短持有期",
                                  SELL_BASIS, C_HOLD))
                del holdings[code]
            elif code not in target and held_years >= min_hold_years and better_available:
                rows.append(_sell(d, code, holdings[code]["weight"],
                                  "R-28 有更好的：安全边际排名已被更高标的取代，且已满足约束 "
                                  "R-32 三年守则的最短评估期(持有%.1f年≥%.1f年)"
                                  % (held_years, min_hold_years), SELL_BASIS, C_HOLD))
                del holdings[code]
        # 第五步 仓位合适：R-16 持股数 3–10 + R-04 归零口径权重上限
        if n_target:
            w = min(1.0 / n_target, cap)
            # 已持仓且仍在目标内：仅在有账面利润时补足剩余笔数（R-37 亏损头寸不加码）
            for code, h in holdings.items():
                if h["left"] <= 0 or code not in target:
                    continue
                px_now = f.loc[code, "price"] if code in f.index else np.nan
                if not (pd.notna(px_now) and px_now >= h["px"]):
                    continue
                step = min(h["plan"] / tranches, cap - h["weight"])
                if step <= 0:
                    h["left"] = 0
                    continue
                h["weight"] += step
                h["left"] -= 1
                h["px"] = float(px_now)
                rows.append(_buy(d, code, h["weight"],
                                 "R-27 补足建仓笔数(剩余%d笔)：仍在目标内且账面利润>0；约束 R-37 禁止摊平"
                                 "（亏损头寸不加码）、R-04 归零口径权重上限%.1f%%、R-16 目标持仓%d只"
                                 % (h["left"], cap * 100, n_target),
                                 BUY_BASIS, "+".join((C_POSITION, C_COUNT, C_NO_AVG_DOWN))))
            for code in target:
                if code in holdings:
                    continue
                r = f.loc[code]
                plan = min(w, cap)
                step = min(plan / tranches, cap)
                rows.append(_buy(d, code, step,
                                 ("R-27 逻辑成立+价格合理：财务门槛五项全过 + 安全边际%.1f%%≥%.0f%%"
                                  "[降级为定性:内在价值=EPS×(8.5+2g) 替代, "
                                  "g=clip(min(营收同比,净利润同比),0,0.20)]；约束 R-16 目标持仓%d只、"
                                  "R-04 归零口径权重上限%.1f%%；内在价值替代值%.2f；建仓%d笔之一"
                                  % (r["margin"] * 100, tier * 100, n_target, cap * 100,
                                     r["fair_value"], tranches)),
                                 BUY_BASIS, "+".join((C_POSITION, C_COUNT))))
                holdings[code] = {"entry": d, "weight": step, "px": float(r["price"]),
                                  "left": tranches - 1, "plan": plan}
    return _signal_rows(rows)


def _buy(d, code, weight, reason, rule_id, constraint_ids):
    return {"date": d, "code": code, "side": "buy", "reason": reason, "rule_id": rule_id,
            "constraint_ids": constraint_ids, "weight": float(weight), "stop_pct": np.nan}


def _sell(d, code, weight, reason, rule_id, constraint_ids):
    return {"date": d, "code": code, "side": "sell", "reason": reason, "rule_id": rule_id,
            "constraint_ids": constraint_ids, "weight": 0.0, "stop_pct": np.nan}


def _review_dates(cal: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """R-40 季度复盘：每季度首个交易日评估一次（基本面数据本身也是季度快照）。"""
    s = pd.Series(cal, index=cal)
    if str(freq).upper().startswith("Q"):
        key = s.index.to_period("Q")
    elif str(freq).upper().startswith("M"):
        key = s.index.to_period("M")
    else:
        key = s.index.to_period("Y")
    return pd.DatetimeIndex(s.groupby(key).first().values)


def params(**kw) -> dict:
    p = dict(PARAMS)
    p.update({k: v for k, v in kw.items() if v is not None})
    return p


if __name__ == "__main__":
    # 冒烟测试：小样本自证可跑（不依赖 config.py / data.loader.py / core.risk_budget.py）
    print("data_dir =", _data_dir())
    print("yjbb_rich =", _yjbb_rich())
    fund = _load_fundamentals()
    print("fundamentals:", fund.shape, "列:", [c for c in _FUND_COLS if c in fund.columns])
    codes = _universe(250, fund)
    print("universe:", len(codes), codes[:3])
    t0 = pd.Timestamp("2022-01-01")
    cal = _calendar(t0, pd.Timestamp("2024-12-31"))
    print("calendar:", len(cal), cal[0].date(), cal[-1].date())
    sig = run(start="2022-01-01", end="2024-12-31", verbose=True)
    print("signals:", len(sig), "buy=%d sell=%d" % ((sig.side == "buy").sum(), (sig.side == "sell").sum()))
    if len(sig):
        print(sig.head(3).to_string(max_colwidth=70))
        # 自检（附录五元数据 2066）：rule_id 只装依据类；约束类编号只在 constraint_ids
        cons = set()
        for cell in sig["constraint_ids"].dropna().astype(str):
            cons |= {x.strip() for x in cell.split("+") if x.strip()}
        bad_basis = [r for r in sig["rule_id"].dropna().astype(str).unique() if r not in (BUY_BASIS, SELL_BASIS)]
        bad_side = (set(sig.loc[sig.side == "buy", "rule_id"]) - {BUY_BASIS}) | (set(sig.loc[sig.side == "sell", "rule_id"]) - {SELL_BASIS})
        print("rule_id 取值:", sorted(sig["rule_id"].unique()), "｜constraint_ids 取值:", sorted(cons))
        print("自检 rule_id 仅依据类: %s｜买入=R-27/卖出=R-28: %s｜constraint_ids 非空: %s"
              % (not bad_basis, not bad_side, bool((sig["constraint_ids"].astype(str).str.strip() != "").all())))
        try:
            import rules as _rules
            print("rules.gate(buy rule_id) 全通过:",
                  all(_rules.gate([r])[0] for r in sig.loc[sig.side == "buy", "rule_id"].unique()))
        except Exception as exc:
            print("rules 闸门自检跳过:", exc)
    # 与 data/loader.financial_gate_row 的口径交叉核对（该模块可能尚未就绪）
    loader = _opt("data.loader")
    if loader is not None and hasattr(loader, "financial_gate_row"):
        f = _pit_slice(fund, pd.Timestamp("2024-06-30"))
        mine = _gate_mask(f)[0]
        mismatch = 0
        for code in f.index[:20]:
            try:
                ref = loader.financial_gate_row(code, f.loc[code, "report_period"])
            except Exception:
                continue
            if bool(ref.get("pass")) != bool(mine.loc[code]):
                mismatch += 1
                print("  口径不一致:", code, ref.get("pass"), bool(mine.loc[code]))
        print("loader.financial_gate_row 交叉核对: 抽样 20 条，不一致 %d 条" % mismatch)
    else:
        print("cross-check skipped: data/loader.py 未就绪")
