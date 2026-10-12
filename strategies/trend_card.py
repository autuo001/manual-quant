"""strategies/trend_card.py — 趋势交易流信号生成（手册第三章 + 附录二交易规则；规则见附录五）

SPEC 歧义处的取舍（逐条记账，便于主控核对）：
1) 区间窗口 N 与缓冲区 ε 手册原文未给（spec/05 行 1483 自注"区间窗口须自定"），故取默认
   N=60 个交易日、ε=0.5%；两者进 PARAMS 供 validate.protocol 的参数敏感性维度扫描（SPEC §3⑦ 要求）。
2) 上下沿严格按 SPEC 字面：上限_t = 过去 N 日最高收盘 ×(1−ε)，下限_t = 过去 N 日最低收盘；两者都
   shift(1)，保证 T 日信号只用 T 日之前的已实现数据（T 日收盘出信号、T+1 开盘成交）。
3) 信号 weight 按 SPEC §3④ 字面理解为「目标权重」（占组合权益比例），加仓信号给出递增的目标权重。
4) R-36 止损双通道：buy 信号带 stop_pct=1%（engine 原生支持时由 engine 执行），同时策略自行在收盘价
   跌破最后买入价×(1−1%) 时发出 sell 信号（engine 不支持 stop_pct 时仍可独立回测）；
   emit_stop_sell=False 可关闭策略侧止损、只交给 engine 原生 R-36。
5) 建仓/加仓节奏用收盘价近似成交价（engine 实际按 T+1 开盘成交，价差体现在回测结果里）。
6) 单笔计划额度 = min(R-04 归零口径 5%, R-38 风险预算 ÷ 止损距)；首笔 = 计划额度 1/n_tranches（R-35）。
   注意：10 万资金下首笔约 1%，对多数 >10 元的票不足 1 手，engine 会按"不足一手"拒单——这是 R-35 的
   1/5 分批与 100 股整手制度在 10 万资金量下的真实冲突，fraction 不可再缩小。
7) 绝不因股价太高而不买进（spec/05 行 1474）→ 不做任何绝对价位过滤；绝不平摊（R-37）→ 加仓要求
   账面利润 > 0，且每次加仓价必须高于上一次买入价（spec/05 行 1485 方向纪律）。
8) 【本策略的买入依据尚未取得可执行资格——不得据此下单】R-34 在 spec/08-规则总表.md 中标为
   **C 级·经验参考**，且属**Ⅱ 区**（附录五元数据 2066：Ⅰ 区＝可直接执行或满足条件后执行；
   Ⅱ 区＝仅作参考或暂不建议作为触发条件的规则，**不得单独构成买入依据**，须逐条按附录六六维协议
   验证、走「规则变更单」才能升级）。其「验证」字段原文即 **未验证（见 V-09）**，V-09（spec/09 行 2149）
   进一步写明：趋势交易规则原文「不建议投入回测资源——不适合多数人，执行偏差主导结果」。
   本策略的买入触发依据恰好就是 R-34（区间被有效突破后才进场），故在 R-34 完成升级之前，
   **本策略产出的全部信号在手册纪律下没有可执行资格**：这不是可以靠调参修好的实现缺陷，而是
   R-34 自身的证据等级所决定的状态。可直接验证：`rules.gate(['R-34'])` 返回 `(False, …)`
   （见文件末冒烟测试）；任何把 R-34 买入信号送进闸门的下游都应被拦截。
   → **这些信号只能作研究/协议演示用途，不得据此下单**；跑出来的指标只能登记为「未验证/研究」，
   不得写入记录表的「已验证」。卖出依据 R-36 同为 C 级·Ⅱ 区（R-37 为 B 级·Ⅰ 区，且只作约束记账）。
9) 记账列拆分（与 strategies/value_card.py 同一口径，落实元数据 2066）：
   - `rule_id` 只装**触发该信号的依据类规则**：
     · 买入恒为 R-34（最小阻力路线区间突破）。加仓笔的逐笔触发条件来自 spec/05 行 1485「步步上涨」
       方向纪律，手册未编号、rules.py 未登记，故不得冒充规则编号；加仓依据仍记 R-34（趋势未终结）。
     · 卖出为 R-36（止损：触发即执行、不商量）或 R-34（收盘跌破区间下沿＝同一条最小阻力路线规则的
       下行方向判据，行 1477「趋势终结时了结交易」；spec/08 未为离场单列编号，故取最近的已登记编号
       并在此声明该归属）。R-28 属价值仓卖出理由，本策略不适用。
   - `constraint_ids` 只装**约束/限定执行方式、不构成依据本身**的规则：R-04（归零口径单标的权重上限）、
     R-35（首笔 ≤ 计划额度 1/5、无账面利润不加仓）、R-36（最后买入价下方 1% 止损，随 buy 信号下挂）、
     R-37（亏损头寸不加码、绝不摊平）、R-38（单笔风险 ≤ 总资金 1%—2%）。这些编号一律不进 rule_id：
     它们的 kind 是 sizing/exit，不是买入触发依据，记进 rule_id 会污染闸门判定（gate 只把
     kind∈{timing,filter} 当触发类）。空字符串表示该信号无约束类规则参与（如跌破下沿的卖出）。
   - R-16（3—10 只）**不记为趋势信号的约束**：spec/08 R-16 写明「适用＝价值仓」、失效条件「指数策略
     不适用」；本模块的持仓上限只是沿用 config.py 的 MAX_POSITIONS 数值（SPEC §① 把该常量标注为
     R-16），属参数借用，不构成 R-16 在趋势仓的适用性主张。
10) 记账列拆分的跨模块影响面（未改他人文件，仅声明）：`validate/protocol.py` 的参数敏感性维度用信号
   `rule_id` 反查可扫参数（`_codes_from_signals`），约束类编号移出 rule_id 后，维④ 对本策略只再扫
   R-34（无可调数值参数 → 不进网格）与 R-36（stop_loss_pct）；R-04 / R-35 / R-38 的档位扫描随之消失。
   SPEC §3⑦ 要求「N 与 ε 必须进 params 以便敏感性测试」，本模块 PARAMS 保留 n_window/eps_band 供直接
   注入，但维④ 不会自动扫到它们——要恢复这几档证据，须由 protocol 侧同时读 constraint_ids 或由
   调用方显式注入（非本文件所有权范围）。
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pandas as pd

_SIGNAL_COLS = ["date", "code", "side", "reason", "rule_id", "constraint_ids", "weight", "stop_pct"]

# 本策略的买入依据 R-34 在手册中属 C 级·Ⅱ 区且「未验证（见 V-09）」→ 无独立可执行资格。
# 该常量供下游/自检直接读取，避免有人只看指标数字就当成可执行策略（见文件头第 8 条）。
EXECUTION_QUALIFIED = False
QUALIFICATION_NOTE = ("R-34（最小阻力路线区间突破）为 C 级·Ⅱ 区、验证＝未验证（见 V-09），"
                      "附录五元数据 2066 规定 Ⅱ 区不得单独构成买入依据 → 本策略信号仅供研究，"
                      "不得据此下单；rules.gate(['R-34']) 返回 False。")

# 依据类规则编号（rule_id 只装这些）：买入依据 = R-34 最小阻力路线区间突破（C 级·Ⅱ 区，见文件头第 8 条）；
# 卖出依据 = R-36 止损（C 级·Ⅱ 区）/ R-34 跌破区间下沿。依据类编号一律不带约束（R-04/R-35/R-36/R-37/R-38）。
BUY_BASIS = "R-34"
SELL_BASIS_STOP = "R-36"   # 止损卖出：触发即执行、不商量（spec/08 R-36，行 2117）
SELL_BASIS_BAND = "R-34"   # 跌破下沿了结：同一最小阻力路线规则的下行判据（spec/05 行 1477/1483）
# 约束类规则编号（约束/限定执行方式，不构成依据本身）——一律不进 rule_id
C_POSITION = "R-04"     # 归零口径：单标的最大潜在损失 ≤ 可投资金融资产 5%（A 级·Ⅰ 区，sizing）
C_FIRST = "R-35"        # 首笔 ≤ 计划额度 1/5、无账面利润不加仓（C 级·Ⅱ 区，sizing）
C_STOP = "R-36"         # 最后买入价下方 1% 止损，随 buy 信号下挂（C 级·Ⅱ 区，exit）
C_NO_AVG_DOWN = "R-37"  # 亏损头寸不加码、绝不摊平（B 级·Ⅰ 区，sizing）
C_RISK = "R-38"         # 单笔风险 ≤ 总资金 1%—2%（C 级·Ⅱ 区，sizing）
# 建仓/加仓信号共用的约束集合；加仓笔另加 R-37（浮盈才加、绝不摊平）
CONSTRAINTS_ENTRY = "+".join((C_POSITION, C_FIRST, C_STOP, C_RISK))
CONSTRAINTS_ADD = "+".join((C_POSITION, C_FIRST, C_STOP, C_NO_AVG_DOWN, C_RISK))

MAIN_PREFIXES = ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688")

_DEFAULTS = {
    "START_DATE": "2016-01-01",
    "END_DATE": "2026-09-30",
    "RISK_ZERO_LOSS": 0.05,
    "RISK_PER_TRADE": 0.015,
    "MAX_POSITIONS": 10,
    "FIRST_TRANCHE": 0.20,
    "STOP_LOSS_PCT": 0.01,
}

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # manual_quant/

# 可调参数（validate.protocol 参数敏感性维度 → 每项扫 3 档）
PARAMS = {
    "n_window": 60,          # 区间窗口 N（手册未给，默认 60 交易日）
    "eps_band": 0.005,       # 缓冲区 ε（手册未给，默认 0.5%）
    "top_k": 5,              # 领头羊横截面取前 k（spec/05 行 1479；手册"5 只左右"）
    "n_tranches": 5,         # R-35 首笔 ≤ 计划额度 1/5 → 5 笔建满
    "add_step": 0.01,        # 加仓步长（spec/05 行 1487：涨 1% 再买）
    "stop_pct": 0.01,        # R-36 最后买入价下方 1%
    "max_weight": None,      # 单笔计划额度上限；None → R-04 归零口径上限（5%）
    "trend_account": None,   # 可选：趋势仓总暴露上限（spec/02 第 3 层"趋势仓 ≤ 可投资金融资产 5%"）；
                             # 默认不启用：10 万资金下 0.05/5=0.2% 首笔不足 1 手，且 SPEC §3⑦ 未要求
    "risk_per_trade": 0.015, # R-38 单笔风险 1%–2% 取中值（换算成权重上限 = 风险预算 ÷ 止损距）
    "max_positions": 10,     # R-16 3–10 只
    "min_list_days": 250,    # 上市满 250 交易日
    # R-36 双通道开关：engine 原生支持 stop_pct（core/engine.py 已实现 R-36 止损）时可置 False，
    # 避免同一止损被 engine 与策略各判一次
    "emit_stop_sell": True,
}


# ---------------------------------------------------------------- 上下文/路径（同 value_card，保持自足）

def _opt(modname: str):
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


# ---------------------------------------------------------------- 数据

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
    """按 code 分块读闭市价 → 宽表（index=交易日, columns=code）；避免 989 万行整表驻留。"""
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


def _universe(min_list_days: int) -> list:
    loader = _opt("data.loader")
    if loader is not None and hasattr(loader, "universe"):
        try:
            codes = [str(c) for c in loader.universe(min_list_days=min_list_days)]
            if codes:
                return sorted(codes)
        except Exception:
            pass
    codes = pd.read_parquet(
        os.path.abspath(os.path.join(_ROOT, os.pardir, "backtest", "data", "parquet", "yjbb.parquet")),
        columns=["股票代码"],
    )["股票代码"].astype(str).unique()
    return sorted({c for c in codes if c[:3] in MAIN_PREFIXES})


# ---------------------------------------------------------------- 区间/信号判定

def _entry_flags(breakout: np.ndarray) -> np.ndarray:
    """突破首日才动手（spec/05 行 1473「等市场变化证明你的观点后再交易」）。"""
    prev = np.vstack([np.zeros((1, breakout.shape[1]), dtype=bool), breakout[:-1]])
    return breakout & ~prev


def _walk(c: np.ndarray, dn: np.ndarray, i0: int, stop_pct: float, add_step: float,
          n_tranches: int):
    """单次持仓的逐笔推演（止损/跌破下沿先于加仓判定，同日多条件按 止损>下沿>加仓）。

    返回 (加仓点[(day, tranche)], 离场日|None, 离场原因)。NaN（停牌）比较恒为 False，天然跳过。
    """
    T = c.shape[0]
    last_buy = c[i0]
    tranche = 1
    t = i0
    adds = []
    while True:
        lo = t + 1
        if lo >= T:
            return adds, None, "open"          # 数据结束仍未离场 → 不发卖出信号，交给 engine 清算
        cc = c[lo:]
        bb = dn[lo:]
        stop_px = last_buy * (1.0 - stop_pct)
        eidx = np.flatnonzero((cc <= stop_px) | (cc < bb))
        aidx = np.flatnonzero(cc >= last_buy * (1.0 + add_step)) if tranche < n_tranches else None
        if aidx is not None and aidx.size and (eidx.size == 0 or aidx[0] < eidx[0]):
            day = lo + int(aidx[0])
            tranche += 1
            adds.append((day, tranche))
            last_buy = c[day]
            t = day
            continue
        if eidx.size:
            day = lo + int(eidx[0])
            return adds, day, ("stop" if cc[eidx[0]] <= stop_px else "band")
        return adds, None, "open"


def _episodes(entry, close, dn, stop_pct, add_step, n_tranches):
    """逐标的顺序推演（与手写逐日循环等价）：进场 → 加仓 → 离场，离场后才允许下一次进场。"""
    T, N = close.shape
    out = []
    for j in range(N):
        cand = np.flatnonzero(entry[:, j])
        if cand.size == 0:
            continue
        c = close[:, j]
        b = dn[:, j]
        cur = 0
        for i0 in cand:
            i0 = int(i0)
            if i0 < cur or not np.isfinite(c[i0]):
                continue
            adds, xd, xr = _walk(c, b, i0, stop_pct, add_step, n_tranches)
            out.append({"j": j, "i0": i0, "exit": xd, "reason": xr, "adds": adds,
                        "i1": xd if xd is not None else T})
            cur = (xd + 1) if xd is not None else T
    return out


def _signal_rows(rows):
    df = pd.DataFrame(rows, columns=_SIGNAL_COLS)
    if df.empty:
        return _empty_signals()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str)
    # 无约束的信号（如跌破下沿的卖出）也保持列存在且为字符串，避免下游 concat 出 NaN
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


# ---------------------------------------------------------------- 主入口

# rules.sensitivity_grid 输出的参数名 → 本模块参数名（否则协议④注入的档位会被静默忽略）
_ALIASES = {
    "stop_loss_pct": "stop_pct",   # R-36
    "max_loss_pct": "max_weight",  # R-04
}


def _alias_kw(kw: dict) -> dict:
    out = {}
    for k, v in kw.items():
        if v is None:
            continue
        out[_ALIASES.get(k, k)] = v
    if "first_tranche" in out:   # R-35 首笔比例 → 建仓笔数（1/首笔比例）
        try:
            out["n_tranches"] = max(1, int(round(1.0 / float(out.pop("first_tranche")))))
        except (TypeError, ZeroDivisionError):
            out.pop("first_tranche", None)
    return out


def _eff_params(kw: dict, cfg=None) -> dict:
    over = {}
    if cfg is not None:
        for k in list(PARAMS) + ["start", "end"]:
            if k == "max_weight":
                continue   # 单笔上限只由 kw 指定，避免与 EngineConfig.max_weight 意外耦合
            v = cfg.get(k) if isinstance(cfg, dict) else getattr(cfg, k, None)
            if v is not None:
                over[k] = v
    over.update(_alias_kw(kw))
    p = dict(PARAMS)
    # 未显式指定时以 config.py 常量为准（SPEC §① 的唯一权威来源）
    if "n_tranches" not in over:
        p["n_tranches"] = max(1, int(round(1.0 / float(_const("FIRST_TRANCHE")))))
    for k, cname in (("stop_pct", "STOP_LOSS_PCT"), ("risk_per_trade", "RISK_PER_TRADE"),
                     ("max_positions", "MAX_POSITIONS")):
        if k not in over:
            p[k] = _const(cname)
    p.update(over)
    return p


def run(calendar=None, cfg=None, **kw) -> pd.DataFrame:
    """趋势交易流信号：最小阻力路线区间突破进场 → 首笔 1/5 → 账面利润>0 才加仓 → 止损/跌破下沿离场。

    kw: start/end/codes/n_window/eps_band/top_k/n_tranches/add_step/stop_pct/trend_account/
        risk_per_trade/max_positions/min_list_days/verbose
    """
    p = _eff_params(kw, cfg)
    start = pd.Timestamp(p.get("start") or _const("START_DATE"))
    end = pd.Timestamp(p.get("end") or _const("END_DATE"))
    n_win = int(p["n_window"])
    eps = float(p["eps_band"])
    stop_pct = float(p["stop_pct"])
    add_step = float(p["add_step"])
    n_tr = max(1, int(p["n_tranches"]))
    top_k = max(1, int(p["top_k"]))
    max_pos = max(1, int(p["max_positions"]))
    min_list_days = int(p["min_list_days"])

    if calendar is not None:
        cal = pd.DatetimeIndex(pd.to_datetime(list(calendar)))
        cal = cal[(cal >= start) & (cal <= end)]
    else:
        cal = _calendar(start, end)
    if len(cal) == 0:
        return _empty_signals()

    codes = [str(c) for c in (p.get("codes") or _universe(min_list_days))]
    # 前推窗口：让首个信号日的 N 日区间已可由已实现数据算出
    pre = cal[0] - pd.Timedelta(days=int(n_win * 3) + 10)
    panel = _close_panel(codes, _calendar(pre, end))
    dates = panel.index
    first_row = int(dates.get_indexer([cal[0]])[0])
    T = panel.shape[0]

    close = panel.to_numpy(dtype=np.float32)
    up = panel.rolling(n_win, min_periods=n_win).max().shift(1) * (1.0 - eps)
    dn = panel.rolling(n_win, min_periods=n_win).min().shift(1)
    breakout = (panel > up).to_numpy()
    entry = _entry_flags(breakout)                        # 突破首日才动手（R-34）
    mom = (panel / panel.shift(n_win) - 1.0).to_numpy()   # 领头羊横截面打分（spec/05 行 1479）
    listed = panel.notna().to_numpy()
    listed_ok = pd.DataFrame(listed.astype(np.float32)).rolling(
        min_list_days, min_periods=min_list_days).sum().to_numpy()

    eps_list = _episodes(entry, close, dn.to_numpy(), stop_pct, add_step, n_tr)
    eps_list = [e for e in eps_list if e["i0"] >= first_row]

    # 横截面 top-k（领头羊）：同一天进场候选按 N 日动量排序
    by_date = {}
    for e in eps_list:
        sc = mom[e["i0"], e["j"]]
        if not np.isfinite(sc):
            sc = -np.inf
        e["score"] = float(sc)
        by_date.setdefault(e["i0"], []).append(e)
    kept = []
    for i0, group in by_date.items():
        group.sort(key=lambda e: -e["score"])
        kept.extend(group[:top_k])

    # 持仓数上限（R-16）+ 趋势账户总暴露（R-04：每只 ≤ 账户/top_k，合计 ≤ 账户）
    # 单笔计划额度：R-04 归零口径上限（5%）与 R-38 风险预算反推上限（1.5%÷1%＝150%）取更严 → 5%
    risk_cap = float(p["risk_per_trade"]) / stop_pct
    cap = float(p["max_weight"]) if p.get("max_weight") else _risk_cap_weight()
    room = (float(p["trend_account"]) / float(top_k)) if p.get("trend_account") else float("inf")
    planned = max(0.0, min(cap, risk_cap, room))
    slots = np.zeros(T + 1, dtype=np.int16)
    kept.sort(key=lambda e: (e["i0"], -e["score"]))
    take = []
    for e in kept:
        if slots[e["i0"]] >= max_pos:
            continue
        take.append(e)
        slots[e["i0"]:e["i1"] + 1] += 1

    tranche_w = planned / n_tr
    risk_pct = tranche_w * stop_pct
    rows = []
    for e in take:
        code = codes[e["j"]]
        i0, i1 = e["i0"], e["i1"]
        n_bars = listed_ok[i0, e["j"]]
        if not np.isfinite(n_bars) or n_bars < min_list_days:
            continue
        rows.append({
            "date": dates[i0], "code": code, "side": "buy",
            "reason": ("R-34 突破区间上沿(过去%d日最高收盘×%.3f，上沿%.2f→收%.2f)；依据 R-34=最小阻力路线"
                       "区间突破(C 级·Ⅱ 区，未取得可执行资格，见文件头第 8 条，仅供研究)；约束 首笔=计划"
                       "额度1/%d(R-35)、计划单笔额度%.2f%%=min(R-04 %.1f%%, R-38 反推 %.1f%%)、持仓≤%d 只"
                       "(max_positions，R-16 适用价值仓故不作趋势仓主张)、最后买入价×(1−%.1f%%)止损(R-36)；"
                       "单笔风险=仓位%.2f%%×止损%.1f%%=%.4f%%≤%.1f%%(R-38)；横截面动量%.1f%%"
                       % (n_win, 1 - eps, float(up.iloc[i0, e["j"]]), float(close[i0, e["j"]]), n_tr,
                          planned * 100, cap * 100, risk_cap * 100, top_k, stop_pct * 100,
                          tranche_w * 100, stop_pct * 100, risk_pct * 100,
                          float(p["risk_per_trade"]) * 100, e["score"] * 100)),
            "rule_id": BUY_BASIS,
            "constraint_ids": CONSTRAINTS_ENTRY,
            "weight": tranche_w, "stop_pct": stop_pct,
        })
        for (day, k) in e["adds"]:
            rows.append({
                "date": dates[day], "code": code, "side": "buy",
                "reason": ("加仓第%d/%d笔：较最后买入价+%.1f%%(逐笔触发来自 spec/05 行 1485「步步上涨」方向纪律，"
                           "手册未编号、不进 rule_id)；依据仍记 R-34(最小阻力路线趋势未终结，C 级·Ⅱ 区，"
                           "未取得可执行资格，仅供研究)；约束 账面利润>0 才加(R-37)、累计笔数≤1/%d(R-35)、"
                           "单笔额度受 R-04/R-38 上限；目标权重累计%.2f%%"
                           % (k, n_tr, add_step * 100, n_tr, k * tranche_w * 100)),
                "rule_id": BUY_BASIS,
                "constraint_ids": CONSTRAINTS_ADD,
                "weight": k * tranche_w, "stop_pct": stop_pct,
            })
        if e["exit"] is not None:
            if e["reason"] == "stop" and not p["emit_stop_sell"]:
                continue   # 交给 engine 原生 stop_pct 执行 R-36
            if e["reason"] == "stop":
                rows.append({"date": dates[e["exit"]], "code": code, "side": "sell",
                             "reason": ("R-36 止损：收盘跌破最后买入价×(1−%.1f%%)，触发即执行不商量——"
                                        "R-36 是本卖出的依据本身（C 级·Ⅱ 区，未取得可执行资格，仅供研究）；"
                                        "约束 R-37 禁止以摊平/补仓替代止损" % (stop_pct * 100)),
                             "rule_id": SELL_BASIS_STOP, "constraint_ids": C_NO_AVG_DOWN,
                             "weight": 0.0, "stop_pct": np.nan})
            else:
                rows.append({"date": dates[e["exit"]], "code": code, "side": "sell",
                             "reason": ("R-34 趋势终结：收盘跌破区间下沿(过去%d日最低收盘)，了结交易"
                                        "(spec/05 行 1477)；下沿方向判据出自同一条最小阻力路线规则"
                                        "(行 1483)，spec/08 未单列离场编号故取 R-34（未取得可执行资格，"
                                        "仅供研究）；无约束类规则参与" % n_win),
                             "rule_id": SELL_BASIS_BAND, "constraint_ids": "",
                             "weight": 0.0, "stop_pct": np.nan})
    if p.get("verbose"):
        print("[trend_card] N=%d ε=%.3f 候选进场 %d → 受 top_k/持仓上限约束 %d；信号 %d"
              % (n_win, eps, len(eps_list), len(take), len(rows)))
    return _signal_rows(rows)


def params(**kw) -> dict:
    p = dict(PARAMS)
    p.update({k: v for k, v in kw.items() if v is not None})
    return p


if __name__ == "__main__":
    # 冒烟测试：小样本自证可跑（不依赖 config.py / data.loader.py / core.risk_budget.py）
    print("!! " + QUALIFICATION_NOTE)
    print("data_dir =", _data_dir())
    t0 = pd.Timestamp("2022-01-01")
    cal = _calendar(t0, pd.Timestamp("2024-12-31"))
    print("calendar:", len(cal), cal[0].date(), cal[-1].date())
    sig = run(start="2022-01-01", end="2024-12-31",
              codes=_universe(250)[:300], verbose=True)
    print("signals:", len(sig), "buy=%d sell=%d" % ((sig.side == "buy").sum(), (sig.side == "sell").sum()))
    if len(sig):
        print(sig.head(3).to_string(max_colwidth=70))
        print("weight max = %.4f (R-04 归零口径上限 5%% 之内)" % sig.weight.max())
        print("stop_pct 非空 =", int(sig.stop_pct.notna().sum()))

        # 自检（附录五元数据 2066 + 文件头第 9 条）：rule_id 只装依据类，约束类只进 constraint_ids
        BASIS = {BUY_BASIS, SELL_BASIS_STOP, SELL_BASIS_BAND}
        CONS = {C_POSITION, C_FIRST, C_STOP, C_NO_AVG_DOWN, C_RISK}
        cons = set()
        for cell in sig["constraint_ids"].astype(str):
            cons |= {x.strip() for x in cell.split("+") if x.strip()}
        bad_basis = sorted(set(sig["rule_id"]) - BASIS)
        bad_side = sorted((set(sig.loc[sig.side == "buy", "rule_id"]) - {BUY_BASIS})
                          | (set(sig.loc[sig.side == "sell", "rule_id"]) - {SELL_BASIS_STOP, SELL_BASIS_BAND}))
        overlap = sorted(cons & BASIS)
        print("rule_id 取值:", sorted(sig["rule_id"].unique()),
              "｜constraint_ids 取值:", sorted(cons) or ["(空)"])
        print("自检 rule_id 仅依据类: %s（越界=%s）｜买入=R-34/卖出∈{R-36,R-34}: %s"
              % (not bad_basis, bad_basis or "无", not bad_side))
        print("自检 constraint_ids ⊆ 约束类: %s（越界=%s）｜与 rule_id 无重叠: %s"
              % (cons <= CONS, sorted(cons - CONS) or "无", not overlap))
        print("自检 constraint_ids 列存在且为字符串:",
              "constraint_ids" in sig.columns and sig["constraint_ids"].map(type).eq(str).all())
        try:
            import rules as _rules
            ok_buy, why_buy = _rules.gate([BUY_BASIS])
            print("rules.gate(['R-34']) =", (ok_buy, why_buy[:60] + "…"))
            print("自检 买入依据被闸门拦截（=R-34 未取得可执行资格，符合文件头第 8 条）:", ok_buy is False)
            ok_cons, _ = _rules.gate([C_POSITION], only_trigger=True)
            print("对照 rules.gate(['R-04'])（约束类，不参与否决）:", ok_cons)
        except Exception as exc:
            print("rules 闸门自检跳过:", exc)
