"""⑥ core/allocation.py — 长期配置与定投（R-09 / R-11 / R-12 / R-13 / R-18 / R-19 / R-24）

对 SPEC 歧义处的取舍（每行一条，均为最贴近原文的固定口径）：
- R-11 原文（04-06／02-卖出与仓位①-3）只给「5 年 25%、30 年 68%」两端点，SPEC 另给
  「10 年 1/3 强、20 年 >50%」：取下表 10 年 0.35、20 年 0.55，其余线性插值，端点外截断。
- R-12 按字面返回 (100−年龄)%，不做截断；边界约束交给 R-09 的 clamp_equity，避免两层各自夹一次。
- R-24 放大器 n 在「E/P > 10%」档默认取 1（资金充裕取 2，签名无该入参，故固定 n=1 并在 reason 注明）。
- R-13「优先用新增资金」在无新增资金入参下的编码：买入腿标记 funded_by="new_cash" 且 priority=1，
  卖出腿标记 funded_by="liquidation" 且 priority=2，由调用方先执行买入腿；偏离判定含端点（0.50→0.55 即触发）。
- 债券腿无数据（SPEC 第 2 节缺失数据）时，run_allocation_backtest 用现金腿 0% 近似（04-01 验证注）。
"""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

# R-11 持有期→权益权重映射表（原文两端点 + SPEC 补充两点，中间线性插值）
HOLDING_YEARS_GRID: list = [
    (1.0, 0.00),
    (2.0, 0.00),
    (5.0, 0.25),
    (10.0, 0.35),
    (20.0, 0.55),
    (30.0, 0.68),
]

# 04-B1 / R-18 / R-19 盈利收益率阈值：10% 源出格雷厄姆「>10% 且大幅高于无风险利率」，6.4% 源出国内债券基金长期平均收益
EY_BUY = 0.10
EY_SELL = 0.064
EY_SELL_TRANCHES = 10.0  # R-19 高估时分 10 份卖出，每次 1 份
N_AMPLIFY_LOW = 1.0      # R-24 低估时 n=1
N_AMPLIFY_RICH = 2.0     # R-24 资金充裕取 2

# 04-01 标准各半（R-09 的标准配置，25%–75% 区间由 clamp_equity 承担）
STANDARD_EQUITY = 0.50

# config.py（① data-loader 所有）可能尚未落地：延迟加载 + SPEC 口径兜底，避免 import 期副作用
_FALLBACK_CFG = SimpleNamespace(
    EQUITY_MIN=0.25,
    EQUITY_MAX=0.75,
    REBAL_BAND=0.05,
    OUT_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"),
)


def _cfg():
    try:
        import config as _c  # noqa: F401
    except ImportError:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            import config as _c  # noqa: F401
        except ImportError:
            return _FALLBACK_CFG
    return _c


def _cfg_value(name: str, default):
    return getattr(_cfg(), name, default)


def equity_ratio_by_holding_years(years: float) -> float:
    """R-11（04-06）：按持有期定权益比例，线性插值并在 1–30 年端点外截断。"""
    y = float(years)
    if math.isnan(y):
        raise ValueError("years 不能为 NaN")
    grid = np.asarray(HOLDING_YEARS_GRID, dtype=float)
    # np.interp 在 xp 范围外取端点值，即「端点外截断」
    return float(np.interp(y, grid[:, 0], grid[:, 1]))


def equity_ratio_by_age(age: float) -> float:
    """R-12（04-09）：(100−年龄)% 仅作起点，不设下限，故此处不做截断。"""
    return (100.0 - float(age)) / 100.0


def clamp_equity(x: float) -> float:
    """R-09（04-01）：权益 25%–75% 硬边界。"""
    lo = float(_cfg_value("EQUITY_MIN", 0.25))
    hi = float(_cfg_value("EQUITY_MAX", 0.75))
    return float(min(max(float(x), lo), hi))


def rebalance_orders(cur: dict, target: dict, band: float = 0.05,
                     prefer_new_cash: bool = True) -> list:
    """R-13（04-01/04-02）：偏离目标 ±5pp 即机械再平衡，优先用新增资金而非卖出。

    返回 list[dict]，每条含：
    `asset, side(buy/sell), delta(目标−当前, 带符号), weight(|delta|), priority(1 买入/2 卖出),`
    `funded_by(new_cash/liquidation), rule_id, reason`。
    优先级：prefer_new_cash 时买入腿 priority=1 先执行（用新增资金补齐低配），卖出腿 priority=2；
    同优先级按偏离幅度从大到小（先撤涨幅最大、仓位最重的部分，02-卖出与仓位①-1）。
    """
    b = float(_cfg_value("REBAL_BAND", 0.05) if band is None else band)
    assets = list(dict.fromkeys(list(target.keys()) + list(cur.keys())))
    orders = []
    for a in assets:
        c = float(cur.get(a, 0.0))
        t = float(target.get(a, 0.0))
        d = t - c
        # 04-01：股票升至 55%（偏离正好 5pp）即调整 → 判定含端点
        if abs(d) < b - 1e-9:
            continue
        is_buy = d > 0
        orders.append({
            "asset": a,
            "side": "buy" if is_buy else "sell",
            "delta": d,
            "weight": abs(d),
            "priority": 1 if (is_buy and prefer_new_cash) else 2,
            "funded_by": "new_cash" if (is_buy and prefer_new_cash) else "liquidation",
            "rule_id": "R-13",
            "reason": "R-13 偏离目标 {:.4f}（阈值 ±{:.4f}）".format(abs(d), b),
        })
    orders.sort(key=lambda o: (o["priority"], -o["weight"]))
    return orders


def dca_plan(monthly_income: float, monthly_expense: float) -> float:
    """04-13（R-23 同段）：每月定投金额 =（月收入 − 月开支）× 50%。"""
    surplus = float(monthly_income) - float(monthly_expense)
    return max(0.0, surplus * 0.5)


def n_amplifier(earnings_yield, base_amount: float) -> tuple:
    """R-24 / R-18 / R-19（04-17、04-B1）：按盈利收益率 E/P 放大或收缩定投金额。

    返回值 (金额, 原因)；正为买入、负为卖出、0 为不动（数据不足）。
    """
    base = float(base_amount)
    if earnings_yield is None or (isinstance(earnings_yield, float) and math.isnan(earnings_yield)):
        return 0.0, "数据不足：无指数盈利收益率 E/P（SPEC 第 2 节列为缺失数据，需指数 PE 序列）"
    ey = float(earnings_yield)
    if ey > EY_BUY:
        n = N_AMPLIFY_LOW
        return base * n, "R-24/R-18 盈利收益率 {:.2%} > 10%：低估定投放大 n={:.0f}（资金充裕可取 {:.0f}）".format(ey, n, N_AMPLIFY_RICH)
    if ey >= EY_SELL:
        return base, "R-18 盈利收益率 {:.2%} 落在 [6.4%, 10%]：定额买入（坚定持有）".format(ey)
    return -base / EY_SELL_TRANCHES, "R-19 盈利收益率 {:.2%} < 6.4%：高估分批卖出，本次卖 1/10 份（{:.2f}）".format(
        ey, base / EY_SELL_TRANCHES)


def _close_series(df) -> pd.Series:
    """df（DataFrame[close] 或 Series）→ 归一化日期索引的收盘价序列；缺列时退化取最后一列。"""
    if df is None:
        return None
    if isinstance(df, pd.Series):
        s = pd.to_numeric(df, errors="coerce").astype(float)
    else:
        col = "close" if "close" in df.columns else df.columns[-1]
        s = pd.to_numeric(df[col], errors="coerce").astype(float)
        if not isinstance(df.index, pd.DatetimeIndex) and "date" in getattr(df, "columns", []):
            s.index = pd.to_datetime(df["date"])
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.dropna()


def _max_drawdown(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1.0)) if len(eq) else 0.0


def _stale_days(eq: np.ndarray) -> int:
    """策略失效期：最长未创新高交易日数（SPEC 第 3-④ stats 口径）。"""
    peak = -np.inf
    stale = worst = 0
    for v in eq:
        if v >= peak:
            peak = v
            stale = 0
        else:
            stale += 1
            worst = max(worst, stale)
    return int(worst)


def _monthly_irr(flows: list) -> float:
    """二分法求月 IRR；flows[0] 为第 0 期（首月投入取负），最后一期含期末市值（正）。"""
    flows = [float(c) for c in flows]  # 统一为 Python float：numpy 标量的布尔量不支持 `-`

    def npv(r):
        # 下界取 −0.9：−0.9999 会让 (1e−4)**95 下溢为 0.0，触发除零
        tot = 0.0
        try:
            for i, cf in enumerate(flows):
                tot += cf / (1.0 + r) ** i
                if not math.isfinite(tot):
                    break
        except (ZeroDivisionError, OverflowError):
            return float("inf")
        return tot

    def sgn(x):
        x = float(x)
        return (x > 0.0) - (x < 0.0)

    lo, hi = -0.9, 2.0
    f_lo, f_hi = npv(lo), npv(hi)
    s_lo, s_hi = sgn(f_lo), sgn(f_hi)
    if s_lo == 0:
        return lo
    if s_hi == 0:
        return hi
    # 用符号比较取代 f_lo*f_mid：24 年期现金流的折现因子可达 1e±300，乘积会溢出
    if s_lo == s_hi:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2.0
        s_mid = sgn(npv(mid))
        if s_mid == 0:
            return mid
        if s_mid == s_lo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def run_allocation_backtest(monthly_amount: float, calendar, index,
                            bond: pd.DataFrame = None) -> dict:
    """④-06 定投＋50/50 再平衡回测（04-12 发薪后第 1 个交易日投、04-01 偏离 ±5pp 再平衡）。

    债券腿（bond）为 None 时按 04-01 验证注用现金腿 0% 近似并写入 notes。
    返回 {"equity", "orders", "stats", "obs_order", "notes"}。
    """
    notes = []
    px = _close_series(index)
    if px is None or px.empty:
        raise ValueError("index 无可用收盘价序列")
    cal = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in calendar]) if calendar is not None else px.index
    cal = pd.DatetimeIndex(sorted(set(cal)))
    px = px.reindex(cal).ffill()
    if px.isna().all():
        raise ValueError("index 与 calendar 无交集")

    bond_px = _close_series(bond) if bond is not None else None
    if bond_px is None:
        notes.append("债券腿无数据（SPEC 第 2 节缺失）：按 04-01 验证注用现金腿 0% 近似，R-09 的债券端收益不可验证")
    else:
        bond_px = bond_px.reindex(cal).ffill()

    target_eq = clamp_equity(STANDARD_EQUITY)  # R-09 标准各半
    band = float(_cfg_value("REBAL_BAND", 0.05))

    # 每月第 1 个交易日为定投日（04-12）
    month_key = pd.Series(cal.year * 100 + cal.month, index=cal)
    dca_days = set(month_key.groupby(month_key).apply(lambda s: s.index[0]))

    cash = 0.0
    eq_units = 0.0
    bond_units = 0.0
    invested = 0.0
    traded = 0.0
    n_orders = 0
    rows = []
    order_rows = []
    monthly_flow = {}

    def _leg(day, asset, side, amount, why):
        """登记一条成交腿：traded／笔数／账本三处同步，避免口径分叉。"""
        nonlocal traded, n_orders
        if amount <= 0:
            return
        traded += amount
        n_orders += 1
        order_rows.append({"date": day, "asset": asset, "side": side,
                           "amount": amount, "rule_id": "R-13", "reason": why})

    for day in cal:
        contrib = 0.0
        p = float(px.loc[day])
        if not np.isfinite(p):
            p = None
        bp = float(bond_px.loc[day]) if bond_px is not None else None
        if day in dca_days:
            # 04-12：定投日固定在发薪后第 1 个交易日；R-13「优先用新增资金」→ 当期入金直接按目标权重配置
            contrib = float(monthly_amount)
            cash += contrib
            invested += contrib
            key = (day.year, day.month)
            monthly_flow[key] = monthly_flow.get(key, 0.0) + contrib
            why = "04-12 定投日按目标权重 {:.2f}/{:.2f} 配置当期入金".format(target_eq, 1.0 - target_eq)
            if p is not None and p > 0:
                buy_eq = contrib * target_eq
                eq_units += buy_eq / p
                cash -= buy_eq
                _leg(day, "equity", "buy", buy_eq, why)
            if bond_px is not None and bp is not None and bp > 0:
                buy_bond = contrib * (1.0 - target_eq)
                bond_units += buy_bond / bp
                cash -= buy_bond
                _leg(day, "bond", "buy", buy_bond, why)

        eq_val = eq_units * p if p is not None else 0.0
        bond_val = bond_units * bp if (bond_px is not None and bp is not None) else 0.0
        total = eq_val + bond_val + cash
        if total > 0:
            w_eq = eq_val / total
            # 04-01：偏离 ±5 个百分点即再平衡（含端点）
            if abs(w_eq - target_eq) >= band - 1e-9:
                why = "R-13 权益占比 {:.4f} 偏离目标 {:.4f}（阈值 ±{:.4f}）".format(w_eq, target_eq, band)
                need = target_eq * total - eq_val
                if need > 0:
                    if bond_units > 0 and bp is not None and bp > 0 and cash < need:
                        # 现金（新增资金）不够时才动债券腿，符合 R-13 的优先序
                        bu = min((need - cash) / bp, bond_units)
                        bond_units -= bu
                        cash += bu * bp
                        _leg(day, "bond", "sell", bu * bp, why)
                    if p is not None and p > 0:
                        spend = min(need, cash)
                        if spend > 0:
                            eq_units += spend / p
                            cash -= spend
                            _leg(day, "equity", "buy", spend, why)
                elif need < 0:
                    if p is not None and p > 0:
                        su = min(-need / p, eq_units)
                        cash += su * p
                        eq_units -= su
                        _leg(day, "equity", "sell", su * p, why)
                    if bond_px is None:
                        notes.append("R-09 债券端缺数据：权益减仓所得滞留现金（0% 收益）")
                # 安全腿归位：有债券数据时把安全腿的现金换成债券（R-09 的债券端），否则由现金腿 0% 近似
                if bond_px is not None and bp is not None and bp > 0:
                    d_bond = (1.0 - target_eq) * total - bond_units * bp
                    if d_bond > 0:
                        buy_bond = min(d_bond, cash)
                        if buy_bond > 0:
                            bond_units += buy_bond / bp
                            cash -= buy_bond
                            _leg(day, "bond", "buy", buy_bond, why)
                    elif d_bond < 0:
                        su = min(-d_bond / bp, bond_units)
                        if su > 0:
                            bond_units -= su
                            cash += su * bp
                            _leg(day, "bond", "sell", su * bp, why)

        eq_val = eq_units * p if p is not None else 0.0
        bond_val = bond_units * bp if (bond_px is not None and bp is not None) else 0.0
        equity = eq_val + bond_val + cash
        rows.append({
            "date": day, "cash": cash, "bond_value": bond_val, "stock_value": eq_val,
            "equity": equity, "exposure": (eq_val / equity if equity > 0 else 0.0),
            "contribution": contrib,
        })

    eqdf = pd.DataFrame(rows).set_index("date")

    eq = eqdf["equity"].to_numpy(dtype=float)
    contrib = eqdf["contribution"].to_numpy(dtype=float)
    stat_ret = np.zeros(len(eq))
    stat_ret[1:] = (np.diff(eq) - contrib[1:]) / np.where(eq[:-1] > 0, eq[:-1], np.nan)
    stat_ret = stat_ret[~np.isnan(stat_ret)]

    months = sorted(monthly_flow.keys())
    flows = [-monthly_flow[k] for k in months]
    if flows:
        flows[-1] += eq[-1]
    r_m = _monthly_irr(flows) if flows else None
    annual = ((1.0 + r_m) ** 12 - 1.0) if r_m is not None and r_m > -0.9999 else None
    mean_eq = float(np.mean(eq)) if len(eq) else 0.0

    stats = {
        # 观测口径优先级（附录六）：最大回撤＞波动＞换手率＞资金占用＞策略失效期＞收益率
        "最大回撤": _max_drawdown(eq),
        "波动(年化)": float(np.std(stat_ret, ddof=1) * math.sqrt(252)) if len(stat_ret) > 1 else 0.0,
        "换手率": (traded / mean_eq) if mean_eq > 0 else 0.0,
        "资金占用": float(np.mean(eqdf["exposure"])) if len(eqdf) else 0.0,
        "策略失效期": _stale_days(eq),
        "总收益": (eq[-1] / invested - 1.0) if invested > 0 else None,
        "年化": annual,
        "总投入": invested,
        "期末市值": float(eq[-1]) if len(eq) else 0.0,
        "交易笔数": int(n_orders),
        "胜率": None,
        "盈亏比": None,
        "口径说明": "换手率=区间总成交额/平均权益；资金占用=平均权益市值占比；"
                    "收益率=期末市值/累计投入−1（简单口径），年化=月现金流二分法 IRR；"
                    "最大回撤与策略失效期基于含定投现金流的组合市值曲线（新增投入会摊薄回撤，未做剔除）；"
                    "配置类无平仓交易，胜率/盈亏比不适用",
    }

    return {
        "equity": eqdf,
        "orders": pd.DataFrame(order_rows, columns=["date", "asset", "side", "amount", "rule_id", "reason"]),
        "stats": stats,
        "obs_order": ["最大回撤", "波动(年化)", "换手率", "资金占用", "策略失效期", "总收益"],
        "notes": sorted(set(notes)),
        "target_equity": target_eq,
        "rebalance_band": band,
    }


if __name__ == "__main__":
    print("== R-11 持有期→权益 ==")
    for y in [0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40]:
        print("  {:>5} 年 → {:.4f}".format(y, equity_ratio_by_holding_years(y)))
    print("== R-12/R-09 ==")
    print("  age 35 →", equity_ratio_by_age(35), "| clamp(0.90) →", clamp_equity(0.90),
          "| clamp(0.10) →", clamp_equity(0.10))
    print("== R-13 偏离 5pp 触发 ==")
    for cur in [{"stock": 0.55, "bond": 0.45}, {"stock": 0.549, "bond": 0.451},
                {"stock": 0.60, "bond": 0.40}, {"stock": 0.50, "bond": 0.50}]:
        print("  cur", cur, "→", rebalance_orders(cur, {"stock": 0.50, "bond": 0.50}))
    print("== 04-13 / R-24 / R-18 / R-19 ==")
    print("  dca_plan(20000,12000) →", dca_plan(20000, 12000))
    for ey in [0.11, 0.10, 0.08, 0.064, 0.05, None]:
        print("  E/P", ey, "→", n_amplifier(ey, 10000.0))

    print("== 合成指数走一遍配置回测 ==")
    cal = pd.bdate_range("2016-01-04", "2023-12-29")
    rng = np.random.default_rng(7)
    synth = pd.DataFrame({"close": 3000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, len(cal))))}, index=cal)
    out = run_allocation_backtest(3000.0, list(cal), synth, bond=None)
    print("  stats:", {k: out["stats"][k] for k in out["obs_order"]})
    print("  总投入 {:.0f} 期末市值 {:.0f} 交易笔数 {}".format(
        out["stats"]["总投入"], out["stats"]["期末市值"], out["stats"]["交易笔数"]))
    print("  曲线行数", len(out["equity"]), "| notes:", out["notes"])

    try:
        from data.loader import load_index, trading_calendar
    except Exception as exc:  # 依赖模块（①/④）并行实现中，缺失时跳过跨模块调用
        print("  跳过真实数据冒烟（%s: %s）" % (type(exc).__name__, exc))
    else:
        try:
            idx = load_index()
            o = run_allocation_backtest(3000.0, trading_calendar(), idx, bond=None)
            print("  真实沪深300 定投：总投入 {:.0f} 期末市值 {:.0f} 最大回撤 {:.2%} IRR {:.4f}".format(
                o["stats"]["总投入"], o["stats"]["期末市值"], o["stats"]["最大回撤"],
                o["stats"]["年化"] or float("nan")))
        except Exception as exc:
            print("  真实数据回测失败（%s: %s）" % (type(exc).__name__, exc))
