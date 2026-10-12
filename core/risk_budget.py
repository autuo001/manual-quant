"""core/risk_budget.py — ⑤ 仓位上限与风险预算（R-04 / R-14 / R-16 / R-35 / R-36 / R-37 / R-38 + 组合压力测试）

对应手册：spec/02-卖出与仓位.md ①-3 仓位上限与风险预算（行 537/539/540/541/552-556）、
①-4 建仓节奏与持股数量（行 473/474-477/487）；spec/03-执行与复盘.md 价值/趋势操作卡（L728/L735-737/L745）。

取舍（SPEC 字面优先，以下几处歧义按此实现）：
1) `max_weight_zero_loss` 返回 R-04 的**金额口径**权重上限（默认 5%），`price` 只用于在 reason 里
   给出整手可行股数与真实归零损失，不改变上限数值；整手后权重由 `shares_for_weight` 另行给出。
2) `fisher_class_cap` 依 SPEC 注释「与 R-04 取更严」返回 min(费雪类上限, 5%)——故 A/B/C 三档均落到 5%；
   费雪原值由 `fisher_class_cap_raw` 提供，调用方需要区分档次时取原值自行取更严。
3) `size_by_risk` 的 `risk_pct` 超出 R-38 的 1%—2% 时按「同层冲突取更保守」(L573) 钳到区间内。
4) `stress_test` 的 `shocks` 按小数口径（-0.40 = -40%）；传入绝对值 >1 的数（如 -40）按百分数处理。
5) 数值口径：R-04 归零损失分母 = 可投资金融资产（行 514-518，禁止混用分母）。
"""

from __future__ import annotations

import os
import sys
from numbers import Real

import pandas as pd

try:
    import config as _cfg
except ImportError:  # 直接 `python3 core/risk_budget.py` 运行时 __file__ 目录不在 sys.path
    _cfg = None
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(_root) and _root not in sys.path:
        sys.path.insert(0, _root)
    try:
        import config as _cfg
    except ImportError:
        _cfg = None


def _cfg_val(name: str, default):
    return getattr(_cfg, name, default) if _cfg is not None else default


LOT_SIZE = 100                     # A 股一手 100 股（交易所制度，非手册阈值）
RISK_ZERO_LOSS_MIN = 0.03          # R-04 归零损失参数区间下限（3%—10% 可自定）
RISK_ZERO_LOSS_MAX = 0.10          # R-04 归零损失参数区间上限
RISK_PER_TRADE_MIN = 0.01          # R-38 单笔风险下限（1%—2%）
RISK_PER_TRADE_MAX = 0.02          # R-38 单笔风险上限
STRESS_TOLERANCE = 0.30            # 行 541/556：只愿承受 30% 组合损失 → 权益上限降到 60%
ZERO_LOSS_TOTAL_CAP = 0.15         # 行 555：个股+窄基+趋势仓全部归零合计 ≤ 15%（10%—20% 可自定）
FISHER_CAP_RAW = {"A": 0.20, "B": 0.10, "C": 0.05}   # R-14 费雪 A≤20% / B 8—10% / C≤5%（B 取区间上限）

BINDING_RULES: dict[str, tuple[str, ...]] = {
    "max_weight_zero_loss": ("R-04",),
    "size_by_risk": ("R-38", "R-36"),
    "plan_tranches": ("R-35",),
    "can_add_position": ("R-16",),
    "can_average_down": ("R-37", "R-35"),
    "stress_test": ("R-04", "R-09"),
    "fisher_class_cap": ("R-14", "R-04"),
    "fisher_class_cap_raw": ("R-14",),
    "shares_for_weight": ("R-04",),
    "can_add_tranche": ("R-35",),
    "check_zero_loss_total": ("R-04",),
    "trend_position_size": ("R-38", "R-04", "R-36"),
}


def rule_ids(func_name: str) -> tuple[str, ...]:
    """SPEC 「所有函数返回中必须能指出绑定的规则编号」：元组型返回值无法携带字符串，故在此登记。"""
    return BINDING_RULES.get(func_name, ())


# --------------------------------------------------------------------------- R-04 归零口径
def _zero_loss_cap() -> float:
    cap = float(_cfg_val("RISK_ZERO_LOSS", 0.05))
    return min(max(cap, RISK_ZERO_LOSS_MIN), RISK_ZERO_LOSS_MAX)


def max_weight_zero_loss(equity: float, price: float) -> tuple[float, str]:
    """R-04：按归零假设，单一标的最大潜在损失 ≤ 可投资金融资产的 5%（区间 3%—10%）。"""
    cap = _zero_loss_cap()
    equity = float(equity)
    price = float(price)
    if equity <= 0 or price <= 0:
        return 0.0, f"R-04 归零口径：权益/价格非正，权重上限 0（参数区间 {RISK_ZERO_LOSS_MIN:.0%}—{RISK_ZERO_LOSS_MAX:.0%}）"
    amount = equity * cap
    shares = shares_for_weight(equity, price, cap)
    reason = (
        f"R-04 归零口径：单一标的潜在损失 ≤ 可投资金融资产 {cap:.1%}"
        f"（参数区间 {RISK_ZERO_LOSS_MIN:.0%}—{RISK_ZERO_LOSS_MAX:.0%}，行 540/462）；"
        f"{equity:,.0f} × {cap:.1%} = {amount:,.2f} 元；价 {price:.2f} 时整手 {shares} 股，"
        f"归零损失 {shares * price:,.2f} 元（{shares * price / equity:.2%}）"
    )
    return cap, reason


def shares_for_weight(equity: float, price: float, weight: float) -> int:
    """把目标权重折算为可成交股数（整手）。R-04/R-14 的上限要落到股数才能与 R-38 比大小。"""
    equity, price, weight = float(equity), float(price), float(weight)
    if equity <= 0 or price <= 0 or weight <= 0:
        return 0
    return int(equity * weight // (price * LOT_SIZE)) * LOT_SIZE


def check_zero_loss_total(weights: dict[str, float]) -> tuple[bool, str]:
    """行 555：个股 + 窄基 + 趋势仓「全部同时归零」的合计损失 ≤ 可投资金融资产的 15%。"""
    total = float(sum((v for v in (weights or {}).values() if isinstance(v, Real))))
    cap = ZERO_LOSS_TOTAL_CAP
    ok = total <= cap + 1e-12
    tail = "通过" if ok else "超限，须剔除归零类标的"
    return ok, (
        f"R-04 归零合计口径（行 555）：当前归零类合计 {total:.2%}，上限 {cap:.0%}"
        f"（可在 10%—20% 间自定）→ {tail}"
    )


# --------------------------------------------------------------------------- R-38 单笔风险
def size_by_risk(equity: float, price: float, stop_pct: float = None,
                 risk_pct: float = None) -> tuple[int, float]:
    """R-38 单笔风险 ≤ 总资金 1%—2%；止损距 R-36 = 最后买入价下方 1%。

    返回 (股数, 该笔实际风险金额)；股数按风险金额/每股止损额向下取整。
    """
    if stop_pct is None:
        stop_pct = _cfg_val("STOP_LOSS_PCT", 0.01)
    if risk_pct is None:
        risk_pct = _cfg_val("RISK_PER_TRADE", 0.015)
    equity, price, stop_pct, risk_pct = (float(equity), float(price),
                                         float(stop_pct), float(risk_pct))
    if equity <= 0 or price <= 0 or stop_pct <= 0 or risk_pct <= 0:
        return 0, 0.0
    # 同层冲突取更保守（L573）：越界参数钳回 R-38 区间，不放大风险
    risk_pct = min(max(risk_pct, RISK_PER_TRADE_MIN), RISK_PER_TRADE_MAX)
    per_share_risk = price * stop_pct
    # 1e-9 容差抵掉 0.015×1e5 这类二进制浮点毛刺，否则 7500 股会被算成 7499
    shares = int(equity * risk_pct / per_share_risk + 1e-9)
    return shares, round(shares * per_share_risk, 6)


def trend_position_size(equity: float, price: float, stop_pct: float = None,
                        risk_pct: float = None) -> tuple[int, float, str]:
    """行 552「两条上限同时算、取更严」：min(R-38 单笔风险, R-04 归零上限)，止损按 R-36。"""
    risk_shares, risk_amount = size_by_risk(equity, price, stop_pct, risk_pct)
    cap_w, _ = max_weight_zero_loss(equity, price)
    cap_shares = shares_for_weight(equity, price, cap_w)
    risk_lots = risk_shares // LOT_SIZE * LOT_SIZE
    shares = min(risk_lots, cap_shares)
    bound = "R-38 单笔风险" if risk_lots <= cap_shares else "R-04 归零上限"
    return shares, round(shares * float(price) * float(stop_pct if stop_pct else
                                                       _cfg_val("STOP_LOSS_PCT", 0.01)), 6), (
        f"R-38+R-04 取更严（行 552）：风险口径 {risk_lots} 股（风险额 {risk_amount:,.2f}）"
        f" vs 归零口径 {cap_shares} 股（权重 {cap_w:.1%}）→ 取 {shares} 股，绑定 {bound}；"
        f"止损位 R-36 = 买入价下方 {float(stop_pct if stop_pct else _cfg_val('STOP_LOSS_PCT', 0.01)):.1%}"
    )


# --------------------------------------------------------------------------- R-35 分批建仓
def plan_tranches(total_value: float, n: int = 5) -> list[float]:
    """R-35：首笔 ≤ 计划额度 1/5（默认 n=5 等分）；n<5 时首笔钳到 1/5、余额在后 n-1 笔等分。"""
    total = float(total_value)
    n = int(n)
    if total <= 0 or n <= 0:
        return []
    if n == 1:
        return [total]  # 一次买入无法满足 1/5，交由调用方决策
    first = float(_cfg_val("FIRST_TRANCHE", 0.20))
    equal = total / n
    if equal <= total * first + 1e-12:
        return [equal] * n
    head = total * first
    return [head] + [(total - head) / (n - 1)] * (n - 1)


def can_add_tranche(position) -> tuple[bool, str]:
    """R-35：首笔没有账面利润就绝不加第二笔（加仓前置条件 = 浮盈 > 0）。"""
    pct, known = _position_pnl_pct(position)
    if not known:
        return False, "R-35：无账面利润信息 → 不加第二笔（取更保守，L573）"
    return (pct > 0), (
        f"R-35：当前浮盈 {pct:+.2%}，" + ("有账面利润，可加下一笔（仍受 R-37 与上限约束）"
                                    if pct > 0 else "无账面利润，禁止加第二笔")
    )


# --------------------------------------------------------------------------- R-16 持股数量
def can_add_position(n_current: int) -> tuple[bool, str]:
    """R-16 价值仓 3—10 只（超过 20 只视为能力薄弱信号，行 2108/474-477）。"""
    mx = int(_cfg_val("MAX_POSITIONS", 10))
    mn = int(_cfg_val("MIN_POSITIONS", 3))
    n = int(n_current)
    if n >= mx:
        return False, (f"R-16：当前 {n} 只，已达上限 {mx} 只 → 禁止加新标的"
                       f"（下限 {mn} 只；>20 只视为能力薄弱）")
    tail = "" if n + 1 >= mn else f"（加后 {n + 1} 只仍低于下限 {mn} 只，需补足分散）"
    return True, f"R-16：组合上限 {mx} 只、下限 {mn} 只；当前 {n} 只 → 可加 1 只{tail}"


# --------------------------------------------------------------------------- R-37 禁止摊平
_PCT_KEYS = ("pnl_pct", "unrealized_pnl_pct", "return_pct", "profit_pct", "浮盈比例", "盈亏比例")
_ABS_KEYS = ("pnl", "unrealized_pnl", "profit", "浮盈", "盈亏")
_COST_KEYS = ("cost", "cost_basis", "avg_price", "avg_cost", "买入均价", "成本")


def _position_pnl_pct(position) -> tuple[float, bool]:
    """把 position（比例数值／dict／对象）归一成浮动盈亏比例；无法判定时返回 (0.0, False)。"""
    if position is None or isinstance(position, bool):
        return 0.0, False
    if isinstance(position, Real):
        return float(position), True   # 约定：裸数值按比例解释（-0.05 = -5%）
    if isinstance(position, dict):
        getter = position.get
    else:
        getter = lambda k, d=None: getattr(position, k, d)  # noqa: E731
    for k in _PCT_KEYS:
        v = getter(k)
        if isinstance(v, Real):
            return float(v), True
    abs_pnl = cost = None
    for k in _ABS_KEYS:
        v = getter(k)
        if isinstance(v, Real):
            abs_pnl = float(v)
            break
    for k in _COST_KEYS:
        v = getter(k)
        if isinstance(v, Real):
            cost = abs(float(v))
            break
    if abs_pnl is not None and cost:
        qty = getter("shares") or getter("qty")
        base = cost * float(qty) if isinstance(qty, Real) and qty else cost
        return abs_pnl / base, True
    return 0.0, False


def can_average_down(position) -> tuple[bool, str]:
    """R-37（安全底线）：亏损头寸不加码、绝不追加保证金；仅浮盈 > 0 时可加。"""
    pct, known = _position_pnl_pct(position)
    if not known:
        return False, ("R-37：无法判定该头寸盈亏（缺浮盈或成本信息）→ 按禁止加码处理"
                       "（同层冲突取更保守，L573）")
    if pct > 0:
        return True, (f"R-37：当前浮盈 {pct:+.2%} > 0，允许加码；"
                      "仍须满足 R-35（首笔无利润不加第二笔）与 R-04/R-14 单标的上限")
    return False, (f"R-37：当前浮动 {pct:+.2%} ≤ 0 → 禁止摊平/加码，"
                   "亏损头寸干脆平仓认赔（行 445/1481）")


# --------------------------------------------------------------------------- 组合压力测试
def stress_test(equity: float, weights: dict[str, float],
                shocks: list[float]) -> pd.DataFrame:
    """行 537/539/541/556：损失% = 仓位% × 跌幅%；权益 75% 遇 -40% → -30%、遇 -50% → -37.5%。

    `weights` 只传受冲击的权益类资产（现金／债券腿不参与），键为该资产权重（分母＝可投资金融资产）；
    `equity` 为可投资金融资产金额，用于折算损失金额。
    """
    equity = float(equity)
    base = float(sum((v for v in (weights or {}).values() if isinstance(v, Real))))
    rows = []
    for s in (shocks or []):
        shock = float(s)
        if abs(shock) > 1.0:                       # 容错：传入 -40 / -50 按百分数解释
            shock = shock / 100.0
        loss_pct = base * shock                    # 行 537：损失% = 仓位% × 跌幅%
        cap_at_shock = min(STRESS_TOLERANCE / abs(shock), float(_cfg_val("EQUITY_MAX", 0.75))) \
            if shock else float("nan")             # 行 556：只愿承受 30% → 权益上限 30%/|跌幅|
        rows.append({
            "shock": round(shock, 6),
            "shock_pct": f"{shock:.1%}",
            "equity_weight": round(base, 6),
            "portfolio_loss_pct": round(loss_pct, 6),
            "loss_amount": round(equity * loss_pct, 2),
            "tolerance": STRESS_TOLERANCE,
            "exceed_tolerance": bool(loss_pct < -STRESS_TOLERANCE - 1e-12),
            "equity_cap_at_shock": round(cap_at_shock, 6),
            "rule_id": "R-04+R-09(L541/556)",
        })
    return pd.DataFrame(rows, columns=[
        "shock", "shock_pct", "equity_weight", "portfolio_loss_pct", "loss_amount",
        "tolerance", "exceed_tolerance", "equity_cap_at_shock", "rule_id"])


# --------------------------------------------------------------------------- R-14 费雪分类上限
def fisher_class_cap_raw(cls: str) -> float:
    """R-14 原值：A 类 ≤20%、B 类 8%—10%（取区间上限 10%）、C 类 ≤5%；分类不适用时回落 R-04。"""
    key = str(cls or "").strip().upper()[:1]
    if key in FISHER_CAP_RAW:
        return FISHER_CAP_RAW[key]
    return _zero_loss_cap()


def fisher_class_cap(cls: str) -> float:
    """R-14 生效上限：与 R-04 取更严（→ 三档均不高于 5%）。"""
    return min(fisher_class_cap_raw(cls), _zero_loss_cap())


# --------------------------------------------------------------------------- 冒烟测试
if __name__ == "__main__":
    print(f"[cfg] source={'config.py' if _cfg is not None else '内置默认(SPEC)'} "
          f"RISK_ZERO_LOSS={_zero_loss_cap():.4f} MAX_POSITIONS={_cfg_val('MAX_POSITIONS', 10)}")

    w, why = max_weight_zero_loss(100_000.0, 50.0)
    print(f"\n[R-04] max_weight_zero_loss(1e5, 50) = {w}\n  {why}")
    assert abs(w - _zero_loss_cap()) < 1e-12 and "R-04" in why
    assert shares_for_weight(100_000.0, 50.0, w) == 100

    sh, amt = size_by_risk(100_000.0, 20.0, stop_pct=0.01, risk_pct=0.015)
    print(f"[R-38] size_by_risk(1e5, 20, 1%, 1.5%) = {sh} 股, 风险额 {amt}")
    assert sh == 7500 and abs(amt - 1500.0) < 1e-6
    sh2, amt2 = size_by_risk(100_000.0, 20.0, stop_pct=0.01, risk_pct=0.05)   # 越界钳到 2%
    assert abs(amt2 - 2000.0) < 1e-6, amt2
    print(f"[R-38] risk_pct=5% 越界 → 钳到 2%: {sh2} 股, 风险额 {amt2}")

    tsh, t_amt, t_why = trend_position_size(100_000.0, 20.0)
    print(f"[行552] trend_position_size = {tsh} 股, 风险额 {t_amt}\n  {t_why}")
    assert tsh == 200, tsh   # 归零口径 5%×1e5/20 = 250→整手 200 股 vs 风险口径 7500 → 取更严

    tr = plan_tranches(100_000.0, 5)
    print(f"[R-35] plan_tranches(1e5, 5) = {tr}")
    assert len(tr) == 5 and abs(sum(tr) - 100_000.0) < 1e-6 and abs(tr[0] - 20_000.0) < 1e-6
    tr3 = plan_tranches(100_000.0, 3)
    print(f"[R-35] plan_tranches(1e5, 3) = {[round(x, 2) for x in tr3]}（首笔钳到 1/5）")
    assert abs(tr3[0] - 20_000.0) < 1e-6 and abs(sum(tr3) - 100_000.0) < 1e-6

    ok, why16 = can_add_position(9)
    ok10, why16b = can_add_position(10)
    print(f"[R-16] n=9 → {ok} | n=10 → {ok10}\n  {why16b}")
    assert ok and not ok10

    print(f"[R-37] 浮亏 -5% → {can_average_down(-0.05)[0]} | 浮盈 +3% → {can_average_down(0.03)[0]}"
          f" | 未知 → {can_average_down({'code': '000001'})[0]}")
    assert not can_average_down(-0.05)[0] and can_average_down(0.03)[0]
    assert not can_average_down({"code": "000001"})[0]
    assert can_average_down({"pnl": 300.0, "avg_price": 10.0, "shares": 1000})[0]

    st = stress_test(1_000_000.0, {"equity": 0.75}, [-0.40, -0.50])
    print("[行541/556] stress_test(1e6, {equity:0.75}, [-40%, -50%]):")
    print(st[["shock_pct", "equity_weight", "portfolio_loss_pct", "loss_amount",
              "exceed_tolerance", "equity_cap_at_shock"]].to_string(index=False))
    assert abs(st.loc[0, "portfolio_loss_pct"] + 0.30) < 1e-9      # 手册：-30%
    assert abs(st.loc[1, "portfolio_loss_pct"] + 0.375) < 1e-9     # 手册：-37.5%
    assert abs(st.loc[0, "equity_cap_at_shock"] - 0.75) < 1e-9     # 手册：权益上限 75%
    assert abs(st.loc[1, "equity_cap_at_shock"] - 0.60) < 1e-9     # 手册：降到 60%
    assert abs(st.loc[1, "loss_amount"] + 375_000.0) < 1e-6
    st_hi = stress_test(1_000_000.0, {"equity": 0.90}, [-0.50])
    assert bool(st_hi.loc[0, "exceed_tolerance"])                  # 90%>60% 超容忍线

    print(f"[R-14] fisher_class_cap A/B/C = {[fisher_class_cap(c) for c in 'ABC']}"
          f" | raw = {[fisher_class_cap_raw(c) for c in 'ABC']}")
    assert [fisher_class_cap(c) for c in "ABC"] == [_zero_loss_cap()] * 3
    assert fisher_class_cap_raw("B") == 0.10 and fisher_class_cap_raw("X") == _zero_loss_cap()

    zok, zwhy = check_zero_loss_total({"000001": 0.05, "300750": 0.05, "trend": 0.06})
    print(f"[行555] check_zero_loss_total(14%通过样例见上/16%) = {zok}\n  {zwhy}")
    assert not zok
    assert check_zero_loss_total({"000001": 0.05, "300750": 0.05, "trend": 0.04})[0]

    print(f"\n[rule_ids] max_weight_zero_loss → {rule_ids('max_weight_zero_loss')}, "
          f"trend_position_size → {rule_ids('trend_position_size')}")
    print("SMOKE OK")
