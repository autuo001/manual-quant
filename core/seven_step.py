"""七步决策法流水线骨架（SPEC §3-⑨；手册第〇章 L125—127）。

取舍（SPEC 未固定的部分，取最贴近字面的做法）：
1. SPEC 只固定「每步返回 (passed, reason, evidence)，任一步否决即终止并记账」，未固定函数名与入参结构
   → 本模块用 step1..step7 + run_pipeline(ctx)，ctx 为 dict；每一步调用前把已完成步骤写入 ctx["_results"]，
   使第六步能对「三个条件必须同时满足」（L166/167）做显式复核。
2. config.py / rules.py / data.loader / core.risk_budget 由其它模块并行实现 → 全部惰性导入；
   常量在 config 缺失时回退 SPEC §3-① 的字面值（仅用于独立冒烟测试，不作为第二套权威来源）。
3. 缺失数据一律否决并标注「数据不足」（SPEC §2：不得臆造）；只有手册明文允许的情形才降级为定性判断。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# 允许 `python3 core/seven_step.py` 直接冒烟测试时也能导入同仓其它模块（非数据路径，无硬编码绝对路径）
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

__all__ = [
    "STEP_META", "PIPELINE", "STRATEGIES", "MARGIN_TIERS", "THREE_GATES",
    "const", "steps_meta", "three_gates", "risk_veto_check", "run_pipeline",
    "step1_capital_allowed", "step2_strategy_selected", "step3_logic_holds",
    "step4_valuation_reasonable", "step5_position_size", "step6_buy_trigger",
    "step7_exit_plan",
]

# L137—145：三套打法先选一套且规则整套使用，中途不得因亏损更换逻辑
STRATEGIES = {"index": "指数定投流", "value": "价值投资流", "trend": "趋势交易流"}

# R-27 安全边际三档（≥30% 一般 / ≥33% 门槛 / ≥50% 经典），取更严
MARGIN_TIERS = ((0.50, "经典（R-27 ≥50%）"), (0.33, "门槛（R-27 ≥33%）"), (0.30, "一般（R-27 ≥30%）"))

# L134/222/276/292：三条总闸优先于手册其他所有规则
THREE_GATES = ("不用杠杆", "留足应急现金", "不孤注一掷")

# SPEC §3-① 字面值，仅在 config.py 未就绪时用于独立冒烟测试
_SPEC_FALLBACK = {
    "RISK_ZERO_LOSS": 0.05, "RISK_PER_TRADE": 0.015,
    "MAX_POSITIONS": 10, "MIN_POSITIONS": 3,
    "FIRST_TRANCHE": 0.20, "STOP_LOSS_PCT": 0.01,
    "REBAL_BAND": 0.05,
}


def const(name: str) -> Any:
    cfg = _config()
    if cfg is not None and hasattr(cfg, name):
        return getattr(cfg, name)
    return _SPEC_FALLBACK.get(name)


def _config():
    try:
        import config  # ① data-loader 所有；本模块只读
        return config
    except Exception:
        return None


def _need(ctx: dict, keys: list) -> list:
    return [k for k in keys if ctx.get(k) is None]


def _fail(reason: str, rule_ids: list, checks: dict, missing: list | None = None) -> tuple:
    ev = {"rule_ids": list(rule_ids), "checks": checks}
    if missing:
        ev["missing"] = list(missing)
    return False, reason, ev


def _pass(reason: str, rule_ids: list, checks: dict, **extra) -> tuple:
    ev = {"rule_ids": list(rule_ids), "checks": checks}
    ev.update(extra)
    return True, reason, ev


def _financial_gate(code: str, report_period: Any):
    """巴菲特财务门槛（data.loader 所有；惰性导入避免 import 时读 parquet）。"""
    try:
        from data.loader import financial_gate_row
    except Exception as exc:  # 模块未就绪或磁盘数据缺失
        return None, f"data.loader 不可用（{type(exc).__name__}）"
    try:
        return financial_gate_row(code, report_period), ""
    except Exception as exc:
        return None, f"financial_gate_row 失败（{exc}）"


def _zero_loss_cap(equity: float, price: float):
    """R-04 归零口径权重上限；优先用 core.risk_budget，缺失时按 SPEC 字面值本地计算。"""
    try:
        from core.risk_budget import max_weight_zero_loss
        return max_weight_zero_loss(equity, price)
    except Exception:
        cap = const("RISK_ZERO_LOSS")
        return cap, f"R-04 归零口径：单标的潜在损失 ≤ {cap:.0%} 可投资金融资产（本地回退，quote={price}）"


def _fisher_cap(cls: str):
    """R-14 费雪分类上限，与 R-04 取更严（B 类 8%—10% 取下限 8%）。"""
    try:
        from core.risk_budget import fisher_class_cap
        return fisher_class_cap(cls)
    except Exception:
        return {"A": 0.20, "B": 0.08, "C": 0.05}.get(str(cls).upper(), const("RISK_ZERO_LOSS"))


def _can_add(n_current: int):
    try:
        from core.risk_budget import can_add_position
        return can_add_position(n_current)
    except Exception:
        maxp = const("MAX_POSITIONS")
        return (n_current < maxp, f"R-16 持仓 {n_current}/{maxp} 只")


# ---------------------------------------------------------------- 逐步实现

def step1_capital_allowed(ctx: dict) -> tuple:
    """第一步｜资金是否允许（L125—134/186/275/292）。"""
    rule_ids = ["R-01", "R-02", "R-03", "R-08"]
    missing = _need(ctx, ["horizon_years", "emergency_fund_months", "high_interest_debt"])
    if missing:
        return _fail("数据不足：缺少 " + "、".join(missing) + "（R-02/R-03 需个人资金时间表，不得臆造）",
                     rule_ids, {}, missing)

    checks = {}
    leverage = ctx.get("leverage", 1.0)  # SPEC §0：回测杠杆倍数恒为 1
    checks["杠杆倍数=1（R-01）"] = leverage <= 1.0
    checks["无高息债务（R-02）"] = not bool(ctx.get("high_interest_debt"))
    months = ctx["emergency_fund_months"]
    checks["应急金≥3 个月支出（R-02）"] = months >= 3
    years = ctx["horizon_years"]
    checks["资金 3 年内不用（R-03）"] = years >= 3
    checks["非孤注一掷（R-08）"] = not bool(ctx.get("single_decision_all_in"))

    if not checks["杠杆倍数=1（R-01）"]:
        return _fail(f"R-01 安全底线：禁止任何杠杆（当前杠杆倍数 {leverage}）", rule_ids, checks)
    if not checks["无高息债务（R-02）"]:
        return _fail("R-02：先还清高息债务（信用卡分期、网贷等）再投资", rule_ids, checks)
    if not checks["应急金≥3 个月支出（R-02）"]:
        return _fail(f"R-02：应急金 {months} 个月 < 3 个月家庭支出，不足不进场", rule_ids, checks)
    if not checks["资金 3 年内不用（R-03）"]:
        if years < 1:
            return _fail("R-03：1 年内要用的钱只放货币基金，不进股市", rule_ids, checks)
        return _fail(f"R-03：资金 {years} 年内要用（<3 年），不买权益", rule_ids, checks)
    if not checks["非孤注一掷（R-08）"]:
        return _fail("R-08：任何时点保留现金缓冲，单一决策不押上全部", rule_ids, checks)

    note = "" if months >= 6 else "（应急金 3—6 个月为区间，取 6 更稳）"
    return _pass(f"资金允许{note}", rule_ids, checks)


def step2_strategy_selected(ctx: dict) -> tuple:
    """第二步｜策略选择：三套打法先选一套，整套使用（L137—145；R-07/L226 路径解锁）。"""
    rule_ids = ["R-07", "R-17"]
    strategy = ctx.get("strategy")
    checks = {"策略∈三套打法": strategy in STRATEGIES}
    if not checks["策略∈三套打法"]:
        return _fail(f"第二步：strategy={strategy!r} 不在 {sorted(STRATEGIES)} 内（L137—145 三套打法先选一套）",
                     rule_ids, checks)
    checks["未中途更换逻辑（L137—145）"] = not bool(ctx.get("switched_midway"))
    checks["理由不跨策略引用（R-07）"] = not bool(ctx.get("cross_strategy_reasons"))
    if not checks["未中途更换逻辑（L137—145）"]:
        return _fail("L137—145：中途不得因亏损更换逻辑（混搭即失效）", rule_ids, checks)
    if not checks["理由不跨策略引用（R-07）"]:
        return _fail("R-07：一个标的只归属一套策略，不得跨策略引用理由", rule_ids, checks)

    if strategy in ("value", "trend"):  # L226：个股进阶路径须同时满足三项前置
        missing = _need(ctx, ["basic_path_months", "weekly_research_time", "read_method_ch2"])
        if missing:
            return _fail("数据不足：进阶路径前置缺少 " + "、".join(missing) + "（L179/226）",
                         rule_ids + ["L226"], checks, missing)
        ok_pre = (ctx["basic_path_months"] >= 6 and bool(ctx["weekly_research_time"])
                  and bool(ctx["read_method_ch2"]))
        checks["进阶路径前置（L226）"] = ok_pre
        if not ok_pre:
            return _fail("L226：进阶路径（个股研究）须完成基础路径 6 个月以上＋每周固定研究时间＋读完第二章",
                         rule_ids + ["L226"], checks)
    return _pass(f"策略＝{STRATEGIES[strategy]}，规则整套使用", rule_ids, checks)


def step3_logic_holds(ctx: dict) -> tuple:
    """第三步｜逻辑是否成立：可证伪化（L151/395/396）＋策略专属门槛。"""
    rule_ids = ["R-05", "R-39"]
    conditions = ctx.get("falsifiable_conditions") or []
    checks = {"可证伪条件 2—3 条（L395）": 2 <= len(conditions) <= 3}
    if len(conditions) < 2:
        return _fail(f"L395：买入逻辑须写成 2—3 条可证伪陈述（当前 {len(conditions)} 条），写不清楚不进入下一步",
                     rule_ids, checks)

    strategy = ctx.get("strategy")
    evidence: dict[str, Any] = {"falsifiable_conditions": list(conditions)}

    if strategy == "value":
        gate, err = _financial_gate(ctx.get("code"), ctx.get("report_period"))
        if gate is None:
            return _fail(f"数据不足：财务门槛无法判定（{err}）", rule_ids, checks)
        evidence["financial_gate"] = gate
        checks["巴菲特财务门槛（毛利率≥40%/净利率>20%/ROE>0/营收同比>0/净利同比>0）"] = bool(gate.get("pass"))
        if gate.get("missing"):
            return _fail("数据不足：财务门槛缺字段 " + "、".join(map(str, gate["missing"])), rule_ids, checks)
        if not gate.get("pass"):
            bad = [k for k, v in gate.get("items", {}).items() if not v[2]]
            return _fail("财务门槛不通过：" + "；".join(bad), rule_ids, checks)
    elif strategy == "trend":
        checks["趋势确认（R-34 有效突破后进场）"] = bool(ctx.get("trend_confirmed"))
        if not checks["趋势确认（R-34 有效突破后进场）"]:
            return _fail("R-34：只在区间被有效突破后才动手（当前未确认）", rule_ids, checks)
    elif strategy == "index":
        # L155/254：估值阈值使用前须按附录二校准，未校准不得作为买卖信号
        calibrated = ctx.get("valuation_calibrated")
        if calibrated is None:
            return _fail("数据不足：指数估值阈值未按附录二校准（L155/254）", rule_ids, checks, ["valuation_calibrated"])
        checks["估值阈值已校准（L155/254）"] = bool(calibrated)
        if not calibrated:
            return _fail("L155/254：未按附录二校准的估值阈值不得作为买卖信号", rule_ids, checks)

    return _pass("逻辑成立且已可证伪化", rule_ids, checks, **evidence)


def step4_valuation_reasonable(ctx: dict) -> tuple:
    """第四步｜估值是否合理：R-27 安全边际三档；无内在价值数据时明确降级为定性（SPEC §3⑦）。"""
    rule_ids = ["R-27", "R-34"]
    iv = ctx.get("intrinsic_value")
    price = ctx.get("price")
    mos = ctx.get("margin_of_safety")

    if mos is None and iv and price:
        mos = 1.0 - float(price) / float(iv)
    if mos is None:
        if ctx.get("qualitative_confirmed") is True:
            return _pass("降级为定性：内在价值区间不可得，已人工确认（R-27 失效条件）", rule_ids,
                         {"安全边际可量化": False}, downgraded="定性", missing=["intrinsic_value"])
        return _fail("数据不足：内在价值区间不可得（现库无 DCF 输入项）→ 按 R-27 降级为定性判断，未人工确认前不买",
                     rule_ids, {"安全边际可量化": False}, ["intrinsic_value"])

    tier = None
    for line, label in MARGIN_TIERS:
        if mos >= line:
            tier = label
            break
    checks = {f"安全边际 {mos:.2%} ≥ 30%（R-27 下限）": mos >= 0.30}
    if tier:
        checks[f"档位＝{tier}"] = True
    if mos < 0.30:
        return _fail(f"R-27：安全边际 {mos:.2%} < 30%（一般档下限），不买；07 章口径「不满足即放弃参与」",
                     rule_ids, checks)
    return _pass(f"价格合理：安全边际 {mos:.2%}，档位 {tier}", rule_ids, checks,
                 margin_of_safety=round(mos, 6), tier=tier)


def step5_position_size(ctx: dict) -> tuple:
    """第五步｜仓位是否合适：R-04 归零口径 ≤5%＋R-16 3—10 只＋R-14 分类上限＋R-35/R-38 趋势仓。"""
    rule_ids = ["R-04", "R-16"]
    missing = _need(ctx, ["equity", "price", "planned_weight", "n_positions"])
    if missing:
        return _fail("数据不足：缺少 " + "、".join(missing) + "（R-04 需权益与价格）", rule_ids, {}, missing)
    if not ctx["equity"] or not ctx["price"]:
        return _fail("R-04：权益或价格非正，无法定仓位", rule_ids, {})

    r04_cap, cap_note = _zero_loss_cap(float(ctx["equity"]), float(ctx["price"]))
    r04_cap = float(r04_cap)
    weight = float(ctx["planned_weight"])
    checks = {f"单标的权重 ≤ {r04_cap:.0%}（R-04 归零口径）": weight <= r04_cap + 1e-12}
    cap, binder = r04_cap, "R-04"

    fisher = ctx.get("fisher_class")
    if fisher:
        fcap = float(_fisher_cap(fisher))
        rule_ids = rule_ids + ["R-14"]
        checks[f"费雪 {fisher} 类 ≤ {fcap:.0%}（R-14，与 R-04 取更严）"] = weight <= fcap + 1e-12
        if fcap < cap:
            cap, binder = fcap, "R-14"

    checks["持仓数 < 10（R-16）"] = bool(_can_add(int(ctx["n_positions"]))[0])
    checks["非摊平加仓（R-37）"] = not bool(ctx.get("is_averaging_down"))

    if weight > cap + 1e-12:
        return _fail(f"{binder}：计划权重 {weight:.2%} > 上限 {cap:.2%}（{cap_note}）", rule_ids, checks)
    if not checks.get("持仓数 < 10（R-16）", True):
        return _fail(f"R-16：持仓已达 {ctx['n_positions']} 只（上限 {const('MAX_POSITIONS')}）", rule_ids, checks)
    if not checks["非摊平加仓（R-37）"]:
        return _fail("R-37 安全底线：亏损头寸不加码、绝不摊平", rule_ids, checks)

    extra: dict[str, Any] = {"weight_cap": round(cap, 6), "sizing_note": cap_note}
    if ctx.get("strategy") == "trend":
        rule_ids = rule_ids + ["R-35", "R-38"]
        if _need(ctx, ["first_tranche_ratio", "risk_pct"]):
            return _fail("数据不足：趋势仓缺少 first_tranche_ratio / risk_pct（R-35/R-38）", rule_ids, checks)
        ft = float(const("FIRST_TRANCHE"))
        checks[f"首笔 ≤ 计划额度 {ft:.0%}（R-35）"] = float(ctx["first_tranche_ratio"]) <= ft + 1e-12
        checks["单笔风险 ∈ [1%, 2%]（R-38）"] = 0.01 - 1e-12 <= float(ctx["risk_pct"]) <= 0.02 + 1e-12
        if not checks[f"首笔 ≤ 计划额度 {ft:.0%}（R-35）"]:
            return _fail(f"R-35：首笔 {ctx['first_tranche_ratio']:.0%} > 计划额度 {ft:.0%}", rule_ids, checks)
        if not checks["单笔风险 ∈ [1%, 2%]（R-38）"]:
            return _fail(f"R-38：单笔风险 {ctx['risk_pct']:.2%} 超出 1%—2%（默认 {const('RISK_PER_TRADE'):.1%}）",
                         rule_ids, checks)
        extra["first_tranche_ratio"] = ctx["first_tranche_ratio"]
        extra["risk_pct"] = ctx["risk_pct"]

    return _pass(f"仓位合适：权重上限 {cap:.2%}", rule_ids, checks, **extra)


def risk_veto_check(ctx: dict) -> tuple:
    """买入前风险否决清单（L166/167「且通过风险否决清单」）：任一命中即不买。"""
    hits = []
    if float(ctx.get("leverage", 1.0)) > 1.0:
        hits.append("R-01 用了杠杆")
    if ctx.get("high_interest_debt") or (ctx.get("emergency_fund_months") is not None
                                         and ctx["emergency_fund_months"] < 3):
        hits.append("R-02 高息债务未清或应急金不足 3 个月")
    if ctx.get("horizon_years") is not None and ctx["horizon_years"] < 3:
        hits.append("R-03 资金 3 年内要用")
    if ctx.get("single_decision_all_in"):
        hits.append("R-08 孤注一掷")
    if ctx.get("is_averaging_down"):
        hits.append("R-37 摊平亏损头寸")
    if ctx.get("data_insufficient"):
        hits.append("数据不足项存在（SPEC §2 不得臆造）")
    return (not hits), ("ok" if not hits else "；".join(hits)), {"hits": hits}


def step6_buy_trigger(ctx: dict) -> tuple:
    """第六步｜买入触发：三条件同时满足（L166/167）＋风险否决清单＋附录五证据闸门。"""
    rule_ids = ["R-39", "R-05"]
    prior = ctx.get("_results", {})
    three = {
        "逻辑成立（第三步）": bool(prior.get("logic_holds", {}).get("passed")),
        "价格合理（第四步）": bool(prior.get("valuation_reasonable", {}).get("passed")),
        "仓位合适（第五步）": bool(prior.get("position_size", {}).get("passed")),
    }
    checks = dict(three)
    if not all(three.values()):
        lack = [k for k, v in three.items() if not v]
        return _fail("L166/167：三个条件缺一不买（缺 " + "、".join(lack) + "）", rule_ids, checks)

    ok_veto, veto_reason, veto_ev = risk_veto_check(ctx)
    checks["通过风险否决清单（L166/167）"] = ok_veto
    if not ok_veto:
        return _fail("风险否决清单命中：" + veto_reason, rule_ids, checks)

    basis = list(ctx.get("rule_ids") or [])
    if not basis:
        return _fail("SPEC §0：任何买入信号必须记账其依据的规则编号（rule_ids 为空）", rule_ids, checks)
    try:
        import rules
        gate_ok, gate_reason = rules.gate(basis)
    except Exception as exc:
        return _fail(f"rules.gate 不可用（{type(exc).__name__}）：附录五闸门无法校验，按安全底线不买入",
                     rule_ids + basis, checks)
    checks[f"证据闸门 rules.gate({'/'.join(basis)})"] = bool(gate_ok)
    if not gate_ok:
        return _fail(f"附录五闸门拦截：{gate_reason}", rule_ids + basis, checks)

    return _pass("买入触发：三条件同时满足 + 风险否决清单通过 + 证据闸门通过",
                 rule_ids + basis, checks, rule_ids_basis=basis)


def step7_exit_plan(ctx: dict) -> tuple:
    """第七步｜退出条件：买入前先写下来（L168），按策略取对应退出规则。"""
    rule_ids = ["R-05", "R-39"]
    exits = ctx.get("exit_conditions") or []
    checks = {"退出条件已写下（L168/R-05）": len(exits) >= 1}
    if not exits:
        return _fail("L168/R-05：买入前必须先写下第七步的退出条件，未写不买", rule_ids, checks)

    strategy = ctx.get("strategy")
    extra = {"exit_conditions": list(exits)}

    if strategy == "value":
        rule_ids = rule_ids + ["R-28", "R-32"]
        reasons = set(ctx.get("sell_reasons") or [])
        allowed = {"买错了", "公司变了", "有更好的"}
        # R-28 三个卖出理由：与盈亏无关；「公司变了」优先于 R-32 三年守则
        checks["卖出理由∈{买错了/公司变了/有更好的}（R-28）"] = bool(reasons & allowed) and reasons <= allowed
        if not checks["卖出理由∈{买错了/公司变了/有更好的}（R-28）"]:
            return _fail(f"R-28：卖出理由须取自三个卖出理由，当前 {sorted(reasons)}", rule_ids, checks)
        plan_years = ctx.get("hold_plan_years")
        extra["evaluation"] = "三年守则（R-32）：以三年为限评估，逻辑未变不因短期波动卖出"
        if plan_years is not None and plan_years < 3:
            extra["note"] = "R-32 最短评估期 3 年，计划持有短于 3 年（「公司变了」优先）"
    elif strategy == "trend":
        rule_ids = rule_ids + ["R-36"]
        stop = ctx.get("stop_pct")
        if stop is None:
            return _fail("数据不足：趋势仓未给止损距（R-36 最后买入价下方 1—1.5%）", rule_ids, checks, ["stop_pct"])
        checks[f"止损距 ∈ (0, 1.5%]（R-36，默认 {const('STOP_LOSS_PCT'):.1%}）"] = 0 < float(stop) <= 0.015 + 1e-12
        if not checks[f"止损距 ∈ (0, 1.5%]（R-36，默认 {const('STOP_LOSS_PCT'):.1%}）"]:
            return _fail(f"R-36：止损距 {stop:.2%} 超出 1—1.5 点（1%）口径，触发即执行不商量", rule_ids, checks)
        extra["stop_pct"] = stop
    elif strategy == "index":
        rule_ids = rule_ids + ["R-13", "R-18", "R-19"]
        band = ctx.get("rebalance_band", const("REBAL_BAND"))
        checks[f"再平衡触发带 ±{const('REBAL_BAND'):.0%}（R-13）"] = abs(float(band) - float(const("REBAL_BAND"))) < 1e-9
        if not checks[f"再平衡触发带 ±{const('REBAL_BAND'):.0%}（R-13）"]:
            return _fail(f"R-13：再平衡触发带须为 ±{const('REBAL_BAND'):.0%}（当前 {band}）", rule_ids, checks)
        extra["exit_rule"] = "指数仓：估值卖出（R-18 >10% 定投／R-19 <6.4% 分 10 份卖出）或再平衡（R-13）"

    return _pass("退出条件已登记", rule_ids, checks, **extra)


PIPELINE: list = [
    step1_capital_allowed, step2_strategy_selected, step3_logic_holds,
    step4_valuation_reasonable, step5_position_size, step6_buy_trigger, step7_exit_plan,
]

# refs = 手册行号出处，供 README / 复盘引用
STEP_META = {
    "step1_capital_allowed": {"no": 1, "key": "capital_allowed", "name": "资金是否允许", "refs": "L125—134/186/275/292"},
    "step2_strategy_selected": {"no": 2, "key": "strategy_selected", "name": "策略选择", "refs": "L137—145/179/226"},
    "step3_logic_holds": {"no": 3, "key": "logic_holds", "name": "逻辑是否成立", "refs": "L150—151/395—396"},
    "step4_valuation_reasonable": {"no": 4, "key": "valuation_reasonable", "name": "估值是否合理", "refs": "L156/289—291"},
    "step5_position_size": {"no": 5, "key": "position_size", "name": "仓位是否合适", "refs": "L161—162/292"},
    "step6_buy_trigger": {"no": 6, "key": "buy_trigger", "name": "买入触发", "refs": "L166—168"},
    "step7_exit_plan": {"no": 7, "key": "exit_plan", "name": "退出条件", "refs": "L166/172—173"},
}


def steps_meta() -> list:
    return [dict(STEP_META[fn.__name__], func=fn.__name__) for fn in PIPELINE]


def three_gates() -> tuple:
    """三条总闸（L134/222/276/292），优先于手册其他所有规则。"""
    return THREE_GATES


def run_pipeline(ctx: dict, steps: list | None = None) -> dict:
    """逐层否决：任一步 (passed=False) 即终止，返回已记账的全部步骤。"""
    ctx = dict(ctx or {})
    results: dict = {}
    ctx["_results"] = results
    order = list(steps or PIPELINE)
    log: list = []
    for fn in order:
        meta = STEP_META.get(fn.__name__, {"no": len(log) + 1, "key": fn.__name__, "name": fn.__name__, "refs": ""})
        passed, reason, evidence = fn(ctx)
        rec = {
            "step": meta["no"], "key": meta["key"], "name": meta["name"], "refs": meta["refs"],
            "passed": bool(passed), "reason": reason, "evidence": evidence,
        }
        results[meta["key"]] = rec
        log.append(rec)
        if not passed:
            return {"code": ctx.get("code"), "strategy": ctx.get("strategy"), "passed": False,
                    "stopped_at": meta["key"], "stopped_step": meta["no"], "reason": reason,
                    "rule_ids": evidence.get("rule_ids", []), "steps": log}
    rule_ids = sorted({rid for rec in log for rid in rec["evidence"].get("rule_ids", [])})
    notes = [rec["evidence"]["note"] for rec in log if rec["evidence"].get("note")]
    return {"code": ctx.get("code"), "strategy": ctx.get("strategy"), "passed": True, "stopped_at": None,
            "stopped_step": None, "reason": "ok", "rule_ids": rule_ids, "steps": log,
            "downgraded": any(rec["evidence"].get("downgraded") for rec in log), "notes": notes}


def _stub_deps() -> list:
    """冒烟测试用桩：仅在本模块独立运行、且真实依赖模块尚未就绪时启用。"""
    installed = []
    for name, attrs in (
        # 桩只近似附录五闸门语义：V-xx 为待验证（D 级）→ 拦截；真实判定由 rules.gate 负责
        ("rules", {"gate": lambda ids: (not any(i.startswith("V-") for i in ids),
                                        "ok" if not any(i.startswith("V-") for i in ids)
                                        else "含待验证（D 级）依据，不得单独构成买入依据")}),
        ("data.loader", {"financial_gate_row": lambda code, rp: {"pass": True, "items": {}, "missing": []}}),
    ):
        try:
            __import__(name)
        except Exception:
            import types
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod
            if name == "data.loader":
                sys.modules.setdefault("data", types.ModuleType("data"))
            installed.append(name)
    return installed


if __name__ == "__main__":
    t0 = time.time()
    stubbed = _stub_deps()
    if stubbed:
        print(f"[warn] 依赖模块未就绪，冒烟测试使用桩：{stubbed}")

    base = {
        "code": "000001", "strategy": "trend",
        "horizon_years": 10, "emergency_fund_months": 6, "high_interest_debt": False, "leverage": 1.0,
        "basic_path_months": 8, "weekly_research_time": True, "read_method_ch2": True,
        "falsifiable_conditions": ["周线跌破前低即认错", "区间上限被否即降级观察"],
        "trend_confirmed": True,
        "equity": 100_000.0, "price": 10.0, "planned_weight": 0.05, "n_positions": 3,
        "first_tranche_ratio": 0.20, "risk_pct": 0.015, "margin_of_safety": 0.35,
        # 买入依据须记账且为 Ⅰ 区 A/B 级（附录五元数据 2066）；R-04/R-37 均为 A·B 级 Ⅰ 区
        "rule_ids": ["R-04", "R-37"], "stop_pct": 0.01,
        "exit_conditions": ["跌破最后一笔买入价下方 1% 即卖"],
    }
    print("steps:", [(m["no"], m["name"], m["refs"]) for m in steps_meta()])
    for label, case in (("应当全过", base),
                        ("第一步否决：杠杆 2 倍", dict(base, leverage=2.0)),
                        ("第四步否决：安全边际 10%", dict(base, margin_of_safety=0.10)),
                        ("第四步：无内在价值数据→降级为定性", dict(base, margin_of_safety=None, qualitative_confirmed=True)),
                        ("第六步：依据含 C 级 R-34/R-35（真实闸门应拦截）", dict(base, rule_ids=["R-34", "R-35"])),
                        ("第六步否决：依据是待验证 V-01", dict(base, rule_ids=["V-01"])),
                        ("第六步否决：未记账 rule_id", dict(base, rule_ids=[])),
                        ("第七步否决：趋势仓未写止损", dict(base, stop_pct=None))):
        r = run_pipeline(case)
        print(f"{label}: passed={r['passed']} stopped_at={r['stopped_at']} :: {r['reason'][:66]}")
    full = run_pipeline(base)
    print("全过用例命中规则:", full["rule_ids"])
    print("降级标记:", full.get("downgraded"), "notes:", full.get("notes"))
    # 价值流路径：财务门槛走真实 data.loader（惰性导入，不得在 import 时读全表）
    value_case = dict(base, strategy="value", code="600519", report_period="20231231",
                      sell_reasons=["公司变了"], hold_plan_years=5,
                      exit_conditions=["基本面恶化（营收/净利同比转负）即卖"])
    value_case.pop("stop_pct"), value_case.pop("first_tranche_ratio"), value_case.pop("risk_pct")
    rv = run_pipeline(value_case)
    ev = next((x for x in rv["steps"] if x["key"] == "logic_holds"), {})
    print(f"价值流: passed={rv['passed']} stopped_at={rv['stopped_at']} :: {rv['reason'][:56]}"
          f"｜财务门槛={ev.get('evidence', {}).get('financial_gate', {}).get('pass')}")
    print("冻结检查: 归零上限=%.0f%% 首笔=%.0f%% 止损=%.1f%%" % (
        const("RISK_ZERO_LOSS") * 100, const("FIRST_TRANCHE") * 100, const("STOP_LOSS_PCT") * 100))
    print("末步记账:", json.dumps(full["steps"][-1], ensure_ascii=False)[:180])
    print("耗时 %.2fs" % (time.time() - t0))
