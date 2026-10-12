"""CLI 编排（SPEC §3-⑨）：python3 run.py {check|value|trend|allocation|protocol|all}。

取舍：SPEC 只固定子命令集合与 check/protocol 的职责，未固定参数形式
→ 用 argparse，策略相关子命令统一 `--strategy {value,trend,all}`，allocation 的月度金额
必须由 --income/--expense 或 --monthly 显式给出（手册外不得臆造金额）；金额缺失时只打印调用方式。
所有跨模块 import 都在函数内惰性执行：依赖模块未就绪时打印缺失原因并返回退出码 2，不抛栈。
check 的数据缺口清单**动态读自 ② data.extra.availability()**：同一台机器联网／离线／缓存三种情形
取到的项不同（SPEC 3-②／4、spec/07 ②），写死清单必然与 output/data_availability.md 脱节。

本轮口径裁决（复核报告 §6.1／§6.2／§6.7，只在本文件内落地）：
- **全区间日历**：value / trend / allocation / check 自检与六维① 一律用
  `data.loader.trading_calendar()` 裁到 config.START_DATE~END_DATE 的**同一份**日历，不按信号区间裁剪。
  engine 的 `calendar=None` 回退会把日历缩成「首条信号日 → 末条信号日+5 交易日」，期末清仓日与
  资金占用／波动的分母随之变化，同一策略会同时存在 1.1873/1.2354/1.3924/1.4373 四个头条收益率
  （复核报告 §6.2）；框架对外只能有一个「全区间」口径。
- **编号透传**：protocol 子命令把策略信号里记账的 rule_id 写进 report 再交给 judge，judge 未给行时
  按规则逐条兜底建行，保证 output/verification_records.csv 的「编号」非空（复核报告 §6.7：
  编号为空 → judge._read_records 会读丢该行，记录表无法累积）。
- **可复现性自检**：check 末尾对 value 信号打乱两次各跑一次 engine，总收益必须逐位相同，不一致即
  退出码 2（复核报告 §6.1：engine 同日行顺序敏感是确定性回归，任何人跑 check 都该立刻发现）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMMANDS = ("check", "value", "trend", "allocation", "protocol", "all")
# SPEC §0 观测口径优先级：最大回撤、波动、换手率、资金占用、策略失效期 ＞ 收益率
OBS_PRIORITY = ("最大回撤", "波动", "换手", "资金占用", "失效", "收益", "年化", "交易笔数", "胜率", "盈亏比")

MISSING_NOTE = "模块未就绪（并行实现中）"
# check 的可复现性自检：两个固定种子 → 同一信号集合、两种行顺序；总收益必须逐位相同（§6.1）
REPRO_SEEDS = (20261012, 20261104)
RECORD_FIELDS = ("编号", "规则", "状态", "数据区间", "基准", "结论", "更新日期")


def _hr(title: str = "") -> None:
    print("\n" + (f"=== {title} " if title else "=== ").ljust(78, "="))


def _imp(modpath: str):
    try:
        return __import__(modpath, fromlist=["*"]), None
    except Exception as exc:
        return None, f"{modpath}: {type(exc).__name__}: {exc}"


def _config():
    return _imp("config")[0]


def _engine_cfg():
    """给六维协议一个 engine 基线配置：否则③交易成本三档无基线可比（SPEC §3-④ s⑥）。"""
    engine, err = _imp("core.engine")
    if engine is None or not hasattr(engine, "EngineConfig"):
        return None, err or "core.engine.EngineConfig 缺失"
    try:
        return engine.EngineConfig(), None
    except Exception as exc:
        return None, f"EngineConfig() 失败：{exc}"


def _full_calendar(cfg=None):
    """全区间日历（唯一口径）：`data.loader.trading_calendar()` 裁到 config.START_DATE~END_DATE。

    与 validate/protocol.py 的 `_slice_cal(ctx.calendar(), start, end)` 同源，故 CLI 与六维① 的
    期末清仓日、资金占用／波动的分母一致（复核报告 §6.2）。为什么裁区间用 config 常量而不是信号
    首末日：按信号裁剪会让区间随策略变，跨命令的数字不可互相引用。返回 (DatetimeIndex, None) 或 (None, 原因)。
    """
    loader, lerr = _imp("data.loader")
    if loader is None:
        return None, f"data.loader 不可用（{lerr}）"
    try:
        import pandas as pd

        cal = pd.DatetimeIndex(pd.to_datetime(list(loader.trading_calendar())))
    except Exception as exc:
        return None, f"trading_calendar() 失败：{type(exc).__name__}: {exc}"
    if len(cal) == 0:
        return None, "trading_calendar() 为空（指数数据不可用）"
    lo = getattr(cfg, "START_DATE", None) if cfg is not None else None
    hi = getattr(cfg, "END_DATE", None) if cfg is not None else None
    if lo:
        cal = cal[cal >= pd.Timestamp(lo)]
    if hi:
        cal = cal[cal <= pd.Timestamp(hi)]
    if len(cal) == 0:
        return None, f"日历在 {lo}~{hi} 内无交易日"
    return cal.sort_values().unique(), None


def _print_calendar(cal, err) -> None:
    if cal is None:
        print(f"[日历] 全区间日历不可用（{err}）→ 退回 engine 默认日历（口径可能与六维① 不一致）")
    else:
        print(f"[日历] 全区间 {cal[0].date()} → {cal[-1].date()}，共 {len(cal)} 个交易日"
              "（loader.trading_calendar 裁到 config 区间，不按信号区间裁剪）")


def _rule_ids_from_signals(sig) -> list:
    """策略信号里记账的依据类规则编号（SPEC §3-⑦：多条用 `+` 连接）→ 记录表的「编号」。"""
    if sig is None or getattr(sig, "empty", True) or "rule_id" not in getattr(sig, "columns", []):
        return []
    seen = []
    for cell in sig["rule_id"].dropna().astype(str):
        for rid in cell.replace("，", "+").split("+"):
            rid = rid.strip()
            if rid and rid not in seen:
                seen.append(rid)
    return seen


def _strategy_probe(strat):
    """包一层策略 run，缓存六维① 那次原始信号，用于把 rule_id 透传进记录表（复核报告 §6.7）。

    六维① 以 `_call_strategy(fn, cal, cfg)`（无额外 kw）调用；六维④ 的浮动档带 kw，不覆盖缓存。
    """
    seen: dict = {}

    def wrapped(calendar=None, cfg=None, **kw):
        out = strat.run(calendar=calendar, cfg=cfg, **kw)
        if not kw:
            seen["signals"] = out
        return out

    return wrapped, seen


def _records_rows(judge, verdict, rule_ids, name, span, rules_mod=None) -> list:
    """判定 → 记录表行；「编号」必填（附录六 7 字段的键，SPEC §0/§3-③ 逐规则维护）。

    judge 已给非空编号的行时直接采用；否则（含 judge 未产出行的情形）按信号里的规则逐条兜底建行 ——
    编号为空的行会被 judge._read_records 过滤掉，记录表无法累积（复核报告 §6.7）。
    """
    raw = verdict.get("row")
    rows = [r for r in (raw if isinstance(raw, list) else [raw]) if isinstance(r, dict)]
    if rows and all(str(r.get("编号") or "").strip() for r in rows):
        return rows
    v = verdict.get("verdict") or verdict.get("判定") or "部分通过"
    state = verdict.get("next_state") or verdict.get("状态建议")
    if not state:
        try:
            state = judge.next_state("未验证", v)
        except Exception:
            state = "未验证"
    today = time.strftime("%Y-%m-%d")
    out = []
    for rid in (rule_ids or [name]):
        rname = name
        rule = getattr(rules_mod, "RULES", {}).get(rid) if rules_mod is not None else None
        if rule is not None:
            rname = getattr(rule, "name", None) or name
        out.append(judge.record_row(rid, rname, state, span, "沪深300", v, today))
    return out


def _verify_records(path: str) -> tuple:
    """回读记录表：7 字段齐备且「编号」非空（复核报告 §6.7 的验收点）。"""
    import csv

    if not path or not os.path.exists(path):
        return False, f"未生成：{path}"
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        got = list(rdr.fieldnames or [])
        data = list(rdr)
    if tuple(got) != RECORD_FIELDS:
        return False, f"字段不符：{got}"
    blanks = sum(1 for r in data if not str(r.get("编号") or "").strip())
    if blanks or not data:
        return False, f"{len(data)} 行中 {blanks} 行「编号」为空"
    return True, f"{len(data)} 行，编号 {sorted({str(r['编号']) for r in data})}"


def _shown_stats(stats: dict) -> list:
    keys = list(stats)

    def rank(k: str) -> tuple:
        for i, name in enumerate(OBS_PRIORITY):
            if name in str(k):
                return (i, str(k))
        return (len(OBS_PRIORITY), str(k))

    return sorted(keys, key=rank)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.4f}"
    return str(v)


def _print_stats(stats: dict, title: str = "stats") -> None:
    print(f"[{title}] 按 SPEC §0 观测口径排序：")
    for k in _shown_stats(stats):
        v = stats[k]
        if isinstance(v, (dict, list)) and not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False)[:120]
        print(f"  {k:<24} {_fmt(v)}")


def _seven_step_selftest() -> bool:
    """规则闸门 + 七步法骨架自检（check 子命令用）。"""
    mod, err = _imp("core.seven_step")
    if mod is None:
        print(f"  [缺失] {err}")
        return False
    stubbed = mod._stub_deps()
    if stubbed:
        print(f"  [warn] 依赖未就绪，自检使用桩：{stubbed}（仅验证流水线逻辑，不代表真实闸门）")
    print("  七步：" + " → ".join(f"{m['no']}.{m['name']}" for m in mod.steps_meta()))
    base = {
        "code": "000001", "strategy": "trend", "horizon_years": 10, "emergency_fund_months": 6,
        "high_interest_debt": False, "leverage": 1.0, "basic_path_months": 8,
        "weekly_research_time": True, "read_method_ch2": True,
        "falsifiable_conditions": ["跌破前低即认错", "区间上限被否即降级"],
        "trend_confirmed": True, "equity": 100000.0, "price": 10.0, "planned_weight": 0.05,
        "n_positions": 3, "first_tranche_ratio": 0.20, "risk_pct": 0.015,
        "margin_of_safety": 0.35, "rule_ids": ["R-04", "R-37"], "stop_pct": 0.01,
        "exit_conditions": ["跌破最后一笔买入价下方 1% 即卖"],
    }
    ok = mod.run_pipeline(base)
    bad = mod.run_pipeline(dict(base, leverage=2.0))
    print(f"  全过用例：passed={ok['passed']}，命中规则 {len(ok['rule_ids'])} 条")
    print(f"  否决用例（杠杆 2 倍）：stopped_at={bad['stopped_at']}｜{bad['reason'][:60]}")
    return bool(ok["passed"] and not bad["passed"])


def _offline_now(extra) -> bool:
    """离线判定与 ② 保持一致（MQ_OFFLINE 的取值口径归它所有）；缺失时退回 SPEC §4 字面规则。"""
    fn = getattr(extra, "_offline", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            pass
    return os.environ.get("MQ_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def _gap_kind(info: dict) -> str:
    """按 ② 的登记判断缺口性质：可用／仅离线不可用／无法获取（README 第 5 节的三分类口径）。

    「仅离线不可用」只能由 note 判定：② 在离线路径写入的原因含 MQ_OFFLINE=1。为什么不能一律
    当成「数据不存在」——离线时它根本没联网（SPEC §4），没试过不等于取不到。
    """
    if info.get("available"):
        return "可用"
    return "仅离线不可用" if "MQ_OFFLINE" in str(info.get("note", "")) else "无法获取"


def _print_gaps(avail: dict, extra) -> None:
    """直读 ② availability() 输出缺口，不回退到任何写死清单（数据通断会随联网/缓存变化）。"""
    if not avail:
        print("  [无法判定] availability() 无返回 → 不列缺口（禁止用写死清单代替）")
        return
    buckets = {"可用": [], "仅离线不可用": [], "无法获取": []}
    for name, info in avail.items():
        buckets[_gap_kind(info)].append((name, info))
    for name, info in buckets["可用"]:
        print(f"  [可用]       {name:<16} 来源={info.get('source') or '—'} 行数={info.get('rows')}")
    for kind in ("仅离线不可用", "无法获取"):
        for name, info in buckets[kind]:
            print(f"  [{kind}] {name:<16} {str(info.get('note', ''))[:150]}")
    print(f"  [汇总] 可用 {len(buckets['可用'])} 项／仅离线不可用 {len(buckets['仅离线不可用'])} 项／"
          f"无法获取 {len(buckets['无法获取'])} 项；影响面见 README 第 5 节，权威清单见 output/data_availability.md")
    if _offline_now(extra):
        print("  [提示] MQ_OFFLINE=1：离线只复用 data/extra/ 下已抓到的缓存，缓存缺失项才记「数据不足」；"
              "去掉 MQ_OFFLINE 重跑本命令才能区分「仅离线不可用」与「无法获取」")


def _reproducibility_selftest(cfg=None) -> bool:
    """可复现性自检（复核报告 §6.1）：同一批 value 信号打乱两次各跑一次 engine，总收益必须逐位相同。

    为什么是「打乱两次」而不是「同顺序跑两次」：同顺序只能发现随机数／时间戳之类的抖动；§6.1 的根因是
    engine 对**同日行顺序**敏感（pandas 默认 quicksort 不稳定 + 无同日 tie-breaker），只有改变输入行
    顺序才能暴露。返回 True=通过（或无信号可比对），False=不可复现／依赖未就绪（check 据此退出码 2）。
    """
    strat, serr = _imp("strategies.value_card")
    engine, eerr = _imp("core.engine")
    if strat is None or engine is None:
        print(f"  [跳过] 依赖未就绪（{serr or eerr}）→ 确定性未被验证")
        return False
    cfg = cfg if cfg is not None else _config()
    cal, calerr = _full_calendar(cfg)
    if cal is None:
        print(f"  [跳过] {calerr} → 确定性未被验证")
        return False
    _print_calendar(cal, calerr)
    try:
        sig = strat.run(calendar=cal, cfg=cfg)
    except Exception as exc:
        print(f"  [失败] value_card.run(): {type(exc).__name__}: {exc}")
        return False
    if sig is None or getattr(sig, "empty", True):
        print("  [跳过] value 信号为空（数据不足）→ 无可比对的成交，确定性未被验证")
        return True
    runs, orders = [], []
    for seed in REPRO_SEEDS:
        shuffled = sig.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        orders.append([(str(d), str(c), str(s)) for d, c, s in
                       zip(shuffled["date"], shuffled["code"], shuffled["side"])])
        try:
            res = engine.run_backtest(shuffled, calendar=cal, cfg=None, liquidate_at_end=True)
        except Exception as exc:
            print(f"  [失败] run_backtest(打乱 seed={seed}): {type(exc).__name__}: {exc}")
            return False
        st = (res or {}).get("stats", {}) if isinstance(res, dict) else {}
        eq = (res or {}).get("equity") if isinstance(res, dict) else None
        final = float(eq["equity"].iloc[-1]) if hasattr(eq, "columns") and len(eq) else None
        runs.append({"seed": seed, "总收益": st.get("总收益"), "交易笔数": st.get("交易笔数"), "期末权益": final})
    a, b = runs
    print(f"  打乱 #{a['seed']}：总收益 {a['总收益']!r}｜交易笔数 {a['交易笔数']}｜期末权益 {a['期末权益']!r}")
    print(f"  打乱 #{b['seed']}：总收益 {b['总收益']!r}｜交易笔数 {b['交易笔数']}｜期末权益 {b['期末权益']!r}")
    if orders[0] == orders[1]:
        print("  [失败] 两次打乱后的行顺序相同 → 自检无效（seed 退化，请换种子）")
        return False
    if a["总收益"] is None or b["总收益"] is None:
        print(f"  [失败] 总收益缺失（{a['总收益']!r} / {b['总收益']!r}）→ 无法断言确定性")
        return False
    if a["总收益"] != b["总收益"]:
        print("  [失败] 可复现性自检失败：同一信号集合仅行顺序不同，engine 结果却不一致")
        print(f"    总收益差 = {float(a['总收益']) - float(b['总收益'])!r}"
              f"（{a['总收益']!r} vs {b['总收益']!r}）")
        print(f"    交易笔数 {a['交易笔数']} vs {b['交易笔数']}｜期末权益 {a['期末权益']!r} vs {b['期末权益']!r}")
        print("    处置：engine 需按 (date, side, code) 稳定排序并加同日 tie-breaker（复核报告 §6.1）；"
              "在此之前 CLI 与六维① 的头条数字不可互相引用")
        return False
    print(f"  可复现性自检通过：打乱两种行顺序后总收益逐位相同（{a['总收益']!r}），"
          f"信号 {len(sig)} 条 / 成交 {a['交易笔数']} 笔")
    return True


def cmd_check(args) -> int:
    _hr("环境")
    print(f"python {sys.version.split()[0]} | offline(MQ_OFFLINE)={os.environ.get('MQ_OFFLINE', '0')}")
    cfg, cerr = _imp("config")
    _hr("config.py（① data-loader 所有）")
    if cfg is None:
        print(f"  [缺失] {cerr} → 路径与常量无法解析；数据/规则自检跳过")
    else:
        for attr in ("DATA_DIR", "YJBB_RICH", "OUT_DIR", "START_DATE", "END_DATE", "IS_END", "OOS_START"):
            if hasattr(cfg, attr):
                val = getattr(cfg, attr)
                mark = ""
                if attr in ("DATA_DIR", "YJBB_RICH", "OUT_DIR"):
                    p = Path(str(val))
                    if p.is_file():
                        mark = f"  [{p.stat().st_size / 1e6:.1f} MB, 存在]"
                    elif p.is_dir():
                        mark = "  [目录存在]"
                    else:
                        mark = "  [不存在]"
                print(f"  {attr:<12} {val}{mark}")
        for attr in ("RISK_ZERO_LOSS", "RISK_PER_TRADE", "MAX_POSITIONS", "MIN_POSITIONS",
                     "FIRST_TRANCHE", "STOP_LOSS_PCT", "EQUITY_MIN", "EQUITY_MAX", "REBAL_BAND"):
            if hasattr(cfg, attr):
                print(f"  {attr:<16} {getattr(cfg, attr)}")
    _hr("数据可用性（② data.extra）")
    extra, eerr = _imp("data.extra")
    avail: dict = {}
    if extra is None:
        print(f"  [缺失] {eerr}")
    else:
        try:
            avail = extra.availability() or {}
        except Exception as exc:
            print(f"  [失败] availability(): {exc}")
        for name, info in avail.items():
            print(f"  {name:<16} available={info.get('available')} rows={info.get('rows')} "
                  f"note={info.get('note', '')}")
        av = Path(str(getattr(cfg, "OUT_DIR", "output"))) / "data_availability.md" if cfg else None
        if av is not None and av.exists():
            print(f"  清单文件：{av}")
    _hr("规则闸门自检（③ rules）")
    rules, rerr = _imp("rules")
    if rules is None:
        print(f"  [缺失] {rerr}")
    else:
        try:
            table = rules.RULES
            z2 = sum(1 for r in table.values() if getattr(r, "zone", 1) == 2)
            cd = sum(1 for r in table.values() if getattr(r, "level", "A") in ("C", "D"))
            print(f"  规则数 {len(table)}｜Ⅱ 区 {z2} 条｜C/D 级 {cd} 条｜可编码 {len(rules.encodable())} 条")
            for probe in (["R-04"], ["V-01"], ["R-04", "V-01"]):
                print(f"  gate({probe}) → {rules.gate(probe)}")
            print(f"  记录表列：{list(rules.dump_table().columns)}")
        except Exception as exc:
            print(f"  [失败] {exc}")
    _hr("七步法骨架自检（⑨ core.seven_step）")
    ok = _seven_step_selftest()
    _hr("可复现性自检（同一信号集合 → 同一结果；复核报告 §6.1）")
    # 这个函数此前只定义未接线，导致「信号行顺序敏感」这类回归不会被 check 发现
    repro_ok = _reproducibility_selftest(cfg)
    ok = bool(ok) and repro_ok
    _hr("数据缺口清单（动态读自 ② data.extra.availability()）")
    _print_gaps(avail, extra)
    return 0 if ok else 2


def _run_strategy(name: str, args) -> tuple:
    """取策略信号并跑 engine 回测；返回 (stats, 退出码)。"""
    modname = {"value": "strategies.value_card", "trend": "strategies.trend_card"}[name]
    strat, serr = _imp(modname)
    if strat is None:
        print(f"[缺失] {serr} → {name} 跳过")
        return None, 2
    cfg = _config()
    cal, calerr = _full_calendar(cfg)
    _print_calendar(cal, calerr)
    try:
        signals = strat.run(calendar=cal, cfg=cfg)
    except TypeError as exc:
        # 策略未接受 calendar= 时退回 SPEC 允许的默认签名，并显式标注口径降级（不静默换口径）
        print(f"[warn] {modname}.run(calendar=…): {exc} → 退回 calendar=None，口径将与六维① 不一致")
        try:
            signals = strat.run(cfg=cfg)
        except Exception as exc2:
            print(f"[失败] {modname}.run(): {type(exc2).__name__}: {exc2}")
            return None, 1
    except Exception as exc:
        print(f"[失败] {modname}.run(): {type(exc).__name__}: {exc}")
        return None, 1
    print(f"[信号] {modname} 共 {len(signals)} 条")
    need = [c for c in ("date", "code", "side", "reason", "rule_id") if c not in getattr(signals, "columns", [])]
    if need:
        print(f"[违约] 信号缺列 {need}（SPEC §3-④ signals 列）")
    else:
        if len(signals):
            print(f"  区间 {signals['date'].min()} → {signals['date'].max()}｜"
                  f"标的 {signals['code'].nunique()} 只｜side 分布 {signals['side'].value_counts().to_dict()}")
            print(f"  依据规则 Top5：{signals['rule_id'].value_counts().head().to_dict()}")
        else:
            print("  无信号（可能数据不足，见 check 子命令）")
    engine, eerr = _imp("core.engine")
    if engine is None:
        print(f"[缺失] {eerr} → 回测跳过")
        return None, 2
    try:
        res = engine.run_backtest(signals, calendar=cal, cfg=None, liquidate_at_end=True)
    except Exception as exc:
        print(f"[失败] run_backtest(): {type(exc).__name__}: {exc}")
        return None, 1
    stats = res.get("stats", {}) if isinstance(res, dict) else {}
    _print_stats(stats, f"{name} 回测")
    return stats, 0


def cmd_value(args) -> int:
    _hr("策略 value_card（七）→ engine（四）")
    return _run_strategy("value", args)[1]


def cmd_trend(args) -> int:
    _hr("策略 trend_card（七）→ engine（四）")
    return _run_strategy("trend", args)[1]


def cmd_allocation(args) -> int:
    _hr("长期配置与定投（⑥ core/allocation.py）")
    alloc, aerr = _imp("core.allocation")
    if alloc is None:
        print(f"[缺失] {aerr}")
        return 2
    monthly = args.monthly
    if monthly is None:
        if args.income is not None and args.expense is not None:
            monthly = alloc.dca_plan(args.income, args.expense)
            print(f"[R-21/L208] 每月定投额＝(收入 {args.income} − 支出 {args.expense})×50% = {monthly:,.2f}")
        else:
            print("[数据不足] 缺参数：请给 --monthly，或 --income 与 --expense（按 (收入−支出)×50% 计算）")
            print("  用法：python3 run.py allocation --income 20000 --expense 12000")
            return 2
    print(f"[输入] 每月定投额 {monthly:,.2f}")
    try:
        print(f"  R-12 年龄起点参考：equity_ratio_by_age(35)={alloc.equity_ratio_by_age(35):.2%}（仅起点）")
        print(f"  R-11 持有期 10 年：equity_ratio_by_holding_years(10)="
              f"{alloc.equity_ratio_by_holding_years(10):.2%}")
        print(f"  R-09 区间截断：clamp_equity(0.9)={alloc.clamp_equity(0.9):.2%}")
    except Exception as exc:
        print(f"  [失败] 分配函数：{exc}")
    try:
        index = None
        loader, lerr = _imp("data.loader")
        if loader is not None:
            index = loader.load_index()
            print(f"[数据] 指数 {len(index)} 行 {index.index.min()} → {index.index.max()}")
        else:
            print(f"[缺失] {lerr} → 定投回测缺指数序列")
        bond = None
        extra, _ = _imp("data.extra")
        if extra is not None:
            try:
                bond = extra.try_fetch_bond_index()
            except Exception as exc:
                print(f"[数据不足] 债券腿获取失败（不中断）：{exc}")
        if bond is None:
            print("[数据不足] 债券腿取不到（联网/缓存皆无）→ 债腿与六维② 60/40 不可完成（SPEC §2）")
        else:
            print("[口径] 债券腿为国债/中债现券指数，≠ 债券基金指数：R-13／60/40 属降级替代，"
                  "引用时须连口径一起写（见 output/data_availability.md 口径②）")
        res = alloc.run_allocation_backtest(monthly, None, index, bond)
        for k, v in (res or {}).items():
            if isinstance(v, dict):
                _print_stats(v, k)
            elif hasattr(v, "shape"):
                print(f"  {k}: {type(v).__name__} shape={getattr(v, 'shape', None)}（明细见 output/）")
            else:
                print(f"  {k}: {v}")
        return 0
    except Exception as exc:
        print(f"[失败] run_allocation_backtest(): {type(exc).__name__}: {exc}")
        return 1


def cmd_protocol(args) -> int:
    _hr("六维验证协议（⑧ validate/protocol.py）+ 判定与记录表（⑩ validate/judge.py）")
    proto, perr = _imp("validate.protocol")
    if proto is None:
        print(f"[缺失] {perr}")
        return 2
    targets = ["value", "trend"] if args.strategy == "all" else [args.strategy]
    ecfg, ecfg_err = _engine_cfg()
    if ecfg is None:
        print(f"[warn] {ecfg_err} → 六维③交易成本将无基线（结果里会标注）")
    judge, jerr = _imp("validate.judge")
    if judge is None:
        print(f"[缺失] {jerr} → 将只输出六维结果，不判定")
    rc = 0
    rows = []
    for name in targets:
        modname = {"value": "strategies.value_card", "trend": "strategies.trend_card"}.get(name)
        strat, serr = _imp(modname) if modname else (None, f"未知策略 {name}")
        if strat is None:
            print(f"[缺失] {serr} → {name} 协议跳过")
            rc = max(rc, 2)
            continue
        try:
            report = proto.run_six_dim(strat.run, name, cfg=ecfg)
        except Exception as exc:
            print(f"[失败] run_six_dim({name}): {type(exc).__name__}: {exc}")
            rc = max(rc, 1)
            continue
        print(f"\n[{name}] 观测口径顺序：{report.get('obs_order')}")
        for dim, metrics in (report.get("dims") or {}).items():
            if isinstance(metrics, dict):
                head = {k: metrics[k] for k in list(metrics)[:6]}
                print(f"  {dim}: {json.dumps(head, ensure_ascii=False, default=str)[:220]}")
            else:
                print(f"  {dim}: {metrics}")
        print(f"  过拟合嫌疑={report.get('overfit_suspect')}｜notes={report.get('notes')}")
        if judge is not None:
            try:
                verdict = judge.judge(report)
                print(f"  判定：{json.dumps(verdict, ensure_ascii=False, default=str)[:300]}")
                v = verdict.get("verdict") or verdict.get("判定") or "部分通过"
                row = verdict.get("row")
                if not row:
                    state = (verdict.get("next_state") or verdict.get("状态建议")
                             or judge.next_state("未验证", v))
                    row = judge.record_row("", name, state, f"{args.start}~{args.end}", "沪深300", v,
                                           time.strftime("%Y-%m-%d"))
                rows.append(row)
            except Exception as exc:
                print(f"  [失败] judge(): {type(exc).__name__}: {exc}")
                rc = max(rc, 1)
    if judge is not None and rows:
        try:
            out = judge.write_records(rows)
            print(f"\n[记录表] {out}")
        except Exception as exc:
            print(f"[失败] write_records(): {exc}")
            rc = max(rc, 1)
    return rc


def cmd_all(args) -> int:
    rcs = []
    # all 作为一键冒烟：allocation 缺月投金额时按默认 3000 元跑（否则该步固定返回 2，
    # 会把整条流水线判成失败，掩盖其它步骤的真实问题）
    if getattr(args, "monthly", None) is None and getattr(args, "income", None) is None:
        args.monthly = 3000.0
    for fn in (cmd_check, cmd_value, cmd_trend, cmd_allocation, cmd_protocol):
        try:
            rcs.append(fn(args))
        except Exception:
            traceback.print_exc()
            rcs.append(1)
    _hr("汇总")
    for name, rc in zip(("check", "value", "trend", "allocation", "protocol"), rcs):
        print(f"  {name:<12} 退出码 {rc}（0 通过 / 1 失败 / 2 依赖模块未就绪）")
    return max(rcs) if rcs else 0


HANDLERS = {"check": cmd_check, "value": cmd_value, "trend": cmd_trend,
            "allocation": cmd_allocation, "protocol": cmd_protocol, "all": cmd_all}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py", description="manual_quant 回测框架 CLI（SPEC §3-⑨）")
    p.add_argument("command", choices=COMMANDS, help="check|value|trend|allocation|protocol|all")
    p.add_argument("--strategy", choices=("value", "trend", "all"), default="all",
                   help="protocol 子命令要验证的策略，默认 all")
    p.add_argument("--monthly", type=float, default=None, help="allocation：每月定投额")
    p.add_argument("--income", type=float, default=None, help="allocation：月收入（R-21 口径）")
    p.add_argument("--expense", type=float, default=None, help="allocation：月支出（R-21 口径）")
    p.add_argument("--start", default=None, help="记录表「数据区间」起（默认取 config.START_DATE）")
    p.add_argument("--end", default=None, help="记录表「数据区间」止（默认取 config.END_DATE）")
    p.add_argument("--offline", action="store_true", help="等价 MQ_OFFLINE=1，禁止联网")
    return p


if __name__ == "__main__":
    t0 = time.time()
    args = build_parser().parse_args()
    if args.offline:
        os.environ["MQ_OFFLINE"] = "1"
    cfg = _config()
    if cfg is not None:
        args.start = args.start or getattr(cfg, "START_DATE", None)
        args.end = args.end or getattr(cfg, "END_DATE", None)
    else:
        args.start = args.start or "?"
        args.end = args.end or "?"
    try:
        rc = HANDLERS[args.command](args)
    except KeyboardInterrupt:
        print("\n[中断] 用户中止")
        rc = 130
    print(f"\n[run.py {args.command}] 退出码 {rc}｜耗时 {time.time() - t0:.1f}s")
    sys.exit(rc)
