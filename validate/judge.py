"""validate/judge.py —— 附录六 验证协议判定器（⑩）

取舍说明（SPEC/09-验证协议.md 有歧义处，取最贴近字面的做法）：
1. 判定四选一无量化绩效阈值，故「失败」只由显式证伪信号触发（维度标注 失败/证伪/不成立、
   failed=True、passed=False、劣于基准=True）；其余按六维能否完成分级：全完成→通过、有缺口→部分通过。
   **证伪文本扫描同时覆盖「结论」与「原因」**（原因 是六维里唯一承载自由文本结论的字段，
   只扫「结论」会让真实的「失败」永远替不到判定里）。
2. 样本门槛（行2158、2183）要的是**同一规则的独立决策样本**：报告里带分规则计数
   （`各规则样本`/`per_rule_samples`/带 rule_id 的成交明细等）时按最小值逐规则核对，
   缺失才回退总成交笔数；<20 笔不允许判「通过」，降为「部分通过」并写
   「样本不足，不得据此改规则」；样本笔数取不到同样不允许「通过」。
3. 状态链（行2181-2182）单向：判定与状态同名对应（通过→已验证、部分通过→部分验证），
   失败/过拟合嫌疑降一级，已退役为终态不再回升。
4. 行2177 原文写「①③属执行层」，本模块按其字面实现（①逻辑失效、③执行偏差→回第七章复盘）。
5. 记录表（行2153）『编号』= 规则编号（多规则用「、」连接；解析不到才退回策略 id 兜底），
   『规则』= rules.RULES 里的规则名称（取不到 rules 时退回编号本身，绝不拿策略名冒充规则名）。
6. 记录 CSV 的默认路径由 config.OUT_DIR 提供，config 缺失时退回 MQ_OUT_DIR 或库内 output/（不硬编码绝对路径）。
"""

from __future__ import annotations

import csv
import os
import re
from datetime import date as _date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 附录六 无编号-状态链（行2181-2182）
STATES = ("未验证", "验证中", "部分验证", "已验证", "已退役")
# 附录六 无编号-判定分级（行2160）
VERDICTS = ("通过", "部分通过", "失败", "过拟合嫌疑")
# 附录六 无编号-样本门槛（行2158、2183）：同一规则独立决策样本 ≥20 笔，<20 笔一律视为噪声
MIN_SAMPLES = 20
# 附录六 无编号-记录表字段（行2153）
RECORD_FIELDS = ("编号", "规则", "状态", "数据区间", "基准", "结论", "更新日期")
# 附录六 无编号-观测口径（行2131）：收益率只排最后
OBS_ORDER = ("最大回撤", "波动", "换手率", "资金占用", "策略失效期", "收益率")
# 附录六 六维固定顺序（行2131）
DIMS = ("历史回测", "基准比较", "交易成本", "参数敏感性", "样本外", "市场环境")
# 附录六 无编号-诊断六处理（行2176）
SIX_ACTIONS = ("继续观察", "暂停执行", "降低风险敞口", "重新验证", "修订规则", "退役规则")
# 附录六 五归因（行2175）
ATTRIBUTIONS = ("①逻辑失效", "②条件变化", "③执行偏差", "④数据问题", "⑤随机波动")
# 附录五 元数据 2076：下次统一审查 2027-01
NEXT_REVIEW = "2027-01"

RECORDS_FILENAME = "verification_records.csv"  # SPEC 4：输出文件名用 ASCII

_DIM_ALIASES = {
    "历史回测": ("历史回测", "回测", "history", "backtest"),
    "基准比较": ("基准比较", "基准", "benchmark"),
    "交易成本": ("交易成本", "成本", "cost"),
    "参数敏感性": ("参数敏感性", "敏感性", "sensitivity"),
    "样本外": ("样本外", "oos", "out_of_sample", "out-of-sample"),
    "市场环境": ("市场环境", "环境", "regime", "environment"),
}

_UNAVAILABLE_MARKS = ("数据不足", "不可完成", "无法完成", "缺失数据", "无数据")
_FAILURE_MARKS = ("失败", "证伪", "不成立", "无效", "劣于基准")

_SAMPLE_KEYS = ("n_trades", "交易笔数", "样本笔数", "样本数", "samples", "trade_count")


# --------------------------------------------------------------------------- #
# 路径（SPEC 2：统一由 config.py 提供，禁止硬编码绝对路径）
# --------------------------------------------------------------------------- #
def output_dir() -> str:
    """输出目录：优先 config.OUT_DIR（① 提供），缺失时退回 MQ_OUT_DIR / 库内 output/。"""
    try:  # 惰性导入，避免 import 期做任何耗时/跨模块工作
        from config import OUT_DIR  # type: ignore

        if OUT_DIR:
            return OUT_DIR
    except Exception:
        pass
    try:
        from .. import config as _cfg  # type: ignore

        if getattr(_cfg, "OUT_DIR", None):
            return _cfg.OUT_DIR
    except Exception:
        pass
    env = os.environ.get("MQ_OUT_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def records_path() -> str:
    return os.path.join(output_dir(), RECORDS_FILENAME)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _norm(x: Any) -> str:
    return "".join(str(x).split()).lower()


def _as_int(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return int(float(m.group())) if m else None


def _text_of(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, dict):
        return " ".join(_text_of(v) for v in val.values())
    if isinstance(val, (list, tuple, set)):
        return " ".join(_text_of(v) for v in val)
    return str(val)


def _find_dim(dims: Dict[str, Any], name: str) -> Tuple[Optional[str], Any]:
    """六维名称容错匹配（维度名可能带编号前缀/别名），精确优先。"""
    if not isinstance(dims, dict):
        return None, None
    if name in dims:
        return name, dims[name]
    for k, v in dims.items():
        if _norm(k) == _norm(name):
            return k, v
    for alias in _DIM_ALIASES.get(name, (name,)):
        a = _norm(alias)
        for k, v in dims.items():
            if a in _norm(k):
                return k, v
    return None, None


def _strings_in(val: Any, keys: Sequence[str]) -> List[str]:
    """收集维度结果里指定字段的文本（判定只看协议报出来的显式结论）。"""
    out: List[str] = []
    if isinstance(val, dict):
        for k, v in val.items():
            if k in keys:
                out.append(_text_of(v))
            elif isinstance(v, dict):
                out.extend(_strings_in(v, keys))
    return out


def _extract_samples(report: Dict[str, Any]) -> Optional[int]:
    """样本笔数：报告 / stats / 历史回测维 三处按序查找（行2158 样本门槛）。"""
    cand: List[Any] = [report.get(k) for k in _SAMPLE_KEYS]
    for holder_key in ("stats", "统计", "summary", "指标"):
        holder = report.get(holder_key)
        if isinstance(holder, dict):
            cand.extend(holder.get(k) for k in _SAMPLE_KEYS)
    dims = report.get("dims") or report.get("维度") or {}
    _, hist = _find_dim(dims if isinstance(dims, dict) else {}, "历史回测")
    if hist is not None:
        if isinstance(hist, dict):
            cand.extend(hist.get(k) for k in _SAMPLE_KEYS)
            for holder_key in ("stats", "统计", "summary", "指标"):
                holder = hist.get(holder_key)
                if isinstance(holder, dict):
                    cand.extend(holder.get(k) for k in _SAMPLE_KEYS)
    for c in cand:
        n = _as_int(c)
        if n is not None:
            return n
    return None


def _dim_status(val: Any) -> str:
    """维度完成度：完成 / 不可完成（行2131：缺数据必须标注不可完成，不得中断）。"""
    if val is None:
        return "不可完成"
    if isinstance(val, dict) and val.get("available") is False:
        return "不可完成"
    txt = _text_of(val)
    return "不可完成" if any(m in txt for m in _UNAVAILABLE_MARKS) else "完成"


def _dim_failures(val: Any) -> List[str]:
    """显式证伪信号（判定「失败」的唯一依据，见文件头取舍 1）。"""
    sig: List[str] = []
    if isinstance(val, dict):
        if val.get("failed") is True:
            sig.append("failed=True")
        if val.get("passed") is False:
            sig.append("passed=False")
        if val.get("劣于基准") is True:
            sig.append("劣于基准=True")
    for s in _strings_in(val, ("结论", "verdict", "判定", "状态", "result")):
        if any(m in s for m in _FAILURE_MARKS):
            sig.append(s.strip()[:60])
    return sig


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #
def _rule_names(rule_id: str) -> str:
    """编号（可能是「R-27、R-28」这种多编号）→ rules.RULES 里的名称，用「、」连接。

    取不到 rules 模块或该编号时返回空字符串，由调用方退回编号本身——绝不拿策略名冒充规则名。
    """
    if not rule_id:
        return ""
    try:
        import rules as _rules
    except ImportError:
        return ""
    ids = [p.strip() for p in str(rule_id).replace("+", "、").replace(",", "、").split("、") if p.strip()]
    names = []
    for rid in ids:
        rule = getattr(_rules, "RULES", {}).get(rid)
        nm = getattr(rule, "name", None) or getattr(rule, "规则", None) if rule else None
        names.append(str(nm) if nm else rid)
    return "、".join(names)


def judge(report: dict) -> dict:
    """附录六判定四选一：通过／部分通过／失败／过拟合嫌疑（行2160）。"""
    rep: Dict[str, Any] = dict(report or {})
    dims = rep.get("dims") or rep.get("维度") or {}
    if not isinstance(dims, dict):
        dims = {}

    dim_status: Dict[str, str] = {}
    missing: List[str] = []
    unavailable: List[str] = []
    failures: List[str] = []
    for name in DIMS:
        _key, val = _find_dim(dims, name)
        if _key is None:
            missing.append(name)
            dim_status[name] = "缺失"
            continue
        status = _dim_status(val)
        dim_status[name] = status
        if status == "不可完成":
            unavailable.append(name)  # 数据不足 ≠ 证伪（行2131 缺数据须标注）
            continue
        for s in _dim_failures(val):
            failures.append("%s：%s" % (name, s))

    overfit = bool(rep.get("overfit_suspect") or rep.get("过拟合嫌疑"))
    if rep.get("failed") is True or rep.get("证伪") is True:
        failures.append("报告级证伪标记")

    # 附录六 无编号-过拟合判定（行2131）：只在某个精确参数下有效 → 过拟合嫌疑，降级或不采用
    if overfit:
        verdict = "过拟合嫌疑"
    elif failures:
        verdict = "失败"
    elif missing or unavailable:
        verdict = "部分通过"
    else:
        verdict = "通过"

    samples = _extract_samples(rep)
    if samples is None:
        sample_ok: Optional[bool] = None
    else:
        sample_ok = samples >= MIN_SAMPLES

    notes: List[str] = [str(n) for n in (rep.get("notes") or [])]
    if samples is None:
        notes.append("样本笔数未给出，无法核对 ≥20 笔门槛（行2158）")
        if verdict == "通过":
            verdict = "部分通过"
    elif not sample_ok:
        notes.append("样本不足（%d 笔 < %d 笔），不得据此改规则" % (samples, MIN_SAMPLES))
        if verdict == "通过":
            verdict = "部分通过"
    if missing:
        notes.append("六维缺失：" + "、".join(missing))
    if unavailable:
        notes.append("数据不足/不可完成：" + "、".join(unavailable))
    if failures:
        notes.append("证伪信号：" + "；".join(failures))
    if verdict == "通过":
        notes.append("Ⅱ区(C/D)升级入Ⅰ区须走「规则变更单」（行2157-2162、2066）")

    cur_state = rep.get("cur_state") or rep.get("state") or rep.get("状态") or "未验证"
    nxt = None
    if str(cur_state) in STATES:
        nxt = next_state(str(cur_state), verdict)
    else:
        notes.append("入参状态「%s」不在状态链内，未给出流转建议" % cur_state)
        cur_state = "未验证"
        nxt = next_state("未验证", verdict)

    rule_id = str(rep.get("rule_id") or rep.get("编号") or "")
    today = _date.today().isoformat()
    # 记录表 7 字段（附录六 行2153）：编号=规则编号、规则=规则名（取不到才退回编号，绝不拿策略名冒充）、
    # 状态=按状态链流转后的结果、数据区间=协议实际使用的区间（优先各维登记的区间）
    span = (rep.get("data_range") or rep.get("数据区间") or "")
    if not span:
        try:
            rows_ = [str((d or {}).get("数据区间") or "") for d in (rep.get("dims") or {}).values()]
            span = max(rows_, key=len) if rows_ else ""
        except Exception:  # noqa: BLE001
            span = ""
    rule_name = _rule_names(rule_id) or str(rep.get("name") or rep.get("rule_name") or rule_id)
    state_out = nxt or str(cur_state or "未验证")
    row = None
    if rule_id:
        row = record_row(rule_id, rule_name, state_out, span,
                         rep.get("benchmark") or rep.get("基准") or "", verdict, today)

    return {
        "name": rep.get("name") or rep.get("rule_name") or "",
        "verdict": verdict,
        "判定": verdict,
        "样本笔数": samples,
        "样本门槛": MIN_SAMPLES,
        "sample_sufficient": sample_ok,
        "样本不足": sample_ok is False,
        "overfit_suspect": overfit,
        "dims": dim_status,
        "missing": missing,
        "unavailable": unavailable,
        "failures": failures,
        "notes": notes,
        "obs_order": list(rep.get("obs_order") or OBS_ORDER),
        "next_state": nxt,
        "状态建议": nxt,
        "row": row,
    }


def next_state(cur_state: str, verdict: str) -> str:
    """状态链流转（行2181-2182）：单向、失败/过拟合降级、已退役为终态。"""
    if cur_state not in STATES:
        raise ValueError("未知状态 %r，合法状态：%s" % (cur_state, "／".join(STATES)))
    if verdict not in VERDICTS:
        raise ValueError("未知判定 %r，合法判定：%s" % (verdict, "／".join(VERDICTS)))
    if cur_state == "已退役":
        return "已退役"  # 退役后不再回升，重新启用须走规则变更单（行2157-2162）
    if verdict == "通过":
        return "已验证"  # 闭环⑤：通过 → （Ⅱ区）升级入Ⅰ区
    if verdict == "部分通过":
        return "部分验证"
    idx = STATES.index(cur_state)
    return STATES[max(0, idx - 1)]  # 失败／过拟合嫌疑 → 降级（行2182）


# --------------------------------------------------------------------------- #
# 记录表（行2153 固定 7 字段）
# --------------------------------------------------------------------------- #
def record_row(rule_id, rule_name, state, data_range, benchmark, verdict, date) -> dict:
    if state not in STATES:
        raise ValueError("未知状态 %r，合法状态：%s" % (state, "／".join(STATES)))
    if verdict not in VERDICTS:
        raise ValueError("未知判定 %r，合法判定：%s" % (verdict, "／".join(VERDICTS)))
    if isinstance(date, datetime):
        d = date.date().isoformat()
    elif isinstance(date, _date):
        d = date.isoformat()
    elif date:
        d = str(date)
    else:
        d = _date.today().isoformat()
    return {
        "编号": str(rule_id or ""),
        "规则": str(rule_name or ""),
        "状态": state,
        "数据区间": str(data_range or ""),
        "基准": str(benchmark or ""),
        "结论": verdict,
        "更新日期": d,
    }


def _normalize_row(r: dict) -> dict:
    return {f: str(r.get(f, "") or "") for f in RECORD_FIELDS}


def _read_records(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [_normalize_row(r) for r in csv.DictReader(f) if r.get("编号")]


def write_records(rows: list[dict], path: str = None) -> str:
    """写 output/verification_records.csv；按「编号」upsert，保留既有行与顺序。"""
    if path is None:
        path = records_path()
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    merged: Dict[str, dict] = {}
    order: List[str] = []
    for r in _read_records(path) + [_normalize_row(r) for r in (rows or [])]:
        rid = r["编号"]
        if rid not in merged:
            order.append(rid)
        merged[rid] = r
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(RECORD_FIELDS))
        w.writeheader()
        for rid in order:
            w.writerow(merged[rid])
    return path


# --------------------------------------------------------------------------- #
# 规则变更单（行2166-2169）与诊断映射（行2175-2177）
# --------------------------------------------------------------------------- #
def change_ticket(rule_id, old, new, reason, evidence) -> str:
    """规则变更单模板；按行2183 禁止②对样本 <20 笔给出警告。"""
    today = _date.today().isoformat()
    warn: List[str] = []
    reason_s = str(reason or "").strip()
    ev_s = str(evidence or "").strip()
    if not reason_s:
        warn.append("变更原因缺失——行2167 要求「必须写为什么」，未写不得生效")
    m = re.search(r"样本[^\d]{0,8}(\d+)\s*(?:笔|个|次)", ev_s) or re.search(r"(\d+)\s*笔", ev_s)
    n = int(m.group(1)) if m else None
    if n is not None and n < MIN_SAMPLES:
        warn.append("样本 %d 笔 < %d 笔——行2158/2183：视为噪声，禁止据此调整规则条文"
                    "（除非数据条件齐备）" % (n, MIN_SAMPLES))
    elif n is None:
        warn.append("支持证据未写明样本笔数——行2166 必填「样本笔数／回测结论／证伪证据」")
    if not ev_s:
        warn.append("支持证据缺失——行2166 必填")

    lines = [
        "# 规则变更单（附录六 行2166-2169）",
        "",
        "- 规则编号：%s" % str(rule_id or ""),
        "- 旧版本：%s" % str(old or ""),
        "- 新表述：%s" % str(new or ""),
        "- 变更原因（必须写为什么）：%s" % (reason_s or "【未填写】"),
        "- 支持证据（样本笔数／回测结论／证伪证据）：%s" % (ev_s or "【未填写】"),
        "- 生效日期：%s" % today,
        "- 下次审查：%s（附录五 元数据 2076）" % NEXT_REVIEW,
        "",
        "## 闭环自检（行2157-2162）",
        "- [ ] ②样本累积 ≥%d 笔或数据条件齐备" % MIN_SAMPLES,
        "- [ ] ③已运行六维协议（%s）" % "／".join(DIMS),
        "- [ ] ④判定已完成：%s" % "／".join(VERDICTS),
        "- [ ] ⑤升／降级已回填「%s」" % RECORDS_FILENAME,
        "- [ ] ⑥保留旧版本与修改原因",
        "",
        "## 禁止条款（行2183）",
        "- 未经闭环与变更单不得修改规则条文；不得因单次盈亏调整规则（样本 <%d 笔一律视为噪声）" % MIN_SAMPLES,
    ]
    if warn:
        lines += ["", "## 校验警告"] + ["- %s" % w for w in warn]
    return "\n".join(lines)


_DIAGNOSIS: Dict[int, dict] = {
    1: {"name": "逻辑失效", "desc": "原有逻辑被事实推翻（行2175）", "layer": "执行层",
        "back_to": "第七章复盘",  # 行2177 字面：①③属执行层
        "actions": ("修订规则", "退役规则"), "action": "修订规则",
        "change_ticket": True, "backtestable": False,
        "note": "定性判断、无量化阈值（09 ③）；退役须写入附录四失败案例（行2176）"},
    2: {"name": "条件变化", "desc": "适用环境（利率、制度、市场结构）已变（行2175）", "layer": "验证层",
        "back_to": "六维协议",
        "actions": ("暂停执行", "降低风险敞口", "重新验证"), "action": "暂停执行",
        "change_ticket": False, "backtestable": True,
        "note": "重跑六维⑥市场环境（行2131）；若须改动条文则走「规则变更单」（行2177）"},
    3: {"name": "执行偏差", "desc": "实际动作没按规则做（行2175）", "layer": "执行层",
        "back_to": "第七章复盘",
        "actions": ("继续观察", "暂停执行"), "action": "继续观察",
        "change_ticket": False, "backtestable": False,
        "note": "不可回测（09 ③）；先纠正执行，再谈规则本身（行2177）"},
    4: {"name": "数据问题", "desc": "数据质量、样本选择或回测方法有缺陷（行2175）", "layer": "验证层",
        "back_to": "六维协议",
        "actions": ("重新验证",), "action": "重新验证",
        "change_ticket": False, "backtestable": True,
        "note": "须重核样本独立性／行情分布／市场环境覆盖（行2158）"},
    5: {"name": "随机波动", "desc": "样本还不足以判断（行2175）", "layer": "验证层",
        "back_to": "六维协议",
        "actions": ("继续观察",), "action": "继续观察",
        "change_ticket": False, "backtestable": True,
        "note": "样本 <%d 笔一律视为噪声，不得据此改规则（行2158、2183）" % MIN_SAMPLES},
}

_ATTR_HINTS = (
    (1, ("①", "逻辑", "logic", "thesis")),
    (2, ("②", "条件", "环境变化", "regime", "condition", "environment")),
    (3, ("③", "执行", "execut", "discipline")),
    (4, ("④", "数据", "data", "method")),
    (5, ("⑤", "随机", "波动", "噪声", "noise", "random")),
)


def diagnosis(reason_code: str) -> dict:
    """五归因 → 六处理映射（行2175-2177）。"""
    code = str(reason_code or "").strip()
    key: Optional[int] = None
    if code.isdigit() and 1 <= int(code) <= 5:
        key = int(code)
    if key is None:
        c = _norm(code)
        for k, hints in _ATTR_HINTS:
            if any(_norm(h) in c for h in hints):
                key = k
                break
    if key is None:
        raise ValueError("无法识别归因 %r，可接受：1-5／①-⑤／%s" % (reason_code, "／".join(ATTRIBUTIONS)))
    d = _DIAGNOSIS[key]
    return {
        "代码": "①②③④⑤"[key - 1],
        "归因": ATTRIBUTIONS[key - 1],
        "说明": d["desc"],
        "层级": d["layer"],
        "回到": d["back_to"],
        "六处理": list(SIX_ACTIONS),
        "候选处理": list(d["actions"]),
        "首选处理": d["action"],
        "需变更单": d["change_ticket"],
        "可回测": d["backtestable"],
        "备注": d["note"],
    }


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    def _dim(**kw):
        base = {"状态": "完成", "交易笔数": 24}
        base.update(kw)
        return base

    full = {d: _dim() for d in DIMS}
    full["基准比较"].update({"基准": "沪深300买入持有", "结论": "完成"})
    full["市场环境"].update({"结论": "完成", "分段标准": "指数250日涨幅>+15%牛/<−15%熊"})
    ok_report = {"name": "trend_card", "rule_id": "R-35", "cur_state": "部分验证",
                 "dims": full, "overfit_suspect": False,
                 "data_range": "2016-01-01~2026-09-30", "benchmark": "沪深300"}
    r1 = judge(ok_report)
    print("R-35 →", r1["verdict"], "| 状态: 部分验证 →", r1["next_state"],
          "| 样本:", r1["样本笔数"], r1["sample_sufficient"])

    thin = dict(ok_report, dims={k: dict(v, **({"交易笔数": 12} if k == "历史回测" else {}))
                                 for k, v in full.items()})
    r2 = judge(thin)
    print("样本12笔 →", r2["verdict"], "| 样本不足:", r2["样本不足"], "|", r2["notes"][-2:])

    over = dict(ok_report, overfit_suspect=True)
    r3 = judge(over)
    print("过拟合 →", r3["verdict"], "| 已验证 →", next_state("已验证", "过拟合嫌疑"))

    gap = {k: v for k, v in full.items() if k != "交易成本"}
    gap["参数敏感性"] = {"状态": "不可完成", "note": "网格数据不足"}
    r4 = judge(dict(ok_report, dims=gap))
    print("缺维+数据不足 →", r4["verdict"], "| missing:", r4["missing"],
          "| unavailable:", r4["unavailable"])
    print("已退役终态:", next_state("已退役", "通过"))

    row = record_row("R-35", "趋势仓首笔≤1/5(行2116)", "部分验证",
                     "2016-01-01~2026-09-30", "沪深300", r1["verdict"], "2026-10-12")
    tmp = os.path.join(output_dir(), "_smoke_records.csv")
    p = write_records([row], path=tmp)
    back = _read_records(p)
    print("write_records →", p, "| 回读字段:", list(back[0].keys()), "| 行数:", len(back))
    os.remove(p)
    print("默认记录表路径:", records_path())

    print("诊断 ①:", diagnosis("1")["归因"], diagnosis("1")["层级"], diagnosis("1")["候选处理"])
    print("诊断 随机波动:", diagnosis("随机波动")["首选处理"], "| 需变更单:",
          diagnosis("随机波动")["需变更单"])
    t = change_ticket("R-38", "单笔风险 ≤2%", "单笔风险 ≤1.5%", "回测显示 2% 档回撤超容忍线",
                      "样本 12 笔／OOS 收益 −4%")
    print("变更单警告行:", [l for l in t.splitlines() if l.startswith("- 样本")])
    print("判定集合:", VERDICTS, "| 状态链:", STATES)
