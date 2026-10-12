# -*- coding: utf-8 -*-
"""数据加载层（SPEC §3①）——只读 SPEC §2 已登记的数据，不重复下载、不臆造缺失数据。

取舍说明（SPEC 有歧义处，按最贴近字面的做法）：
1. 比率字段单位统一用「百分数」：`gross_margin` 直接取源列「销售毛利率」（源库本身即百分数，
   SPEC 括号写作「(销售毛利率)」），故 `net_margin` 也按 净利润/营收×100 输出；
   `roe`/`rev_yoy`/`profit_yoy` 同为源库百分数。财务门槛阈值同单位（40 / 20 / 0）。
   注意 `net_margin` 是 5.0 而非 0.05。
2. `load_fundamentals()` 输出 SPEC §3① 列出的 14 列（顺序与 SPEC 列序逐字一致），并在**末尾追加**
   2 列：`bps`（每股净资产，源列直取）与 `ocf_ratio`（每股经营现金流量/每股收益）。
   这两列为**主控新增**：价值策略算 PB 与现金流排雷都要用它们——spec/01 序38「PE/PB 组合象限」、
   序36「盈利不稳定或周期性品种改用 PB 处于历史底部判断」需要 PB，而 SPEC §2 的富字段里本来就有
   「每股净资产」列（此前未派生，白丢数据）；现金流排雷对应 spec/01 序45 ⑤「经营现金流须长期为正
   且与净利润规模匹配」与 spec/07 序44 案例四清单 G「经营现金流与利润背离」。新增列追加在末尾
   （不插入中段），是为了让 SPEC 前 14 列的顺序保持不变，下游按列名取用/按列名断言均不受影响。
3. `ocf_ratio` 是现金流质量的**近似**口径：现库无现金流量表，也没有净利润绝对值的逐笔对应，
   只能用「每股经营现金流量/每股收益」替代（手册本身只给关系不给公式，故不上任何自定义阈值）。
   EPS 为 0 时留 NaN（不臆造）；EPS < 0（亏损）时比值符号会翻转，调用方须先用 ROE/净利同比
   门槛过滤盈利能力之后再拿它排雷，否则会把「亏损+经营现金流为负」误读成比值正常。
4. `financial_gate_row(code, report_period=None)`：report_period 省略/None/NaT 时**自动取该股票
   最新一期已公告报告期**（ann_date 非空者中 report_period 最大者），并在返回 dict 的
   `report_period` 字段里标明实际用了哪一期（该股票无任何已公告记录时为 None）；显式传期号时
   行为与从前完全一致。改这个的原因：原签名必填，而 core/seven_step.py 用 ctx.get("report_period")
   取值可能为 None，落到 `NaT.strftime` 会抛 ValueError: NaTType does not support strftime。
5. `profit_growth_streak` 用年度报告期（report_period 以 1231 结尾）的「净利润-同比增长>0」
   连续期数，近似手册「EPS 连续 10 年增长」（现库无 EPS 绝对值历史，见 spec/04 序80）；
   同比字段缺失即中断计数（保守，不外推）。
6. `universe` 按 [START_DATE, END_DATE] 窗口内的 bar 数折算「上市天数」。
7. 直接 `import pyarrow` 的原因：`pd.read_parquet(filters=...)` 本身就依赖 pyarrow 引擎，
   而按 code 过滤读行组是 SPEC §4「989 万行必须按需切片、禁止无脑 concat 全表」的落地手段；
   未新增任何第三方依赖（pyarrow 是 pandas 读 parquet 的既有依赖），且带 pandas 回退分支。
"""

import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

try:  # 允许三种导入姿势：run.py 平铺导入 / data.loader / 直接执行本文件
    import config as _cfg
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config as _cfg

try:
    import pyarrow as _pa
    import pyarrow.compute as _pc
    import pyarrow.parquet as _pq
    _HAS_ARROW = True
except ImportError:  # pragma: no cover - pyarrow 随 pandas 安装，这里只做兜底
    _HAS_ARROW = False

__all__ = [
    "load_daily", "load_daily_all", "load_index", "load_fundamentals",
    "trading_calendar", "universe", "profit_growth_streak", "financial_gate_row",
    "GATE_ITEMS", "FIN_GATE_THRESHOLDS",
]

_DAILY_COLS = ["open", "close", "volume"]
# 前 14 列 = SPEC §3① 的必给派生列，顺序逐字不动；末 2 列为主控新增（见文件头 2/3 条）
_FUND_COLS = [
    "code", "name", "report_period", "ann_date", "eps", "revenue", "net_profit",
    "net_margin", "gross_margin", "roe", "ocfps", "rev_yoy", "profit_yoy", "industry",
    "bps", "ocf_ratio",
]

# 富字段 yjbb → 派生列（列名以 SPEC §2 列的富字段清单为准）
_YJBB_SRC = [
    "股票代码", "股票简称", "每股收益", "营业总收入-营业总收入", "营业总收入-同比增长",
    "净利润-净利润", "净利润-同比增长", "每股净资产", "净资产收益率", "每股经营现金流量",
    "销售毛利率", "所处行业", "最新公告日期", "report_period",
]

# 巴菲特财务门槛（阈值与手册原文逐字一致，见 spec/04 序36/42 与 SPEC §3①）
# 元组 = (名称, 派生列, 阈值, 比较符)；阈值与实际值同为百分数
GATE_ITEMS = (
    ("毛利率", "gross_margin", 40.0, "ge"),      # 手册：毛利率 ≥ 40%（spec/04 序36，L1364）
    ("净利率", "net_margin", 20.0, "gt"),        # 手册：净利率 > 20%（spec/04 序42，L1364）
    ("ROE", "roe", 0.0, "gt"),                   # 手册：ROE > 0
    ("营收同比", "rev_yoy", 0.0, "gt"),          # 手册：营收同比 > 0
    ("净利润同比", "profit_yoy", 0.0, "gt"),     # 手册：净利润同比 > 0
)
FIN_GATE_THRESHOLDS = {name: thr for name, _col, thr, _op in GATE_ITEMS}


def _norm_code(code) -> str:
    """统一的 6 位 A 股代码口径（daily_all / yjbb 均为 str6）。"""
    s = str(code).strip()
    if s.endswith(".0"):                      # 防止从 Excel/CSV 带出的 600519.0
        s = s[:-2]
    return s.zfill(6)


def _empty_frame(cols) -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float32") for c in cols})
    return df.set_index(pd.DatetimeIndex([], name="date"))


_EMPTY_DAILY = _empty_frame(_DAILY_COLS)


# ---------------------------------------------------------------------------
# 行情
# ---------------------------------------------------------------------------
@lru_cache(maxsize=512)
def _daily_cached(code: str) -> pd.DataFrame:
    # SPEC §4：按 code 过滤读 parquet 行组，不整表载入（989 万行）
    df = pd.read_parquet(
        _cfg.DAILY_ALL_PARQUET,
        columns=["date"] + _DAILY_COLS,
        filters=[("code", "==", code)],
    )
    if df.empty:
        return _EMPTY_DAILY
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    return df.set_index("date")[_DAILY_COLS]


def load_daily(code: str) -> pd.DataFrame:
    """单只个股前复权日线：index=date 升序，cols=[open, close, volume]。"""
    return _daily_cached(_norm_code(code)).copy()


def load_daily_all() -> dict:
    """一次性载入全部个股（SPEC 明确标注内存较大，供离线批处理用）。

    路径与列序同 load_daily；调用方不要在高频循环里反复调用。
    """
    df = pd.read_parquet(_cfg.DAILY_ALL_PARQUET, columns=["code", "date"] + _DAILY_COLS)
    df = df.drop_duplicates(subset=["code", "date"]).sort_values(["code", "date"], kind="stable")
    return {
        code: g.set_index("date")[_DAILY_COLS]
        for code, g in df.groupby("code", sort=True)
    }


@lru_cache(maxsize=1)
def _index_cached() -> pd.DataFrame:
    df = pd.read_parquet(_cfg.INDEX_PARQUET, columns=["date", "open", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    return df.set_index("date")[["open", "close"]]


def load_index() -> pd.DataFrame:
    """沪深300 日线：index=date 升序，cols=[open, close]（high/low/volume 不在契约内）。"""
    return _index_cached().copy()


@lru_cache(maxsize=1)
def _calendar_cached() -> tuple:
    idx = _index_cached()
    start, end = pd.Timestamp(_cfg.START_DATE), pd.Timestamp(_cfg.END_DATE)
    return tuple(idx.index[(idx.index >= start) & (idx.index <= end)])


def trading_calendar() -> list:
    """交易日历（以沪深300 指数交易日为准），限定在 [START_DATE, END_DATE]。"""
    return list(_calendar_cached())


# ---------------------------------------------------------------------------
# 基本面
# ---------------------------------------------------------------------------
def _derive_fundamentals(raw: pd.DataFrame) -> pd.DataFrame:
    rev = pd.to_numeric(raw["营业总收入-营业总收入"], errors="coerce")
    profit = pd.to_numeric(raw["净利润-净利润"], errors="coerce")
    eps = pd.to_numeric(raw["每股收益"], errors="coerce")
    ocfps = pd.to_numeric(raw["每股经营现金流量"], errors="coerce")
    out = pd.DataFrame({
        "code": raw["股票代码"].map(_norm_code),
        "name": raw["股票简称"].astype(str),
        "report_period": raw["report_period"].astype(str).str.strip().str[:8],
        "ann_date": pd.to_datetime(raw["最新公告日期"], errors="coerce"),
        "eps": eps,
        "revenue": rev,
        "net_profit": profit,
        # 净利率 = 净利润/营收（营收为 0 时留 NaN，不臆造）
        "net_margin": profit / rev.where(rev != 0) * 100.0,
        "gross_margin": pd.to_numeric(raw["销售毛利率"], errors="coerce"),
        "roe": pd.to_numeric(raw["净资产收益率"], errors="coerce"),
        "ocfps": ocfps,
        "rev_yoy": pd.to_numeric(raw["营业总收入-同比增长"], errors="coerce"),
        "profit_yoy": pd.to_numeric(raw["净利润-同比增长"], errors="coerce"),
        "industry": raw["所处行业"].astype(str),
        # 每股净资产：PB 与「内在价值」可得替代的输入（spec/01 序36/38，R-27 降级为定性时仍需 PB 定位）
        "bps": pd.to_numeric(raw["每股净资产"], errors="coerce"),
        # 现金流质量近似 = 每股经营现金流量/每股收益，对应 spec/01 序45⑤「经营现金流须长期为正
        # 且与净利润规模匹配」；EPS 为 0 时留 NaN（除零不臆造），亏损股符号会翻转，须先过盈利门槛
        "ocf_ratio": ocfps / eps.where(eps != 0),
    })
    return out[_FUND_COLS]


@lru_cache(maxsize=1)
def _fundamentals_cached() -> pd.DataFrame:
    """读富字段业绩报表；缺失时退回同目录瘦字段版并打印警告（不崩，缺列由门槛报「数据不足」）。

    为什么允许降级：SPEC §4 要求「缺数据只标注数据不足、不得中断」；外部用户可能只有瘦字段表，
    此时毛利率/净利率等门槛会因缺字段而不通过并登记 missing，而不是抛异常。
    """
    path = _cfg.YJBB_RICH
    if not os.path.isfile(path):
        thin = getattr(_cfg, "YJBB_THIN", os.path.join(_cfg.DATA_DIR, "yjbb.parquet"))
        if os.path.isfile(thin) and os.path.abspath(thin) != os.path.abspath(path):
            print("[警告] 富字段业绩报表缺失（%s），退回瘦字段版 %s："
                  "毛利率/净利率/ROE 等门槛将登记为缺字段" % (path, thin))
            path = thin
        else:
            print("[警告] 业绩报表不可用（%s 与 %s 均不存在）：财务门槛相关规则将全部登记为「数据不足」"
                  % (path, thin))
    raw = pd.read_parquet(path, columns=[c for c in _YJBB_SRC if c != "每股净资产"] or None) \
        if os.path.isfile(path) else pd.DataFrame(columns=_YJBB_SRC)
    return _derive_fundamentals(raw)


def load_fundamentals() -> pd.DataFrame:
    """富字段业绩报表 → 派生列（快照表，(code, report_period) 唯一）。"""
    return _fundamentals_cached().copy()


@lru_cache(maxsize=1)
def _gate_frame() -> pd.DataFrame:
    f = _fundamentals_cached()
    cols = [c for _n, c, _t, _o in GATE_ITEMS]
    return f.set_index(["code", "report_period"])[cols].sort_index()


@lru_cache(maxsize=1)
def _latest_announced_period() -> pd.Series:
    """每只股票最新一期「已公告」报告期：ann_date 非空者中 report_period 最大者。

    供 financial_gate_row 在未指定报告期时兜底（盘中原报期可能尚未公告，故按 ann_date 过滤）。
    """
    f = _fundamentals_cached()
    ann = f[f["ann_date"].notna()]
    return ann.groupby("code")["report_period"].max()


@lru_cache(maxsize=1)
def _annual_yoy() -> pd.DataFrame:
    f = _fundamentals_cached()
    ann = f[f["report_period"].str.endswith("1231")][["code", "report_period", "profit_yoy"]]
    return ann.sort_values(["code", "report_period"]).set_index("code")


def _norm_period(report_period) -> str:
    if isinstance(report_period, str):
        s = report_period.strip()
        if len(s) == 8 and s.isdigit():
            return s
        return pd.Timestamp(s).strftime("%Y%m%d")
    if isinstance(report_period, (int, np.integer)):
        return str(int(report_period))[:8]
    return pd.Timestamp(report_period).strftime("%Y%m%d")


def profit_growth_streak(code: str, periods: int = 10) -> int:
    """连续正增长期数（近似手册「EPS 连续 10 年增长」，spec/04 序43/80 L1364）。

    取年度报告期，按最近年份回溯累计「净利润-同比增长 > 0」的连续期数，上限 periods。
    """
    code = _norm_code(code)
    try:
        sub = _annual_yoy().loc[[code]]
    except KeyError:
        return 0
    n = 0
    for yoy in sub.sort_values("report_period", ascending=False)["profit_yoy"].tolist():
        if pd.isna(yoy) or yoy <= 0:
            break
        n += 1
        if n >= periods:
            break
    return n


def _is_missing_period(report_period) -> bool:
    """None / np.nan / pd.NaT 统一视为「未指定报告期」。

    非标量（数组/列表）交给 _norm_period 按原逻辑处理，这里不接管。
    """
    if report_period is None:
        return True
    try:
        return bool(pd.isna(report_period))
    except (TypeError, ValueError):
        return False


def financial_gate_row(code, report_period=None) -> dict:
    """巴菲特财务门槛逐项判定（SPEC §3①）。

    items: {名称: (阈值, 实际值, 是否通过)}；阈值与实际值同为百分数。
    字段缺失 → 实际值 None、是否通过 False，并记入 missing。
    (code, period) 无记录时五项全部记入 missing（missing==全部名称即「无此记录」）。

    report_period 省略/None/NaT → 自动取该股票最新一期已公告报告期（调用方如
    core/seven_step.py 的 ctx.get("report_period") 常为 None），返回 dict 的
    `report_period` 字段标明实际用了哪期，无已公告记录时为 None；显式传期号时行为与从前一致。
    """
    code6 = _norm_code(code)
    if _is_missing_period(report_period):
        latest = _latest_announced_period().get(code6)
        period = None if latest is None or pd.isna(latest) else str(latest)
    else:
        period = _norm_period(report_period)

    row = None
    if period is not None:
        try:
            row = _gate_frame().loc[(code6, period)]
        except KeyError:
            row = None

    items, missing = {}, []
    for name, col, thr, op in GATE_ITEMS:
        actual = None if row is None else row[col]
        if actual is None or pd.isna(actual):
            missing.append(name)
            items[name] = (thr, None, False)
            continue
        actual = float(actual)
        passed = actual >= thr if op == "ge" else actual > thr
        items[name] = (thr, actual, bool(passed))
    return {"report_period": period,
            "pass": all(v[2] for v in items.values()) and len(missing) == 0,
            "items": items, "missing": missing}


# ---------------------------------------------------------------------------
# 票池
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _bar_counts() -> pd.Series:
    start, end = pd.Timestamp(_cfg.START_DATE), pd.Timestamp(_cfg.END_DATE)
    if _HAS_ARROW:
        tbl = _pq.read_table(_cfg.DAILY_ALL_PARQUET, columns=["code", "date"])
        mask = _pc.and_(
            _pc.greater_equal(tbl["date"], _pa.scalar(start)),
            _pc.less_equal(tbl["date"], _pa.scalar(end)),
        )
        vc = _pc.value_counts(tbl.filter(mask)["code"])
        s = pd.Series({r["values"]: r["counts"] for r in vc.to_pylist()}, name="bars")
    else:  # pragma: no cover
        df = pd.read_parquet(_cfg.DAILY_ALL_PARQUET, columns=["code", "date"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        s = df.groupby("code").size().rename("bars")
    return s.sort_index()


def universe(min_list_days: int = 250) -> list:
    """可用票池：窗口内 bar 数 ≥ min_list_days 的代码（默认约一年交易日）。"""
    counts = _bar_counts()
    return counts.index[counts >= min_list_days].tolist()


if __name__ == "__main__":
    print("== config 路径 ==")
    print("DATA_DIR", _cfg.DATA_DIR, "| YJBB_RICH", _cfg.YJBB_RICH)

    cal = trading_calendar()
    print("== trading_calendar ==", len(cal), cal[0].date(), "->", cal[-1].date())

    d = load_daily("000001")
    print("== load_daily(000001) ==", d.shape, list(d.columns))
    print(d.head(2).to_string())

    idx = load_index()
    print("== load_index ==", idx.shape, list(idx.columns), idx.index[-1].date())

    uni = universe(250)
    print("== universe(250) ==", len(uni), uni[:5])

    f = load_fundamentals()
    print("== load_fundamentals ==", f.shape, list(f.columns))
    print(f.tail(1)[["code", "report_period", "ann_date", "eps", "net_margin",
                     "gross_margin", "roe", "rev_yoy", "profit_yoy"]].to_string(index=False))

    # 600036 连续 10 年正增长；600519/000001 最近一年净利同比转负 → 应为 0
    for c in ("600036", "600519", "000001"):
        print("profit_growth_streak(%s) =" % c, profit_growth_streak(c))

    rp = "20241231"   # 用固定年报期，便于人工复核
    for c in ("600519", "000651"):
        g = financial_gate_row(c, rp)
        print("gate(%s, %s) pass=%s missing=%s" % (c, rp, g["pass"], g["missing"]))
        for k, v in g["items"].items():
            print("   ", k, v)
    nan_row = f[(f["report_period"] == rp) & f["gross_margin"].isna()]
    if len(nan_row):
        c = nan_row["code"].iloc[0]
        print("gate 缺字段样例 gate(%s, %s) missing=%s" % (c, rp, financial_gate_row(c, rp)["missing"]))

    # 新增列 bps / ocf_ratio（主控为 PB 与现金流排雷追加）
    print("== 新增列 ==", [c for c in ("bps", "ocf_ratio") if c in f.columns],
          "| bps 非空 %d | ocf_ratio 非空 %d" % (f["bps"].notna().sum(), f["ocf_ratio"].notna().sum()))

    # report_period 缺省：回归此前 ValueError: NaTType does not support strftime
    for r in (None, pd.NaT):
        g = financial_gate_row("600519", r)
        print("financial_gate_row('600519', %r) -> period=%s pass=%s missing=%s"
              % (r, g["report_period"], g["pass"], g["missing"]))
    print("financial_gate_row('600519') ->", financial_gate_row("600519")["report_period"],
          "| 该股最新已公告期 =", _latest_announced_period().get("600519"))
    print("无记录 code 的缺省期:", financial_gate_row("999999")["report_period"])
    print("gate 无记录样例:", financial_gate_row("600519", "19991231")["missing"])
