"""④ core/engine.py —— 事件驱动回测引擎（T 日收盘信号 → T+1 开盘成交）。

SPEC 未明确处的取舍（逐条）：
1. daily_all.parquet 只有 open/close，无 high/low：涨跌停只能按「T+1 开盘价是否等于涨跌停价」判定，
   止损只能按「收盘价跌破止损价」判定（无法判盘中触及），前复权价与实际涨跌停价存在近似误差。
2. weight 语义按 SPEC 字面为「目标权重」：买入补足到目标权重、卖出减到目标权重、无 weight 的卖出=清仓；
   缺省时取 1/MAX_POSITIONS（R-16 上限 10 只 → 10%）；A 股按 100 股整数倍取整。
   另留 cfg.weight_mode="delta" 供分批加仓（R-35）类信号使用：weight=本次下单占当前权益比例 / 卖出比例。
3. 停牌（当日无 bar）：默认顺延，最多顺延 5 个交易日后放弃（SPEC 允许"顺延或放弃"二者之一）。
4. 止损只对「买入信号自带 stop_pct」的持仓生效——R-36 属趋势仓；价值仓不得改用止损
   （03-执行与复盘 L584、R-28），故不设全局默认止损。
5. 单标的权重上限不在此处硬编码：R-04 归零口径由 core/risk_budget.max_weight_zero_loss 负责，
   引擎只提供可选的 cfg.max_weight 硬顶。R-16 持仓数（3—10 只）同样不在此处拦截，
   由 core/risk_budget.can_add_position 或策略侧负责（引擎只按 signals 执行）。
6. calendar 缺省时取「首条信号日 → 末条信号日」的指数交易日，再各留 cfg.tail_days=5 个交易日，
   保证末条信号仍有 T+1 可成交、期末清仓有真实收盘价。
7. 返回除 SPEC 三个键外，附加 signals_rejected（未成交/被拒信号审计表）；stats 同时给出
   SPEC 中文键与 ASCII 别名，便于下游按任一写法取用。
8. 期末清仓（liquidate_at_end）按最后一个交易日的收盘价成交（不存在 T+1 开盘价）。
9. 胜率/盈亏比按已平仓卖出笔计算（买入笔 pnl=0）：无亏损笔时盈亏比为 inf，无平仓笔时为 NaN；
   最大回撤输出为正数幅度（0.25 = 回撤 25%）；换手率=Σ成交额/平均权益（累计口径，另给年化值）。
10. A 股 100 股整手：目标金额不足一手时该信号记为「资金不足」并放弃（例：10 万本金买不起一手茅台）。
11. 【排序契约 —— 可复现性的前提】「同一信号集合 → 同一结果」是本引擎的硬约束：独立验证曾对同一批
    value 信号跑出 1.1873 / 1.2354 / 1.3924 / 1.4373 四个总收益（成交 206 与 222 笔），根因是同日信号
    行顺序不稳定（旧代码 `sort_values("date")` 用 pandas 默认 quicksort，不稳定且无同日 tie-breaker），
    顺序一变→现金约束下的拒单与整手取整就跟着变。故引擎对调用方给的**任意行顺序**先重排再执行：
      · 入场排序键 = (date↑, code↑, side：sell 优先于 buy, rule_id↑)，**必须 kind="stable"**
        （见 _SIGNAL_SORT_COLS）；同一天同一票的信号由此获得唯一顺序。
      · 成交阶段按 (到期日↑, sell 优先, 入场序号↑) 处理当日到期订单：同日卖单先成交、卖出现金先到账，
        同日买单再按上面的键顺序依次消耗现金并各自整手取整（`seq` = 入场排序后的全局序号）。
      · 因此 equity 与 trades 是「信号集合」的纯函数：调用方行顺序（含其它模块用 quicksort 预排序、
        并发执行、DataFrame 拼接顺序）都不改变任何数字。修改本文件时不得破坏此契约；
        `_determinism_selfcheck()` 是本契约的回归自检（`python3 core/engine.py` 会跑）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# manual_quant 根目录：从 __file__ 推导，禁止硬编码绝对路径
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

try:  # ① 所有者：data-loader；本模块必须能在 config.py 尚未落地时独立加载
    import config as _config
except ImportError:  # pragma: no cover
    _config = None

_SPEC_DEFAULTS = {
    "DATA_DIR": str(_PKG_ROOT.parent / "six_layer_backtest" / "data" / "parquet"),
    "INITIAL_CAPITAL": 100_000.0,
    "COMMISSION_RATE": 0.00025,
    "COMMISSION_MIN": 5.0,
    "STAMP_TAX": 0.0005,
    "SLIPPAGE": 0.001,
    "LIMIT_PCT_MAIN": 0.10,
    "LIMIT_PCT_GEM": 0.20,
    "MAX_POSITIONS": 10,
    "MIN_POSITIONS": 3,
    "STOP_LOSS_PCT": 0.01,
    "START_DATE": "2016-01-01",
    "END_DATE": "2026-09-30",
}


def _c(name: str):
    """取 config 常量，缺失时回落到 SPEC 默认值（保证模块可独立加载）。"""
    if _config is not None and hasattr(_config, name):
        return getattr(_config, name)
    return _SPEC_DEFAULTS[name]


def _c_path(name: str, default_file: str) -> Path:
    return Path(_c(name)) / default_file


@dataclass
class EngineConfig:
    """全部字段 None 表示取 config.py / SPEC 默认值（SPEC 第 3-④ 节）。"""

    initial_capital: float = None
    commission_rate: float = None
    commission_min: float = None
    stamp_tax: float = None
    slippage: float = None
    limit_pct_main: float = None
    limit_pct_gem: float = None
    lot_size: int = 100
    default_weight: float = None       # 信号未给 weight 时的目标权重
    max_weight: float = None           # 单标的目标权重硬顶（R-04 由 risk_budget 负责，这里只做兜底）
    weight_mode: str = "target"        # SPEC 字面为 target；"delta" 供 R-35 分批加仓信号使用
    stop_loss_pct: float = None        # 仅当信号未带 stop_pct 且此值非 None 时才布防
    honor_stops: bool = True           # R-36 触发即执行
    max_roll_days: int = 5             # 停牌/涨跌停顺延上限，超过即放弃
    tail_days: int = 5                 # 日历自动扩展到末条信号之后的交易天数（留出 T+1 空间）
    limit_tol: float = 1e-4            # 涨跌停判定相对容差（浮点/前复权误差）

    def resolved(self) -> "EngineConfig":
        return EngineConfig(
            initial_capital=_c("INITIAL_CAPITAL") if self.initial_capital is None else self.initial_capital,
            commission_rate=_c("COMMISSION_RATE") if self.commission_rate is None else self.commission_rate,
            commission_min=_c("COMMISSION_MIN") if self.commission_min is None else self.commission_min,
            stamp_tax=_c("STAMP_TAX") if self.stamp_tax is None else self.stamp_tax,
            slippage=_c("SLIPPAGE") if self.slippage is None else self.slippage,
            limit_pct_main=_c("LIMIT_PCT_MAIN") if self.limit_pct_main is None else self.limit_pct_main,
            limit_pct_gem=_c("LIMIT_PCT_GEM") if self.limit_pct_gem is None else self.limit_pct_gem,
            lot_size=self.lot_size,
            default_weight=(1.0 / float(_c("MAX_POSITIONS"))) if self.default_weight is None else self.default_weight,
            max_weight=self.max_weight,
            weight_mode=self.weight_mode,
            stop_loss_pct=self.stop_loss_pct,
            honor_stops=self.honor_stops,
            max_roll_days=self.max_roll_days,
            tail_days=self.tail_days,
            limit_tol=self.limit_tol,
        )


# ---------------------------------------------------------------- 价格访问（惰性）

_PRICE_CACHE: dict = {}


def _bar_file() -> Path:
    if _config is not None and hasattr(_config, "DAILY_ALL_PARQUET"):
        return Path(_config.DAILY_ALL_PARQUET)
    return _c_path("DATA_DIR", "daily_all.parquet")


def _index_file() -> Path:
    if _config is not None and hasattr(_config, "INDEX_PARQUET"):
        return Path(_config.INDEX_PARQUET)
    return _c_path("DATA_DIR", "index.parquet")


class _Book:
    """单只标的的 bar 序列，提供 O(1)/O(log n) 的按日查询。"""

    __slots__ = ("dates", "open", "close", "pos")

    def __init__(self, df: pd.DataFrame):
        df = df.sort_values("date")
        self.dates = df["date"].to_numpy(dtype="datetime64[ns]")
        self.open = df["open"].to_numpy(dtype="float64")
        self.close = df["close"].to_numpy(dtype="float64")
        self.pos = {d: i for i, d in enumerate(df["date"])}

    def idx(self, d) -> int | None:
        return self.pos.get(d)

    def prev_close(self, i: int) -> float | None:
        return float(self.close[i - 1]) if i > 0 else None

    def close_on_or_before(self, d) -> float | None:
        # 停牌日沿用最后一个可得收盘价（估值不中断）
        j = int(np.searchsorted(self.dates, np.datetime64(d), side="right")) - 1
        return None if j < 0 else float(self.close[j])


def _load_books(codes: set, start, end) -> dict:
    """惰性加载：只切片 signals 涉及的 code（SPEC §4 禁止全表 concat）。"""
    key = (tuple(sorted(codes)), pd.Timestamp(start), pd.Timestamp(end))
    cached = _PRICE_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_PRICE_CACHE) > 8:  # 六维协议会连跑多组区间/成本档，避免缓存无界增长
        _PRICE_CACHE.clear()
    path = _bar_file()
    if not codes or not path.exists():
        _PRICE_CACHE[key] = {}
        return {}
    df = pd.read_parquet(
        str(path), columns=["code", "date", "open", "close"],
        filters=[("code", "in", sorted(codes))],
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    books = {c: _Book(g) for c, g in df.groupby("code", sort=False)}
    _PRICE_CACHE[key] = books
    return books


def trading_calendar(start=None, end=None, tail_days: int = 0) -> list:
    """以指数交易日为准（SPEC ① data/loader.trading_calendar 同口径）。"""
    path = _index_file()
    if not path.exists():
        return []
    idx = pd.read_parquet(str(path), columns=["date"])
    days = sorted(pd.to_datetime(idx["date"]).tolist())
    if start is not None:
        days = [d for d in days if d >= pd.Timestamp(start)]
    if end is not None:
        keep = [d for d in days if d <= pd.Timestamp(end)]
        rest = [d for d in days if d > pd.Timestamp(end)][: max(0, int(tail_days))]
        days = keep + rest
    return days


# ---------------------------------------------------------------- 成交规则

def _limit_pct(code: str, cfg: EngineConfig) -> float:
    # SPEC 只给两档：主板 ±10%、创业板/科创板（300/688）±20%
    prefixes = _c("GEM_PREFIXES") if _config is not None and hasattr(_config, "GEM_PREFIXES") else ("300", "688")
    return cfg.limit_pct_gem if str(code)[:3] in tuple(prefixes) else cfg.limit_pct_main


def _at_limit_up(open_px: float, prev_close: float | None, code: str, cfg: EngineConfig) -> bool:
    if prev_close is None or prev_close <= 0:
        return False
    cap = prev_close * (1.0 + _limit_pct(code, cfg))
    return open_px >= cap - max(0.01, cap * cfg.limit_tol)


def _at_limit_down(open_px: float, prev_close: float | None, code: str, cfg: EngineConfig) -> bool:
    if prev_close is None or prev_close <= 0:
        return False
    floor_ = prev_close * (1.0 - _limit_pct(code, cfg))
    return open_px <= floor_ + max(0.01, floor_ * cfg.limit_tol)


def _fee(amount: float, side: str, cfg: EngineConfig) -> float:
    # 佣金万2.5（最低 5 元）＋卖出印花税 0.05%
    if amount <= 0:
        return 0.0
    comm = max(amount * cfg.commission_rate, cfg.commission_min)
    tax = amount * cfg.stamp_tax if side == "sell" else 0.0
    return comm + tax


def _affordable_lots(cash: float, price: float, want_shares: int, cfg: EngineConfig) -> int:
    lot = int(cfg.lot_size)
    if price <= 0 or want_shares < lot:
        return 0
    est = int(cash / (price * (1.0 + cfg.commission_rate)))
    shares = min(int(want_shares), int(est // lot) * lot)
    while shares >= lot and shares * price + _fee(shares * price, "buy", cfg) > cash:
        shares -= lot
    return max(0, shares)


# ---------------------------------------------------------------- 主流程

def _normalize_signals(signals: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    notes = []
    if signals is None or len(signals) == 0:
        return pd.DataFrame(columns=["date", "code", "side", "reason", "rule_id", "weight", "stop_pct"]), ["信号为空"]
    df = signals.copy()
    for col in ("date", "code", "side"):
        if col not in df.columns:
            raise ValueError(f"signals 缺少必需列：{col}")
    for col, default in (("reason", ""), ("rule_id", ""), ("weight", np.nan), ("stop_pct", np.nan)):
        if col not in df.columns:
            df[col] = default
            notes.append(f"signals 缺少可选列 {col}，已置默认值")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # 容错：code 可能是 int/float（如 1.0），统一成 6 位字符串
    df["code"] = df["code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    side_map = {"buy": "buy", "sell": "sell", "买": "buy", "卖": "sell", "买入": "buy", "卖出": "sell"}
    df["side"] = df["side"].astype(str).str.strip().str.lower().map(lambda s: side_map.get(s, s))
    bad = ~df["side"].isin(["buy", "sell"])
    if bad.any():
        notes.append(f"丢弃 {int(bad.sum())} 条 side 非 buy/sell 的信号")
        df = df[~bad]
    # 排序契约（文件头第 11 条）：按「内容」确定性排序 → 调用方行顺序不影响结果。
    # 键 = date↑, code↑, sell 优先, rule_id↑, weight↑, stop_pct↑, reason↑；全部列都进键，
    # 于是「顺序歧义」只剩完全重复行，再用 drop_duplicates 消掉 → 顺序唯一。
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["stop_pct"] = pd.to_numeric(df["stop_pct"], errors="coerce")
    df["reason"] = df["reason"].astype(str)
    df["rule_id"] = df["rule_id"].astype(str)
    df["_side_pri"] = (df["side"] != "sell").astype(int)      # 0=sell 先, 1=buy 后
    key = ["date", "code", "_side_pri", "rule_id", "weight", "stop_pct", "reason"]
    df = (df.sort_values(key, kind="stable", na_position="last")
            .drop_duplicates(subset=["date", "code", "side", "rule_id", "weight", "stop_pct", "reason"])
            .drop(columns=["_side_pri"])
            .reset_index(drop=True))
    df["seq"] = np.arange(len(df))                            # 入场序号：同日现金分配的确定顺序
    return df, notes


def _empty_equity(cal, capital: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"date": list(cal), "cash": capital, "market_value": 0.0,
         "equity": capital, "exposure": 0.0, "n_positions": 0}
    )


def run_backtest(signals: pd.DataFrame, calendar: list | None = None,
                 cfg: EngineConfig = None, liquidate_at_end: bool = True) -> dict:
    cfg = (cfg or EngineConfig()).resolved()
    sig, notes = _normalize_signals(signals)
    rejected: list = []

    if calendar is None:
        if len(sig):
            calendar = trading_calendar(sig["date"].min(), sig["date"].max(), tail_days=cfg.tail_days)
        else:
            calendar = trading_calendar(_c("START_DATE"), _c("END_DATE"))
    cal = sorted({pd.Timestamp(d) for d in calendar})
    if not cal:
        res = {"equity": _empty_equity([], cfg.initial_capital),
               "trades": _empty_trades(), "stats": _stats(_empty_equity([], cfg.initial_capital), cfg, []),
               "signals_rejected": pd.DataFrame(),
               "notes": notes + ["日历为空：指数数据不可用或区间无交易日"]}
        return res

    books = _load_books(set(sig["code"]) if len(sig) else set(), cal[0], cal[-1])
    # 无任何 bar 的代码（错码/未上市/已退市）与日历外日期当场拒绝，不占用停牌顺延
    no_bar = [c for c in sig["code"].unique() if c not in books]
    if no_bar:
        notes.append(f"{len(no_bar)} 个代码在行情中无数据，已拒绝：{sorted(no_bar)[:5]}")
    keep_mask = sig["code"].isin(books.keys()) & sig["date"].isin(cal)
    for row in sig[~keep_mask].itertuples(index=False):
        why = "行情无该代码" if row.code not in books else "信号日期不在交易日历内"
        rejected.append({"date": row.date, "code": row.code, "side": row.side, "reason": row.reason,
                         "rule_id": row.rule_id, "reject_reason": why})
    sig = sig[keep_mask]
    by_day: dict = {}
    for row in sig.itertuples(index=False):
        by_day.setdefault(row.date, []).append(row)

    cash = float(cfg.initial_capital)
    positions: dict = {}      # code -> {shares, cost_amount, avg_cost, last_buy_price, stop_price, stop_pct, entry_date}
    pending: list = []
    trades: list = []
    rows: list = []
    next_day = {d: cal[i + 1] for i, d in enumerate(cal[:-1])}

    for d in cal:
        # ---- 1) T+1 开盘成交（含停牌/涨跌停顺延）
        # 执行顺序契约：到期日↑ → 卖出优先（卖出现金先到账）→ 入场序号↑（文件头第 11 条）
        pending.sort(key=lambda od: (od["due"], 0 if od["side"] == "sell" else 1, od["seq"]))
        keep: list = []
        for od in pending:
            if od["due"] > d:
                keep.append(od)
                continue
            book = books.get(od["code"])
            i = book.idx(d) if book else None
            if i is None:
                if od["rolls"] < cfg.max_roll_days and d in next_day:
                    od["rolls"] += 1
                    od["due"] = next_day[d]
                    keep.append(od)
                else:
                    rejected.append(_reject(od, d, "停牌：顺延超期，放弃"))
                continue
            open_px = float(book.open[i])
            if not np.isfinite(open_px) or open_px <= 0:
                rejected.append(_reject(od, d, "无效开盘价"))
                continue
            px = open_px * (1.0 + cfg.slippage) if od["side"] == "buy" else open_px * (1.0 - cfg.slippage)
            prev_close = book.prev_close(i)
            if od["side"] == "buy":
                if _at_limit_up(open_px, prev_close, od["code"], cfg):
                    # R-34/附录二：开盘一字涨停买不到，顺延或放弃
                    if od["rolls"] < cfg.max_roll_days and d in next_day:
                        od["rolls"] += 1
                        od["due"] = next_day[d]
                        keep.append(od)
                    else:
                        rejected.append(_reject(od, d, "开盘涨停：顺延超期，放弃"))
                    continue
                pos = positions.get(od["code"])
                held = pos["shares"] if pos else 0
                equity_now = cash + _market_value(positions, books, d)
                sized = int(od["weight"] * equity_now / px // cfg.lot_size) * cfg.lot_size if od["weight"] else 0
                want = sized if cfg.weight_mode == "delta" else max(0, sized - held)  # target：补足到目标权重
                shares = _affordable_lots(cash, px, want, cfg)
                if shares <= 0:
                    rejected.append(_reject(od, d, "资金不足或已达目标权重"))
                    continue
                amount = shares * px
                fee = _fee(amount, "buy", cfg)
                cash -= amount + fee
                if pos is None:
                    positions[od["code"]] = {
                        "shares": shares, "cost_amount": amount + fee, "avg_cost": (amount + fee) / shares,
                        "last_buy_price": px, "stop_price": None, "stop_pct": None, "entry_date": d,
                    }
                else:
                    pos["shares"] += shares
                    pos["cost_amount"] += amount + fee
                    pos["avg_cost"] = pos["cost_amount"] / pos["shares"]
                    pos["last_buy_price"] = px
                pos = positions[od["code"]]
                if od["stop_pct"] is not None:  # R-36 最后一笔买入价下方 stop_pct
                    pos["stop_pct"] = float(od["stop_pct"])
                    pos["stop_price"] = pos["last_buy_price"] * (1.0 - pos["stop_pct"])
                elif cfg.stop_loss_pct is not None and pos["stop_price"] is None:
                    pos["stop_pct"] = float(cfg.stop_loss_pct)
                    pos["stop_price"] = pos["last_buy_price"] * (1.0 - pos["stop_pct"])
                trades.append(_trade(d, od, px, shares, amount, fee, 0.0))
            else:
                if _at_limit_down(open_px, prev_close, od["code"], cfg):
                    if od["rolls"] < cfg.max_roll_days and d in next_day:
                        od["rolls"] += 1
                        od["due"] = next_day[d]
                        keep.append(od)
                    else:
                        rejected.append(_reject(od, d, "开盘跌停：顺延超期，放弃"))
                    continue
                pos = positions.get(od["code"])
                if pos is None or pos["shares"] <= 0:
                    rejected.append(_reject(od, d, "无持仓可卖"))
                    continue
                held = pos["shares"]
                if od["weight"] and cfg.weight_mode == "delta":
                    shares = int(od["weight"] * held)  # delta：weight 表示卖出持仓的比例
                elif od["weight"]:
                    equity_now = cash + _market_value(positions, books, d)
                    target = int(od["weight"] * equity_now / px // cfg.lot_size) * cfg.lot_size
                    shares = max(0, held - target)
                else:
                    shares = held
                if shares <= 0:
                    rejected.append(_reject(od, d, "目标权重≥现持仓，无需卖出"))
                    continue
                amount = shares * px
                fee = _fee(amount, "sell", cfg)
                pnl = (px - pos["avg_cost"]) * shares - fee
                cash += amount - fee
                pos["shares"] -= shares
                pos["cost_amount"] -= pos["avg_cost"] * shares
                if pos["shares"] <= 0:
                    positions.pop(od["code"], None)
                trades.append(_trade(d, od, px, shares, amount, fee, pnl))
        pending = keep

        # ---- 2) 收盘估值
        mv = _market_value(positions, books, d)
        equity = cash + mv
        rows.append({"date": d, "cash": cash, "market_value": mv, "equity": equity,
                     "exposure": (mv / equity) if equity > 0 else 0.0, "n_positions": len(positions)})

        # ---- 3) 收盘后：先登记 T 日信号（→T+1），再检查止损（R-36 触发即执行，同样次日开盘）
        queued = {od["code"] for od in pending}
        for row in by_day.get(d, []):
            if row.code in queued and row.side == "sell":
                continue
            weight = None if pd.isna(row.weight) else float(row.weight)
            if weight is None and row.side == "buy":
                weight = cfg.default_weight
            if weight is not None and cfg.max_weight is not None:
                weight = min(weight, cfg.max_weight)
            stop_pct = None if pd.isna(row.stop_pct) else float(row.stop_pct)
            if d not in next_day:
                rejected.append({"date": d, "code": row.code, "side": row.side, "reason": row.reason,
                                 "rule_id": row.rule_id, "reject_reason": "日历末尾无 T+1，无法成交"})
                continue
            pending.append({"code": row.code, "side": row.side, "due": next_day[d], "rolls": 0,
                            "seq": int(row.seq),
                            "reason": row.reason, "rule_id": row.rule_id, "weight": weight, "stop_pct": stop_pct})
            queued.add(row.code)

        if cfg.honor_stops and d in next_day:
            for code, pos in list(positions.items()):
                if pos["stop_price"] is None or code in queued:
                    continue
                book = books.get(code)
                close_px = book.close_on_or_before(d) if book else None
                if close_px is not None and close_px <= pos["stop_price"]:
                    pending.append({"code": code, "side": "sell", "due": next_day[d], "rolls": 0,
                                    "seq": 10 ** 9 + len(pending),   # 止损单排在当日所有信号单之后
                                    "reason": "stop_loss", "rule_id": "R-36", "weight": None, "stop_pct": None})
                    queued.add(code)

    equity_df = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades, columns=_TRADE_COLS) if trades else _empty_trades()

    # ---- 4) 期末清仓（不存在 T+1 开盘价 → 用最后一日收盘价）
    if liquidate_at_end and positions:
        d = cal[-1]
        for code, pos in list(positions.items()):
            book = books.get(code)
            close_px = book.close_on_or_before(d) if book else None
            if close_px is None:
                rejected.append({"date": d, "code": code, "side": "sell", "reason": "liquidate_at_end",
                                 "rule_id": "", "reject_reason": "期末无收盘价，无法清仓"})
                continue
            px = float(close_px) * (1.0 - cfg.slippage)
            shares = pos["shares"]
            amount = shares * px
            fee = _fee(amount, "sell", cfg)
            pnl = (px - pos["avg_cost"]) * shares - fee
            cash += amount - fee
            row = {"date": d, "code": code, "side": "sell", "price": px, "shares": float(shares),
                   "amount": amount, "fee": fee, "pnl": pnl, "reason": "liquidate_at_end", "rule_id": ""}
            trades.append(row)
            positions.pop(code, None)
        trades_df = pd.DataFrame(trades, columns=_TRADE_COLS)
        equity_df.loc[equity_df.index[-1], ["cash", "market_value", "equity", "exposure", "n_positions"]] = [
            cash, 0.0, cash, 0.0, 0]

    stats = _stats(equity_df, cfg, trades_df)
    stats["notes"] = notes
    return {
        "equity": equity_df,
        "trades": trades_df,
        "stats": stats,
        "signals_rejected": pd.DataFrame(rejected) if rejected else pd.DataFrame(
            columns=["date", "code", "side", "reason", "rule_id", "reject_reason"]),
    }


_TRADE_COLS = ["date", "code", "side", "price", "shares", "amount", "fee", "pnl", "reason", "rule_id"]


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=_TRADE_COLS)


def _trade(d, od, price, shares, amount, fee, pnl) -> dict:
    return {"date": d, "code": od["code"], "side": od["side"], "price": price, "shares": float(shares),
            "amount": amount, "fee": fee, "pnl": pnl, "reason": od["reason"], "rule_id": od["rule_id"]}


def _determinism_selfcheck(sig: pd.DataFrame, n: int = 3) -> None:
    """排序契约回归自检：同一信号集用不同行顺序重跑，结果必须逐位相同。

    这是「同一信号集合 → 同一结果」的唯一守卫（文件头第 11 条）。
    """
    rng = np.random.default_rng(20261012)
    base = run_backtest(sig, cfg=EngineConfig())
    base_eq = base["equity"]["equity"].to_numpy()
    base_rets = [base["stats"]["总收益"]]
    for i in range(n - 1):
        order = rng.permutation(len(sig))
        shuffled = sig.iloc[order].reset_index(drop=True)
        out = run_backtest(shuffled, cfg=EngineConfig())
        eq = out["equity"]["equity"].to_numpy()
        assert len(eq) == len(base_eq) and np.allclose(eq, base_eq, rtol=0, atol=0), \
            f"排序契约被破坏：第 {i + 1} 次打乱后权益曲线不同"
        t1, t2 = base["trades"], out["trades"]
        assert len(t1) == len(t2), f"排序契约被破坏：成交笔数 {len(t1)} vs {len(t2)}"
        base_rets.append(out["stats"]["总收益"])
    print(f"[determinism] 打乱 {n} 次的总收益均为 " +
          " / ".join(f"{r:.6f}" for r in base_rets) + " → 排序契约通过")


def _reject(od, d, why) -> dict:
    return {"date": d, "code": od["code"], "side": od["side"], "reason": od["reason"],
            "rule_id": od["rule_id"], "reject_reason": why}


def _market_value(positions: dict, books: dict, d) -> float:
    mv = 0.0
    for code, pos in positions.items():
        book = books.get(code)
        px = book.close_on_or_before(d) if book else None
        if px is not None:
            mv += pos["shares"] * px
    return mv


def _stats(equity: pd.DataFrame, cfg: EngineConfig, trades: pd.DataFrame) -> dict:
    """观测口径优先级照抄附录六：最大回撤/波动/换手率/资金占用/策略失效期 ＞ 收益率。"""
    nan = float("nan")
    if equity is None or len(equity) == 0:
        return {"总收益": nan, "年化": nan, "最大回撤": nan, "波动(年化)": nan, "换手率": nan,
                "资金占用": nan, "策略失效期": nan, "交易笔数": 0, "胜率": nan, "盈亏比": nan,
                "total_return": nan}
    eq = equity["equity"].to_numpy(dtype="float64")
    n = len(eq)
    total_return = eq[-1] / cfg.initial_capital - 1.0 if cfg.initial_capital else nan
    years = max((n - 1) / 252.0, 1e-9)
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if 1.0 + total_return > 0 else -1.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    max_drawdown = float(-dd.min()) if n else nan  # 正数幅度，如 0.25 = 回撤 25%
    rets = pd.Series(eq).pct_change().dropna()
    ann_vol = float(rets.std(ddof=1) * np.sqrt(252.0)) if len(rets) > 1 else nan
    turnover_amt = float(trades["amount"].sum()) if len(trades) else 0.0
    avg_equity = float(eq.mean()) if n else nan
    turnover = turnover_amt / avg_equity if avg_equity else nan
    occupation = float(equity["exposure"].mean())
    under = eq < peak
    longest, cur = 0, 0
    for flag in under:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    sells = trades[trades["side"] == "sell"] if len(trades) else _empty_trades()
    closed = sells["pnl"].to_numpy(dtype="float64") if len(sells) else np.array([])
    wins = closed[closed > 0]
    losses = closed[closed < 0]
    win_rate = float(len(wins) / len(closed)) if len(closed) else nan
    if len(losses) and len(wins):
        pl_ratio = float(wins.mean() / abs(losses.mean()))
    elif len(wins) and not len(losses):
        pl_ratio = float("inf")
    else:
        pl_ratio = nan
    stats = {
        "总收益": total_return, "年化": annual_return, "最大回撤": max_drawdown, "波动(年化)": ann_vol,
        "换手率": turnover, "资金占用": occupation, "策略失效期": float(longest),
        "交易笔数": int(len(trades)), "胜率": win_rate, "盈亏比": pl_ratio,
        "total_return": total_return, "annual_return": annual_return, "max_drawdown": max_drawdown,
        "annual_volatility": ann_vol, "turnover": turnover, "capital_occupation": occupation,
        "longest_underwater": float(longest), "n_trades": int(len(trades)),
        "n_round_trips": int(len(closed)), "win_rate": win_rate, "profit_loss_ratio": pl_ratio,
        "final_equity": float(eq[-1]), "initial_capital": float(cfg.initial_capital),
        "n_days": int(n), "years": years,
        "current_underwater_days": float(cur), "turnover_annual": turnover / years if years else nan,
        "total_fee": float(trades["fee"].sum()) if len(trades) else 0.0,
    }
    return stats


if __name__ == "__main__":
    import time

    t0 = time.time()
    cfg0 = EngineConfig()
    # 涨跌停判定为纯函数，先自证（主板 ±10%、创业板/科创板 ±20%）
    assert _at_limit_up(11.0, 10.0, "000001", cfg0.resolved())
    assert not _at_limit_up(10.9, 10.0, "000001", cfg0.resolved())
    assert not _at_limit_up(10.9, 10.0, "300750", cfg0.resolved())   # 创业板 20% 上限
    assert _at_limit_up(12.0, 10.0, "300750", cfg0.resolved())
    assert _at_limit_down(9.0, 10.0, "000001", cfg0.resolved())
    print("[smoke] 涨跌停判定：主板/创业板两档 OK")

    cal = trading_calendar("2024-01-01", "2024-12-31")
    print(f"[smoke] 日历 {len(cal)} 个交易日 {cal[0].date()} → {cal[-1].date()}" if cal else "[smoke] 日历为空")

    sig = pd.DataFrame([
        # ① 价值仓：买入信号不带 stop_pct → 引擎不布防（R-28 三个卖出理由才是退出信号）
        {"date": "2024-01-02", "code": "000001", "side": "buy", "weight": 0.20,
         "reason": "value_entry", "rule_id": "R-04+R-16"},
        {"date": "2024-06-05", "code": "000001", "side": "sell", "reason": "better_idea", "rule_id": "R-28"},
        # ② 趋势仓：2024-01-02 收盘出信号，T+1（01-03 开盘 7.56）成交后随即跌破 1% → 应触发 R-36 止损
        {"date": "2024-01-02", "code": "601318", "side": "buy", "weight": 0.20, "stop_pct": 0.01,
         "reason": "breakout", "rule_id": "R-34+R-35"},
        # ③ 2024-09-30 收盘出信号 → T+1 = 2024-10-08 一字涨停开盘 → 买不到（顺延/放弃）
        {"date": "2024-09-30", "code": "601318", "side": "buy", "weight": 0.20, "stop_pct": 0.01,
         "reason": "breakout", "rule_id": "R-34"},
        # ⑤ 期末仍持仓 → 应走 liquidate_at_end 清仓（注：茅台一手 15 万，10 万本金买不起，
        #    会被记为「资金不足或已达目标权重」，故此处用价格较低的标的）
        {"date": "2024-11-01", "code": "600036", "side": "buy", "weight": 0.20,
         "reason": "value_entry", "rule_id": "R-04"},
        # ④ 错码 → 应进 signals_rejected
        {"date": "2024-03-01", "code": "999999", "side": "buy", "weight": 0.10,
         "reason": "bad_code", "rule_id": "R-04"},
    ])
    try:
        out = run_backtest(sig, cfg=EngineConfig())
        st = out["stats"]
        tr = out["trades"]
        print(f"[smoke] 交易笔数={st['交易笔数']} 期末权益={st['final_equity']:.0f} "
              f"总收益={st['总收益']:.2%} 最大回撤={st['最大回撤']:.2%} 波动={st['波动(年化)']:.2%} "
              f"资金占用={st['资金占用']:.2%} 换手率={st['换手率']:.2f} 策略失效期={st['策略失效期']:.0f}日 "
              f"胜率={st['胜率']}")
        print("[smoke] trades:\n" + tr.to_string(index=False))
        print("[smoke] equity head:\n" + out["equity"].head(3).to_string(index=False))
        print("[smoke] rejected:\n" + out["signals_rejected"].to_string(index=False))
        assert st["n_trades"] >= 2, "应有成交"
        assert (tr["side"] == "sell").sum() >= 1
        assert "stop_loss" in set(tr["reason"]), "R-36 止损未触发"
        assert (tr.loc[tr["reason"] == "stop_loss", "rule_id"] == "R-36").all()
        assert "999999" in set(out["signals_rejected"]["code"]), "错码应被拒绝"
        assert "liquidate_at_end" in set(tr["reason"]), "期末清仓未执行"
        print("[smoke] 断言全部通过：R-36 止损触发、错码拒绝、T+1 开盘成交、期末清仓")

        # 排序契约回归自检（文件头第 11 条）：同一信号集，任意行顺序必须给出逐位相同的结果
        _determinism_selfcheck(sig)

        # 顺延/放弃两种口径：max_roll_days=0 时涨停开盘必须直接放弃
        strict = run_backtest(sig, cfg=EngineConfig(max_roll_days=0), liquidate_at_end=False)
        rej = strict["signals_rejected"]
        print("[smoke] max_roll_days=0 时被拒：\n" + rej.to_string(index=False))
        assert rej["reject_reason"].str.contains("涨停").any(), "一字涨停应被拒绝"
        late = out["trades"]
        print(f"[smoke] 默认口径下 601318 成交日期："
              f"{sorted(late.loc[late['code'] == '601318', 'date'].dt.date.unique())}（涨停日 2024-10-08 顺延）")
    except FileNotFoundError as e:  # 数据未就绪时不得中断加载
        print(f"[smoke] 跳过（数据缺失）：{e}")
    print(f"[smoke] 耗时 {time.time() - t0:.1f}s")
