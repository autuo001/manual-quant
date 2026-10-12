# -*- coding: utf-8 -*-
"""② data/extra.py —— 尝试获取并登记「数据不足」的外部数据（SPEC 第 3-② 节）

对应手册规则 / 契约：
- SPEC.md 3-②：try_fetch_index_pe / try_fetch_bond_yield / try_fetch_bond_index / availability / earnings_yield
- SPEC.md 2：缺失数据不得臆造，只能标注「数据不足」；路径统一由 config.py 提供
- SPEC.md 4：MQ_OFFLINE=1 不联网；模块 import 时不做耗时工作
- spec/07-数据限制与失败案例.md ②：盈利收益率三档、阈值校准公式(10 年期国债利率)、博格公式、四个估值指标、股债再平衡
- spec/09-验证协议.md ②：V-01 盈利收益率(V-01 阈值 10%／6.4%)、V-02 10 年期国债利率与债券基金指数(区间 2010—2026)、六维② 60/40 基准的债券腿

SPEC 未明确处的取舍（每行一条）：
[1] `pd.DataFrame | None` 在 Python 3.9 无法求值，写成语义等价的 Optional[pd.DataFrame]；
[2] 「45 秒硬超时 + 3 次重试」按字面取 3 次重试（不含首次），即单个数据源最多 4 次尝试；
[3] 「腾讯 → 东财 → 新浪」按同类行情端点的优先级落地：沪深300 PE 在这三家均无 akshare 接口，
    PE 用 乐咕乐股(整体法加权) → 中证指数官网；债券指数三个源 2026-10-12 复测**均能取到**
    （新浪 0.2 秒 5768 行、中债 1.3 秒 6193 行、腾讯 32 秒 5738 行），腾讯此前报 SSLError 属瞬时
    故障而非不可达，故顺序按「秒级优先」取 新浪 → 中债 → 腾讯；实际采用的数据源一律写进
    availability() 的 source 字段；
[4] earnings_yield() 返回 1/PE 的小数口径（0.074 = 7.4%），与 SPEC 字面「E/P 序列（1/PE）」一致，
    需要百分数时用 earnings_yield(pct=True)；
[5] availability() 默认顺带写出 output/data_availability.md（SPEC 要求产出该文件），可用 write_report=False 关闭。
    条目结构按 SPEC 3-② 固定为 {available, source, rows, note} 四键，故 latest_date 不进条目字典，
    只进 md 表格（由模块级 _LATEST 承载），避免破坏 run.py 等下游对该结构的依赖；
[6] 磁盘缓存（data/extra/<name>.parquet + 同名 .meta.json）：SPEC §2 把「指数 PE／国债收益率／债券基金指数」
    列为缺失数据，但三者实测均可取，故成功结果落盘、读取时缓存优先；MQ_OFFLINE=1 亦复用已抓数据
    —— 同时满足 SPEC §4「离线不联网」与 spec/09② V-01（E/P 10%／6.4% 三档）、六维②60/40 债券腿的
    离线复核需求；缓存缺失才登记「数据不足」，绝不臆造。缓存目录可用 MQ_EXTRA_CACHE_DIR 覆盖；
[7] 只缓存成功结果，失败不落盘：失败多为瞬时网络问题，落盘会让下一次离线误判为「数据不足」；
[8] try_fetch_* 增加关键字参数 refresh=False（默认沿用「缓存优先」）。SPEC 只固定无参调用形式，
    默认值不变即对既有调用方完全兼容；refresh=True 供强制联网重取（冒烟测试与人工复核用）；
[9] 债券指数口径（必须随数据登记，不得含糊）：实测取到的是 **上证国债指数 sh000012**（新浪/腾讯源），
    它是上交所上市国债现券按发行量加权的**指数**，与 spec/09② V-02 与六维② 所需的「债券基金指数」
    **不是同一口径**（不含债基申赎/管理费，也不含金融债、可转债、同业存单等持仓）。spec/07②
    「股债再平衡」允许债券端用「债券基金指数**或国债代表资产**」，故它可用于 R-13／60/40，
    但 V-02 的字面口径（2010—2026 债券基金指数）只算部分满足，须在报告与记录表标注为降级替代；
    备选 中债-新综合财富指数（含票息再投资，更接近债基净值口径）列第二顺位。
"""

import io
import json
import os
import signal
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# ---- 路径：优先 config.py（① 所有者），缺失时按 SPEC 的相对默认值自派生，不硬编码绝对路径 ----
_ROOT_DIR = Path(__file__).resolve().parent.parent  # manual_quant/
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
try:
    import config as _config
except ImportError:  # config.py 尚未就绪时本模块仍须可独立加载
    _config = None

_TIMEOUT_SEC = 45  # SPEC 3-②：45 秒硬超时
_RETRY_TIMES = 3  # SPEC 3-②：3 次重试（不含首次）
_EMPTY_FLAG = "数据不足"

# 进程内缓存：导入时不取数，首次调用才联网；失败结果同样缓存，避免复跑重复等待
_FRAME_CACHE: Dict[str, Optional[pd.DataFrame]] = {}
_META_CACHE: Dict[str, dict] = {}
_AKSHARE = None

# 数据项 key → (缓存文件名, 必备列)：磁盘缓存名固定为 SPEC/任务约定的 ASCII 名
_CACHE_NAMES: Dict[str, Tuple[str, str]] = {
    "指数PE(沪深300)": ("index_pe", "pe"),
    "10年期国债收益率": ("bond_yield", "yield_10y"),
    "债券基金/国债指数": ("bond_index", "close"),
}
# 最新日期只在 md 表格里出现（SPEC 3-② 锁定了 availability() 的四键结构）
_LATEST: Dict[str, str] = {}


def _out_dir() -> Path:
    env = os.environ.get("MQ_OUT_DIR")
    if env:
        return Path(env)
    if _config is not None and getattr(_config, "OUT_DIR", None):
        return Path(_config.OUT_DIR)
    return _ROOT_DIR / "output"  # SPEC §2 默认 ../six_layer_backtest/... 的同级替代：manual_quant/output


def _offline() -> bool:
    # SPEC 4：MQ_OFFLINE=1 时不联网；每次读取环境变量，便于同一进程内切换测试
    return os.environ.get("MQ_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def _ak():
    global _AKSHARE
    if _AKSHARE is None:
        import akshare  # 惰性导入：模块 import 阶段不做耗时工作（SPEC 4）

        _AKSHARE = akshare
    return _AKSHARE


def clear_cache() -> None:
    """清空**进程内**缓存（测试或强制重新取数用）；磁盘缓存见 clear_disk_cache()。"""
    _FRAME_CACHE.clear()
    _META_CACHE.clear()
    _LATEST.clear()


# ---------- 磁盘缓存（取舍[6][7]）：抓到的数据落盘，MQ_OFFLINE=1 也能复用 ----------


def _cache_dir() -> Path:
    """缓存目录：MQ_EXTRA_CACHE_DIR 可覆盖，默认 manual_quant/data/extra（本模块自有目录）。"""
    env = os.environ.get("MQ_EXTRA_CACHE_DIR")
    if env:
        return Path(env)
    # 与 config.py 同口径：按本文件位置推导，不写绝对路径（SPEC 4）
    return Path(__file__).resolve().parent / "extra"


def _cache_paths(cache_name: str) -> Tuple[Path, Path]:
    base = _cache_dir() / cache_name
    return base.with_suffix(".parquet"), base.with_suffix(".meta.json")


def _load_cache(cache_name: str, required_col: str) -> Tuple[Optional[pd.DataFrame], dict]:
    """读磁盘缓存；文件缺失/损坏/缺列一律按「未命中」处理，不抛异常。"""
    parquet, sidecar = _cache_paths(cache_name)
    if not parquet.exists():
        return None, {}
    try:
        df = pd.read_parquet(parquet)
        if df.empty or required_col not in df.columns:
            return None, {}
        if not isinstance(df.index, pd.DatetimeIndex):
            # 索引未被 parquet 还原时才退回 date 列；否则宁可判未命中，也不把 RangeIndex 转成 1970 年假日期
            if "date" not in df.columns:
                return None, {}
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]
        if df.empty:
            return None, {}
        meta = {}
        if sidecar.exists():
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        return df.sort_index(), meta
    except Exception:
        return None, {}


def _save_cache(cache_name: str, df: pd.DataFrame, meta: dict) -> Optional[str]:
    """写 parquet + .meta.json（source/口径/抓取时间随之落盘，离线时才能如实登记来源）。

    失败只放弃缓存、不中断流程（SPEC 3-②：失败不得中断）；失败结果不落盘（取舍[7]）。
    """
    try:
        parquet, sidecar = _cache_paths(cache_name)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet)  # pyarrow 已在环境内（pandas 自带 parquet 引擎）
        payload = dict(meta)
        payload.update(
            {
                "cache_file": str(parquet),
                "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "first_date": f"{df.index[0]:%Y-%m-%d}",
                "latest_date": f"{df.index[-1]:%Y-%m-%d}",
                "columns": [str(c) for c in df.columns],
            }
        )
        sidecar.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        return str(parquet)
    except Exception:
        return None


def clear_disk_cache(names: Optional[List[str]] = None) -> List[str]:
    """删除磁盘缓存（默认三个数据项全删），返回被删文件路径；供强制重取数/测试用。"""
    targets = names or [v[0] for v in _CACHE_NAMES.values()]
    removed: List[str] = []
    for cache_name in targets:
        for path in _cache_paths(cache_name):
            if path.exists():
                path.unlink()
                removed.append(str(path))
    return removed


def cache_status() -> Dict[str, dict]:
    """登记磁盘缓存现状（供离线自检与报告用）。"""
    out: Dict[str, dict] = {}
    for key, (cache_name, _col) in _CACHE_NAMES.items():
        parquet, sidecar = _cache_paths(cache_name)
        meta = {}
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        out[key] = {
            "cache_file": str(parquet),
            "exists": parquet.exists(),
            "source": str(meta.get("source", "")),
            "rows": int(meta.get("rows", 0) or 0),
            "latest_date": str(meta.get("latest_date", "")),
            "fetched_at": str(meta.get("fetched_at", "")),
        }
    return out


class FetchTimeout(RuntimeError):
    """硬超时。SPEC 3-②：网络请求必须带 45 秒硬超时。"""


def _call_timed(fn: Callable, *args, **kwargs):
    """带硬超时的调用。

    主线程用 SIGALRM：它能在阻塞 socket 上真正中断（信号处理器抛异常时 PEP 475 不再重试系统调用）；
    非主线程退化为守护线程 join，超时后不回收线程但不会挂住进程退出。
    """
    if threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM"):

        def _on_alarm(signum, frame):
            raise FetchTimeout(f"超过 {_TIMEOUT_SEC} 秒未返回")

        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, _TIMEOUT_SEC)
        try:
            return fn(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    box: Dict[str, object] = {}

    def _run():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 —— 失败必须只登记不中断
            box["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(_TIMEOUT_SEC)
    if worker.is_alive():
        raise FetchTimeout(f"超过 {_TIMEOUT_SEC} 秒未返回")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("value")


def _fetch_with_retry(source: str, fn: Callable, *args, **kwargs) -> Tuple[Optional[pd.DataFrame], str]:
    """SPEC 3-②：失败不得中断流程。返回 (结果 or None, 最后一次失败原因)。"""
    last_error = "未知错误"
    for _ in range(1 + _RETRY_TIMES):  # 取舍[2]：首次 + 3 次重试
        buf = io.StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(buf):  # akshare 的 tqdm/警告不进主输出
                raw = _call_timed(fn, *args, **kwargs)
        except FetchTimeout as exc:
            last_error = f"硬超时: {exc}"
            continue
        except Exception as exc:  # 网络/接口/解析异常一律降级为「数据不足」
            last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            continue
        if raw is None or len(raw) == 0:
            last_error = "接口返回空表"
            continue
        return raw, ""
    return None, last_error


# ---------- 列解析与收尾 ----------


def _num(raw: pd.DataFrame, col: str) -> pd.Series:
    if col not in raw.columns:
        return pd.Series(float("nan"), index=raw.index, dtype="float64")
    return pd.to_numeric(raw[col], errors="coerce").astype("float64")


def _finish(raw: pd.DataFrame) -> pd.DataFrame:
    """统一收尾：DatetimeIndex 升序、按日去重（SPEC 无指定，与 loader 的 index=date 口径一致）。"""
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").set_index("date")
    df.index.name = "date"
    return df


def _freq_label(index) -> str:
    if len(index) < 3:
        return "频率未知"
    gap = pd.Series(index).diff().dt.days.dropna().median()
    return "月频" if gap >= 20 else "日频"


def _index_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """行情类接口（腾讯/新浪/中债）→ 统一列 [close]。

    中债指数接口（bond_new_composite_index_cbond）把指数点位放在 value 列，故一并接受。
    """
    cols = {str(c).lower(): c for c in raw.columns}
    if "date" in cols:
        dates = pd.to_datetime(raw[cols["date"]], errors="coerce")
    else:  # 部分接口把日期放在索引上
        dates = pd.to_datetime(raw.index, errors="coerce")
    close_col = cols.get("close") or cols.get("value")
    if close_col is None:
        raise ValueError(f"缺少 close/value 列：{list(raw.columns)}")
    out = pd.DataFrame({"date": dates, "close": pd.to_numeric(raw[close_col], errors="coerce")})
    return out.dropna(subset=["close"])


# ---------- 数据源候选（顺序即优先级） ----------


def _pe_legulegu() -> pd.DataFrame:
    # 乐咕乐股：2005-04 起月频，含整体法加权市盈率，与 R-23/R-24 盈利收益率的三档阈值同源口径
    raw = _ak().stock_index_pe_lg(symbol="沪深300")
    out = pd.DataFrame({"date": pd.to_datetime(raw["日期"], errors="coerce")})
    out["pe_ttm"] = _num(raw, "滚动市盈率")
    out["pe_static"] = _num(raw, "静态市盈率")
    return out


def _pe_csindex() -> pd.DataFrame:
    # 中证指数官网口径：市盈率2 = 滚动(TTM)、市盈率1 = 静态；接口只返回最近约 20 个交易日
    raw = _ak().stock_zh_index_value_csindex(symbol="000300")
    out = pd.DataFrame({"date": pd.to_datetime(raw["日期"], errors="coerce")})
    out["pe_ttm"] = _num(raw, "市盈率2")
    out["pe_static"] = _num(raw, "市盈率1")
    return out


def _yield_eastmoney() -> pd.DataFrame:
    # 起始 2010-01-01 对应 spec/09-验证协议.md ② V-02 的区间 2010—2026
    raw = _ak().bond_zh_us_rate(start_date="20100101")
    out = pd.DataFrame({"date": pd.to_datetime(raw["日期"], errors="coerce")})
    out["yield_10y"] = _num(raw, "中国国债收益率10年")
    return out


def _yield_cbond() -> pd.DataFrame:
    # 中国债券信息网收益率曲线（备选源，不在 SPEC 列举的腾讯/东财/新浪之内，可能长时间无响应）
    end = datetime.now()
    start = end - timedelta(days=300)  # 接口限制：查询区间须小于一年
    raw = _ak().bond_china_yield(
        start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d")
    )
    df = raw
    if "曲线名称" in df.columns:
        df = df[df["曲线名称"].astype(str).str.contains("国债")]
    out = pd.DataFrame({"date": pd.to_datetime(df["日期"], errors="coerce")})
    out["yield_10y"] = _num(df, "10年")
    return out


def _bond_index_tencent() -> pd.DataFrame:
    return _index_frame(_ak().stock_zh_index_daily_tx(symbol="sh000012"))


def _bond_index_sina() -> pd.DataFrame:
    return _index_frame(_ak().stock_zh_index_daily(symbol="sh000012"))


def _bond_index_cbond() -> pd.DataFrame:
    # 中债-新综合财富指数（中债财富口径含票息再投资，最贴合 R-13 股债再平衡的债券腿）
    return _index_frame(_ak().bond_new_composite_index_cbond(indicator="财富", period="总值"))


# 口径声明（取舍[9]）：必须随数据一起登记，否则下游会把国债指数当债券基金指数用
_BOND_SCOPE_NOTE = (
    "口径：国债/中债现券指数，≠ spec/09② V-02 与六维② 所要求的「债券基金指数」"
    "（不含债基申赎/管理费与金融债、可转债、同业存单等持仓）；spec/07② 允许债券端用"
    "「债券基金指数或国债代表资产」，故可用于 R-13／60/40，但 V-02 字面口径只算部分满足，"
    "属降级替代，须在结果与记录表标注"
)


_PE_CANDIDATES: Tuple[Tuple[str, Callable, str], ...] = (
    ("乐咕乐股(akshare stock_index_pe_lg)", _pe_legulegu, "整体法加权；TTM 优先、缺 TTM 退回静态"),
    ("中证指数官网(akshare stock_zh_index_value_csindex)", _pe_csindex, "市盈率2=滚动(TTM)、市盈率1=静态"),
)
_YIELD_CANDIDATES: Tuple[Tuple[str, Callable, str], ...] = (
    ("东方财富(akshare bond_zh_us_rate)", _yield_eastmoney, "中债国债收益率曲线 10 年期"),
    ("中国债券信息网(akshare bond_china_yield)", _yield_cbond, "国债收益率曲线，仅返回一年内区间"),
)
_BOND_INDEX_CANDIDATES: Tuple[Tuple[str, Callable, str], ...] = (
    # 顺序按「实测耗时」而非 SPEC 列举顺序：2026-10-12 复测三源均可用（新浪 0.2s/5768 行、
    # 中债 1.3s/6193 行、腾讯 32s/5738 行），腾讯只是慢，此前 SSLError 属瞬时故障；
    # 故按秒级优先取 新浪 → 中债 → 腾讯，三源全保留，避免单源失效即判「数据不足」。
    # 口径见 _BOND_SCOPE_NOTE：取到的是国债指数，不是 spec/09② V-02 要的债券基金指数。
    ("新浪财经(akshare stock_zh_index_daily)", _bond_index_sina, f"上证国债指数 sh000012；{_BOND_SCOPE_NOTE}"),
    (
        "中国债券信息网(akshare bond_new_composite_index_cbond)",
        _bond_index_cbond,
        f"中债-新综合财富指数(总值，含票息再投资，更接近债基净值口径)；{_BOND_SCOPE_NOTE}",
    ),
    (
        "腾讯证券(akshare stock_zh_index_daily_tx)",
        _bond_index_tencent,
        f"上证国债指数 sh000012；{_BOND_SCOPE_NOTE}",
    ),
)


def _finalize_pe(df: pd.DataFrame) -> pd.DataFrame:
    df = _finish(df)
    df["pe"] = df["pe_ttm"].combine_first(df["pe_static"])  # 取舍[4] 配套：主口径 TTM
    return df.dropna(subset=["pe"])


def _drop_missing(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return _finish(df).dropna(subset=[col])


def _run_sources(
    key: str,
    label: str,
    candidates: Tuple[Tuple[str, Callable, str], ...],
    finalize: Callable[[pd.DataFrame], pd.DataFrame],
    refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """取数主路径：进程内缓存 → 磁盘缓存 → （离线）登记「数据不足」/（在线）按优先级联网。

    全部数据源失败即登记「数据不足」，不抛异常（SPEC 3-②）。
    取舍[6]：缓存命中即返回 —— MQ_OFFLINE=1 也能用上已抓到的数据；只有缓存缺失才登记「数据不足」。
    """
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]

    cache_name, required_col = _CACHE_NAMES[key]

    if not refresh:
        cached, cache_meta = _load_cache(cache_name, required_col)
        if cached is not None:
            base_note = str(cache_meta.get("note") or f"{label}：来自磁盘缓存")
            _META_CACHE[key] = {
                "available": True,
                "source": str(cache_meta.get("source") or f"磁盘缓存 {cache_name}.parquet"),
                "rows": int(len(cached)),
                "note": f"{base_note}；来自磁盘缓存 {cache_name}.parquet"
                f"（抓取于 {cache_meta.get('fetched_at', '未知时间')}）",
            }
            _LATEST[key] = f"{cached.index[-1]:%Y-%m-%d}"
            _FRAME_CACHE[key] = cached
            return cached

    if _offline():
        _META_CACHE[key] = {
            "available": False,
            "source": "",
            "rows": 0,
            "note": f"{_EMPTY_FLAG}：MQ_OFFLINE=1 且无本地缓存 {cache_name}.parquet"
            f"（SPEC 4 不联网；联网成功过一次后本项即可离线复用）",
        }
        _LATEST.pop(key, None)
        _FRAME_CACHE[key] = None
        return None

    failures: List[str] = []
    for source, fn, hint in candidates:
        raw, err = _fetch_with_retry(source, fn)
        if raw is None:
            failures.append(f"{source} 失败({err})")
            continue
        try:
            df = finalize(raw)
        except Exception as exc:  # 字段改名/结构变化 → 换下一个源
            failures.append(f"{source} 解析失败({type(exc).__name__}: {str(exc)[:80]})")
            continue
        if df is None or df.empty:
            failures.append(f"{source} 无有效行")
            continue
        note = f"{hint}；{_freq_label(df.index)}；覆盖 {df.index[0]:%Y-%m-%d}～{df.index[-1]:%Y-%m-%d}"
        _META_CACHE[key] = {"available": True, "source": source, "rows": int(len(df)), "note": note}
        _LATEST[key] = f"{df.index[-1]:%Y-%m-%d}"
        _FRAME_CACHE[key] = df
        # source/口径随 parquet 一起落盘，离线读缓存时才能如实登记来源（取舍[7]：失败不落盘）
        _save_cache(
            cache_name,
            df,
            {"name": cache_name, "key": key, "label": label, "source": source,
             "rows": int(len(df)), "note": note},
        )
        return df

    _META_CACHE[key] = {
        "available": False,
        "source": "",
        "rows": 0,
        "note": f"{_EMPTY_FLAG}：" + ("；".join(failures) if failures else "无可用数据源"),
    }
    _LATEST.pop(key, None)
    _FRAME_CACHE[key] = None
    return None


# ---------- 对外接口（SPEC 3-②） ----------


def try_fetch_index_pe(refresh: bool = False) -> Optional[pd.DataFrame]:
    """沪深300 市盈率（用于盈利收益率 E/P）。

    返回 index=date、列 [pe, pe_ttm, pe_static]；pe 为主口径（整体法加权 TTM 优先）。
    数据落盘 data/extra/index_pe.parquet，命中即返回（MQ_OFFLINE=1 同样命中）；
    缓存缺失且离线时返回 None 并登记「数据不足」，绝不臆造。refresh=True 强制联网重取。
    """
    return _run_sources(
        "指数PE(沪深300)", "指数PE(沪深300)", _PE_CANDIDATES, _finalize_pe, refresh=refresh
    )


def try_fetch_bond_yield(refresh: bool = False) -> Optional[pd.DataFrame]:
    """10 年期国债收益率。返回 index=date、列 [yield_10y]，单位为百分数（1.68 = 1.68%）。

    数据落盘 data/extra/bond_yield.parquet；供 spec/07② 阈值校准公式（定投线 ≈ max(2×10Y, 6%)、
    卖出线 ≈ 10Y + 2.5%~3%）与 spec/09② V-02（区间 2010—2026）使用。
    """
    return _run_sources(
        "10年期国债收益率",
        "10年期国债收益率",
        _YIELD_CANDIDATES,
        lambda df: _drop_missing(df, "yield_10y"),
        refresh=refresh,
    )


def try_fetch_bond_index(refresh: bool = False) -> Optional[pd.DataFrame]:
    """债券腿指数（供 R-13 股债再平衡与六维②60/40）。返回 index=date、列 [close]。

    数据落盘 data/extra/bond_index.parquet。**口径警告（取舍[9]）**：实测取到的是上证国债指数
    sh000012（或备选中债-新综合财富指数），**不是** spec/09② V-02 所要求的「债券基金指数」；
    spec/07② 允许债券端用「国债代表资产」，故可用于 60/40，但 V-02 只算部分满足，须标注降级替代。
    """
    return _run_sources(
        "债券基金/国债指数",
        "债券基金/国债指数",
        _BOND_INDEX_CANDIDATES,
        lambda df: _drop_missing(df, "close"),
        refresh=refresh,
    )


def earnings_yield(pct: bool = False) -> Optional[pd.Series]:
    """盈利收益率 E/P（SPEC 3-②：1/PE）。

    默认返回小数（0.074 = 7.4%），与 SPEC 字面「E/P 序列（1/PE）」一致；
    取百分数口径（供 V-01／R-23 的 10%／6.4% 三档直接比较）时用 earnings_yield(pct=True)。
    频率与 PE 源一致；PE ≤ 0 的区间口径无效，已剔除。
    """
    pe = try_fetch_index_pe()
    if pe is None or pe.empty or "pe" not in pe.columns:
        return None
    pe = pe["pe"].astype("float64")
    pe = pe[pe > 0]
    if pe.empty:
        return None
    ey = 1.0 / pe
    ey.name = "earnings_yield"
    return ey * 100.0 if pct else ey


# 既有数据缺口（SPEC §2「缺失数据」+ spec/09② 明确无覆盖项），一律登记不臆造
_KNOWN_MISSING: Tuple[Tuple[str, str], ...] = (
    ("股息率", "无长期序列：中证指数官网接口仅返回最近约 20 个交易日，长度不足以回测；影响 spec/07② 四个估值指标、博格公式，spec/09② V-08/V-10"),
    ("资产负债表科目", "缺货币资金/有息负债/其他应收/资本开支等科目；影响 spec/07② 净流动资产策略、债券覆盖倍数与清算价值折扣、财报排雷清单 G（存贷双高）"),
    ("美股数据", "无美股行情与规模/价值因子序列；影响 spec/07② 西格尔美国长期参数、非 A 股市场案例数字，spec/09② V-04"),
    ("通胀/CPI", "无 CPI 序列；影响 spec/09② V-05 实际取用率（4% 取用规则的中国样本自测）"),
)


def availability(write_report: bool = True) -> Dict[str, dict]:
    """登记全部外部数据可用性：名称 → {"available","source","rows","note"}（SPEC 3-②）。

    顺序：进程内缓存 → 磁盘缓存 → 联网（缓存命中或 MQ_OFFLINE=1 时都不联网）。
    条目结构按 SPEC 固定为四键，latest_date 只进 output/data_availability.md 表格。
    默认把结果写入 output/data_availability.md，供 README「已知数据缺口与不可验证项」引用。
    """
    items: Dict[str, dict] = {}

    try_fetch_index_pe()
    items["指数PE(沪深300)"] = dict(
        _META_CACHE.get(
            "指数PE(沪深300)", {"available": False, "source": "", "rows": 0, "note": _EMPTY_FLAG}
        )
    )

    ey = earnings_yield()
    if ey is None or len(ey) == 0:
        items["盈利收益率(E/P)"] = {
            "available": False,
            "source": "",
            "rows": 0,
            "note": f"{_EMPTY_FLAG}：无指数 PE 序列，1/PE 不可得（影响 spec/09② V-01）",
        }
    else:
        src = items["指数PE(沪深300)"].get("source", "")
        items["盈利收益率(E/P)"] = {
            "available": True,
            "source": f"{src}(1/PE)" if src else "1/PE",
            "rows": int(ey.notna().sum()),
            "note": f"小数口径（0.074 = 7.4%）；覆盖 {ey.index[0]:%Y-%m-%d}～{ey.index[-1]:%Y-%m-%d}",
        }

    try_fetch_bond_yield()
    items["10年期国债收益率"] = dict(
        _META_CACHE.get(
            "10年期国债收益率",
            {"available": False, "source": "", "rows": 0, "note": f"{_EMPTY_FLAG}：无可用数据源"},
        )
    )

    try_fetch_bond_index()
    items["债券基金/国债指数"] = dict(
        _META_CACHE.get(
            "债券基金/国债指数",
            {"available": False, "source": "", "rows": 0, "note": f"{_EMPTY_FLAG}：无可用数据源"},
        )
    )

    for name, note in _KNOWN_MISSING:
        items[name] = {"available": False, "source": "", "rows": 0, "note": f"{_EMPTY_FLAG}：{note}"}

    if write_report:
        _write_report(items)
    return items


# ---------- 产出 output/data_availability.md ----------


def _render_report(items: Dict[str, dict]) -> str:
    cached = [k for k, (n, _c) in _CACHE_NAMES.items() if _cache_paths(n)[0].exists()]
    if _offline():
        status = (
            "MQ_OFFLINE=1（不联网；下列结果来自磁盘缓存，缓存缺失项才登记「数据不足」）"
            if cached
            else "MQ_OFFLINE=1（不联网且无磁盘缓存，外部数据一律登记为数据不足）"
        )
    else:
        status = "已读缓存/尝试联网（候选源：乐咕乐股/中证指数官网估值 + 东财国债收益率 + 新浪/中债/腾讯债券指数）"
    cache_line = (
        "- 磁盘缓存：`data/extra/<name>.parquet` + 同名 `.meta.json`（记 source／口径／抓取时间）；"
        "读取时缓存优先，MQ_OFFLINE=1 亦复用；当前已有缓存："
        + ("、".join(f"`{_CACHE_NAMES[k][0]}.parquet`" for k in cached) if cached else "无")
        + "；刷新用 `refresh=True` 或删除缓存文件"
    )
    lines = [
        "# 数据可用性报告（manual_quant / data/extra.py 产出）",
        "",
        f"- 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- 联网状态：{status}",
        f"- 网络约束：{_TIMEOUT_SEC} 秒硬超时 + {_RETRY_TIMES} 次重试，失败只登记不中断（SPEC 第 3-② 节）",
        cache_line,
        "- 候选源顺序：PE 乐咕乐股→中证指数官网；10 年期国债收益率 东财→中国债券信息网；"
        "债券指数 新浪→中债→腾讯（2026-10-12 复测三源均可用：0.2s／1.3s／32s，按耗时排序）",
        "- 口径①：盈利收益率 E/P = 1/PE（小数，0.074 = 7.4%）；PE 主口径为整体法加权滚动(TTM)市盈率",
        f"- 口径②（债券腿，引用时必须一并写出）：{_BOND_SCOPE_NOTE}",
        "- 复现：`python3 manual_quant/data/extra.py`（联网抓取 + 落盘 + 离线复核）或 `run.py check`",
        "",
        "| 数据项 | 可用 | 来源 | 行数 | 最新日期 | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    for name, info in items.items():
        latest = _LATEST.get(name)
        if not latest and name == "盈利收益率(E/P)":
            latest = _LATEST.get("指数PE(沪深300)", "")
        lines.append(
            "| {name} | {ok} | {src} | {rows} | {latest} | {note} |".format(
                name=name,
                ok="是" if info["available"] else "否",
                src=info["source"] or "—",
                rows=info["rows"],
                latest=latest or "—",
                note=str(info["note"]).replace("|", "/"),
            )
        )
    available = [n for n, i in items.items() if i["available"]]
    missing = [n for n, i in items.items() if not i["available"]]
    lines += [
        "",
        "## 汇总",
        "",
        f"- 可用：{('、'.join(available) if available else '无')}",
        f"- 数据不足：{('、'.join(missing) if missing else '无')}",
        "",
        "## 缺口的实际影响（README「已知数据缺口与不可验证项」须与本表一致）",
        "",
    ]
    for name in missing:
        lines.append(f"- **{name}**：{items[name]['note']}")
    if items.get("债券基金/国债指数", {}).get("available"):
        lines += [
            "",
            "## 已可用但属降级口径的项（引用时须连同口径一起写）",
            "",
            f"- **债券基金/国债指数**：{items['债券基金/国债指数']['note']}",
        ]
    lines += [
        "",
        "> 以上缺口项按 SPEC 第 2 节要求只登记、不臆造；相应规则不得据此单独构成买入依据",
        "> （SPEC 第 0 节证据分级闸门：Ⅱ 区 C/D 级不得单独构成买入依据）。",
        "",
    ]
    return "\n".join(lines)


def _write_report(items: Dict[str, dict], path: Optional[str] = None) -> str:
    target = Path(path) if path else (_out_dir() / "data_availability.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_render_report(items), encoding="utf-8")
    return str(target)


def write_availability_report(path: Optional[str] = None) -> str:
    """显式重建 output/data_availability.md，返回文件路径。"""
    return _write_report(availability(write_report=False), path)


if __name__ == "__main__":
    import tempfile
    import time

    print("== data/extra.py 冒烟测试 ==")
    saved_offline = os.environ.get("MQ_OFFLINE")
    saved_cache_dir = os.environ.get("MQ_EXTRA_CACHE_DIR")

    def _restore_env() -> None:
        for name, value in (("MQ_OFFLINE", saved_offline), ("MQ_EXTRA_CACHE_DIR", saved_cache_dir)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    # ① 离线 + 空缓存目录：不联网、全部登记「数据不足」、不抛异常（SPEC 4）
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MQ_EXTRA_CACHE_DIR"] = tmp
        os.environ["MQ_OFFLINE"] = "1"
        clear_cache()
        offline_items = availability(write_report=False)
        assert all(
            set(v) == {"available", "source", "rows", "note"} for v in offline_items.values()
        ), "availability() 字段必须固定为 available/source/rows/note"
        assert not any(v["available"] for v in offline_items.values()), "空缓存 + 离线必须全部「数据不足」"
        assert earnings_yield() is None
        assert try_fetch_bond_index() is None
        print(f"[离线/空缓存] 条目 {len(offline_items)} 个，全部登记为「数据不足」（无网络请求）")

    _restore_env()
    print(f"[缓存目录] {_cache_dir()}")
    online_rows: Dict[str, int] = {}

    # ② 联网抓取 + 落盘；失败只登记不中断，md 以本次结果为准（refresh=True 强制重取）
    if saved_offline in ("1", "true", "yes", "on"):
        print("[联网] 跳过：外部环境已设 MQ_OFFLINE=1")
    else:
        clear_cache()
        t0 = time.time()
        for fn in (try_fetch_index_pe, try_fetch_bond_yield, try_fetch_bond_index):
            fn(refresh=True)
        items = availability(write_report=False)  # 命中进程内缓存，不会二次联网
        path = _write_report(items)
        print(f"[联网] 耗时 {time.time() - t0:.1f}s，报告 {path}")
        for name, info in items.items():
            print(f"  - {name}: 可用={info['available']} 行数={info['rows']} 来源={info['source'] or '—'}")
        online_rows = {
            n: items[n]["rows"] for n in _CACHE_NAMES if items.get(n, {}).get("available")
        }
        ey = earnings_yield(pct=True)
        if ey is None:
            print("  - E/P：数据不足（按 SPEC 记 None，不得臆造）")
        else:
            print(f"  - E/P 尾值 {ey.iloc[-1]:.2f}%（{ey.index[-1]:%Y-%m-%d}，百分数口径；V-01 阈值 10%／6.4%）")
        bond = try_fetch_bond_index()
        if bond is not None:
            print(f"  - 债券指数尾值 {bond['close'].iloc[-1]:.2f}（{bond.index[-1]:%Y-%m-%d}）")
            print(f"  - 债券腿口径：{_BOND_SCOPE_NOTE}")
        for key, st in cache_status().items():
            print(f"  - 缓存 {key}: exists={st['exists']} rows={st['rows']} latest={st['latest_date']}")

    # ③ 离线 + 有缓存：缓存命中即返回数据（本次修正的 offline 语义）。
    # 期望行数优先取本次联网结果，MQ_OFFLINE=1 单跑时退回缓存 sidecar 自己记录的行数。
    expected = {k: st["rows"] for k, st in cache_status().items() if st["exists"] and st["rows"]}
    for name, rows in online_rows.items():
        expected[name] = rows
    if expected:
        os.environ["MQ_OFFLINE"] = "1"
        clear_cache()
        off_items = availability(write_report=False)
        for name, rows in expected.items():
            got = off_items[name]
            assert got["available"] and got["rows"] == rows, f"离线未命中缓存: {name} {got} != {rows}"
            assert set(got) == {"available", "source", "rows", "note"}
            assert got["source"], "离线命中缓存时必须如实带出来源（source 不能为空）"
        assert earnings_yield() is not None, "离线应能用缓存的 PE 算出 E/P"
        assert try_fetch_bond_index() is not None
        print(
            f"[离线/命中缓存] {len(expected)} 项可用且行数与缓存一致："
            + "、".join(f"{n}={expected[n]}" for n in expected)
        )
        print(f"  - 离线读到的债券腿尾值 {try_fetch_bond_index()['close'].iloc[-1]:.2f}")
    else:
        print("[离线/命中缓存] 无缓存可验（先删缓存跑一次联网即可生成）")
    _restore_env()
    print("== 冒烟结束 ==")
