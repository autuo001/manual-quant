# -*- coding: utf-8 -*-
"""manual_quant 全局配置（SPEC §3①）——所有路径与数值阈值的唯一出处。

取舍与补充说明（SPEC 有歧义处按最贴近字面的做法；见 ① 所有者：data-loader 备注）：
1. SPEC 只规定 `MQ_DATA_DIR` / `MQ_OUT_DIR` 两个环境变量；本文件另导出
   `DAILY_ALL_PARQUET` / `INDEX_PARQUET` / `YJBB_THIN` / `GEM_PREFIXES` / `OFFLINE`，
   目的是落实 SPEC §2「路径统一由 config.py 提供」与 §4「MQ_OFFLINE=1 不联网」，
   避免各模块各自拼路径、各自发明前缀表。
2. 默认路径按 config.py 自身位置推导（不是硬编码绝对路径），故 cwd 变化不影响导入。
3. OUT_DIR 在 import 时创建：SPEC §1 目录树把它列为运行产物目录，各模块直接
   `open(OUT_DIR/...)` 写文件，目录缺失会直接抛错；创建开销可忽略，不违反
   「import 时不做耗时工作」。
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径（环境变量可覆盖）
# ---------------------------------------------------------------------------
_MQ_DIR = Path(__file__).resolve().parent      # manual_quant/
_PROJECT_DIR = _MQ_DIR.parent                  # 项目根

# SPEC §2：个股/指数日线数据根目录，默认 ../six_layer_backtest/data/parquet
DATA_DIR: str = os.environ.get("MQ_DATA_DIR") or str(
    (_PROJECT_DIR / "six_layer_backtest" / "data" / "parquet").resolve()
)
# SPEC §2：业绩报表必须用富字段版本（瘦字段缺财务字段，勿用）
YJBB_RICH: str = str((_PROJECT_DIR / "backtest" / "data" / "parquet" / "yjbb.parquet").resolve())
OUT_DIR: str = os.environ.get("MQ_OUT_DIR") or str((_MQ_DIR / "output").resolve())

DAILY_ALL_PARQUET: str = os.path.join(DATA_DIR, "daily_all.parquet")
INDEX_PARQUET: str = os.path.join(DATA_DIR, "index.parquet")
YJBB_THIN: str = os.path.join(DATA_DIR, "yjbb.parquet")   # SPEC §2 瘦字段，仅留作对照，勿用

os.makedirs(OUT_DIR, exist_ok=True)

# SPEC §4：MQ_OFFLINE=1 时全框架不联网（缺数据只标注「数据不足」）
OFFLINE: bool = os.environ.get("MQ_OFFLINE", "").strip() not in ("", "0", "false", "False")

# ---------------------------------------------------------------------------
# 资金与交易成本
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 100_000.0
COMMISSION_RATE = 0.00025
COMMISSION_MIN = 5.0
STAMP_TAX = 0.0005          # 仅卖出
SLIPPAGE = 0.001            # 双边

# SPEC §3④：主板 ±10%、创业板/科创板（300/688）±20%
LIMIT_PCT_MAIN = 0.10
LIMIT_PCT_GEM = 0.20
GEM_PREFIXES = ("300", "688")

# ---------------------------------------------------------------------------
# 区间
# ---------------------------------------------------------------------------
START_DATE = "2016-01-01"
END_DATE = "2026-09-30"
# 附录六⑤ 样本外切分：IS 段 / OOS 段
IS_END = "2023-12-31"
OOS_START = "2024-01-01"

# ---------------------------------------------------------------------------
# 风险与仓位（手册规则编号见 SPEC §3）
# ---------------------------------------------------------------------------
RISK_ZERO_LOSS = 0.05       # R-04 归零损失上限（手册可在 3%–10% 自定）
RISK_PER_TRADE = 0.015      # R-38 单笔风险（1%–2% 取中值）
MAX_POSITIONS = 10          # R-16
MIN_POSITIONS = 3           # R-16
FIRST_TRANCHE = 0.20        # R-35 首笔 ≤ 计划额度 1/5
STOP_LOSS_PCT = 0.01        # R-36 最后买入价下方 1%

# R-09/R-13 格雷厄姆标准配置：权益 25%–75%，再平衡偏离 ±5pp
EQUITY_MIN = 0.25
EQUITY_MAX = 0.75
REBAL_BAND = 0.05


if __name__ == "__main__":
    # 冒烟测试：路径可达 + 常量与 SPEC §3① 逐字一致（防止后续被误改）
    import numbers

    print("MQ_DIR      :", _MQ_DIR)
    print("DATA_DIR    :", DATA_DIR, os.path.isdir(DATA_DIR))
    print("YJBB_RICH   :", YJBB_RICH, os.path.isfile(YJBB_RICH))
    print("OUT_DIR     :", OUT_DIR, os.path.isdir(OUT_DIR))
    for p in (DAILY_ALL_PARQUET, INDEX_PARQUET, YJBB_THIN):
        print("  exists", os.path.isfile(p), p)
    print("OFFLINE     :", OFFLINE)

    spec = {
        "INITIAL_CAPITAL": 100_000.0, "COMMISSION_RATE": 0.00025, "COMMISSION_MIN": 5.0,
        "STAMP_TAX": 0.0005, "SLIPPAGE": 0.001, "LIMIT_PCT_MAIN": 0.10, "LIMIT_PCT_GEM": 0.20,
        "RISK_ZERO_LOSS": 0.05, "RISK_PER_TRADE": 0.015, "MAX_POSITIONS": 10,
        "MIN_POSITIONS": 3, "FIRST_TRANCHE": 0.20, "STOP_LOSS_PCT": 0.01,
        "EQUITY_MIN": 0.25, "EQUITY_MAX": 0.75, "REBAL_BAND": 0.05,
    }
    for k, v in spec.items():
        got = globals()[k]
        assert got == v, (k, got, v)
    assert (START_DATE, END_DATE) == ("2016-01-01", "2026-09-30")
    assert (IS_END, OOS_START) == ("2023-12-31", "2024-01-01")
    print("SPEC §3① 常量自检通过：%d 项数值 + 区间 2 组" % len(spec))
