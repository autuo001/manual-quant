"""③ rules.py —— 规则库 / 证据闸门 / 参数敏感性 / 记录表（SPEC 第 3-③ 节）

规则清单逐条抄自 spec/08-规则总表.md（R-01~R-41，行号即该文件所记原文行号），
V-01~V-10 抄自 spec/09-验证协议.md 并按 SPEC 登记为待验证条目。

闸门语义·主控裁决（2026-10-12）：附录五元数据 2066「Ⅱ 区不得单独构成买入依据」针对的是
**买入依据／触发条件**，而非仓位与卖出约束。因此 gate() 默认只把 kind∈TRIGGER_KINDS（timing/filter）
的规则当作触发依据参与否决；kind 为 sizing/exit/allocation/behavior 的规则（如 R-16 持仓只数、
R-04 归零口径仓位上限、R-36 止损）只是被引用的**约束**，不参与否决。这修正了「value_card 信号
R-04+R-16+R-27 因依据里含 R-16（C 级·sizing）而被全量拦截」的假阳性——R-16 是仓位约束，不是买入触发。
only_trigger=False 保留旧严格语义（依据里出现任何 Ⅱ 区/C·D 级即否决），供协议/判定侧按需回归。

实现取舍（SPEC 未明说处，取最贴近手册/spec 字面的做法，逐条记录）：
1. zone 按附录五分区（手册行 116 / spec/00-总纲与编者观点.md:20,77：「Ⅰ 已可执行（A/B 级）」／
   「Ⅱ 经验与待验证（C/D 级）」，并与元数据 2066「可直接执行」↔ 2067「A、B 可直接执行」逐字对应）：
   A/B 级 → zone=1；C/D 级 → zone=2；V-01~V-10 属「待验证规则」（证据不足、不得作为触发条件）→ 一律 zone=2。
   因此 gate() 的 zone 判定与 level 判定对 R-* 等价；V-06/V-07（B 级）在 only_trigger=False 时仍按等待
   验证拦截，默认语义下二者 kind=allocation 属约束类、不参与否决（见上「闸门语义·主控裁决」）。
2. executable 按 spec/08 ①节标题「可直接编码为回测规则（含明确阈值或公式）」取：①节=True，②③节=False
   （②节仅缺外部数据、③节纯定性，均无法编码为回测逻辑）。zone 与 executable 因此不是同一维度。
3. Rule.name 在 spec/08 中未被提取（每行只有 9 个字段，原文 2076 的「名称」缺失），
   此处按「执行」语义自拟短名，不改动任何阈值文本。
4. V-02 / V-09 在 spec/09 中未标证据等级 → 按 SPEC「level 按 spec 文件所标，D 级标记」记 D；其余照表所标。
   spec/09「证据等级」列的来源括注（如「C（书中回测，R-18/R-19）」）并入 verify 字段，level 只保留字母。
5. sensitivity_grid() 每个可调参数的 3 档各返回一条 {rule_id,param,base,delta,mode,tier,value,values,varies}
   （逐档一行，便于 validate/protocol.py 直接按 {param: value} 注入并区分「基准」档与浮动档）。
   V-01 按 spec/09 用 ±2 个百分点绝对步长，其余按 SPEC 的 ±20% 相对步长。
6. dump_table() 的「状态」由各条「验证」文本映射到状态链词汇（未验证/验证中/部分验证/已验证/已退役），
   「结论」为占位，留待 validate/judge.py 回填。
7. config.py 由 data-loader 负责，并行开发期可能尚不存在 → 用 SPEC 第 3-① 节的默认值兜底，
   保证本模块可独立导入、不因缺文件而失败。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:  # config.py 属 data-loader；缺失时用 SPEC 默认值兜底（见文件头取舍 6）
    import config as _config
except ImportError:  # pragma: no cover
    _config = None

_CFG_FALLBACK = {"START_DATE": "2016-01-01", "END_DATE": "2026-09-30"}


def _cfg(name: str):
    val = getattr(_config, name, None) if _config is not None else None
    return val if val is not None else _CFG_FALLBACK.get(name, "")


@dataclass
class Rule:
    id: str
    name: str
    level: str          # A/B/C/D
    nature: str         # 安全底线/决策原则/经验参考/行为建议/待验证规则
    zone: int           # 1=Ⅰ区可执行, 2=Ⅱ区仅参考
    applies: str
    action: str
    fail: str
    verify: str
    needs_data: str
    line: int
    kind: str           # filter/sizing/exit/timing/allocation/behavior
    executable: bool
    params: dict


# 名称缺失于 spec/08，自拟（见文件头取舍 2）
_NAMES = {
    "R-01": "不使用杠杆／保证金／配资",
    "R-02": "先还清高息债务并备足应急金",
    "R-03": "三年内要用的钱不买权益",
    "R-04": "归零口径单标的最大潜在损失上限",
    "R-05": "买入前写下可证伪条件与退出条件",
    "R-06": "按流程与规则复盘",
    "R-07": "一个标的只归属一套策略",
    "R-08": "任何时点保留现金缓冲",
    "R-09": "权益比例 25%—75%",
    "R-10": "资金使用时间链条",
    "R-11": "按持有期定权益",
    "R-12": "年龄起点法（100−年龄）%",
    "R-13": "偏离 ±5 个百分点再平衡",
    "R-14": "费雪分类仓位上限",
    "R-15": "凯利折扣 1/4—1/2",
    "R-16": "持仓 3—10 只（20 只警戒）",
    "R-17": "多策略独立账户",
    "R-18": "盈利收益率 >10% 定投",
    "R-19": "盈利收益率 <6.4% 分 10 份卖出",
    "R-20": "指数定投估值评估",
    "R-21": "按利率选长债基／短债或货币",
    "R-22": "10 年期国债 3.5%／3% 切换债基",
    "R-23": "定投按月（月频与周频差异很小）",
    "R-24": "低估时按 n 放大器放大金额",
    "R-25": "退休取用率 ≤4%（中国 3%）",
    "R-26": "低估时分红再投入",
    "R-27": "安全边际三档（30/33/50%）",
    "R-28": "只按三个理由卖出",
    "R-29": "成长股逐条打分（第 15 点一票否决）",
    "R-30": "逆势加仓仅限无造假嫌疑标的",
    "R-31": "薪酬断层／外聘 CEO 触发警惕",
    "R-32": "三年评估期",
    "R-33": "价格 ≤ 净流动资产 2/3",
    "R-34": "区间有效突破后才进场",
    "R-35": "首笔 ≤ 计划额度 1/5",
    "R-36": "最后买入价下方 1%（1—1.5 点）止损",
    "R-37": "亏损头寸不加码、绝不摊平",
    "R-38": "单笔风险 ≤ 总资金 1%—2%",
    "R-39": "按第〇章顺序判断，顺序不跳跃",
    "R-40": "季度复盘＋年度审查",
    "R-41": "冷静期 24—72 小时等行为约束",
    "V-01": "盈利收益率阈值",
    "V-02": "债基利率切换（R-22，需先补数据）",
    "V-03": "再平衡",
    "V-04": "持有期比例表",
    "V-05": "取用率",
    "V-06": "单标的风险预算",
    "V-07": "凯利折扣",
    "V-08": "定投频率与 n 放大器",
    "V-09": "趋势交易规则",
    "V-10": "安全边际三档",
}

# ① 节：可直接编码为回测规则（9 字段，含"需数据"）
# (id, level, nature, applies, action, fail, verify, needs_data, line, kind, params)
_SEC1 = [
    ("R-01", "A", "安全底线", "全部策略", "任何资金不使用杠杆、保证金、配资", "无",
     "已完成（附录四案例二）", "无（杠杆率恒设为 1）", 2080, "filter", {}),
    ("R-04", "A", "安全底线", "个股等可能归零标的",
     "按归零假设使单一标的最大潜在损失 ≤ 可投资金融资产的 5%（默认值，可在 3%—10% 间自定）", "无",
     "压力测试通过即有效", "无（组合权重上限 3%—10%，默认 5%）", 2083, "sizing", {"max_loss_pct": 0.05}),
    ("R-11", "B", "经验参考", "长期资金", "按持有期定权益（5 年 25%、30 年 68% 等）", "市场制度剧变",
     "部分（美国 1802—2012）", "无（持有期→权益权重映射表；原文仅给 5 年 25%、30 年 68% 两点）",
     2089, "allocation", {"ratio_5y": 0.25, "ratio_30y": 0.68}),
    ("R-12", "C", "经验参考", "参考值", "（100−年龄）% 仅作起点，不设下限", "临近用钱即下调",
     "未验证", "无（投资者年龄为外部参数）", 2106, "allocation", {}),
    ("R-13", "B", "经验参考", "多资产组合",
     "偏离目标 ±5 个百分点即调整，每年至少检查一次，优先用新增资金", "交易成本过高的小账户",
     "部分（1991—2000 美国对比）", "无（阈值带 ±5pp、检查频率≥1 次/年）", 2090, "allocation",
     {"band": 0.05, "checks_per_year": 1}),
    ("R-15", "D", "待验证规则", "仓位参考", "取 1/4—1/2 折扣，并与 R-04 取更严", "胜率不可估计时弃用",
     "未验证（数学性质，非回测项）", "无（胜率、赔率由策略自身回测产生，折扣系数 1/4—1/2）",
     2119, "sizing", {"discount_lo": 0.25, "discount_hi": 0.50}),
    ("R-16", "C", "经验参考", "价值仓", "3—10 只；超过 20 只视为能力薄弱信号", "指数策略不适用",
     "部分（分散收益递减研究）", "无（持仓数约束 3—10、警戒线 20）", 2108, "sizing",
     {"min_positions": 3, "max_positions": 10, "warn_positions": 20}),
    ("R-23", "B", "经验参考", "定投", "按月与按周差异很小，按发薪节奏选按月", "极端波动的短周期",
     "部分（书中对比）", "无（月频 vs 周频可同口径对照）", 2093, "allocation", {}),
    ("R-32", "C", "经验参考", "价值仓评估周期", "以三年为限评估，逻辑未变不因短期波动卖出",
     '公司永久沉沦（"公司变了"优先）', "未验证", "无（最短评估/持有期 3 年）", 2113, "exit",
     {"eval_years": 3}),
    ("R-35", "C", "经验参考", "趋势仓", "首笔 ≤ 计划额度 1/5，无账面利润不加仓", "无",
     "未验证", "无（首笔上限 20% 计划额度，加仓前置条件=账面利润>0）", 2116, "sizing",
     {"first_tranche": 0.20, "n_tranches": 5}),
    ("R-36", "C", "经验参考", "趋势仓", "最后一笔买入价下方 1—1.5 点（或 1%），触发即执行、不商量",
     "流动性枯竭时改用市价一次性了结", "未验证", "无（止损距 1—1.5 点或 1%）", 2117, "exit",
     {"stop_loss_pct": 0.01}),
    ("R-37", "B", "安全底线", "全部（趋势仓尤甚）", "亏损头寸不加码、绝不追加保证金", "无",
     "部分（利弗莫尔案例＋总闸逻辑）", "无（加仓条件=浮盈>0）", 2098, "sizing",
     {"min_unrealized_profit": 0.0}),
    ("R-38", "C", "经验参考", "趋势仓", "单笔风险 ≤ 总资金 1%—2%", "无",
     "未验证", "无（单笔风险预算 1%—2%）", 2118, "sizing", {"risk_per_trade": 0.015}),
]

# ② 节：需要外部数据才能实现
_SEC2 = [
    ("R-02", "A", "安全底线", "全部", "先还清高息债务 → 备足应急金（3—6 个月支出）→ 再投资", "无",
     "程序性", "个人高息债务余额与利率、月度支出（应急金=3—6 个月支出）", 2081, "filter",
     {"emergency_months_lo": 3, "emergency_months_hi": 6}),
    ("R-03", "A", "安全底线", "全部权益投资", "3 年内要用的钱不买权益，1 年内只放货币基金", "无",
     "程序性", "每笔资金的使用时间表（3 年内外／1 年内外）＋货币基金收益率", 2082, "filter",
     {"no_equity_years": 3, "cash_only_years": 1}),
    ("R-09", "B", "经验参考", "长期资金", "权益 25%—75%，标准各半", "用钱时间 <3 年时退出该框架",
     "部分（美国历史经验）", "债券资产序列（债券基金指数／国债收益率），现有数据无债券端", 2088,
     "allocation", {"equity_min": 0.25, "equity_max": 0.75}),
    ("R-14", "C", "经验参考", "价值仓", "A 类 ≤20%、B 类 8%—10%、C 类 ≤5%，与 R-04 取更严",
     "费雪分类不适用时改用 R-04", "未验证", "费雪 A/B/C 分类标签（定性分类，无对应数据源）", 2107,
     "sizing", {"cap_a": 0.20, "cap_b_lo": 0.08, "cap_b_hi": 0.10, "cap_c": 0.05}),
    ("R-18", "D", "待验证规则", "指数定投", "盈利收益率 >10%（按附录二校准后）定投", "阈值误校",
     "部分（书中回测，V-01）", "指数盈利收益率（E/P＝1/PE），沪深300 指数 PE 不在现有数据边界内",
     2120, "timing", {"yield_threshold": 0.10}),
    ("R-19", "D", "待验证规则", "指数", "盈利收益率 <6.4%（校准后）分 10 份卖出", "阈值误校",
     "部分（V-01）", "指数盈利收益率（E/P）＋分 10 份的批次账务", 2121, "timing",
     {"yield_threshold": 0.064, "n_batches": 10}),
    ("R-20", "B", "经验参考", "指数定投估值", '按"股息率＋PE 变化＋盈利变化"评估，优先股息率高、PE 历史低位时买入',
     "盈利长期为负的指数", "部分（书中复盘）", "指数股息率、指数 PE 及其历史分位", 2091, "timing", {}),
    ("R-21", "B", "经验参考", "债券基金", "利率高选长债基、利率低选短债或货币", "利率管制或产品缺失",
     "原理性", "市场利率／国债收益率、长债基与短债基（货币基金）净值序列", 2092, "allocation", {}),
    ("R-22", "D", "待验证规则", "债基选择", "10 年期国债 >3.5% 选长债、<3% 选短债或货币", "利率环境变化",
     "未验证（V-02）", "10 年期国债收益率、长债基与短债基净值序列", 2122, "allocation",
     {"yield_long_threshold": 0.035, "yield_short_threshold": 0.03}),
    ("R-24", "C", "经验参考", "定投", "低估时按 n＝1（资金充裕取 2）放大金额", "无估值判断能力时改用定额",
     "部分（书中回测）", "低估判定所需的估值指标（指数 PE／盈利收益率）", 2110, "allocation",
     {"n": 1, "n_ample": 2}),
    ("R-25", "D", "待验证规则", "退休取用", "年取 ≤4%（高估值或中国环境建议 3%，或采用动态取用）", "长期熊市",
     "部分（美国研究，V-05）", "通胀（CPI）序列＋长期股债组合序列（现有日线仅 2016—2026）", 2123,
     "allocation", {"withdrawal_rate": 0.04, "withdrawal_rate_cn": 0.03}),
    ("R-26", "B", "经验参考", "指数与基金", "低估时分红再投入；不低估时投其他低估品种", "需要用钱时",
     "部分（西格尔数据）", "分红／股息率、低估判定指标（指数 PE／盈利收益率）", 2094, "allocation", {}),
    ("R-27", "B", "决策原则", "个股", "≥30%（一般）／≥33%（门槛）／≥50%（经典）三档取更严",
     "轻资产高成长不适用 DCF 时降级为定性判断", "部分（格雷厄姆-多德传统）",
     "内在价值估计所需科目（自由现金流、增长率、折现率）或无风险利率", 2095, "filter",
     {"margin_general": 0.30, "margin_gate": 0.33, "margin_classic": 0.50}),
    ("R-31", "C", "经验参考", "个股", "薪酬断层、外聘 CEO 触发警惕（美国语境，中国化用）", "家族企业语境差异",
     "未验证", "高管薪酬数据、CEO 任职与履历（是否为外部聘任）", 2112, "behavior", {}),
    ("R-33", "C", "经验参考", "深度价值", "价格 ≤ 净流动资产 2/3，必须先过清单 G 排雷", "市场容量不足",
     "部分（历史）", "资产负债表科目（流动资产、总负债、股本）", 2114, "filter",
     {"nca_multiple": 2 / 3}),
]

# ③ 节：纯定性／行为类，不可回测（原文该节仅 8 字段，无"需数据"）
_SEC3 = [
    ("R-05", "A", "决策原则", "每次买入", "买入前写下 2—3 条可证伪条件与退出条件，未写不买", "无",
     "程序性", "", 2084, "behavior", {"falsify_conditions_lo": 2, "falsify_conditions_hi": 3}),
    ("R-06", "A", "行为建议", "复盘", '按"是否按流程、是否按规则"评价，不看单次盈亏', "无",
     "程序性", "", 2085, "behavior", {}),
    ("R-07", "A", "决策原则", "全部", "一个标的只归属一套策略，不得跨策略引用理由", "无",
     "程序性", "", 2086, "behavior", {}),
    ("R-08", "A", "安全底线", "全部", "任何时点保留现金缓冲，单一决策不押上全部", "无",
     "程序性", "", 2087, "sizing", {}),
    ("R-10", "C", "经验参考", "长期资金", "用钱时间 → 持有期 → 25%—75% → 年龄参考", "出现明确证伪",
     "未验证", "", 2105, "allocation", {}),
    ("R-17", "C", "决策原则", "多策略并行", "各策略独立账户、互不补贴", "无",
     "程序性", "", 2109, "behavior", {}),
    ("R-28", "B", "决策原则", "价值仓", '只按"买错了／公司变了／有更好的"卖出，与盈亏无关',
     "发现造假时立即处理（并入 R-30）", "部分", "", 2096, "exit", {}),
    ("R-29", "C", "经验参考", "成长股", "逐条打分，第 15 点（诚信）一票否决", "周期股、金融股不适用",
     "未验证", "", 2111, "filter", {"score_items": 15}),
    ("R-30", "B", "安全底线", "逆势加仓", "仅限无反身性、无造假嫌疑的标的；出现嫌疑即禁止加仓并先排雷",
     "流动性枯竭标的", "部分（历史案例）", "", 2097, "filter", {}),
    ("R-34", "C", "经验参考", "趋势仓", "区间被有效突破后才进场", "震荡市反复假突破",
     "未验证（见 V-09）", "", 2115, "timing", {}),
    ("R-39", "A", "决策原则", "全部", "按第〇章顺序判断，顺序不跳跃", "无",
     "程序性", "", 2099, "behavior", {}),
    ("R-40", "A", "行为建议", "全部", "季度复盘＋年度审查，规则变更走变更单", "无",
     "程序性", "", 2100, "behavior", {"review_per_year": 4, "audit_per_year": 1}),
    ("R-41", "B", "行为建议", "全部决策", "冷静期 24—72 小时、双向阅读、书面化、机械再平衡、不参与投票型决策",
     "无", "部分（行为金融研究）", "", 2101, "behavior",
     {"cooldown_hours_lo": 24, "cooldown_hours_hi": 72}),
]

# spec/09 的 V-01~V-10：登记为待验证条目（zone=2, executable=False, nature=待验证规则）
# kind 只取 SPEC 允许的 timing／allocation：择时与阈值类→timing，配置与资金分配类→allocation
# (id, level, applies, action, fail, verify, needs_data, line, kind, params)
_V_ROWS = [
    ("V-01", "C", "盈利收益率阈值",
     "阈值 10%／6.4%；敏感性测试阈值 ±2 个百分点（即 8%~12%、4.4%~8.4%）；区间 2005—2026 全样本独立复现 + 六维协议",
     "阈值 ±2 个百分点浮动后不再有效 → 过拟合嫌疑，降级或不采用", "部分（书中回测，R-18/R-19）",
     "盈利收益率（E/P）所需的 PE、EPS 或净利润绝对值 + 总市值/股本；现有数据不能覆盖", 2133, "timing",
     {"eps_yield_buy": 0.10, "eps_yield_sell": 0.064}),
    ("V-02", "D", "债基选择（R-22）",
     "待补数据后复现 R-22：10 年期国债 >3.5% 选长债、<3% 选短债或货币（区间 2010—2026）",
     "—", "未验证（R-22 所引；spec/09 未标等级，按 SPEC 记 D）",
     "10 年期国债利率、债券基金指数（区间 2010—2026）；现有数据不能覆盖", 2135, "allocation",
     {"yield_long_threshold": 0.035, "yield_short_threshold": 0.03}),
    ("V-03", "C", "再平衡",
     "仓位区间 25%—75%；偏离 ±5% 触发再平衡；待做 A 股／中债样本（已部分验证：1991—2000 美国对比，R-13）",
     "参数（25/75/±5%）浮动后失效", "部分（1991—2000 美国对比）", "中债样本（债券资产序列）", 2137,
     "allocation", {"equity_min": 0.25, "equity_max": 0.75, "band": 0.05}),
    ("V-04", "C", "持有期比例表",
     "待复现持有期与胜率/收益比例的对应表（参考区间美国 1802—2012，R-11）；须做参数敏感性",
     "—", "部分（美国 1802—2012）", "美国 1802—2012 长期股债收益序列（含分红再投资）；现有数据不能覆盖",
     2139, "allocation", {}),
    ("V-05", "C", "取用率", "基准 4% 取用率；待做中国样本自测 3% 与动态取用", "—",
     "部分（美国研究）", "中国样本的长期组合收益序列、通胀/CPI、动态取用规则输入；现有数据不能覆盖",
     2141, "allocation", {"withdrawal_rate": 0.04, "withdrawal_rate_cn": 0.03}),
    ("V-06", "B", "单标的风险预算",
     "单标的归零损失上限 5%（仓位敞口上限）；非回测项，做情景压测（第五章第 4 层）：假设该标的归零，测算组合损失 ≤5%",
     "—", "未验证（非回测项，风险预算设计）", "", 2143, "allocation", {"max_loss_pct": 0.05}),
    ("V-07", "B", "凯利折扣",
     "数学性质 + 敏感性测试；不做历史回测。折扣系数数值原文未在本段给出，需从对应规则章节取值",
     "—", "未验证（数学性质）", "", 2145, "allocation", {"discount_lo": 0.25, "discount_hi": 0.50}),
    ("V-08", "C", "定投频率与 n 放大器",
     "待验证定投频率（周／月等频率对比）；n 放大器定义见 R-23/R-24（正文其他章节）", "—",
     "部分（书中对比，R-23/R-24）",
     "定投标的的股息/分红再投资数据、R-23/R-24 的 n 定义；定投价格序列可用沪深300日线", 2147,
     "allocation", {"n": 1, "n_ample": 2}),
    ("V-09", "D", "趋势交易规则",
     "原文不建议投入回测资源——不适合多数人，执行偏差主导结果；如验证须先定义机械化规则再跑六维协议（R-34）",
     "执行偏差不可回测", "未验证（不适合多数人；spec/09 未标等级，按 SPEC 记 D）", "", 2149, "timing", {}),
    ("V-10", "C", "安全边际三档",
     "三档分档阈值须做敏感性测试；三档具体阈值原文未在本段给出，需从对应规则章节取值",
     "分档阈值浮动后失效", "部分（格雷厄姆-多德传统）", "安全边际三档所需基本面；现有数据不能覆盖",
     2151, "timing", {"margin_general": 0.30, "margin_gate": 0.33, "margin_classic": 0.50}),
]


def _build_rules() -> Dict[str, Rule]:
    rules: Dict[str, Rule] = {}
    # zone 按附录五分区（A/B→Ⅰ 区，C/D→Ⅱ 区，见文件头取舍 1）；executable 按 spec/08 ①节归属
    for section, rows in ((1, _SEC1), (2, _SEC2), (3, _SEC3)):
        for rid, level, nature, applies, action, fail, verify, needs_data, line, kind, params in rows:
            rules[rid] = Rule(rid, _NAMES[rid], level, nature, 1 if level in ("A", "B") else 2,
                              applies, action, fail, verify, needs_data, line, kind,
                              section == 1, dict(params))
    for rid, level, applies, action, fail, verify, needs_data, line, kind, params in _V_ROWS:
        rules[rid] = Rule(rid, _NAMES[rid], level, "待验证规则", 2, applies, action, fail, verify,
                          needs_data, line, kind, False, dict(params))
    return rules


RULES: Dict[str, Rule] = _build_rules()

# spec/09：V-01 的阈值敏感性用 ±2 个百分点绝对步长，而非 SPEC 默认的 ±20% 相对
_ABS_STEP = {"V-01": 0.02}


def get(rule_id: str) -> Optional[Rule]:
    return RULES.get(rule_id)


# 买入「触发依据」的 kind 白名单（主控裁决，见文件头）：timing=择时/阈值触发，filter=筛选/估值闸门。
# 其余 kind（sizing/exit/allocation/behavior）是仓位与卖出约束，被引用时不参与闸门否决。
TRIGGER_KINDS = {"timing", "filter"}


def gate(rule_ids: List[str], only_trigger: bool = True) -> Tuple[bool, str]:
    """附录五元数据 2066：Ⅱ 区或 C/D 级规则不得单独构成买入依据。

    only_trigger=True（默认，主控裁决）：只有 kind∈TRIGGER_KINDS 的规则算「买入触发依据」并参与否决；
    kind 为 sizing/exit/allocation/behavior 的规则视为约束、不参与否决（str 里说明哪些被忽略）。
    only_trigger=False：旧严格语义，依据里出现任一 Ⅱ 区/C·D 级即否决（向后兼容）。
    返回 (是否通过, 说明)：说明含被否决的触发规则、或被忽略的约束规则。
    """
    ids = list(dict.fromkeys(rule_ids or []))
    if not ids:
        return False, "无独立买入依据：买入信号必须记账其依据的规则编号（附录五元数据 2066）"
    unknown = [r for r in ids if r not in RULES]
    if unknown:
        return False, "未登记的规则编号：" + "、".join(unknown)
    triggers = [r for r in ids if (not only_trigger) or RULES[r].kind in TRIGGER_KINDS]
    constraints = [r for r in ids if r not in triggers]
    blocked = []
    for r in triggers:
        rule = RULES[r]
        if rule.zone == 2:
            blocked.append("%s(%s 级·Ⅱ 区·%s)" % (r, rule.level, rule.kind))
        elif rule.level in ("C", "D"):
            blocked.append("%s(%s 级·%s)" % (r, rule.level, rule.kind))
    ignored = ("；约束类规则不参与否决，已忽略：" + "、".join(
        "%s(%s·%s 级)" % (r, RULES[r].kind, RULES[r].level) for r in constraints)) if constraints else ""
    if blocked:
        return False, ("Ⅱ 区／C·D 级不得单独构成买入依据（仅计触发类 %s）：%s（附录五元数据 2066，"
                       "需先走六维协议与规则变更单升级）%s"
                       % ("/".join(sorted(TRIGGER_KINDS)), "、".join(blocked), ignored))
    if not triggers:  # 依据里全是约束类：不否决，但提示触发条件未被记账（触发条件由七步法三条件承担）
        return True, ("ok：依据 %s 均为约束类（非 %s），按主控裁决不参与否决；"
                      "提示——其中未记账任何触发类依据"
                      % ("、".join("%s(%s)" % (r, RULES[r].kind) for r in ids),
                         "/".join(sorted(TRIGGER_KINDS))))
    return True, ("ok" + ignored)


def encodable() -> List[str]:
    """executable=True 的编号（spec/08 ① 节）。"""
    return sorted([r.id for r in RULES.values() if r.executable])


def by_kind(kind: str) -> List[Rule]:
    return sorted([r for r in RULES.values() if r.kind == kind], key=lambda r: r.id)


def by_level(level: str) -> List[Rule]:
    return sorted([r for r in RULES.values() if r.level == level], key=lambda r: r.id)


def sensitivity_grid(rule_id: str, rel: float = 0.20) -> List[dict]:
    """附录六④：阈值上下浮动。默认 ±rel 相对步长，V-01 按 spec/09 用 ±2 个百分点绝对。

    每个可调参数的 3 档各返回一条记录：{rule_id, param, base, delta, mode, tier, value, values, varies}。
    tier 取 低档／基准／高档，基准档标为「基准」——validate/protocol.py 以此区分基准与浮动档；
    value 为该档取值（供引擎按 {param: value} 注入），values 为同参数 3 档全量。
    非数值参数与零值参数（无法按比例浮动）不进入网格。
    """
    rule = RULES.get(rule_id)
    if rule is None:
        raise KeyError("未登记的规则编号：%s" % rule_id)
    abs_step = _ABS_STEP.get(rule_id)
    out = []
    for name in sorted(rule.params):
        base = rule.params[name]
        if isinstance(base, bool) or not isinstance(base, (int, float)):
            continue
        if abs_step is not None:
            mode, step = "absolute", abs_step
        else:
            if base == 0:
                continue
            mode, step = "relative", abs(base) * rel
        step = round(step, 6)
        raw = [base - step, base, base + step]
        if isinstance(base, int):
            values = sorted({int(round(v)) for v in raw})
        else:
            values = sorted({round(v, 6) for v in raw})
        for tier, value in zip(("低档", "基准", "高档"), values):
            out.append({"rule_id": rule_id, "param": name, "base": base, "delta": step,
                        "mode": mode, "tier": tier, "value": value, "values": values,
                        "varies": len(values) > 1})
    return out


def _state_from_verify(verify: str) -> str:
    """把规则表的「验证」文本映射到附录六状态链词汇。"""
    if "已完成" in verify:
        return "已验证"
    if "部分" in verify:
        return "部分验证"
    return "未验证"


def dump_table() -> pd.DataFrame:
    """附录六记录表：固定 7 字段。状态取自规则表验证状态，结论留待 validate/judge.py 回填。"""
    data_range = "%s~%s" % (_cfg("START_DATE"), _cfg("END_DATE"))
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    rows = []
    for rid in sorted(RULES, key=lambda x: (x[0], int(x[2:]))):
        rule = RULES[rid]
        rows.append({"编号": rid, "规则": rule.name, "状态": _state_from_verify(rule.verify),
                     "数据区间": data_range, "基准": "沪深300（简单持有）", "结论": "待验证",
                     "更新日期": today})
    return pd.DataFrame(rows, columns=["编号", "规则", "状态", "数据区间", "基准", "结论", "更新日期"])


if __name__ == "__main__":
    r_ids = [k for k in RULES if k.startswith("R-")]
    v_ids = [k for k in RULES if k.startswith("V-")]
    assert len(r_ids) == 41 and len(v_ids) == 10, (len(r_ids), len(v_ids))
    print("RULES=%d（R-%02d..R-%02d %d 条 / V %d 条）" % (len(RULES), 1, 41, len(r_ids), len(v_ids)))
    print("encodable(%d): %s" % (len(encodable()), ",".join(encodable())))
    print("zone1/zone2:", sum(1 for r in RULES.values() if r.zone == 1),
          "/", sum(1 for r in RULES.values() if r.zone == 2))
    for k in ("filter", "sizing", "exit", "timing", "allocation", "behavior"):
        print("  kind=%-10s n=%2d  %s" % (k, len(by_kind(k)), ",".join(r.id for r in by_kind(k))))

    # 附录五闸门（主控裁决）：只有触发类（timing/filter）的 A/B 级（Ⅰ 区）规则能单独构成买入依据（手册行 116）；
    # sizing/exit/allocation/behavior 属约束，被引用时不否决——value_card 的 R-04+R-16+R-27 因此可通过。
    clean = [rid for rid in sorted(RULES) if RULES[rid].kind in TRIGGER_KINDS and gate([rid])[0]]
    strict = [rid for rid in sorted(RULES) if gate([rid], only_trigger=False)[0]]
    print("可单独构成买入依据的触发类规则 %d 条: %s" % (len(clean), ",".join(clean)))
    print("旧严格语义（only_trigger=False）可单独构成买入依据 %d 条" % len(strict))
    print("gate(['R-04'])                       ->", gate(["R-04"]))
    print("gate(['R-04','R-01'])                ->", gate(["R-04", "R-01"]))
    print("gate(['R-04','R-16','R-27'])         ->", gate(["R-04", "R-16", "R-27"]))
    print("gate(['R-04','R-16','R-27'],False)   ->", gate(["R-04", "R-16", "R-27"], only_trigger=False))
    print("gate(['R-27','R-16'])                ->", gate(["R-27", "R-16"]))
    print("gate(['R-04','R-16'])                ->", gate(["R-04", "R-16"]))
    print("gate(['R-34','R-35'])                ->", gate(["R-34", "R-35"]))
    print("gate(['V-01'])                       ->", gate(["V-01"]))
    print("gate([])                             ->", gate([]))
    print("gate(['R-99'])                       ->", gate(["R-99"]))
    # 主控裁决的回归断言：仓位约束类 C 级规则不再把信号全量拦截；旧严格语义可显式取回
    assert gate(["R-04", "R-16", "R-27"])[0] is True, gate(["R-04", "R-16", "R-27"])
    assert gate(["R-04", "R-16", "R-27"], only_trigger=False)[0] is False
    assert gate(["R-04", "R-16"])[0] is True and gate(["R-04", "R-16"])[1] != "ok"
    assert gate(["R-16"])[0] is True  # 纯约束依据：不否决但从说明里可见
    print("闸门语义回归断言 OK")

    for rid in ("V-01", "R-16"):
        grid = sensitivity_grid(rid)
        print("sensitivity_grid('%s') -> %d 行" % (rid, len(grid)))
        for row in grid:
            print("    %-14s tier=%-3s value=%-8s values=%s mode=%s" %
                  (row["param"], row["tier"], row["value"], row["values"], row["mode"]))

    # 与 validate/protocol.py 的消费口径对齐校验（该模块按 param/value/tier 逐档消费）
    try:
        from validate.protocol import _norm_variant
        rows = sensitivity_grid("R-04")
        norms = [_norm_variant(r) for r in rows]
        # protocol 对「基准」档返回 None（见 protocol.py:_norm_variant 注释：dim① 已有基准行，不重复跑），
        # 故只要求浮动档能被归一化；若对方改为回显基准行也一并接受。
        floats = [n for n in norms if n]
        assert floats and all(n["参数"] == "max_loss_pct" and n["档"] for n in floats), norms
        print("protocol._norm_variant 兼容 OK：浮动档 %s；基准档 %d 行由 protocol 跳过"
              % ([n["值"] for n in floats], len(rows) - len(floats)))
    except ImportError as exc:
        print("跳过 protocol 兼容检查：%s" % exc)

    tbl = dump_table()
    print("dump_table shape=%s cols=%s" % (tbl.shape, list(tbl.columns)))
    print(tbl.head(3).to_string(index=False))
    print("状态分布:", tbl["状态"].value_counts().to_dict())

    try:  # 跨模块冒烟：这些模块由他人并行实现，缺失时跳过而不失败
        from core import risk_budget  # noqa: F401
        print("core.risk_budget 可用（可选校验：R-04 上限 %.2f）" % RULES["R-04"].params["max_loss_pct"])
    except ImportError as exc:
        print("跳过跨模块检查（尚未就绪）：%s" % exc)
