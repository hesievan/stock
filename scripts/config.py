"""
全局配置文件 — 牛市热度指数
"""
import os
from dataclasses import dataclass, field
from typing import List

# ── Tushare 配置（已弃用，全部指标改用 akshare） ─────────
# 保留用于向后兼容，不再必需
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# ── 数据路径 ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INDEX_HISTORY_PATH = os.path.join(DATA_DIR, "index_history.csv")
CHART_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "chart.png")

# ── 邮件配置（从环境变量读取） ─────────────────────────────
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.qq.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
MAIL_TO = os.getenv("MAIL_TO", "")
MAIL_FROM = os.getenv("MAIL_FROM", SMTP_USER)  # 默认同 SMTP_USER

# ── 指数参数 ──────────────────────────────────────────────
@dataclass
class Indicator:
    """单个指标的定义"""
    id: str                     # 唯一标识符，也用于列名和文件名
    name: str                   # 中文名称
    weight: float = 0.10        # 权重（默认10%）
    raw_file: str = ""          # data/raw/ 下的历史数据文件名

    def __post_init__(self):
        if not self.raw_file:
            self.raw_file = f"{self.id}.csv"


# 全部 10 个指标
# 核心梯队（各12%）：换手率、两融、PE、股债利差、成交额M2比、巴菲特指标
# 辅助梯队（各7%）：开户数、存款市值比、基金年化、股债基金比
INDICATORS: List[Indicator] = [
    # 核心梯队（各12%，快速+长历史）
    Indicator(id="turnover_rate", name="沪深换手率", weight=0.12),
    Indicator(id="margin_ratio", name="两融余额市值比", weight=0.12),
    Indicator(id="pe_valuation", name="大盘估值-PE", weight=0.12),
    Indicator(id="equity_bond_spread", name="股债利差", weight=0.12),
    Indicator(id="turnover_m2", name="成交额M2比", weight=0.12),
    Indicator(id="buffett_ratio", name="巴菲特指标", weight=0.12),
    # 辅助梯队（各7%，短历史/慢响应）
    Indicator(id="account_open", name="上交所个人开户数", weight=0.07),
    Indicator(id="deposit_market_ratio", name="存款市值比", weight=0.07),
    Indicator(id="fund_3y_annual", name="偏股基金三年年化收益率", weight=0.07),
    Indicator(id="equity_bond_fund_ratio", name="股债基金比", weight=0.07),
]

INDICATOR_MAP = {ind.id: ind for ind in INDICATORS}

# ── 热度等级定义 ──────────────────────────────────────────
HEAT_LEVELS = [
    (0, 20, "很冷", "历史极值低位，长期布局窗口"),
    (20, 40, "偏冷", "市场偏冷，关注低估机会"),
    (40, 60, "适中", "无显著超买超卖，常规仓位"),
    (60, 80, "偏热", "短期过热风险上升，注意控仓"),
    (80, 100, "很热", "逃顶风险显著，建议降低仓位"),
]


def get_heat_level(index_value: float) -> dict:
    """根据指数值返回热度等级信息"""
    for lo, hi, label, desc in HEAT_LEVELS:
        if lo <= index_value < hi:
            return {"level": label, "description": desc, "range": f"{lo}-{hi}"}
    # 恰好 100
    return {"level": "很热", "description": "逃顶风险显著，建议降低仓位", "range": "80-100"}
