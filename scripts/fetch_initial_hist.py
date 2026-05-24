"""
第一阶段：数据采集与历史数据初始化
采集全部10个指标自起始日的历史日度数据，存入 data/raw/ 下各自 CSV。

数据源策略：
  - akshare 免费数据（已验证 v1.18.63 可用函数）
  - tushare 补充日频数据（自动 65s 延时应对免费版频率限制）

运行方式：
    python scripts/fetch_initial_hist.py

调试子集：
    python scripts/fetch_initial_hist.py --indicators fund_3y_annual,equity_bond_spread
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import RAW_DIR, TUSHARE_TOKEN, INDICATORS

# ── 全局速率限制 ─────────────────────────────────────────
LAST_TUSHARE_CALL = 0.0
TUSHARE_MIN_INTERVAL = 65  # 免费版 1次/分钟 → 65秒间隔


def tushare_call(pro, func_name: str, **kwargs):
    """带速率限制的 tushare API 调用"""
    global LAST_TUSHARE_CALL
    elapsed = time.time() - LAST_TUSHARE_CALL
    if elapsed < TUSHARE_MIN_INTERVAL:
        wait = TUSHARE_MIN_INTERVAL - elapsed
        print(f"  ⏳ 等待 {wait:.0f}s（tushare 频率限制）…")
        time.sleep(wait)

    api_func = getattr(pro, func_name, None)
    if api_func is None:
        print(f"  ✗ tushare 无此接口: {func_name}")
        return pd.DataFrame()

    print(f"  📡 tushare.{func_name}({kwargs})")
    try:
        df = api_func(**kwargs)
        LAST_TUSHARE_CALL = time.time()
        return df
    except Exception as e:
        LAST_TUSHARE_CALL = time.time()
        raise e


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def safe_read_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df
    return pd.DataFrame()


def save_dataframe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"  ✓ 已保存 {len(df)} 行 → {path}")


def resample_to_daily(series: pd.Series, method: str = "ffill") -> pd.Series:
    daily_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    daily = series.reindex(daily_idx).sort_index()
    return daily.ffill() if method == "ffill" else daily.interpolate(method="linear")


def get_tushare_pro():
    import tushare as ts
    if not TUSHARE_TOKEN:
        print("  ⚠ TUSHARE_TOKEN 未设置，tushare 数据不可用")
        return None
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api()


# ═══════════════════════════════════════════════════════════
# 指标 1：巴菲特指标  (akshare 月频 + 季频 → 日频)
# ═══════════════════════════════════════════════════════════

def fetch_buffett_ratio() -> pd.DataFrame:
    """
    巴菲特指标 = A股总市值 / 最近4个季度滚动GDP
    数据源：
      - A股总市值：akshare `macro_china_stock_market_cap`（月频），或 tushare daily_basic（日频）
      - GDP：akshare `macro_china_gdp`（季频）
    """
    print("\n[1/10] 巴菲特指标 (buffett_ratio)")

    # ── A股总市值（akshare 月频插值） ──
    try:
        import akshare as ak
        df = ak.macro_china_stock_market_cap()
        df = df.rename(columns={"数据日期": "date", "市价总值-上海": "mv_sh", "市价总值-深圳": "mv_sz"})
        # 处理中文日期格式 "2026年05月份"
        df["date"] = df["date"].str.replace("年", "-").str.replace("月份", "").str.replace("月", "")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["mv_sh"] = pd.to_numeric(df["mv_sh"], errors="coerce")
        df["mv_sz"] = pd.to_numeric(df["mv_sz"], errors="coerce")
        df["total_mv"] = (df["mv_sh"] + df["mv_sz"]) * 1e8  # 亿元→元
        total_mv = resample_to_daily(df["total_mv"], method="ffill").to_frame()
        print(f"  akshare 月频总市值插值: {len(total_mv)} 行")
    except Exception as e:
        print(f"  ✗ 总市值数据均失败: {e}")
        return pd.DataFrame()

    # ── GDP ──
    try:
        import akshare as ak
        df_gdp = ak.macro_china_gdp()
        # 只保留单季度数据（标签不含 "-"）
        single_q = df_gdp[~df_gdp["季度"].str.contains("-", na=False)]
        gdp = single_q[["季度", "国内生产总值-绝对值"]].copy()
        gdp.columns = ["quarter_label", "gdp"]
        # 解析 "2008年第1季度" → 季末日期
        def parse_single_quarter(q):
            year = q[:4]
            qnum_str = q[q.index("第")+1:q.index("季度")]
            qnum = int(qnum_str)
            month = qnum * 3
            return pd.Timestamp(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0)
        gdp["date"] = gdp["quarter_label"].apply(parse_single_quarter)
        gdp = gdp.set_index("date").sort_index()
        # 去重（防止同一个季度有多行）
        gdp = gdp[~gdp.index.duplicated(keep="last")]
        gdp["gdp"] = pd.to_numeric(gdp["gdp"], errors="coerce") * 1e8  # 亿元→元

        # 4个季度滚动总和
        gdp["gdp_4q"] = gdp["gdp"].rolling(window=4, min_periods=4).sum()
        gdp_4q_daily = resample_to_daily(gdp["gdp_4q"].dropna(), method="ffill")
        print(f"  GDP 单季度: {len(gdp)} 条, 滚动4Q插值后: {len(gdp_4q_daily)} 日")
    except Exception as e:
        print(f"  ✗ GDP 采集失败: {e}")
        return pd.DataFrame()

    # ── 合并 ──
    result = total_mv.join(gdp_4q_daily, how="inner")
    result["value"] = result["total_mv"] / result["gdp_4q"]
    result = result[["value"]].dropna()
    result.index.name = "date"
    print(f"  共 {len(result)} 行")
    return result


# ═══════════════════════════════════════════════════════════
# 指标 2：成交额M2比  (tushare 日频 + akshare M2 月频)
# ═══════════════════════════════════════════════════════════

def fetch_turnover_m2() -> pd.DataFrame:
    """
    成交额M2比 = 沪深两市日成交额 / M2余额

    数据源：
      - 成交额：stock_zh_index_daily_tx 日频沪市+深市（同换手率）
      - M2：macro_china_money_supply 月频向前填充
    """
    print("\n[2/10] 成交额M2比 (turnover_m2)")

    import akshare as ak

    # ── 沪深两市日成交额（与换手率共享相同数据源） ──
    try:
        print("  获取沪市成交额(sh000001)…")
        df_sh = ak.stock_zh_index_daily_tx(symbol='sh000001')
        print("  获取深市成交额(sz399106)…")
        df_sz = ak.stock_zh_index_daily_tx(symbol='sz399106')

        sh_amt = df_sh[["date", "amount"]].copy()
        sh_amt.columns = ["date", "amt_sh"]
        sh_amt["date"] = pd.to_datetime(sh_amt["date"])
        sh_amt = sh_amt.set_index("date").sort_index()

        sz_amt = df_sz[["date", "amount"]].copy()
        sz_amt.columns = ["date", "amt_sz"]
        sz_amt["date"] = pd.to_datetime(sz_amt["date"])
        sz_amt = sz_amt.set_index("date").sort_index()

        amount = sh_amt.join(sz_amt, how="outer").fillna(0)
        amount["total_amount"] = (amount["amt_sh"] + amount["amt_sz"]) * 1e4  # 万元→元
        amount = amount[["total_amount"]].sort_index()
        print(f"  日频成交额: {len(amount)} 行")
    except Exception as e:
        print(f"  ✗ 成交额采集失败: {e}")
        return pd.DataFrame()

    # ── M2（akshare 月频） ──
    try:
        import akshare as ak
        df_m2 = ak.macro_china_money_supply()
        m2 = df_m2[["月份", "货币和准货币(M2)-数量(亿元)"]].copy()
        m2.columns = ["date", "m2"]
        m2["date"] = pd.to_datetime(m2["date"].str.replace("年", "-").str.replace("月份", ""))
        m2 = m2.set_index("date").sort_index()
        m2["m2"] = pd.to_numeric(m2["m2"], errors="coerce") * 1e8  # 亿元→元
        m2_daily = resample_to_daily(m2["m2"].dropna(), method="ffill")
        print(f"  M2: {len(m2)} 月 → {len(m2_daily)} 日")
    except Exception as e:
        print(f"  ✗ M2 采集失败: {e}")
        return pd.DataFrame()

    result = amount.join(m2_daily, how="inner")
    result["value"] = result["total_amount"] / result["m2"]
    result = result[["value"]].dropna()
    result.index.name = "date"
    print(f"  共 {len(result)} 行")
    return result


# ═══════════════════════════════════════════════════════════
# 指标 3：换手率  (tushare 日频)
# ═══════════════════════════════════════════════════════════

def fetch_turnover_rate(pro=None) -> pd.DataFrame:
    """
    沪深换手率 = (沪市成交额 + 深市成交额) / 总市值 × 100%

    数据源：
      - 沪市成交额：stock_zh_index_daily_tx('sh000001')（上证综指，覆盖全部沪市）
      - 深市成交额：stock_zh_index_daily_tx('sz399106')（深证综指，覆盖全部深市）
      - 总市值：macro_china_stock_market_cap 月频插值

    ⚠ 修正说明：原代码只用了沪市成交额（缺深市）且未做万元→元转换，
      导致换手率被低估 1000~3000 倍。现已修复。
    """
    print("\n[3/10] 沪深换手率 (turnover_rate)")

    try:
        import akshare as ak

        # ── 沪市日成交额 ──
        print("  获取沪市成交额(sh000001)…")
        df_sh = ak.stock_zh_index_daily_tx(symbol='sh000001')
        sh_amount = df_sh[["date", "amount"]].copy()
        sh_amount.columns = ["date", "amt_sh"]
        sh_amount["date"] = pd.to_datetime(sh_amount["date"])
        sh_amount = sh_amount.set_index("date").sort_index()
        sh_amount["amt_sh"] = sh_amount["amt_sh"].astype(float)

        # ── 深市日成交额 ──
        print("  获取深市成交额(sz399106)…")
        df_sz = ak.stock_zh_index_daily_tx(symbol='sz399106')
        sz_amount = df_sz[["date", "amount"]].copy()
        sz_amount.columns = ["date", "amt_sz"]
        sz_amount["date"] = pd.to_datetime(sz_amount["date"])
        sz_amount = sz_amount.set_index("date").sort_index()
        sz_amount["amt_sz"] = sz_amount["amt_sz"].astype(float)

        # ── 合并两市成交额 ──
        amount = sh_amount.join(sz_amount, how="outer").fillna(0)
        # amount 单位是万元 → ×1e4 转元
        amount["total_amount"] = (amount["amt_sh"] + amount["amt_sz"]) * 1e4

        print(f"  日成交额: {len(amount)} 行")

        # ── 月频总市值（插值到日） ──
        df_cap = ak.macro_china_stock_market_cap()
        df_cap = df_cap.rename(columns={
            "数据日期": "date", "市价总值-上海": "mv_sh", "市价总值-深圳": "mv_sz"
        })
        df_cap["date"] = df_cap["date"].str.replace("年", "-").str.replace("月份", "").str.replace("月", "")
        df_cap["date"] = pd.to_datetime(df_cap["date"])
        df_cap = df_cap.set_index("date").sort_index()
        for c in ["mv_sh", "mv_sz"]:
            df_cap[c] = pd.to_numeric(df_cap[c], errors="coerce")
        df_cap["total_mv"] = (df_cap["mv_sh"] + df_cap["mv_sz"]) * 1e8  # 亿元→元
        mv_daily = resample_to_daily(df_cap["total_mv"].dropna(), method="ffill").to_frame()
        print(f"  总市值（月频→日）: {len(mv_daily)} 行")

        # ── 计算换手率 ──
        result = amount.join(mv_daily, how="inner")
        result["value"] = result["total_amount"] / result["total_mv"] * 100  # %
        result = result[["value"]].dropna()
        # 过滤异常值
        result = result[result["value"] < 20]
        result.index.name = "date"
        print(f"  共 {len(result)} 行, 换手率均值={result['value'].mean():.4f}%")
        return result

    except Exception as e:
        print(f"  ✗ 换手率计算失败: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════
# 指标 4：两融余额市值比  (tushare 日频)
# ═══════════════════════════════════════════════════════════

def fetch_margin_ratio() -> pd.DataFrame:
    # 全用 akshare，无 tushare 依赖
    """
    两融余额市值比 = (沪市两融余额 + 深市两融余额) / A股总市值
    使用 akshare macro_china_market_margin_sh/sz（日频）
    """
    print("\n[4/10] 两融余额市值比 (margin_ratio)")

    try:
        import akshare as ak

        print("  采集沪市两融…")
        df_sh = ak.macro_china_market_margin_sh()
        margin_sh = df_sh[["日期", "融资融券余额"]].copy()
        margin_sh.columns = ["date", "margin_sh"]
        margin_sh["date"] = pd.to_datetime(margin_sh["date"])
        margin_sh = margin_sh.set_index("date").sort_index()
        margin_sh["margin_sh"] = pd.to_numeric(margin_sh["margin_sh"], errors="coerce")

        print("  采集深市两融…")
        df_sz = ak.macro_china_market_margin_sz()
        margin_sz = df_sz[["日期", "融资融券余额"]].copy()
        margin_sz.columns = ["date", "margin_sz"]
        margin_sz["date"] = pd.to_datetime(margin_sz["date"])
        margin_sz = margin_sz.set_index("date").sort_index()
        margin_sz["margin_sz"] = pd.to_numeric(margin_sz["margin_sz"], errors="coerce")

        margin = margin_sh.join(margin_sz, how="outer").fillna(0)
        margin["margin_total"] = margin["margin_sh"] + margin["margin_sz"]
        print(f"  两融余额: {len(margin)} 日, 范围 {margin.index[0].date()} ~ {margin.index[-1].date()}")
    except Exception as e:
        print(f"  ✗ 两融采集失败: {e}")
        return pd.DataFrame()

    # ── 总市值（用 akshare 月频插值） ──
    try:
        import akshare as ak
        df_cap = ak.macro_china_stock_market_cap()
        df_cap = df_cap.rename(columns={"数据日期": "date", "市价总值-上海": "mv_sh", "市价总值-深圳": "mv_sz"})
        df_cap["date"] = df_cap["date"].str.replace("年", "-").str.replace("月份", "").str.replace("月", "")
        df_cap["date"] = pd.to_datetime(df_cap["date"])
        df_cap = df_cap.set_index("date").sort_index()
        for c in ["mv_sh", "mv_sz"]:
            df_cap[c] = pd.to_numeric(df_cap[c], errors="coerce")
        df_cap["total_mv"] = (df_cap["mv_sh"] + df_cap["mv_sz"]) * 1e8  # 元
        mv_daily = resample_to_daily(df_cap["total_mv"].dropna(), method="ffill").to_frame()
        print(f"  总市值（月频→日）: {len(mv_daily)} 行")
    except Exception as e:
        print(f"  ✗ 总市值失败: {e}")
        return pd.DataFrame()

    result = margin.join(mv_daily, how="inner")
    result["value"] = result["margin_total"] / result["total_mv"] * 100  # %
    result = result[["value"]].dropna()
    result.index.name = "date"
    print(f"  共 {len(result)} 行")
    return result


# ═══════════════════════════════════════════════════════════
# 指标 5：大盘估值-PE  (tushare 日频)
# ═══════════════════════════════════════════════════════════

def fetch_pe_valuation() -> pd.DataFrame:
    """
    大盘估值-PE = 上证指数平均市盈率
    使用 akshare stock_market_pe_lg（月频，1997~至今）

    注意：原使用 stock_zh_index_hist_csindex('000300') 的沪深300滚动市盈率
    但该数据源仅更新到 2024-06。改为有完整历史（1997~至今）的上证指数PE。
    """
    print("\n[5/10] 大盘估值-PE (pe_valuation)")

    try:
        import akshare as ak
        print("  获取上证指数月频市盈率…")
        df = ak.stock_market_pe_lg()
        pe = df[["日期", "平均市盈率"]].copy()
        pe.columns = ["date", "value"]
        pe["date"] = pd.to_datetime(pe["date"])
        pe = pe.set_index("date").sort_index()
        pe["value"] = pd.to_numeric(pe["value"], errors="coerce")
        pe = pe.dropna()
        # 月频插值到日
        daily = resample_to_daily(pe["value"], method="ffill")
        result = daily.to_frame("value")
        result.index.name = "date"
        print(f"  上证指数 平均市盈率: {len(pe)} 月, 插值后 {len(result)} 日")
        print(f"  范围 {pe.index[0].date()} ~ {pe.index[-1].date()}")
        return result
    except Exception as e:
        print(f"  ✗ PE 采集失败: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════
# 指标 6：上交所开户数  (akshare 月频)
# ═══════════════════════════════════════════════════════════

def fetch_account_open() -> pd.DataFrame:
    """
    上交所新增投资者开户数（月频）

    数据源优先级：
      1. data/raw/account_open_user.csv（用户手动维护，若有）
      2. akshare stock_account_statistics_em（官方，截至 2023-08）

    用户可自行在 account_open_user.csv 中追加新行，格式：
        date,value
        2024-01-01,1500000
        2024-02-01,1200000
        ...
    值单位为"人"（非万户），即上交所公布的月度新增投资者数量。
    """
    print("\n[6/10] 上交所个人开户数 (account_open)")

    user_csv = os.path.join(RAW_DIR, "account_open_user.csv")

    # ── 优先读取用户手动维护的 CSV ──
    if os.path.exists(user_csv):
        try:
            df_user = pd.read_csv(user_csv, index_col=0, parse_dates=True)
            if not df_user.empty and "value" in df_user.columns:
                df_user = df_user.sort_index()
                df_user["value"] = pd.to_numeric(df_user["value"], errors="coerce")
                df_user = df_user.dropna()
                start = df_user.index.min().strftime("%Y-%m-%d")
                end = df_user.index.max().strftime("%Y-%m-%d")
                print(f"  📄 读取用户 CSV: {len(df_user)} 行, {start} ~ {end}")
                daily = resample_to_daily(df_user["value"], method="ffill")
                result = daily.to_frame("value")
                result.index.name = "date"
                return result
        except Exception as e:
            print(f"  ⚠ 用户 CSV 读取失败: {e}，回退到 akshare")

    # ── 备用：akshare（数据截至 2023-08） ──
    print("  ⚠ 未找到 account_open_user.csv，使用 akshare 数据（截至 2023-08）")
    try:
        import akshare as ak
        df = ak.stock_account_statistics_em()
        acc = df[["数据日期", "新增投资者-数量"]].copy()
        acc.columns = ["date", "value"]
        acc["date"] = pd.to_datetime(acc["date"])
        acc = acc.set_index("date").sort_index()
        acc["value"] = pd.to_numeric(acc["value"], errors="coerce") * 10000  # 万→人
        acc = acc.dropna()
        daily = resample_to_daily(acc["value"], method="ffill")
        result = daily.to_frame("value")
        result.index.name = "date"
        print(f"  {len(acc)} 月 → {len(result)} 日")
        return result
    except Exception as e:
        print(f"  ✗ 开户数采集失败: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════
# 指标 7：股债利差  (akshare 日频国债 + tushare PE)
# ═══════════════════════════════════════════════════════════

def fetch_equity_bond_spread() -> pd.DataFrame:
    """
    股债利差 = (1 / 上证指数 PE) - (10年期国债收益率 / 100)

    数据源：
      - PE: stock_market_pe_lg 上证指数月频市盈率（1997~至今）
      - 国债收益率: bond_zh_us_rate 中国国债收益率10年（日频）
    """
    print("\n[7/10] 股债利差 (equity_bond_spread)")

    import akshare as ak

    # ── 10年期国债收益率（akshare 日频） ──
    try:
        print("  采集10年期国债收益率…")
        df_bond = ak.bond_zh_us_rate()
        bond = df_bond[["日期", "中国国债收益率10年"]].copy()
        bond.columns = ["date", "bond_yield"]
        bond["date"] = pd.to_datetime(bond["date"])
        bond = bond.set_index("date").sort_index()
        bond["bond_yield"] = pd.to_numeric(bond["bond_yield"], errors="coerce")
        print(f"  国债收益率: {len(bond)} 日, 范围 {bond.index[0].date()} ~ {bond.index[-1].date()}")
    except Exception as e:
        print(f"  ✗ 国债收益率采集失败: {e}")
        return pd.DataFrame()

    # ── 上证指数 PE（月频→日频） ──
    try:
        print("  获取上证指数月频市盈率…")
        df = ak.stock_market_pe_lg()
        pe = df[["日期", "平均市盈率"]].copy()
        pe.columns = ["date", "pe_ttm"]
        pe["date"] = pd.to_datetime(pe["date"])
        pe = pe.set_index("date").sort_index()
        pe["pe_ttm"] = pd.to_numeric(pe["pe_ttm"], errors="coerce")
        # 月频插值到日
        pe_daily = resample_to_daily(pe["pe_ttm"].dropna(), method="ffill").to_frame()
        print(f"  上证PE: {len(pe)} 月 → {len(pe_daily)} 日")
    except Exception as e:
        print(f"  ✗ PE 采集失败: {e}")
        return pd.DataFrame()

    result = pe_daily.join(bond, how="inner")
    result["value"] = (1 / result["pe_ttm"]) - (result["bond_yield"] / 100)
    result = result[["value"]].dropna()
    result.index.name = "date"
    print(f"  共 {len(result)} 行")
    return result


# ═══════════════════════════════════════════════════════════
# 指标 8：存款市值比  (akshare 月频 + tushare 总市值)
# ═══════════════════════════════════════════════════════════

def fetch_deposit_market_ratio() -> pd.DataFrame:
    print("\n[8/10] 存款市值比 (deposit_market_ratio)")

    # ── 住户存款（akshare 月频） ──
    try:
        import akshare as ak
        df = ak.macro_rmb_deposit()
        # 找"储蓄存款"列（注意：列名虽含"新增"，实际值是存量余额）
        # 验证：2026-04 值 ≈ 171万亿，接近中国住户存款实际余额（~150万亿）
        dep_col = [c for c in df.columns if "储蓄存款" in str(c) and "数量" in str(c)]
        if dep_col:
            dep = df[["月份", dep_col[0]]].copy()
            dep.columns = ["date", "deposit"]
            dep["date"] = pd.to_datetime(dep["date"])
            dep = dep.set_index("date").sort_index()
            # 值已经是存量余额（亿元），无需 cumsum()
            dep["deposit"] = pd.to_numeric(dep["deposit"], errors="coerce") * 1e8  # 亿元→元
            print(f"  住户存款余额: {len(dep)} 月, 最新={dep['deposit'].iloc[-1]/1e8:.0f}亿")
            deposit_daily = resample_to_daily(dep["deposit"].dropna(), method="ffill")
        else:
            print(f"  可用列: {df.columns.tolist()}")
            # 备用：使用"新增存款-数量"（也是存量余额）
            dep = df[["月份", "新增存款-数量"]].copy()
            dep.columns = ["date", "deposit"]
            dep["date"] = pd.to_datetime(dep["date"])
            dep = dep.set_index("date").sort_index()
            dep["deposit"] = pd.to_numeric(dep["deposit"], errors="coerce") * 1e8
            deposit_daily = resample_to_daily(dep["deposit"].dropna(), method="ffill")
            print(f"  总存款（代理）: {len(dep)} 月")
    except Exception as e:
        print(f"  ✗ 存款采集失败: {e}")
        return pd.DataFrame()

    # ── 总市值（akshare 月频插值） ──
    try:
        import akshare as ak
        df_cap = ak.macro_china_stock_market_cap()
        df_cap = df_cap.rename(columns={"数据日期": "date",
                                        "市价总值-上海": "mv_sh",
                                        "市价总值-深圳": "mv_sz"})
        df_cap["date"] = df_cap["date"].str.replace("年", "-").str.replace("月份", "").str.replace("月", "")
        df_cap["date"] = pd.to_datetime(df_cap["date"])
        df_cap = df_cap.set_index("date").sort_index()
        for c in ["mv_sh", "mv_sz"]:
            df_cap[c] = pd.to_numeric(df_cap[c], errors="coerce")
        df_cap["total_mv"] = (df_cap["mv_sh"] + df_cap["mv_sz"]) * 1e8
        total_mv = resample_to_daily(df_cap["total_mv"].dropna(), method="ffill").to_frame()
        print(f"  akshare 月频总市值: {len(df_cap)} 月 → {len(total_mv)} 日")
    except Exception as e:
        print(f"  ✗ 总市值采集失败: {e}")
        return pd.DataFrame()

    result = deposit_daily.to_frame("deposit").join(total_mv, how="inner")
    result["value"] = result["deposit"] / result["total_mv"] * 100  # 百分比
    result = result[["value"]].dropna()
    result.index.name = "date"
    print(f"  共 {len(result)} 行")
    return result


# ═══════════════════════════════════════════════════════════
# 指标 9：偏股基金三年年化  (akshare 日频 930950)
# ═══════════════════════════════════════════════════════════

def fetch_fund_3y_annual() -> pd.DataFrame:
    print("\n[9/10] 偏股基金三年年化收益率 (fund_3y_annual)")

    try:
        import akshare as ak
        print("  获取 930950 中证偏股基金指数…")
        df = ak.stock_zh_index_hist_csindex(symbol="930950")
        nav = df[["日期", "收盘"]].copy()
        nav.columns = ["date", "nav"]
        nav["date"] = pd.to_datetime(nav["date"])
        nav = nav.set_index("date").sort_index()
        nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
        nav = nav.dropna()
        print(f"  930950 净值: {len(nav)} 日, 范围 {nav.index[0].date()} ~ {nav.index[-1].date()}")

        # 日收益率
        nav["daily_return"] = nav["nav"].pct_change()

        # 滚动三年年化（750个交易日）
        window = 750
        nav["cum_return"] = (1 + nav["daily_return"]).rolling(window=window).apply(
            lambda x: x.prod(), raw=True
        )
        nav["value"] = (nav["cum_return"] ** (252 / window) - 1) * 100  # %
        nav = nav.dropna(subset=["value"])

        result = nav[["value"]].copy()
        result.index.name = "date"
        print(f"  三年年化收益率: {len(result)} 行（窗口{window}个交易日）")
        return result

    except Exception as e:
        print(f"  ✗ 偏股基金指数采集失败: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════
# 指标 10：股债基金比  (akshare 月频基金规模)
# ═══════════════════════════════════════════════════════════

def fetch_equity_bond_fund_ratio() -> pd.DataFrame:
    """
    股债基金比 = 股票型基金总规模 / 债券型基金总规模

    数据来源：
      - 历史数据：使用 中证股票基金指数(H11021) / 中证债券基金指数(H11023) 比值代理
        （指数价格比反映了股债基金相对表现，与规模比高度相关）
      - 最新数据：从 fund_aum_hist_em 获取天天基金网全市场分类规模快照
    """
    print("\n[10/10] 股债基金比 (equity_bond_fund_ratio)")

    import akshare as ak
    import pandas as pd

    # ── 历史数据：H11021/H11023 指数比值 ──
    print("  获取中证股票/债券基金指数比值（历史代理）…")
    try:
        df_eq = ak.stock_zh_index_hist_csindex(symbol="H11021")  # 中证股票基金指数
        df_bd = ak.stock_zh_index_hist_csindex(symbol="H11023")  # 中证债券基金指数

        eq_close = df_eq[["日期", "收盘"]].copy()
        eq_close.columns = ["date", "eq_close"]
        eq_close["date"] = pd.to_datetime(eq_close["date"])

        bd_close = df_bd[["日期", "收盘"]].copy()
        bd_close.columns = ["date", "bd_close"]
        bd_close["date"] = pd.to_datetime(bd_close["date"])

        merged = eq_close.merge(bd_close, on="date", how="inner")
        merged = merged.set_index("date").sort_index()
        merged["eq_close"] = pd.to_numeric(merged["eq_close"], errors="coerce")
        merged["bd_close"] = pd.to_numeric(merged["bd_close"], errors="coerce")
        merged["value"] = merged["eq_close"] / merged["bd_close"]
        merged = merged.dropna(subset=["value"])

        print(f"  指数比值: {len(merged)} 日, 范围 {merged.index[0].date()} ~ {merged.index[-1].date()}")
        print(f"  最新比值: {merged['value'].iloc[-1]:.4f}")
        result = merged[["value"]].copy()
    except Exception as e:
        print(f"  ✗ 指数数据采集失败: {e}")
        result = pd.DataFrame()

    # ── 补充最新实际规模快照 ──
    try:
        print("  获取天天基金网全市场分类规模（最新快照）…")
        df_scale = ak.fund_aum_hist_em()
        if "股票型" in df_scale.columns and "债券型" in df_scale.columns:
            total_eq = pd.to_numeric(df_scale["股票型"], errors="coerce").sum()
            total_bd = pd.to_numeric(df_scale["债券型"], errors="coerce").sum()
            if total_bd > 0 and total_eq > 0:
                scale_ratio = total_eq / total_bd
                today = pd.Timestamp.now().normalize()
                print(f"  实际规模比: 股票型≈{total_eq:.0f}亿 / 债券型≈{total_bd:.0f}亿 = {scale_ratio:.4f}")

                # 修正指数比值使其对齐到实际规模比（用缩放因子）
                if not result.empty:
                    latest_idx = result.index[-1]
                    latest_ratio = result.loc[latest_idx, "value"]
                    scale_factor = scale_ratio / latest_ratio
                    # 对历史数据应用缩放因子，使最新值对齐实际规模
                    result["value"] = result["value"] * scale_factor
                    print(f"  已对齐实际规模比（缩放因子={scale_factor:.4f}）")

                    # 如果最新快照日期不在结果中，追加一行
                    if today not in result.index:
                        new_row = pd.DataFrame({"value": [scale_ratio]}, index=[today])
                        new_row.index.name = "date"
                        result = pd.concat([result, new_row])
                        result = result.sort_index()
    except Exception as e:
        print(f"  规模快照获取失败（仅使用指数比值）: {e}")

    if result.empty:
        print("  ⚠ 无法获取任何数据")
        return pd.DataFrame()

    result.index.name = "date"
    print(f"  共 {len(result)} 行")
    return result


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

FETCH_FUNCTIONS = {
    "buffett_ratio": fetch_buffett_ratio,
    "turnover_m2": fetch_turnover_m2,
    "turnover_rate": fetch_turnover_rate,
    "margin_ratio": fetch_margin_ratio,
    "pe_valuation": fetch_pe_valuation,
    "account_open": fetch_account_open,
    "equity_bond_spread": fetch_equity_bond_spread,
    "deposit_market_ratio": fetch_deposit_market_ratio,
    "fund_3y_annual": fetch_fund_3y_annual,
    "equity_bond_fund_ratio": fetch_equity_bond_fund_ratio,
}

# 全部指标现已使用 akshare（不依赖 tushare）
AKSHARE_ONLY = {ind.id for ind in INDICATORS}


def main():
    parser = argparse.ArgumentParser(description="采集牛市热度指数历史数据")
    parser.add_argument("--indicators", type=str, default=None,
                        help="指标ID逗号分隔，如 buffett_ratio,turnover_rate")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已有数据文件")
    args = parser.parse_args()

    ind_ids = [i.strip() for i in args.indicators.split(",")] if args.indicators else \
              [ind.id for ind in INDICATORS]

    pro = get_tushare_pro()
    os.makedirs(RAW_DIR, exist_ok=True)

    for ind_id in ind_ids:
        if ind_id not in FETCH_FUNCTIONS:
            print(f"  ⚠ 未知指标: {ind_id}")
            continue

        raw_path = os.path.join(RAW_DIR, f"{ind_id}.csv")
        if os.path.exists(raw_path) and not args.overwrite:
            existing = safe_read_csv(raw_path)
            if len(existing) > 0:
                print(f"\n[{ind_id}] 已有 {len(existing)} 行，跳过（--overwrite 覆盖）")
                continue

        print(f"\n>>> [{ind_id}] 开始采集…")
        try:
            fn = FETCH_FUNCTIONS[ind_id]
            df = fn()
            if df is not None and not df.empty:
                save_dataframe(df, raw_path)
            else:
                print(f"  ⚠ {ind_id} 无数据")
        except Exception as e:
            print(f"  ✗ {ind_id} 异常: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ 采集完成！")
