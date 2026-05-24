"""
第二/三阶段核心：每日指数合成

功能：
1. 读取当日各指标最新值（或沿用最近一期）
2. 计算每个指标的当日百分位（相对于全部历史数据）
3. 处理缺失值（权重重分配）
4. 加权合成指数 → 判断热度等级
5. 追加到 index_history.csv
6. 可选：生成热度仪表图

运行方式：
    TUSHARE_TOKEN=xxx python scripts/update_index.py
    TUSHARE_TOKEN=xxx python scripts/update_index.py --date 2024-01-15  # 指定日期
    TUSHARE_TOKEN=xxx python scripts/update_index.py --backtest          # 回测模式
"""

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    RAW_DIR, INDEX_HISTORY_PATH, CHART_PATH,
    INDICATORS, INDICATOR_MAP, get_heat_level,
    TUSHARE_TOKEN,
)
from scripts.calc_percentile import calc_percentile


# ═══════════════════════════════════════════════════════════
# 数据读取
# ═══════════════════════════════════════════════════════════

def read_raw_data(indicator_id: str) -> pd.DataFrame:
    """读取某个指标的原始历史数据"""
    path = os.path.join(RAW_DIR, f"{indicator_id}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df.sort_index()


def get_latest_values(target_date: Optional[str] = None) -> Dict[str, Optional[float]]:
    """
    获取目标日期（默认今天）各指标的最新有效值。
    若当日无数据，则沿用最近一期。
    """
    if target_date:
        target = pd.Timestamp(target_date)
    else:
        target = pd.Timestamp.now().normalize()

    values: Dict[str, Optional[float]] = {}

    for ind in INDICATORS:
        df = read_raw_data(ind.id)
        if df.empty or "value" not in df.columns:
            print(f"  ⚠ {ind.id}: 无历史数据")
            values[ind.id] = None
            continue

        # 找到 target 当日或最近一期
        mask = df.index <= target
        if mask.any():
            latest = df.loc[mask].iloc[-1]
            values[ind.id] = float(latest["value"])
            # 检查是否就是 target 日的数据
            if latest.name == target:
                print(f"  ✓ {ind.id}: {latest.name.date()} = {latest['value']:.4f}")
            else:
                print(f"  ~ {ind.id}: 沿用 {latest.name.date()} 值 = {latest['value']:.4f}")
        else:
            print(f"  ⚠ {ind.id}: 无 <= {target.date()} 的数据")
            values[ind.id] = None

    return values


# ═══════════════════════════════════════════════════════════
# 百分位计算
# ═══════════════════════════════════════════════════════════

def calc_all_percentiles(
    values: Dict[str, Optional[float]],
) -> Tuple[Dict[str, float], List[str]]:
    """
    计算每个指标当前值的百分位。

    Returns
    -------
    percentiles : dict
        各指标百分位
    missing : list
        缺失（无法计算百分位）的指标ID列表
    """
    percentiles: Dict[str, float] = {}
    missing: List[str] = []

    for ind in INDICATORS:
        current = values.get(ind.id)
        if current is None or (isinstance(current, float) and np.isnan(current)):
            missing.append(ind.id)
            reason = "NaN" if (isinstance(current, float) and np.isnan(current)) else "无有效值"
            print(f"  ✗ {ind.id}: {reason}，标记缺失")
            continue

        df = read_raw_data(ind.id)
        if df.empty or "value" not in df.columns:
            missing.append(ind.id)
            print(f"  ✗ {ind.id}: 无历史数据，标记缺失")
            continue

        history = df["value"].dropna().values
        if len(history) == 0:
            missing.append(ind.id)
            print(f"  ✗ {ind.id}: 历史数据为空，标记缺失")
            continue

        # 百分位不包含当前值自身（确保分母 = 历史总数不含当前值）
        history_excl = history[~np.isclose(history, current, rtol=1e-4)]
        if len(history_excl) == 0:
            missing.append(ind.id)
            print(f"  ✗ {ind.id}: 排除当前值后无历史数据")
            continue
        pct = calc_percentile(current, history_excl)
        percentiles[ind.id] = pct
        print(f"    {ind.id} 百分位: {pct:.2f} (当前={current:.4f}, 历史={len(history_excl)}个点)")

    return percentiles, missing


# ═══════════════════════════════════════════════════════════
# 指数合成
# ═══════════════════════════════════════════════════════════

def compute_index(
    percentiles: Dict[str, float],
    missing: List[str],
) -> float:
    """
    加权合成指数：基于 INDICATORS 中各指标的自定义权重。
    缺失指标的权重按比例均分给有效指标。

    Parameters
    ----------
    percentiles : dict
        有效指标的百分位（0–100）
    missing : list
        缺失的指标ID列表

    Returns
    -------
    float
        合成指数（0–100）。若全部缺失则返回 NaN。
    """
    n_total = len(INDICATORS)  # = 10
    n_valid = n_total - len(missing)

    if n_valid == 0:
        return np.nan

    # 缺失指标的权重均分给有效指标
    missing_weight = sum(ind.weight for ind in INDICATORS if ind.id in missing)
    total_weight = 1.0 - missing_weight  # 有效指标原始权重之和
    # 重分配后：每个有效指标的权重 = 原始权重 + 分到的缺失权重
    scale = 1.0 / total_weight if total_weight > 0 else 0
    weighted_sum = sum(
        percentiles[ind.id] * ind.weight
        for ind in INDICATORS
        if ind.id not in missing
    )
    index_value = weighted_sum * scale
    # 解释：
    #   weighted_sum = Σ(pct_i × w_i)    → 范围 0~100
    #   scale = 1 / Σw_有效               → 使总权重回归1.0
    #   最终 = (Σpct_i×w_i) / Σw_有效     → 归一化到0~100


    return index_value


def get_index_history() -> pd.DataFrame:
    """读取已有的指数历史"""
    if os.path.exists(INDEX_HISTORY_PATH) and os.path.getsize(INDEX_HISTORY_PATH) > 0:
        df = pd.read_csv(INDEX_HISTORY_PATH, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df
    return pd.DataFrame()


def save_index_history(index_df: pd.DataFrame):
    """保存指数历史"""
    os.makedirs(os.path.dirname(INDEX_HISTORY_PATH), exist_ok=True)
    index_df.to_csv(INDEX_HISTORY_PATH)
    print(f"\n✅ 已保存指数历史到 {INDEX_HISTORY_PATH} ({len(index_df)} 行)")


def generate_chart(index_df: pd.DataFrame, save_path: str = CHART_PATH):
    """生成简报仪表图（代替旧版简单走势图）"""
    try:
        from scripts.briefing import generate_briefing
        latest = index_df.iloc[-1]
        latest.name = index_df.index[-1]
        generate_briefing(latest, index_df, save_path=save_path)
        print(f"📊 简报已保存到 {save_path}")
    except Exception as e:
        print(f"  ⚠ 生成简报失败: {e}")


# ═══════════════════════════════════════════════════════════
# 单日更新
# ═══════════════════════════════════════════════════════════

def update_single_day(target_date: str) -> Optional[pd.Series]:
    """
    计算指定日期的指数。

    Returns
    -------
    pd.Series with columns: index_value, status, heat_desc, 各指标百分位...
    如无法计算则返回 None。
    """
    print(f"\n{'='*50}")
    print(f"📅 计算 {target_date} 的指数")
    print(f"{'='*50}")

    # 1. 获取最新值
    values = get_latest_values(target_date)
    if not any(v is not None for v in values.values()):
        print("  ✗ 所有指标均无数据，无法计算")
        return None

    # 2. 计算百分位
    percentiles, missing = calc_all_percentiles(values)

    if len(percentiles) == 0:
        print("  ✗ 无法计算任何指标的百分位")
        return None

    # 3. 合成指数
    index_value = compute_index(percentiles, missing)
    if np.isnan(index_value):
        print("  ✗ 指数合成失败")
        return None

    # 4. 热度等级
    heat = get_heat_level(index_value)

    print(f"\n{'─'*50}")
    print(f"🔥 指数值: {index_value:.2f}")
    print(f"📊 等级: {heat['level']} ({heat['range']})")
    print(f"💬 {heat['description']}")
    if missing:
        print(f"⚠ 缺失指标: {', '.join(missing)}")
    print(f"{'─'*50}")

    # 5. 构建行数据
    row = {
        "index_value": round(index_value, 2),
        "status": heat["level"],
        "heat_desc": heat["description"],
    }
    # 各指标百分位
    for ind in INDICATORS:
        col_name = ind.id
        if ind.id in percentiles:
            row[col_name] = round(percentiles[ind.id], 2)
        else:
            row[col_name] = np.nan

    return pd.Series(row, name=pd.Timestamp(target_date))


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="牛市热度指数更新")
    parser.add_argument("--date", type=str, default=None,
                        help="目标日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--backtest", action="store_true",
                        help="回测模式：对历史全部日期逐个计算")
    parser.add_argument("--chart", action="store_true", default=True,
                        help="生成图表（默认生成）")
    args = parser.parse_args()

    if args.backtest:
        # 回测模式：遍历所有有数据的日期
        print("🔁 回测模式：计算全部历史指数")
        all_dates = set()
        for ind in INDICATORS:
            df = read_raw_data(ind.id)
            if not df.empty:
                all_dates.update(df.index.strftime("%Y-%m-%d"))
        all_dates = sorted(all_dates)
        print(f"共 {len(all_dates)} 个待计算日期")

        index_rows = []
        for i, date_str in enumerate(all_dates):
            row = update_single_day(date_str)
            if row is not None:
                index_rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{len(all_dates)}")

        if index_rows:
            index_df = pd.DataFrame(index_rows)
            index_df.index.name = "date"
            save_index_history(index_df)
            if args.chart:
                generate_chart(index_df)
        return

    # 单日更新模式
    target = args.date or datetime.now().strftime("%Y-%m-%d")
    row = update_single_day(target)
    if row is None:
        print("❌ 无法计算指数")
        sys.exit(1)

    # 追加到历史文件
    index_df = get_index_history()
    if target in index_df.index.strftime("%Y-%m-%d"):
        print(f"  ℹ {target} 已有数据，更新覆盖")
        index_df = index_df.drop(pd.Timestamp(target), errors="ignore")

    new_row_df = row.to_frame().T
    new_row_df.index.name = "date"
    index_df = pd.concat([index_df, new_row_df])
    index_df = index_df.sort_index()
    save_index_history(index_df)

    if args.chart:
        generate_chart(index_df)


if __name__ == "__main__":
    main()
