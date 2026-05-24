"""
回测验证脚本

功能：
1. 计算 2005-01-01 至昨日的每日指数（复用 update_index.py 的回测能力）
2. 验证 5 个关键时间点的阈值
3. 生成回测图表
4. 输出回测报告

验证点：
  - 2007.10 牛市顶部 → 预期指数 ≥80
  - 2008.12 底部 → 预期指数 ≤20
  - 2015.06 杠杆牛顶部 → 预期指数 ≥80
  - 2018.12 底部 → 预期指数 ≤20
  - 2021.02 茅指数顶部 → 预期指数 ≥70

运行方式：
    TUSHARE_TOKEN=xxx python scripts/backtest.py
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import INDEX_HISTORY_PATH, CHART_PATH, RAW_DIR, INDICATORS
from scripts.update_index import (
    update_single_day, get_index_history, save_index_history, generate_chart
)


# ── 验证点 ──────────────────────────────────────────────
VERIFICATION_POINTS = [
    {"label": "2007牛市顶部", "date": "2007-10-16", "expected_min": 80, "expected_max": 100},
    {"label": "2008底部",     "date": "2008-12-01", "expected_min": 0,  "expected_max": 20},
    {"label": "2015杠杆牛顶部", "date": "2015-06-12", "expected_min": 80, "expected_max": 100},
    {"label": "2018底部",     "date": "2018-12-01", "expected_min": 0,  "expected_max": 20},
    {"label": "2021茅指数顶部", "date": "2021-02-18", "expected_min": 70, "expected_max": 100},
]


def run_backtest() -> pd.DataFrame:
    """执行回测，返回完整的指数历史 DataFrame"""
    print("=" * 60)
    print("   牛市热度指数回测")
    print("=" * 60)

    # 检查是否已有计算好的历史
    index_df = get_index_history()
    if len(index_df) > 0:
        print(f"  发现已有指数历史 ({len(index_df)} 行)")
        print("  如需重新计算，请删除 data/index_history.csv 后运行")

        # 检查是否需要补充
        earliest = index_df.index.min()
        if earliest > pd.Timestamp("2005-01-01"):
            print(f"  历史最早日期 {earliest.date()} > 2005-01-01，需要补充")
        else:
            return index_df

    # 获取所有可计算日期
    # 注意：月频指标的原始 CSV 只有月末日期，但 get_latest_values()
    # 通过向前填充支持任意日期。所以需要用日频指标填补中间日期。
    print("\n正在扫描原始数据范围…")

    # 收集所有原始数据日期
    all_raw_dates = set()
    for ind in INDICATORS:
        path = os.path.join(RAW_DIR, f"{ind.id}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if not df.empty:
                for d in df.index:
                    all_raw_dates.add(d)

    all_raw_dates = sorted(all_raw_dates)
    all_raw_dates = [d for d in all_raw_dates if d >= pd.Timestamp("2005-01-01")]

    if len(all_raw_dates) == 0:
        print("❌ 没有原始数据，请先运行 fetch_initial_hist.py 采集数据")
        sys.exit(1)

    # 找到覆盖最全的日频指标的日期范围，补全月频指标中间缺失的交易日
    # 用换手率（覆盖面最广的日频指标）作为交易日期基准
    all_dates = set(all_raw_dates)
    turnover_path = os.path.join(RAW_DIR, "turnover_rate.csv")
    if os.path.exists(turnover_path):
        df_tr = pd.read_csv(turnover_path, index_col=0, parse_dates=True)
        if not df_tr.empty:
            # 取换手率覆盖范围内的所有交易日
            tr_dates = set(df_tr.index)
            tr_min, tr_max = min(tr_dates), max(tr_dates)
            # 将其他的月频指标日期也用日频交易日期补全
            for d in all_raw_dates:
                if tr_min <= d <= tr_max:
                    # 找到 d 之后最近的交易日
                    pass
            # 合并所有换手率的交易日期（在换手率覆盖范围内）
            all_dates.update(d for d in tr_dates if tr_min <= d)

    all_dates = sorted(all_dates)
    print(f"共 {len(all_dates)} 个待计算日期（原始数据 {len(all_raw_dates)} 个，补全了 {len(all_dates)-len(all_raw_dates)} 个交易日）")

    if len(all_dates) == 0:
        print("❌ 没有原始数据，请先运行 fetch_initial_hist.py 采集数据")
        sys.exit(1)

    # 逐个计算（使用 update_index 的单日更新）
    index_rows = []
    for i, date in enumerate(all_dates):
        date_str = date.strftime("%Y-%m-%d")
        row = update_single_day(date_str)
        if row is not None:
            index_rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{len(all_dates)}")

    if not index_rows:
        print("❌ 未能计算任何日期的指数")
        sys.exit(1)

    index_df = pd.DataFrame(index_rows)
    index_df.index.name = "date"
    index_df = index_df.sort_index()
    save_index_history(index_df)
    return index_df


def validate_points(index_df: pd.DataFrame) -> list:
    """验证 5 个关键时间点"""
    print("\n" + "=" * 60)
    print("   关键时间点验证")
    print("=" * 60)

    results = []
    for vp in VERIFICATION_POINTS:
        dt = pd.Timestamp(vp["date"])
        # 查找最近的有效值
        mask = index_df.index <= dt
        if mask.any():
            closest = index_df.loc[mask].iloc[-1]
            actual = closest["index_value"]
            actual_date = closest.name

            passed = vp["expected_min"] <= actual <= vp["expected_max"]
            status = "✅" if passed else "❌"
            detail = (
                f"  预期: [{vp['expected_min']}-{vp['expected_max']}], "
                f"实际: {actual:.2f} ({actual_date.date()})"
            )
            print(f"  {status} {vp['label']} ({vp['date']})")
            print(f"    {detail}")
            results.append({
                "label": vp["label"],
                "target_date": vp["date"],
                "actual_date": str(actual_date.date()),
                "expected_range": f"{vp['expected_min']}-{vp['expected_max']}",
                "actual_value": round(actual, 2),
                "passed": passed,
            })
        else:
            print(f"  ⚠ {vp['label']}: 无可用数据")
            results.append({
                "label": vp["label"],
                "target_date": vp["date"],
                "actual_date": "",
                "expected_range": f"{vp['expected_min']}-{vp['expected_max']}",
                "actual_value": None,
                "passed": False,
            })

    return results


def compute_stats(index_df: pd.DataFrame) -> dict:
    """计算统计摘要"""
    values = index_df["index_value"].dropna()
    return {
        "数据范围": f"{index_df.index.min().date()} → {index_df.index.max().date()}",
        "总天数": len(index_df),
        "平均值": round(values.mean(), 2),
        "中位数": round(values.median(), 2),
        "标准差": round(values.std(), 2),
        "最小值": round(values.min(), 2),
        "最小值日期": str(values.idxmin().date()),
        "最大值": round(values.max(), 2),
        "最大值日期": str(values.idxmax().date()),
        "很热占比(≥80)": f"{round((values >= 80).mean() * 100, 1)}%",
        "偏热占比(60-80)": f"{round(((values >= 60) & (values < 80)).mean() * 100, 1)}%",
        "适中占比(40-60)": f"{round(((values >= 40) & (values < 60)).mean() * 100, 1)}%",
        "偏冷占比(20-40)": f"{round(((values >= 20) & (values < 40)).mean() * 100, 1)}%",
        "很冷占比(<20)": f"{round((values < 20).mean() * 100, 1)}%",
    }


def print_report(validation: list, stats: dict):
    """打印回测报告"""
    print("\n" + "=" * 60)
    print("   回测报告")
    print("=" * 60)

    print("\n📊 统计概览:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n✅ 关键点验证:")
    for r in validation:
        icon = "✅" if r["passed"] else "❌"
        print(f"  {icon} {r['label']}: 预期 {r['expected_range']}, "
              f"实际 {r['actual_value']} ({r['actual_date']})")

    passed = sum(1 for r in validation if r["passed"])
    total = len(validation)
    print(f"\n  验证通过率: {passed}/{total}")


def main():
    # 1. 运行回测
    index_df = run_backtest()

    # 2. 生成图表
    print("\n📊 生成回测图表…")
    generate_chart(index_df)
    print(f"  图表已保存到 {CHART_PATH}")

    # 3. 验证关键点
    validation = validate_points(index_df)

    # 4. 统计
    stats = compute_stats(index_df)

    # 5. 报告
    print_report(validation, stats)

    # 6. 保存验证结果
    results_df = pd.DataFrame(validation)
    results_path = os.path.join(os.path.dirname(INDEX_HISTORY_PATH), "backtest_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n📄 验证结果已保存到 {results_path}")


if __name__ == "__main__":
    main()
