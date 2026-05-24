"""
百分位计算模块

历史百分位定义：
    （小于当前值的历史数据个数）/（历史数据总个数）× 100
- 严格小于当前值（等于时不计数）
- 取值范围 0–100
- 若当前值 > 所有历史值，百分位=100
- 若当前值 ≤ 所有历史值中最小值，百分位=0
"""

import numpy as np
import pandas as pd
from typing import Union


def calc_percentile(
    value: Union[float, int],
    history: Union[np.ndarray, pd.Series, list],
) -> float:
    """
    计算单个值在历史序列中的百分位（严格小于）。

    Parameters
    ----------
    value : float/int
        当前值
    history : array-like
        历史数据序列

    Returns
    -------
    float
        百分位（0–100）
    """
    arr = np.asarray(history, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan

    n_less = np.sum(arr < value)
    total = len(arr)
    percentile = (n_less / total) * 100.0
    return percentile


def calc_percentile_series(
    series: pd.Series,
    min_periods: int = 1,
) -> pd.Series:
    """
    对时间序列计算滚动百分位：每个时间点的值相对于截至前一日的全部历史。

    注意：这与需求中的定义略有不同——需求是"历史百分位"，
    即当日值相对于历史全部数据的百分位（而非滚动）。
    但初始化和回测时需要一次性计算全部历史百分位，
    此时用 rolling(expanding) 更合适。

    为了回测/初始化场景：每个 t 日的百分位 = 该值相对于 [起始日, t-1] 日的百分位。

    Parameters
    ----------
    series : pd.Series
        时间序列，index 为日期
    min_periods : int
        最少需要多少历史数据点才能计算百分位（默认1）

    Returns
    -------
    pd.Series
        百分位序列
    """
    series = series.sort_index().dropna()
    result = pd.Series(index=series.index, dtype=float, name="percentile")

    for i in range(len(series)):
        if i == 0:
            # 第一天没有历史数据，百分位为 NaN
            result.iloc[i] = np.nan
        else:
            history = series.iloc[:i].values  # 到前一天为止
            current = series.iloc[i]
            if len(history) >= min_periods:
                result.iloc[i] = calc_percentile(current, history)
            else:
                result.iloc[i] = np.nan

    return result


def calc_percentile_simple(
    series: pd.Series,
) -> pd.Series:
    """
    简化版：每个值相对于整个序列的百分位（包含自身）。
    用于最终一次性计算：当日百分位 = 该值相对于包含自身在内的全部历史。

    注意：这样做会使最新百分位略微偏高（自身计入分母），
    但数据量足够大时差异可忽略。
    对照《需求说明书》，严格定义是"小于当前值的历史数据个数/历史总个数"，
    即不包括当日自身。本函数用于快速初始化，正式 update_index 中应使用
    calc_percentile_series 或 calc_percentile。
    """
    series = series.sort_index().dropna()
    values = series.values
    result = np.array([
        np.sum(values[:i] < values[i]) / max(i, 1) * 100.0
        if i > 0 else np.nan
        for i in range(len(values))
    ])
    return pd.Series(result, index=series.index, name="percentile")


if __name__ == "__main__":
    # 简单测试
    import numpy as np
    test_history = [10, 20, 30, 40, 50]
    for val in [5, 10, 25, 50, 60]:
        p = calc_percentile(val, test_history)
        print(f"value={val:>3}, history={test_history}, percentile={p:.1f}")

    # 测试序列
    ts = pd.Series(
        [10, 20, 15, 30, 25, 40, 35, 50, 45, 60],
        index=pd.date_range("2020-01-01", periods=10, freq="D"),
    )
    result = calc_percentile_series(ts)
    print("\n滚动百分位测试:")
    for date, p in result.items():
        print(f"  {date.date()}: value={ts[date]:.0f}, percentile={p:.1f}" if not np.isnan(p) else
              f"  {date.date()}: value={ts[date]:.0f}, percentile=NaN")
