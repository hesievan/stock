"""
生成 HTML 格式的每日指数报告

输出：
  docs/index.html — 自包含 HTML，无外部依赖

替代 matplot 简报图。支持手机/桌面浏览。
"""

import os
import sys
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    INDEX_HISTORY_PATH, CHART_PATH, RAW_DIR,
    INDICATORS, get_heat_level,
)

REPORT_PATH = os.path.join(os.path.dirname(CHART_PATH), "index.html")


def _read_data() -> tuple:
    """读取最新指数和各指标数据"""
    df = pd.read_csv(INDEX_HISTORY_PATH, index_col=0, parse_dates=True)
    latest = df.iloc[-1]
    latest.name = df.index[-1]
    return df, latest


def _make_sparkline(index_df: pd.DataFrame) -> tuple:
    """生成迷你走势图的 HTML，返回 (bars_html, labels_html)"""
    values = index_df["index_value"].dropna()
    n = len(values)
    # 采样：最多 240 个点（21年 × ~12点/年）
    max_bars = 240
    step = max(1, n // max_bars)
    sampled = values.iloc[::-1][::step][::-1]  # 从尾到头采样，再反转

    min_v = sampled.min()
    max_v = sampled.max()
    span = max(max_v - min_v, 1)

    # 每 N 个 bar 显示一个年份标签
    label_interval = max(1, len(sampled) // 10)  # 约10个标签

    bars = []
    labels = []
    for i, (dt, v) in enumerate(sampled.items()):
        height_pct = max((v - min_v) / span * 100, 1)
        color = _heat_color(v)
        date_str = dt.strftime("%Y-%m-%d")
        bars.append(
            f'<div class="bar" style="height:{height_pct:.0f}%;background:{color};" '
            f'title="{date_str}: {v:.1f}"></div>'
        )
        # x 轴标签：每 label_interval 个 bar 显示一个年份
        if i % label_interval == 0 or i == len(sampled) - 1:
            fmt = "%Y-%m" if i < label_interval or i >= len(sampled) - label_interval else "%Y"
            labels.append(
                f'<span class="xlabel" style="flex:{label_interval};text-align:{"left" if i==0 else "right" if i>=len(sampled)-label_interval else "center"};">'
                f'{dt.strftime(fmt)}</span>'
            )

    bars_html = "\n".join(bars)
    labels_html = "\n".join(labels)
    return bars_html, labels_html


def _heat_color(val: float) -> str:
    """百分位对应的颜色"""
    if val >= 80:
        return "#F44336"
    if val >= 60:
        return "#FF9800"
    if val >= 40:
        return "#9E9E9E"
    if val >= 20:
        return "#4CAF50"
    return "#00BCD4"


def _heat_gradient(val: float) -> str:
    """从冷到热的渐变色条"""
    # 蓝(0) → 绿(25) → 灰(50) → 橙(75) → 红(100)
    r = min(255, int(val * 2.55 * 2)) if val > 50 else int(val * 2.55)
    g = max(0, 255 - int(val * 5.1)) if val < 50 else max(0, 255 - int((val - 50) * 5.1))
    b = max(0, 255 - int(val * 5.1))
    return f"rgb({min(r,255)},{min(g,255)},{min(b,255)})"


def _indicator_bars(latest: pd.Series) -> str:
    """生成各指标百分位的横向条形图 HTML"""
    rows = []
    for ind in INDICATORS:
        val = latest.get(ind.id)
        if pd.isna(val) or val is None:
            rows.append(f"""
    <tr class="ind-row">
      <td class="ind-name">{ind.name}</td>
      <td class="ind-val muted">N/A</td>
      <td class="ind-bar">
        <div class="bar-track"><div class="bar-fill na" style="width:0%"></div></div>
      </td>
    </tr>""")
        else:
            v = float(val)
            color = _heat_color(v)
            bg = _heat_gradient(v)
            rows.append(f"""
    <tr class="ind-row">
      <td class="ind-name">{ind.name}</td>
      <td class="ind-val" style="color:{color};font-weight:bold;">{v:.1f}</td>
      <td class="ind-bar">
        <div class="bar-track">
          <div class="bar-fill" style="width:{v:.1f}%;background:linear-gradient(90deg,{bg},{color});"></div>
          <span class="bar-label">{v:.0f}</span>
        </div>
      </td>
    </tr>""")
    return "\n".join(rows)


def _stats_table(index_df: pd.DataFrame) -> str:
    """生成历史统计信息"""
    v = index_df["index_value"]
    return f"""
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">数据范围</div>
        <div class="stat-value">{index_df.index[0].strftime('%Y-%m-%d')} ~ {index_df.index[-1].strftime('%Y-%m-%d')}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">交易日数</div>
        <div class="stat-value">{len(v)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">平均值</div>
        <div class="stat-value">{v.mean():.1f}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">最大值</div>
        <div class="stat-value" style="color:#F44336;">{v.max():.1f}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">最小值</div>
        <div class="stat-value" style="color:#00BCD4;">{v.min():.1f}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">当前值</div>
        <div class="stat-value" style="color:{_heat_color(v.iloc[-1])};">
          {v.iloc[-1]:.1f} / 100
        </div>
      </div>
    </div>
    """


def _heat_legend() -> str:
    """热度等级图例"""
    items = [
        (0, "00BCD4", "很冷"),
        (25, "4CAF50", "偏冷"),
        (50, "9E9E9E", "适中"),
        (75, "FF9800", "偏热"),
        (100, "F44336", "很热"),
    ]
    segments = ""
    for pct, color, label in items:
        segments += f'<div class="legend-item"><span class="legend-dot" style="background:#{color};"></span>{label}</div>\n'
    return segments


def generate_report() -> str:
    """生成完整的 HTML 报告，返回 HTML 字符串"""
    df, latest = _read_data()
    index_val = latest["index_value"]
    heat = get_heat_level(index_val)
    today_str = latest.name.strftime("%Y-%m-%d")

    sparkline, xlabels = _make_sparkline(df)
    ind_bars = _indicator_bars(latest)
    stats = _stats_table(df)
    legend = _heat_legend()

    color_map = {"很冷": "#00BCD4", "偏冷": "#4CAF50", "适中": "#9E9E9E",
                 "偏热": "#FF9800", "很热": "#F44336"}
    main_color = color_map.get(heat["level"], "#9E9E9E")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>牛市热度指数 · {today_str}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
    background: #f0f2f5;
    color: #333;
    padding: 16px;
    line-height: 1.5;
  }}
  .container {{ max-width: 800px; margin: 0 auto; }}

  /* ── 卡片 ── */
  .card {{
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    padding: 24px 20px;
    margin-bottom: 16px;
  }}
  .card-title {{
    font-size: 13px;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    font-weight: 600;
  }}

  /* ── 顶部大数字 ── */
  .hero {{
    text-align: center;
    padding: 28px 20px 20px;
  }}
  .hero .date {{ font-size: 14px; color: #999; margin-bottom: 8px; }}
  .hero .big-number {{
    font-size: 72px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -2px;
  }}
  .hero .big-number .unit {{
    font-size: 24px;
    font-weight: 400;
    color: #aaa;
  }}
  .hero .level-badge {{
    display: inline-block;
    padding: 6px 20px;
    border-radius: 20px;
    font-size: 20px;
    font-weight: 700;
    color: #fff;
    margin-top: 8px;
  }}
  .hero .desc {{
    font-size: 14px;
    color: #666;
    margin-top: 8px;
  }}

  /* ── 走势图 ── */
  .chart-box {{ position: relative; overflow-x: auto; }}
  .chart {{
    display: flex;
    align-items: flex-end;
    height: 160px;
    gap: 1px;
    min-width: 100%;
  }}
  .chart .bar {{
    flex: 1;
    min-width: 2px;
    border-radius: 1px 1px 0 0;
    opacity: 0.85;
    transition: opacity 0.15s;
  }}
  .chart .bar:hover {{ opacity: 1; }}

  /* ── 图例 ── */
  .legend {{
    display: flex;
    gap: 16px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 8px 0 0;
    font-size: 12px;
    color: #666;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }}

  /* ── 指标条形 ── */
  .ind-table {{ width: 100%; border-collapse: collapse; }}
  .ind-row td {{ padding: 6px 4px; }}
  .ind-name {{
    font-size: 13px;
    color: #555;
    white-space: nowrap;
    width: 140px;
    padding-right: 12px !important;
  }}
  .ind-val {{
    font-size: 15px;
    width: 48px;
    text-align: right;
    padding-right: 12px !important;
    font-variant-numeric: tabular-nums;
  }}
  .ind-val.muted {{ color: #bbb !important; }}
  .ind-bar {{ width: auto; }}
  .bar-track {{
    position: relative;
    width: 100%;
    height: 20px;
    background: #f0f0f0;
    border-radius: 10px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 10px;
    transition: width 0.3s;
  }}
  .bar-fill.na {{ background: #e8e8e8 !important; }}
  .bar-label {{
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 11px;
    color: #666;
    font-weight: 600;
  }}

  /* ── 统计 ── */
  .stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px;
  }}
  .stat-card {{
    background: #f8f9fa;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
  }}
  .stat-label {{ font-size: 12px; color: #999; margin-bottom: 4px; }}
  .stat-value {{ font-size: 16px; font-weight: 700; color: #333; }}

  /* ── 验证点 ── */
  .vp-list {{ list-style: none; }}
  .vp-list li {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #f0f0f0;
    font-size: 14px;
  }}
  .vp-list li:last-child {{ border: none; }}
  .vp-pass {{ color: #4CAF50; }}

  /* ── 响应式 ── */
  @media (max-width: 600px) {{
    .hero .big-number {{ font-size: 56px; }}
    .ind-name {{ font-size: 12px; width: 100px; }}
    .chart {{ height: 120px; }}
  }}

  /* ── 区段着色 ── */
  .zone-labels {{
    display: flex;
    justify-content: space-between;
    margin-top: -4px;
    font-size: 10px;
    color: #bbb;
    padding: 0 2px;
  }}

  /* ── X 轴年份 ── */
  .xaxis {{
    display: flex;
    width: 100%;
    margin-top: 2px;
    padding: 0 1px;
  }}
  .xaxis .xlabel {{
    font-size: 10px;
    color: #999;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
</style>
</head>
<body>
<div class="container">

  <!-- ═══ 大数字 ═══ -->
  <div class="card hero">
    <div class="date">📅 {today_str}</div>
    <div class="big-number" style="color:{main_color};">
      {index_val:.1f}<span class="unit">/100</span>
    </div>
    <div class="level-badge" style="background:{main_color};">{heat['level']}</div>
    <div class="desc">{heat['description']}</div>
  </div>

  <!-- ═══ 走势图 ═══ -->
  <div class="card">
    <div class="card-title">📈 全历史走势</div>
    <div class="chart-box">
      <div class="chart">
        {sparkline}
      </div>
      <div class="xaxis">
        {xlabels}
      </div>
    </div>
    <div class="zone-labels">
      <span>0</span><span>很冷</span><span>偏冷</span><span>适中</span><span>偏热</span><span>很热</span><span>100</span>
    </div>
    <div class="legend">{legend}</div>
  </div>

  <!-- ═══ 各指标百分位 ═══ -->
  <div class="card">
    <div class="card-title">📊 各指标百分位</div>
    <table class="ind-table">
      {ind_bars}
    </table>
  </div>

  <!-- ═══ 统计 ═══ -->
  <div class="card">
    <div class="card-title">📋 历史统计</div>
    {stats}
  </div>

  <!-- ═══ 回测验证 ═══ -->
  <div class="card">
    <div class="card-title">✅ 回测验证</div>
    <ul class="vp-list">
      <li><span>2007.10 牛市顶部</span><span class="vp-pass">50.35</span></li>
      <li><span>2008.12 底部</span><span class="vp-pass">56.25</span></li>
      <li><span>2015.06 杠杆牛顶部</span><span class="vp-pass">84.11 ✅</span></li>
      <li><span>2018.12 底部</span><span class="vp-pass">28.96</span></li>
      <li><span>2021.02 茅指数顶部</span><span class="vp-pass">70.02 ✅</span></li>
    </ul>
  </div>

  <!-- ═══ 页脚 ═══ -->
  <div class="card" style="text-align:center;color:#999;font-size:12px;">
    <p>🤖 牛市热度指数 · 10个指标合成 · 核心12% / 辅助7%</p>
    <p>数据来源: akshare · 更新时间: {today_str} · 不构成投资建议</p>
  </div>

</div>
</body>
</html>"""
    return html


def main():
    """主入口：生成 HTML 报告并保存"""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    html = generate_report()
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    file_size = len(html.encode("utf-8"))
    print(f"📄 HTML 报告已生成 → {REPORT_PATH}")
    print(f"   大小: {file_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
