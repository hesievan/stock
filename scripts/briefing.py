"""
牛市热度指数 — 每日简报生成器

生成：
  1. `docs/briefing.png` — 简报仪表图（指数走势 + 指标雷达 + 热力条）
  2. 邮件 HTML 正文（含内嵌图片）

适用于每日更新和邮件预警。
"""

import os
import sys
import warnings
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    INDEX_HISTORY_PATH, CHART_PATH, RAW_DIR,
    INDICATORS, get_heat_level,
)


# ── 字体配置 ────────────────────────────────────────────
# 按优先级选择可用的中文字体（跨平台：macOS / Linux GitHub Actions / Windows）
_CJK_FONTS = [
    "Heiti TC", "PingFang SC", "Hiragino Sans GB",
    "Noto Sans CJK SC", "Noto Sans CJK", "WenQuanYi Micro Hei",
    "Microsoft YaHei", "SimHei",
    "Apple LiGothic", "Hei", "Kai",
    "DejaVu Sans",
]
_AVAILABLE_CJK = None


def _get_cjk_font():
    global _AVAILABLE_CJK
    if _AVAILABLE_CJK is not None:
        return _AVAILABLE_CJK
    candidates = [f.name for f in fm.fontManager.ttflist]
    for name in _CJK_FONTS:
        if name in candidates:
            _AVAILABLE_CJK = name
            print(f"  [font] 使用字体: {name}")
            return name
    _AVAILABLE_CJK = "DejaVu Sans"
    print("  [font] ⚠ 未找到中文字体，使用 DejaVu Sans（中文可能显示为方框）")
    return _AVAILABLE_CJK


def _setup_style():
    """全局 matplotlib 样式"""
    font = _get_cjk_font()
    plt.rcParams.update({
        "font.sans-serif": [font, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "#F8F9FA",
        "axes.facecolor": "#FFFFFF",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.color": "#CCCCCC",
    })
    return font


# ═══════════════════════════════════════════════════════════
# 简报图
# ═══════════════════════════════════════════════════════════

def generate_briefing(
    latest: pd.Series,
    index_df: pd.DataFrame,
    save_path: str = None,
) -> bytes:
    """
    生成简报仪表图，返回 PNG bytes。

    layout:
      ┌──────────────────────────────────┐
      │  牛市热度指数 · 每日简报          │
      │  ├── 大数字 + 仪表盘             │
      │  ├── 全历史走势 + 区间着色        │
      │  └── 10指标百分位横向热力条       │
      └──────────────────────────────────┘
    """
    font = _setup_style()
    today_str = datetime.now().strftime("%Y-%m-%d")

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.8, 0.9],
                          hspace=0.06, wspace=0.15)

    # ── 标题行 ──────────────────────────────────────────
    title_fig = fig.add_subplot(gs[0, :])
    title_fig.axis("off")
    _draw_title_panel(title_fig, latest, index_df, today_str, font)

    # ── 走势图 ──────────────────────────────────────────
    ax_chart = fig.add_subplot(gs[1, :])
    _draw_history_chart(ax_chart, index_df, latest, font)

    # ── 指标热力条 ──────────────────────────────────────
    ax_bars = fig.add_subplot(gs[2, :])
    _draw_indicator_bars(ax_bars, latest, font)

    # 保存或返回
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"📊 简报已保存 → {save_path}")

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _draw_title_panel(ax, latest, index_df, today_str, font):
    """顶部标题区域：大数字 + 等级 + 关键信息"""
    index_val = latest["index_value"]
    heat = get_heat_level(index_val)

    # 背景渐变
    color_map = {
        "很冷": "#00BCD4", "偏冷": "#4CAF50",
        "适中": "#9E9E9E", "偏热": "#FF9800", "很热": "#F44336",
    }
    main_color = color_map.get(heat["level"], "#9E9E9E")

    # 大数字
    ax.text(0.02, 0.55, f"{index_val:.1f}", fontsize=64,
            fontweight="bold", color=main_color, va="center", fontfamily=font)
    ax.text(0.02 + 0.12, 0.55, "/ 100", fontsize=18,
            color="#666666", va="center", fontfamily=font)

    # 等级标签
    ax.text(0.30, 0.70, f"等级：{heat['level']}", fontsize=20,
            fontweight="bold", color=main_color, va="center", fontfamily=font)
    ax.text(0.30, 0.38, heat["description"], fontsize=12,
            color="#555555", va="center", fontfamily=font)

    # 基本信息
    info_items = [
        f"📅 {today_str}",
        f"📊 数据范围: {index_df.index[0].strftime('%Y-%m-%d')} ~ {index_df.index[-1].strftime('%Y-%m-%d')}",
        f"📈 {len(index_df)} 个交易日",
    ]
    for i, text in enumerate(info_items):
        ax.text(0.60, 0.78 - i * 0.16, text, fontsize=11,
                color="#444444", va="center", fontfamily=font)

    # 分隔线
    ax.axhline(y=0.1, xmin=0.02, xmax=0.98, color="#DDDDDD", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _draw_history_chart(ax, index_df, latest, font):
    """指数历史走势图（带区间着色）"""
    values = index_df["index_value"].dropna()

    # 主曲线
    ax.plot(values.index, values.values, color="#1565C0",
            linewidth=1.2, alpha=0.85, zorder=3)

    # 30日均线
    ma30 = values.rolling(30).mean()
    ax.plot(ma30.index, ma30.values, color="#FF6F00",
            linewidth=1.5, linestyle="--", alpha=0.7, zorder=4,
            label="30日均线")

    # 热度区间着色
    zones = [
        (0, 20, "#00BCD4", "很冷"),
        (20, 40, "#4CAF50", "偏冷"),
        (40, 60, "#E0E0E0", "适中"),
        (60, 80, "#FF9800", "偏热"),
        (80, 100, "#F44336", "很热"),
    ]
    for lo, hi, color, label in zones:
        ax.axhspan(lo, hi, alpha=0.08, color=color, zorder=0)
    # 添加图例标签（只放几个）
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#00BCD4", alpha=0.25, label="很冷 (0-20)"),
        Patch(facecolor="#4CAF50", alpha=0.25, label="偏冷 (20-40)"),
        Patch(facecolor="#E0E0E0", alpha=0.5, label="适中 (40-60)"),
        Patch(facecolor="#FF9800", alpha=0.25, label="偏热 (60-80)"),
        Patch(facecolor="#F44336", alpha=0.25, label="很热 (80-100)"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper left",
                    fontsize=7, ncol=5, framealpha=0.8)
    leg.set_zorder(10)

    # 标记关键历史点
    key_points = {
        "2015-06-12": "2015顶",
        "2018-12-01": "2018底",
        "2021-02-18": "2021顶",
    }
    for date_str, label in key_points.items():
        dt = pd.Timestamp(date_str)
        if dt in values.index:
            val = values[dt]
            ax.annotate(
                f"{label}\n{val:.0f}",
                xy=(dt, val),
                xytext=(8, 12), textcoords="offset points",
                fontsize=7, color="#333333",
                fontfamily=font,
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                         edgecolor="none", alpha=0.7),
            )

    # 标记当前点
    today = latest.name
    if today in values.index:
        cur_val = values[today]
        ax.scatter([today], [cur_val], color="#D32F2F", s=60,
                   zorder=10, marker="o", edgecolors="white", linewidth=1.5)
        ax.annotate(
            f"当前 {cur_val:.1f}",
            xy=(today, cur_val),
            xytext=(12, -18), textcoords="offset points",
            fontsize=9, fontweight="bold", color="#D32F2F",
            fontfamily=font,
            arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.2),
        )

    ax.set_ylim(-2, 102)
    ax.set_ylabel("指数 (0-100)", fontsize=10, fontfamily=font)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(labelsize=8)


def _draw_indicator_bars(ax, latest, font):
    """10个指标百分位横向热力条"""
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(INDICATORS) - 0.5)
    ax.set_xlabel("百分位 (0-100)", fontsize=9, fontfamily=font)
    ax.invert_yaxis()
    ax.tick_params(left=False, labelleft=False)
    ax.grid(axis="x", alpha=0.3)

    # 背景区间
    for lo, hi, color in [
        (0, 20, "#00BCD4"), (20, 40, "#4CAF50"),
        (40, 60, "#E0E0E0"), (60, 80, "#FF9800"), (80, 100, "#F44336"),
    ]:
        ax.axvspan(lo, hi, alpha=0.06, color=color, zorder=0)

    for i, ind in enumerate(INDICATORS):
        val = latest.get(ind.id)
        if pd.isna(val) or val is None:
            continue
        val = float(val)

        # 颜色根据百分位
        color = (
            "#F44336" if val >= 80 else
            "#FF9800" if val >= 60 else
            "#9E9E9E" if val >= 40 else
            "#4CAF50" if val >= 20 else
            "#00BCD4"
        )

        # 条形
        ax.barh(i, val, height=0.55, color=color, alpha=0.85, zorder=3,
                edgecolor="white", linewidth=0.5)

        # 标签（右侧数值，防止 val=100 时被裁剪）
        label_x = min(val + 1.5, 97)
        ax.text(label_x, i, f"{val:.1f}", va="center",
                fontsize=7, color="#333333", fontfamily=font)

    # 指标名称（左侧，用 yticklabels 避免被坐标轴裁剪）
    ax.set_yticks(range(len(INDICATORS)))
    ax.set_yticklabels([ind.name for ind in INDICATORS],
                       fontsize=7.5, fontfamily=font)

    # 百分位刻度
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(labelsize=7)


# ═══════════════════════════════════════════════════════════
# 邮件 HTML
# ═══════════════════════════════════════════════════════════

def build_email_html(latest: pd.Series, image_cid: str = "briefing") -> str:
    """
    构建简报邮件 HTML 正文。
    图片使用 CID 引用（内嵌附件），文本为指标明细表。
    """
    index_val = latest["index_value"]
    heat = get_heat_level(index_val)
    date_str = latest.name.strftime("%Y-%m-%d")

    # 等级颜色
    level_colors = {"很冷": "#00BCD4", "偏冷": "#4CAF50",
                    "适中": "#9E9E9E", "偏热": "#FF9800", "很热": "#F44336"}
    main_color = level_colors.get(heat["level"], "#9E9E9E")

    # 指标明细表
    rows_html = ""
    for ind in INDICATORS:
        val = latest.get(ind.id)
        if pd.isna(val) or val is None:
            rows_html += f"""
<tr>
  <td style="padding:5px 10px;border-bottom:1px solid #eee;font-size:12px;">{ind.name}</td>
  <td style="padding:5px 10px;border-bottom:1px solid #eee;text-align:center;color:#999;">N/A</td>
  <td style="padding:5px 10px;border-bottom:1px solid #eee;text-align:center;">—</td>
</tr>"""
        else:
            v = float(val)
            bar_color = (
                "#F44336" if v >= 80 else "#FF9800" if v >= 60 else
                "#9E9E9E" if v >= 40 else "#4CAF50" if v >= 20 else "#00BCD4"
            )
            bar_len = max(1, int(v / 5))
            bar_html = "█" * bar_len
            rows_html += f"""
<tr>
  <td style="padding:5px 10px;border-bottom:1px solid #eee;font-size:12px;">{ind.name}</td>
  <td style="padding:5px 10px;border-bottom:1px solid #eee;text-align:center;font-weight:bold;font-size:13px;">{v:.1f}</td>
  <td style="padding:5px 10px;border-bottom:1px solid #eee;text-align:left;">
    <span style="color:{bar_color};font-size:11px;letter-spacing:1px;">{bar_html}</span>
  </td>
</tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:20px;background:#F5F5F5;font-family:'Microsoft YaHei','PingFang SC',Arial,sans-serif;">
<table align="center" width="620" cellpadding="0" cellspacing="0"
       style="background:#FFFFFF;border-radius:10px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.08);">

  <!-- 头部 -->
  <tr>
    <td style="padding:24px 28px 12px;text-align:center;">
      <h1 style="margin:0;font-size:22px;color:#333333;">📊 牛市热度指数 · 每日简报</h1>
    </td>
  </tr>

  <!-- 大数字 -->
  <tr>
    <td style="padding:8px 28px 4px;text-align:center;">
      <span style="font-size:56px;font-weight:bold;color:{main_color};">{index_val:.1f}</span>
      <span style="font-size:16px;color:#999;"> / 100</span>
    </td>
  </tr>

  <!-- 等级 -->
  <tr>
    <td style="padding:0 28px 12px;text-align:center;">
      <span style="display:inline-block;padding:4px 18px;border-radius:20px;
                   background:{main_color};color:white;font-size:16px;font-weight:bold;">
        {heat['level']}
      </span>
      <span style="display:block;margin-top:4px;font-size:13px;color:#666;">
        {heat['description']}
      </span>
    </td>
  </tr>

  <!-- 简报图（内嵌 CID） -->
  <tr>
    <td style="padding:8px 20px;text-align:center;">
      <img src="cid:{image_cid}" alt="牛市热度指数简报图"
           style="max-width:100%;border-radius:6px;border:1px solid #eee;" />
    </td>
  </tr>

  <!-- 指标明细 -->
  <tr>
    <td style="padding:16px 28px 8px;">
      <h3 style="margin:0 0 10px;font-size:15px;color:#333;">📋 各指标百分位明细</h3>
      <table width="100%" cellpadding="0" cellspacing="0"
             style="border-collapse:collapse;font-size:13px;">
        <tr style="background:#F8F9FA;">
          <th style="padding:7px 10px;border-bottom:2px solid #ddd;text-align:left;font-size:12px;">指标</th>
          <th style="padding:7px 10px;border-bottom:2px solid #ddd;text-align:center;font-size:12px;">百分位</th>
          <th style="padding:7px 10px;border-bottom:2px solid #ddd;text-align:center;font-size:12px;">热力</th>
        </tr>
        {rows_html}
      </table>
    </td>
  </tr>

  <!-- 脚注 -->
  <tr>
    <td style="padding:16px 28px 20px;border-top:1px solid #eee;">
      <p style="margin:0;font-size:11px;color:#999;text-align:center;">
        📊 牛市热度指数 v1.0 · 10个指标等权合成<br>
        🤖 数据来源: akshare · 每日 {date_str} 更新<br>
        此邮件由 GitHub Actions 自动发送
      </p>
    </td>
  </tr>
</table>
</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    """独立运行：生成简报图并打印邮件 HTML 示例"""
    if not os.path.exists(INDEX_HISTORY_PATH):
        print("❌ 指数历史文件不存在，请先运行 backtest.py 或 update_index.py")
        return

    df = pd.read_csv(INDEX_HISTORY_PATH, index_col=0, parse_dates=True)
    latest = df.iloc[-1]
    latest.name = df.index[-1]

    png_bytes = generate_briefing(latest, df, save_path=CHART_PATH)
    print(f"简报图大小: {len(png_bytes)/1024:.0f} KB")

    print("\n--- 邮件 HTML 预览（前 300 字符）---")
    html = build_email_html(latest)
    print(html[:300] + "…")
    print("\n✅ 简报生成完成！")


if __name__ == "__main__":
    main()
