"""
邮件预警脚本 — 发送每日简报邮件

触发条件：
  - 指数 ≥80（很热，逃顶风险）
  - 指数 ≤20（很冷，布局窗口）

邮件内容：
  - HTML 报告（docs/index.html，响应式布局）

运行方式：
    SMTP_USER=xxx@qq.com SMTP_PASS=xxx MAIL_TO=receiver@example.com \\
    python scripts/send_alert.py

选项：
    --force     强制发送（无视触发条件）
    --dry-run   仅打印邮件内容，不实际发送
"""

import argparse
import os
import smtplib
import sys
import warnings
from email.mime.text import MIMEText
from email.utils import formatdate
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import (
    INDEX_HISTORY_PATH, CHART_PATH,
    INDICATORS, get_heat_level,
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS,
    MAIL_TO, MAIL_FROM,
)

REPORT_PATH = os.path.join(os.path.dirname(CHART_PATH), "index.html")


def get_today_index() -> pd.Series | None:
    """获取最新一天的指数"""
    if not os.path.exists(INDEX_HISTORY_PATH):
        print(f"⚠ 指数历史文件不存在: {INDEX_HISTORY_PATH}")
        return None, None
    df = pd.read_csv(INDEX_HISTORY_PATH, index_col=0, parse_dates=True)
    if df.empty:
        return None, None
    latest = df.iloc[-1]
    latest.name = df.index[-1]
    return latest, df


def should_alert(index_value: float) -> tuple[bool, str]:
    """判断是否应发送预警"""
    if index_value >= 80:
        return True, f"🔴 很热 ({index_value:.2f}) — 逃顶风险显著"
    elif index_value <= 20:
        return True, f"🔵 很冷 ({index_value:.2f}) — 历史极值低位"
    return False, ""


def read_report_html() -> str:
    """读取已生成的 HTML 报告"""
    if not os.path.exists(REPORT_PATH):
        print(f"⚠ HTML 报告不存在: {REPORT_PATH}")
        print(f"  请先运行 python scripts/generate_report.py")
        return ""
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def send_email(subject: str, html_body: str, dry_run: bool = False):
    """发送 HTML 邮件"""

    # dry-run 优先（无需 SMTP 配置）
    if dry_run:
        print(f"\n{'=' * 50}")
        print(f"📧 [DRY RUN] 邮件预览")
        print(f"  From: {MAIL_FROM or SMTP_USER or '(未设置)'}")
        print(f"  To: {MAIL_TO or '(未设置)'}")
        print(f"  Subject: {subject}")
        print(f"  HTML 长度: {len(html_body.encode('utf-8'))} 字符")
        print(f"{'=' * 50}")
        return True

    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS or not MAIL_TO:
        print("⚠ 邮件配置不完整，请设置 SMTP_HOST, SMTP_USER, SMTP_PASS, MAIL_TO")
        return False

    msg = MIMEText(html_body, "html", "utf-8")
    msg["From"] = MAIL_FROM or SMTP_USER
    msg["To"] = MAIL_TO
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)

    try:
        print(f"📧 正在发送邮件到 {MAIL_TO}…")
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
                server.login(SMTP_USER, SMTP_PASS)
                server.sendmail(msg["From"], msg["To"], msg.as_string())
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.sendmail(msg["From"], msg["To"], msg.as_string())
        print(f"✅ 邮件已发送到 {MAIL_TO}")
        return True
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="牛市热度指数邮件预警")
    parser.add_argument("--force", action="store_true",
                        help="强制发送（无视触发条件）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印邮件内容，不实际发送")
    args = parser.parse_args()

    latest, index_df = get_today_index()
    if latest is None:
        print("❌ 无法获取最新指数")
        sys.exit(1)

    index_value = latest["index_value"]
    date_str = latest.name.strftime("%Y-%m-%d")
    print(f"📅 {date_str} 指数: {index_value:.2f}")

    need_alert, reason = should_alert(index_value)
    if not need_alert and not args.force:
        print(f"ℹ 指数 {index_value:.2f} 在正常范围，无需预警（触发条件: ≥80 或 ≤20）")
        sys.exit(0)

    if args.force:
        reason = f"强制发送 — 当日指数 {index_value:.2f}"
    print(f"🔔 预警触发: {reason}")

    html_body = read_report_html()
    if not html_body:
        print("❌ 无法读取 HTML 报告")
        sys.exit(1)

    print(f"  HTML 报告大小: {len(html_body.encode('utf-8')) / 1024:.0f} KB")

    heat = get_heat_level(index_value)
    subject = f"⚠ 牛市热度指数预警 | {date_str} | {index_value:.1f} {heat['level']}"
    send_email(subject, html_body, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
