"""
邮件预警脚本 — 发送每日简报邮件

触发条件：
  - 指数 ≥80（很热，逃顶风险）
  - 指数 ≤20（很冷，布局窗口）

邮件内容：
  - 简报仪表图（内嵌 CID 图片）
  - 各指标百分位明细表

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
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
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
from scripts.briefing import generate_briefing, build_email_html


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


def send_email(subject: str, html_body: str, image_bytes: bytes,
               dry_run: bool = False):
    """发送带内嵌图片的 HTML 邮件"""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS or not MAIL_TO:
        print("⚠ 邮件配置不完整，请设置以下环境变量:")
        print("  SMTP_HOST, SMTP_USER, SMTP_PASS, MAIL_TO")
        return False

    msg = MIMEMultipart("related")
    msg["From"] = MAIL_FROM or SMTP_USER
    msg["To"] = MAIL_TO
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)

    # HTML 正文
    msg_alternative = MIMEMultipart("alternative")
    msg.attach(msg_alternative)

    text_body = f"牛市热度指数每日简报，请查看 HTML 版本邮件。\n{subject}"
    msg_alternative.attach(MIMEText(text_body, "plain", "utf-8"))
    msg_alternative.attach(MIMEText(html_body, "html", "utf-8"))

    # 内嵌简报图（CID）
    image = MIMEImage(image_bytes, _subtype="png")
    image.add_header("Content-ID", "<briefing>")
    image.add_header("Content-Disposition", "inline", filename="briefing.png")
    msg.attach(image)

    if dry_run:
        print(f"\n{'='*50}")
        print(f"📧 [DRY RUN] 邮件预览")
        print(f"  From: {msg['From']}")
        print(f"  To: {msg['To']}")
        print(f"  Subject: {msg['Subject']}")
        print(f"  简报图大小: {len(image_bytes)/1024:.0f} KB")
        print(f"  HTML 长度: {len(html_body)} 字符")
        print(f"{'='*50}")
        return True

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

    # 获取最新指数
    latest, index_df = get_today_index()
    if latest is None:
        print("❌ 无法获取最新指数")
        sys.exit(1)

    index_value = latest["index_value"]
    date = latest.name
    date_str = date.strftime("%Y-%m-%d")

    print(f"📅 {date_str} 指数: {index_value:.2f}")

    # 判断是否需要预警
    need_alert, reason = should_alert(index_value)
    if not need_alert and not args.force:
        print(f"ℹ 指数 {index_value:.2f} 在正常范围，无需预警")
        print(f"  触发条件: 指数 ≥80 或 ≤20")
        sys.exit(0)

    if args.force:
        reason = f"强制发送 — 当日指数 {index_value:.2f}"
        print(f"🔔 强制发送模式")

    print(f"🔔 预警触发: {reason}")

    # 生成简报图
    print("📊 生成简报图…")
    try:
        png_bytes = generate_briefing(latest, index_df)
        print(f"  简报图大小: {len(png_bytes)/1024:.0f} KB")
    except Exception as e:
        print(f"  ⚠ 简报图生成失败: {e}")
        png_bytes = None

    # 构建邮件
    heat = get_heat_level(index_value)
    subject = (f"⚠ 牛市热度指数预警 | {date_str} | "
               f"{index_value:.1f} {heat['level']}")

    html_body = build_email_html(latest, image_cid="briefing")

    if png_bytes:
        send_email(subject, html_body, png_bytes, dry_run=args.dry_run)
    else:
        # 无图片时发送纯 HTML
        print("  ⚠ 无简报图，发送纯文本邮件")
        html_body = build_email_html(latest)
        # 创建一个透明 1x1 占位图
        import io
        png_bytes = b""
        send_email(subject, html_body, png_bytes, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
