"""Send generated briefings through optional delivery channels."""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Callable

from .models import AppConfig, BriefingResult

LOGGER = logging.getLogger(__name__)
TELEGRAM_LIMIT = 4096
SMTP_PASSWORD_ENV = "SMTP_PASSWORD"


def notify(output_path: Path, briefing: BriefingResult, config: AppConfig) -> None:
    """Deliver the saved briefing through enabled channels."""

    LOGGER.info("Markdown briefing saved to %s", output_path)
    _try_channel("email", lambda: _send_email(output_path, briefing, config))
    _try_channel("telegram", lambda: _send_telegram(output_path, briefing, config))


def _try_channel(name: str, sender: Callable[[], None]) -> None:
    """Run a channel sender without blocking others on failure."""

    try:
        sender()
    except Exception as exc:
        LOGGER.warning("Failed to send %s notification: %s", name, exc)


def _send_email(output_path: Path, briefing: BriefingResult, config: AppConfig) -> None:
    """Send the briefing over SMTP when enabled."""

    settings = config.output.email
    if not settings.enabled:
        return
    password = os.getenv(SMTP_PASSWORD_ENV)
    if not password:
        raise RuntimeError(f"Environment variable '{SMTP_PASSWORD_ENV}' is required for email delivery")
    if not settings.smtp_host or not settings.sender or not settings.recipients:
        raise RuntimeError("Email delivery requires smtp_host, sender, and recipients")
    message = _build_email_message(output_path, briefing, config)
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
        smtp.starttls()
        smtp.login(settings.sender, password)
        smtp.sendmail(settings.sender, settings.recipients, message.as_string())


def _build_email_message(output_path: Path, briefing: BriefingResult, config: AppConfig) -> MIMEMultipart:
    """Build a multipart email for the briefing."""

    message = MIMEMultipart("alternative")
    settings = config.output.email
    subject_date = briefing.generated_at.strftime("%Y-%m-%d")
    message["Subject"] = f"📰 每日头条简报 — {subject_date}"
    message["From"] = settings.sender
    message["To"] = ", ".join(settings.recipients)
    markdown_text = output_path.read_text(encoding="utf-8")
    html_text = _markdown_to_html(markdown_text)
    message.attach(MIMEText(markdown_text, "plain", "utf-8"))
    message.attach(MIMEText(html_text, "html", "utf-8"))
    return message


def _markdown_to_html(markdown_text: str) -> str:
    """Convert Markdown to HTML for richer emails."""

    try:
        import markdown

        return markdown.markdown(markdown_text)
    except ImportError:
        return f"<pre>{markdown_text}</pre>"


def _send_telegram(output_path: Path, briefing: BriefingResult, config: AppConfig) -> None:
    """Send the briefing via Telegram Bot API when enabled."""

    del briefing
    settings = config.output.telegram
    if not settings.enabled:
        return
    bot_token = os.getenv(settings.bot_token_env)
    chat_id = os.getenv(settings.chat_id_env)
    if not bot_token or not chat_id:
        raise RuntimeError("Telegram delivery requires bot token and chat id environment variables")
    text = output_path.read_text(encoding="utf-8")
    for chunk in _split_message(text, TELEGRAM_LIMIT):
        _post_telegram_message(bot_token, chat_id, chunk)


def _post_telegram_message(bot_token: str, chat_id: str, text: str) -> None:
    """Post a single message chunk to Telegram."""

    import requests

    response = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=10,
    )
    response.raise_for_status()


def _split_message(text: str, limit: int) -> list[str]:
    """Split long messages on line boundaries when possible."""

    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        split_at = split_at if split_at > 0 else limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    return chunks
