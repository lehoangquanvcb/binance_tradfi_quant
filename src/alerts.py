"""Telegram/email alert helpers. Safe by default: dry-run unless credentials are set."""
from __future__ import annotations
import os, smtplib, requests
from email.mime.text import MIMEText
from .database import log_event

def send_telegram(message: str) -> bool:
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        log_event('ALERT_DRY_RUN', 'TELEGRAM_NOT_CONFIGURED', message)
        return False
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    r = requests.post(url, json={'chat_id': chat_id, 'text': message}, timeout=10)
    ok = r.ok
    log_event('ALERT', 'TELEGRAM_SENT' if ok else 'TELEGRAM_FAILED', message)
    return ok

def send_email(subject: str, body: str) -> bool:
    host = os.getenv('SMTP_HOST')
    user = os.getenv('SMTP_USER')
    password = os.getenv('SMTP_PASSWORD')
    to_addr = os.getenv('ALERT_EMAIL_TO')
    if not all([host, user, password, to_addr]):
        log_event('ALERT_DRY_RUN', 'EMAIL_NOT_CONFIGURED', subject + '\n' + body)
        return False
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = to_addr
    with smtplib.SMTP_SSL(host, int(os.getenv('SMTP_PORT', '465'))) as server:
        server.login(user, password)
        server.sendmail(user, [to_addr], msg.as_string())
    log_event('ALERT', 'EMAIL_SENT', subject)
    return True

def send_risk_alert(message: str) -> None:
    send_telegram(message)
    send_email('Binance TradFi Quant Platform Alert', message)
