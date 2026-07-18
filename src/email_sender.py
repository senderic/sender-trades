"""Email distribution for pipeline reports via Gmail SMTP."""

from __future__ import annotations

import logging
import os
import smtplib
from collections.abc import Sequence
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from src.models.recommendation import DirectionalForecast

logger = logging.getLogger(__name__)

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  line-height: 1.6;
  color: #1a1a1a;
  max-width: 680px;
  margin: 0 auto;
  padding: 20px;
  background-color: #f8f9fa;
}
.container {
  background-color: #ffffff;
  border-radius: 8px;
  padding: 32px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
h1 {
  color: #0d1117;
  font-size: 24px;
  border-bottom: 2px solid #58a6ff;
  padding-bottom: 8px;
  margin-top: 0;
}
h2 {
  color: #1f6feb;
  font-size: 18px;
  margin-top: 28px;
  border-bottom: 1px solid #e1e4e8;
  padding-bottom: 6px;
}
p { margin: 8px 0; font-size: 14px; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 12px 0;
  font-size: 13px;
}
th, td {
  border: 1px solid #d0d7de;
  padding: 8px 12px;
  text-align: left;
}
th {
  background-color: #f0f3f6;
  font-weight: 600;
}
.up { color: #1a7f37; font-weight: 600; }
.down { color: #cf222e; font-weight: 600; }
.sideways { color: #9a6700; font-weight: 600; }
.footer {
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid #e1e4e8;
  font-size: 12px;
  color: #8b949e;
  text-align: center;
}
"""


def render_forecast_html(forecast: DirectionalForecast) -> str:
    rows = ""
    for f in forecast.forecasts:
        style = (
            "up"
            if f.up_confidence > f.down_confidence
            else ("down" if f.down_confidence > f.up_confidence else "sideways")
        )
        move_str = f"{f.expected_move_pct:+.1f}%"
        parts = []
        if f.up_sources:
            parts.append("↑" + ", ".join(f.up_sources))
        if f.down_sources:
            parts.append("↓" + ", ".join(f.down_sources))
        src_str = "<br>".join(parts) if parts else "—"
        rows += f"""<tr>
  <td>{f.asset}</td>
  <td class="up">{f.up_confidence:.0%}</td>
  <td class="down">{f.down_confidence:.0%}</td>
  <td class="sideways">{f.sideways_confidence:.0%}</td>
  <td class="{style}">{move_str}</td>
  <td>{src_str}</td>
</tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>{CSS}</style>
</head>
<body>
<div class="container">
<h1>sender-trades — Directional Forecast</h1>
<p>Generated at {forecast.generated_at.strftime("%Y-%m-%d %H:%M UTC")}</p>
<table>
<thead>
<tr><th>Asset</th><th>UP</th><th>DOWN</th><th>SIDE</th><th>MOVE</th><th>Sources</th></tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
<div class="footer">
sender-trades &mdash; 0DTE Intraday Options Trading System<br>
</div>
</div>
</body>
</html>"""


def send_email(
    forecast: DirectionalForecast,
    subject: str = "sender-trades — Directional Forecast",
    recipients: Sequence[str] | None = None,
    dry_run: bool = False,
) -> dict[str, bool]:
    user = os.environ.get("GMAIL_USER", "")
    password = os.environ.get("GMAIL_APP_PASSWORD", "")
    raw_recipients = os.environ.get("RECIPIENT_EMAIL", "") if recipients is None else ",".join(recipients)
    to = [r.strip() for r in raw_recipients.split(",") if r.strip()] or ([user] if user else [])

    if not user or not password:
        logger.warning("GMAIL_USER or GMAIL_APP_PASSWORD not set — skipping email")
        return {}

    if not to or not to[0]:
        logger.warning("No recipient email — skipping email")
        return {}

    if dry_run:
        logger.info("Dry-run — would send email to %s", to)
        return dict.fromkeys(to, True)

    html = render_forecast_html(forecast)
    plain = f"sender-trades Directional Forecast\n\n{forecast.table()}\n\n---\nsender-trades"

    msg = MIMEMultipart("alternative")
    msg["From"] = user
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject
    msg.attach(MIMEText(plain, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    results: dict[str, bool] = {}
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        for r in to:
            masked = r[:3] + "***" + r[r.index("@") :] if "@" in r else "***"
            logger.info("Forecast sent to %s", masked)
            results[r] = True
    except Exception as e:
        logger.error("Failed to send forecast email: %s", e)
        for r in to:
            results[r] = False

    sent = sum(1 for v in results.values() if v)
    logger.info("Email: %d/%d sent", sent, len(to))
    return results
