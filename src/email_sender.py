"""Email distribution for pipeline reports via Gmail SMTP."""

from __future__ import annotations

import logging
import os
import smtplib
from collections.abc import Sequence
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from src.models.recommendation import DirectionalForecast, PredictionOutcome
from src.timezone import format_la, today_local

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
.vibe { font-size: 14px; padding: 10px 14px; background: #f0f6ff; border-radius: 6px; border-left: 4px solid #58a6ff; margin: 12px 0; }
.outcome-card { font-size: 13px; padding: 10px 14px; border-radius: 6px; margin: 8px 0; }
.outcome-card p { margin: 4px 0; }
.outcome-success { background: #f0faf1; border-left: 4px solid #1a7f37; }
.outcome-fail { background: #faf0f0; border-left: 4px solid #cf222e; }
.outcome-unknown { background: #faf8f0; border-left: 4px solid #9a6700; }
.outcome-asset { font-weight: 600; }
.badge-success { display: inline-block; background: #1a7f37; color: #fff; border-radius: 4px; padding: 1px 8px; font-size: 11px; font-weight: 600; }
.badge-fail { display: inline-block; background: #cf222e; color: #fff; border-radius: 4px; padding: 1px 8px; font-size: 11px; font-weight: 600; }
.footer {
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid #e1e4e8;
  font-size: 12px;
  color: #8b949e;
  text-align: center;
}
"""


def render_forecast_html(
    forecast: DirectionalForecast,
    yesterday_outcomes: list[PredictionOutcome] | None = None,
) -> str:
    rows = ""
    for f in forecast.forecasts:
        if f.direction is None:
            direction_str = "—"
            style = "sideways"
            conf_str = "—"
            move_str = "—"
        else:
            direction_str = f.direction
            style = f.direction.lower()
            conf_str = f"{f.confidence:.0%}"
            move_str = f"{f.predicted_move_pct:+.1f}%"
        drivers = f.rationale or ("<br>".join(f.sources) if f.sources else "—")
        if f.sources and f.rationale:
            sources_str = " · ".join(f.sources)
            drivers += f'<br><span style="font-size:0.78rem;color:#8b949e">{sources_str}</span>'
        rows += f"""<tr>
  <td>{f.asset}</td>
  <td class="{style}">{direction_str}</td>
  <td>{conf_str}</td>
  <td class="{style}">{move_str}</td>
  <td>{drivers}</td>
</tr>"""

    vibe = ""
    if forecast.market_vibe:
        vibe = f'<p class="vibe"><strong>Market Vibe:</strong> {forecast.market_vibe}</p>'

    yesterday_html = _render_yesterday_section(yesterday_outcomes)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>{CSS}</style>
</head>
<body>
<div class="container">
<h1>sender-trades — Intraday Prediction</h1>
<p>Generated at {format_la(forecast.generated_at)}</p>
{vibe}
<table>
<thead>
<tr><th>Asset</th><th>Direction</th><th>Confidence</th><th>Pred. Move</th><th>Key Drivers of the Prediction</th></tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
{yesterday_html}
<div class="footer">
sender-trades &mdash; 0DTE Intraday Prediction Engine<br>
Powered by opencode LLM + Market Research
</div>
</div>
</body>
</html>"""


def _render_yesterday_section(
    outcomes: list[PredictionOutcome] | None,
) -> str:
    if not outcomes:
        return ""

    cards = ""
    for o in outcomes:
        if o.result == "success":
            badge = '<span class="badge-success">SUCCESS</span>'
            card_class = "outcome-success"
        elif o.result == "fail":
            badge = '<span class="badge-fail">FAIL</span>'
            card_class = "outcome-fail"
        else:
            badge = ""
            card_class = "outcome-unknown"

        details_html = o.details.replace(" | ", "<br>")
        pred_move = f"{o.confidence:.0%} confidence"
        cards += f"""<div class="outcome-card {card_class}">
  <p><span class="outcome-asset">{o.asset}</span> — Predicted <strong>{o.predicted_direction}</strong> ({pred_move}) {badge}</p>
  <p>{details_html}</p>
</div>"""

    return f"""<h2>Yesterday's Prediction Recap</h2>
{cards}"""


def send_email(
    forecast: DirectionalForecast,
    subject: str = "sender-trades — Directional Forecast",
    recipients: Sequence[str] | None = None,
    dry_run: bool = False,
    correlation_id: str = "",
    log_dir: str | Path | None = None,
    yesterday_outcomes: list[PredictionOutcome] | None = None,
) -> dict[str, bool]:
    user = os.environ.get("GMAIL_USER", "")
    password = os.environ.get("GMAIL_APP_PASSWORD", "")
    raw_recipients = (
        os.environ.get("RECIPIENT_EMAIL", "") if recipients is None else ",".join(recipients)
    )
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

    html = render_forecast_html(forecast, yesterday_outcomes=yesterday_outcomes)

    plain_parts = [f"sender-trades Directional Forecast\n\n{forecast.table()}"]
    if yesterday_outcomes:
        plain_parts.append("\n\nYesterday's Prediction Recap:")
        for o in yesterday_outcomes:
            result_label = {"success": "SUCCESS", "fail": "FAIL", "unknown": "UNKNOWN"}.get(
                o.result, "?"
            )
            plain_parts.append(
                f"  {o.asset}: {o.predicted_direction} ({o.confidence:.0%}) — {result_label}"
            )
            plain_parts.append(f"  {o.details}")
    plain_parts.append("\n---\nsender-trades")
    plain = "\n".join(plain_parts)

    # Persist the rendered HTML body to the run's log directory so it can
    # be inspected post-hoc by correlation_id -- useful for audit and for
    # the trial runs documented in LESSONS_LEARNED.md. Failure to write
    # the file is non-fatal; the email itself still goes out.
    if log_dir is not None and correlation_id:
        try:
            day_dir = Path(log_dir).expanduser() / today_local().isoformat()
            day_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"-{correlation_id}" if correlation_id else ""
            (day_dir / f"email{suffix}.html").write_text(html, encoding="utf-8")
            (day_dir / f"email{suffix}.txt").write_text(plain, encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to persist email body to log dir: %s", e)

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
