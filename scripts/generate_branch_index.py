"""Generate a branch-index landing page for the public/ directory."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

PUBLIC_DIR = Path("public")
PUBLIC_DIR.mkdir(exist_ok=True)
INDEX = PUBLIC_DIR / "index.html"

BRANCHES = sorted(
    [d.name for d in PUBLIC_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")],
    reverse=True,
)

DEFAULT_BRANCH = "main"

CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: #0d1117; color: #e6edf3; margin: 0; padding: 0; line-height: 1.6;
}
.container { max-width: 720px; margin: 0 auto; padding: 60px 24px; }
h1 { font-size: 1.5rem; margin-bottom: 4px; }
h1 span { color: #58a6ff; }
.subtitle { color: #8b949e; font-size: 0.9rem; margin-bottom: 32px; }
.branches { display: flex; flex-direction: column; gap: 8px; }
.branch {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 20px; background: #161b22; border: 1px solid #30363d;
  border-radius: 8px; text-decoration: none; color: #e6edf3;
  transition: border-color 0.1s;
}
.branch:hover { border-color: #58a6ff; }
.branch .name { font-weight: 600; }
.branch .name .default-tag {
  display: inline-block; margin-left: 8px; padding: 1px 8px;
  border-radius: 10px; font-size: 0.7rem; font-weight: 500;
  background: rgba(63,185,80,0.12); color: #3fb950;
}
.branch .meta { color: #8b949e; font-size: 0.8rem; }
footer { text-align: center; padding: 40px 0; color: #484f58; font-size: 0.8rem; }
"""

NOW = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

if not BRANCHES:
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>sender-trades &mdash; Branch Previews</title><style>{CSS}</style></head>
<body><div class="container"><h1>sender-<span>trades</span> Branch Previews</h1>
<p class="subtitle">No branches deployed yet.</p>
<footer>Generated {NOW}</footer></div></body></html>"""
else:
    items = []
    for b in BRANCHES:
        is_default = b == DEFAULT_BRANCH
        tag = ' <span class="default-tag">default</span>' if is_default else ""
        link = b if is_default else b
        items.append(
            f'<a class="branch" href="./{link}/"><span class="name">{b}{tag}</span><span class="meta">{link} &rarr;</span></a>'
        )

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>sender-trades &mdash; Branch Previews</title><style>{CSS}</style></head>
<body><div class="container"><h1>sender-<span>trades</span> Branch Previews</h1>
<p class="subtitle">Per-branch site deployments. Push to <code>main</code> or <code>feature/**</code> to auto-deploy.</p>
<div class="branches">{''.join(items)}</div>
<footer>Generated {NOW}</footer></div></body></html>"""

INDEX.write_text(html)
print(f"Generated {INDEX} with {len(BRANCHES)} branches: {BRANCHES}")
