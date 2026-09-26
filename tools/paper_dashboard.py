#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/paper_dashboard.py
#  M5 — a SINGLE-FILE, self-contained web dashboard for the paper book.
#
#  Renders one HTML document with no external assets (no CDN, no fetch, no
#  images): inline CSS, an inline SVG equity curve, and plain tables. Open it
#  directly in any browser, or serve it from the sidecar.
#
#  Panels: equity curve · open positions · today's P&L · feed health ·
#          recent fills · kill-switch trips.
#
#  Usage:
#    python tools/paper_dashboard.py --out state/dashboard.html
# =============================================================================
import argparse
import datetime as dt
import html
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import paper_report as pr  # noqa: E402

DISCLAIMER = ("Educational analytics. Losses are minimized by discipline, never "
              "eliminated. Past performance does not guarantee future results. "
              "Paper trading only — no real orders are ever placed.")


def _svg_curve(curve, w=920, h=260, pad=34):
    if len(curve) < 2:
        return "<p class='muted'>Not enough history for an equity curve yet.</p>"
    ts = [p[0] for p in curve]
    eq = [p[1] for p in curve]
    lo, hi = min(eq), max(eq)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    t0, t1 = min(ts), max(ts)
    span = (t1 - t0) or 1
    pts = []
    for t, e in curve:
        x = pad + (t - t0) / span * (w - 2 * pad)
        y = h - pad - (e - lo) / (hi - lo) * (h - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    first = dt.datetime.fromtimestamp(t0, dt.timezone.utc).strftime("%Y-%m-%d")
    last = dt.datetime.fromtimestamp(t1, dt.timezone.utc).strftime("%Y-%m-%d")
    up = eq[-1] >= eq[0]
    stroke = "#e5484d" if up else "#2fbf71"          # CN convention: up = red
    return f"""<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img"
     aria-label="equity curve" preserveAspectRatio="none">
  <rect x="0" y="0" width="{w}" height="{h}" fill="none"/>
  <line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#2a2f3a" stroke-width="1"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#2a2f3a" stroke-width="1"/>
  <polyline fill="none" stroke="{stroke}" stroke-width="2"
            points="{' '.join(pts)}"/>
  <text x="{pad}" y="{pad-10}" fill="#8b93a7" font-size="12">{hi:,.2f}</text>
  <text x="{pad}" y="{h-pad+18}" fill="#8b93a7" font-size="12">{lo:,.2f}</text>
  <text x="{w-pad}" y="{h-pad+18}" fill="#8b93a7" font-size="12"
        text-anchor="end">{last}</text>
  <text x="{pad}" y="{h-pad+18}" fill="#8b93a7" font-size="12"
        text-anchor="start">{first}</text>
</svg>"""


def _kpi(label, value, cls=""):
    return (f"<div class='kpi'><div class='kpi-label'>{html.escape(label)}</div>"
            f"<div class='kpi-value {cls}'>{html.escape(value)}</div></div>")


def render_html(a, generated_ts=None):
    generated_ts = generated_ts or a["last_ts"] or 0
    gen = dt.datetime.fromtimestamp(generated_ts, dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC") if generated_ts else "—"

    feeds = pr.feed_rows(a["feeds"])
    feeds_ok = sum(1 for f in feeds if not f["abstain"])

    kpis = "".join([
        _kpi("Equity", pr.fmt_money(a["last_equity"])),
        _kpi("Today's P&L", pr.fmt_signed(a["day_pnl"]),
             pr.pnl_class(a["day_pnl"])),
        _kpi("Today %", pr.fmt_pct(a["day_pct"]), pr.pnl_class(a["day_pct"])),
        _kpi("Open positions", str(len(a["positions"]))),
        _kpi("Kill-switch trips", str(a["halts"])),
        _kpi("Feeds OK", f"{feeds_ok}/{len(feeds)}"),
    ])

    # Open positions
    if a["positions"]:
        rows = []
        for sym in sorted(a["positions"]):
            p = a["positions"][sym]
            rows.append(
                f"<tr><td>{html.escape(sym)}</td><td class='num'>{p['qty']:.4f}</td>"
                f"<td class='num'>{p['avg']:.2f}</td>"
                f"<td class='num'>{p['last']:.2f}</td>"
                f"<td class='num {pr.pnl_class(p['unrealized'])}'>"
                f"{pr.fmt_signed(p['unrealized'])}</td></tr>")
        pos_html = ("<table><thead><tr><th>symbol</th><th>qty</th><th>avg</th>"
                    "<th>last</th><th>unrealized</th></tr></thead><tbody>"
                    + "".join(rows) + "</tbody></table>")
    else:
        pos_html = "<p class='muted'>The book is flat.</p>"

    # Feed health
    if feeds:
        rows = []
        for f in feeds:
            state = "ABSTAIN" if f["abstain"] else f["state"]
            badge = "warn" if f["abstain"] else "ok"
            rows.append(
                f"<tr><td>{html.escape(f['symbol'])}</td>"
                f"<td>{html.escape(f['asset'])}</td>"
                f"<td>{html.escape(f['provider'])}</td>"
                f"<td><span class='badge {badge}'>{html.escape(state)}</span></td>"
                f"<td class='num {pr.pnl_class(f['change_pct_24h'])}'>"
                f"{pr.fmt_pct(f['change_pct_24h'])}</td>"
                f"<td class='num'>{f['last']:.4g}</td>"
                f"<td>{html.escape(f['reason'])}</td></tr>")
        feed_html = ("<table><thead><tr><th>symbol</th><th>asset</th>"
                     "<th>provider</th><th>state</th><th>24h</th><th>last</th>"
                     "<th>reason</th></tr></thead><tbody>"
                     + "".join(rows) + "</tbody></table>")
    else:
        feed_html = ("<p class='muted'>No status heartbeat — run "
                     "<code>tools/paper_loop.py</code>.</p>")

    # Recent fills (last 20)
    recent = list(reversed(a["today_fills"]))[:20] or \
        list(reversed([r for r in pr.load_journal(a["journal"])
                       if r["kind"] == "fill"]))[:20]
    if recent:
        rows = []
        for r in recent:
            when = dt.datetime.fromtimestamp(
                r["ts"], dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
            rows.append(
                f"<tr><td>{when}</td><td>{html.escape(r['ticker'])}</td>"
                f"<td class='num'>{r['qty']:.4f}</td>"
                f"<td class='num'>{r['price']:.2f}</td>"
                f"<td class='num {pr.pnl_class(r['pnl'])}'>"
                f"{pr.fmt_signed(r['pnl'])}</td>"
                f"<td>{html.escape(r['reason'])}</td></tr>")
        fills_html = ("<table><thead><tr><th>ts (UTC)</th><th>symbol</th>"
                      "<th>qty</th><th>price</th><th>pnl</th><th>reason</th>"
                      "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
    else:
        fills_html = "<p class='muted'>No fills recorded.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OmniSeed — paper dashboard</title>
<style>
  :root {{
    --bg:#0e1116; --panel:#161b23; --line:#242b36; --fg:#e6e9ef;
    --muted:#8b93a7; --up:#e5484d; --down:#2fbf71; --accent:#5b9dff;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  header {{ padding:20px 24px; border-bottom:1px solid var(--line); }}
  h1 {{ margin:0 0 4px; font-size:20px; }}
  .muted {{ color:var(--muted); }}
  main {{ padding:20px 24px 48px; max-width:1100px; margin:0 auto; }}
  section {{ margin-bottom:28px; }}
  h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:.06em;
        color:var(--muted); margin:0 0 12px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
           gap:12px; }}
  .kpi {{ background:var(--panel); border:1px solid var(--line);
          border-radius:10px; padding:12px 14px; }}
  .kpi-label {{ color:var(--muted); font-size:12px; }}
  .kpi-value {{ font-size:20px; font-weight:600; margin-top:4px; }}
  .card {{ background:var(--panel); border:1px solid var(--line);
           border-radius:10px; padding:14px; overflow:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line);
           white-space:nowrap; }}
  th {{ color:var(--muted); font-weight:600; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .up {{ color:var(--up); }} .down {{ color:var(--down); }}
  .flat {{ color:var(--muted); }}
  .badge {{ padding:2px 8px; border-radius:999px; font-size:12px; }}
  .badge.ok {{ background:rgba(47,191,113,.15); color:var(--down); }}
  .badge.warn {{ background:rgba(229,72,77,.15); color:var(--up); }}
  footer {{ color:var(--muted); font-size:12px; border-top:1px solid var(--line);
            padding:16px 24px; }}
  code {{ background:#0b0e13; padding:1px 5px; border-radius:4px; }}
</style>
</head>
<body>
<header>
  <h1>OmniSeed — paper trading dashboard</h1>
  <div class="muted">mode: {html.escape(str(a['status'].get('mode', 'paper')))}
    · generated {gen} · day {html.escape(a['day'])} UTC
    · journal <code>{html.escape(a['journal'])}</code></div>
</header>
<main>
  <section><h2>Headline</h2><div class="kpis">{kpis}</div></section>
  <section><h2>Equity curve</h2><div class="card">{_svg_curve(a['curve'])}</div></section>
  <section><h2>Open positions</h2><div class="card">{pos_html}</div></section>
  <section><h2>Feed health</h2><div class="card">{feed_html}</div></section>
  <section><h2>Recent fills</h2><div class="card">{fills_html}</div></section>
  <section><h2>Risk</h2><div class="card">
    <table><tbody>
      <tr><td>Kill-switch trips (lifetime)</td><td class="num">{a['halts']}</td></tr>
      <tr><td>ABSTAIN events</td><td class="num">{a['abstains']}</td></tr>
      <tr><td>Risk-gate refusals</td><td class="num">{a['refused']}</td></tr>
      <tr><td>Signals journalled</td><td class="num">{a['signals']}</td></tr>
      <tr><td>Records</td><td class="num">{a['records']}</td></tr>
    </tbody></table></div></section>
</main>
<footer>{html.escape(DISCLAIMER)}</footer>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="OmniSeed single-file paper dashboard")
    ap.add_argument("--journal", default="state/paper_journal.csv")
    ap.add_argument("--status", default="state/paper_status.json")
    ap.add_argument("--out", default="state/dashboard.html")
    a = ap.parse_args()

    analysis = pr.analyze(a.journal, a.status)
    doc = render_html(analysis)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[dashboard] wrote {a.out} ({len(doc)} bytes, "
          f"equity {pr.fmt_money(analysis['last_equity'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
