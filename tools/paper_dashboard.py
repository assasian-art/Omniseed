#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/paper_dashboard.py
#  M5/T2 — a SINGLE-FILE, self-contained web dashboard for the paper book.
#
#  Renders one HTML document with no external assets (no CDN, no fetch of
#  third-party code, no images): inline CSS, an inline SVG equity curve, plain
#  tables, and one inline script.
#
#  Panels: equity curve · open positions (entry / stop / confidence) ·
#          today's P&L · regime status · feed health · recent fills ·
#          kill-switch trips.
#
#  Two ways to view it
#  -------------------
#   * Static  — write the file and open it. The page detects `file://` and
#               says so; the numbers are a snapshot.
#   * Live    — `python tools/serve_dashboard.py` re-renders on every request
#               and exposes /api/state; the page then polls that endpoint and
#               updates the KPIs, the curve, the positions table and the regime
#               table in place, without a reload.
#
#  Usage:
#    python tools/paper_dashboard.py --out state/dashboard.html
#    python tools/serve_dashboard.py            # live, http://127.0.0.1:8787
# =============================================================================
import argparse
import datetime as dt
import html
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import paper_report as pr  # noqa: E402

DISCLAIMER = ("Educational analytics. Losses are minimized by discipline, never "
              "eliminated. Past performance does not guarantee future results. "
              "Paper trading only — no real orders are ever placed.")

DEFAULT_REFRESH_MS = 15000


# ---------------------------------------------------------------------------
# Small formatters
# ---------------------------------------------------------------------------
def _money_or_dash(x):
    return "—" if x is None else pr.fmt_money(x)


def _pct_or_dash(x, digits=2):
    return "—" if x is None else f"{x * 100.0:.{digits}f}%"


def _confidence_cell(c):
    """A confidence score as a number plus a meter.

    `None` renders as an em dash, never as 0%: a position opened before the
    engine recorded provenance has an UNKNOWN confidence, and showing 0% would
    claim it was a coin flip.
    """
    if c is None:
        return "<td class='num muted'>—</td>"
    pct = max(0.0, min(1.0, float(c))) * 100.0
    return (f"<td class='num'>{pct:.0f}%"
            f"<span class='meter'><i style='width:{pct:.0f}%'></i></span></td>")


# ---------------------------------------------------------------------------
# Equity curve (inline SVG)
# ---------------------------------------------------------------------------
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
    return f"""<svg id="eq-svg" viewBox="0 0 {w} {h}" width="100%" height="{h}"
     role="img" aria-label="equity curve" preserveAspectRatio="none">
  <rect x="0" y="0" width="{w}" height="{h}" fill="none"/>
  <line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#2a2f3a" stroke-width="1"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#2a2f3a" stroke-width="1"/>
  <polyline id="eq-poly" fill="none" stroke="{stroke}" stroke-width="2"
            points="{' '.join(pts)}"/>
  <text id="eq-hi" x="{pad}" y="{pad-10}" fill="#8b93a7" font-size="12">{hi:,.2f}</text>
  <text id="eq-lo" x="{pad}" y="{h-pad+18}" fill="#8b93a7" font-size="12">{lo:,.2f}</text>
  <text id="eq-last" x="{w-pad}" y="{h-pad+18}" fill="#8b93a7" font-size="12"
        text-anchor="end">{last}</text>
  <text id="eq-first" x="{pad}" y="{h-pad+18}" fill="#8b93a7" font-size="12"
        text-anchor="start">{first}</text>
</svg>"""


def _kpi(label, value, cls="", kpi_id=""):
    idattr = f" id=\"{kpi_id}\"" if kpi_id else ""
    return (f"<div class='kpi'><div class='kpi-label'>{html.escape(label)}</div>"
            f"<div class='kpi-value {cls}'{idattr}>{html.escape(value)}</div></div>")


# ---------------------------------------------------------------------------
# Open positions: entry · stop · confidence (the T2 requirement)
# ---------------------------------------------------------------------------
def positions_rows_html(positions):
    if not positions:
        return ("<tr><td colspan='7' class='muted'>The book is flat.</td></tr>")
    rows = []
    for sym in sorted(positions):
        p = positions[sym]
        last = p.get("last")
        unrl = p.get("unrealized", 0.0)
        stop = p.get("stop")
        # Distance to the stop, as the engine would measure it.
        if stop is not None and last:
            dist = (stop / last - 1.0) * 100.0
            dist_html = f"<span class='muted'>{dist:+.2f}%</span>"
        else:
            dist_html = "<span class='muted'>—</span>"
        rows.append(
            f"<tr><td>{html.escape(sym)}</td>"
            f"<td class='num'>{p.get('qty', 0.0):.4f}</td>"
            f"<td class='num'>{_money_or_dash(p.get('avg'))}</td>"
            f"<td class='num'>{_money_or_dash(stop)}</td>"
            f"<td class='num'>{dist_html}</td>"
            + _confidence_cell(p.get("confidence")) +
            f"<td class='num'>{_money_or_dash(last)}</td>"
            f"<td class='num {pr.pnl_class(unrl)}'>{pr.fmt_signed(unrl)}</td>"
            f"</tr>")
    return "".join(rows)


def positions_table_html(positions):
    return ("<table><thead><tr><th>symbol</th><th>qty</th><th>entry</th>"
            "<th>stop</th><th>to stop</th><th>confidence</th><th>last</th>"
            "<th>unrealized</th></tr></thead><tbody id='pos-body'>"
            + positions_rows_html(positions) + "</tbody></table>")


# ---------------------------------------------------------------------------
# Regime status (M8 engine, surfaced through the loop's heartbeat)
# ---------------------------------------------------------------------------
_REGIME_CLASS = {"trend_up": "up", "trend_down": "down", "high_vol": "warn",
                 "range": "flat"}


def regime_rows_html(regime):
    if not regime:
        return ("<tr><td colspan='6' class='muted'>No regime reading in the "
                "heartbeat. Run <code>tools/paper_loop.py</code> (needs "
                "<code>tools/monster/regime.py</code>) to populate it.</td></tr>")

    def num(x, digits):
        return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "—"

    rows = []
    for sym in sorted(regime):
        r = regime[sym] or {}
        label = str(r.get("label") or "?")
        cls = _REGIME_CLASS.get(label, "flat")
        rows.append(
            f"<tr><td>{html.escape(sym)}</td>"
            f"<td><span class='badge {cls}'>{html.escape(label)}</span></td>"
            f"<td>{html.escape(str(r.get('direction') or '—'))}</td>"
            f"<td class='num'>{num(r.get('trend_score'), 3)}</td>"
            f"<td class='num'>{num(r.get('vol_score'), 2)}</td>"
            f"<td class='num'>{'yes' if r.get('stressed') else 'no'}</td></tr>")
    return "".join(rows)


def regime_table_html(regime):
    return ("<table><thead><tr><th>symbol</th><th>regime</th><th>direction</th>"
            "<th>trend</th><th>vol</th><th>stressed</th></tr></thead>"
            "<tbody id='regime-body'>" + regime_rows_html(regime)
            + "</tbody></table>")


# ---------------------------------------------------------------------------
# The live updater (inlined; no external script)
# ---------------------------------------------------------------------------
def _live_script(refresh_ms):
    return """
<script>
(function () {
  var POLL = __REFRESH_MS__;
  var note = document.getElementById('live-note');
  if (location.protocol === 'file:') {
    if (note) note.textContent = 'static snapshot — run tools/serve_dashboard.py for live updates';
    return;
  }
  function money(x) { return (x === null || x === undefined) ? '\\u2014'
      : Number(x).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}); }
  function signed(x) { var v = Number(x || 0); return (v >= 0 ? '+' : '-') + money(Math.abs(v)); }
  function pnlClass(x) { return x > 0 ? 'up' : (x < 0 ? 'down' : 'flat'); }
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) {
      return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]; }); }
  function isoDay(ts) { return new Date(ts * 1000).toISOString().slice(0, 10); }
  function setText(id, txt) { var e = document.getElementById(id); if (e) e.textContent = txt; }

  function drawCurve(c) {
    var poly = document.getElementById('eq-poly');
    if (!poly || !c || c.length < 2) return;
    var W = 920, H = 260, P = 34;
    var ts = c.map(function (p) { return p[0]; }), eq = c.map(function (p) { return p[1]; });
    var lo = Math.min.apply(null, eq), hi = Math.max.apply(null, eq);
    if (hi - lo < 1e-9) hi = lo + 1;
    var t0 = Math.min.apply(null, ts), t1 = Math.max.apply(null, ts), span = (t1 - t0) || 1;
    poly.setAttribute('points', c.map(function (p) {
      return (P + (p[0] - t0) / span * (W - 2 * P)).toFixed(1) + ',' +
             (H - P - (p[1] - lo) / (hi - lo) * (H - 2 * P)).toFixed(1);
    }).join(' '));
    poly.setAttribute('stroke', eq[eq.length - 1] >= eq[0] ? '#e5484d' : '#2fbf71');
    setText('eq-hi', money(hi)); setText('eq-lo', money(lo));
    setText('eq-first', isoDay(t0)); setText('eq-last', isoDay(t1));
  }

  function drawPositions(pos) {
    var body = document.getElementById('pos-body');
    if (!body) return;
    var syms = Object.keys(pos || {}).sort();
    if (!syms.length) { body.innerHTML = "<tr><td colspan='7' class='muted'>The book is flat.</td></tr>"; return; }
    body.innerHTML = syms.map(function (s) {
      var p = pos[s], stop = p.stop, dist = (stop !== null && stop !== undefined && p.last)
          ? ((stop / p.last - 1) * 100).toFixed(2) + '%' : '\\u2014';
      var conf = (p.confidence === null || p.confidence === undefined) ? "<span class='muted'>\\u2014</span>"
          : Math.round(p.confidence * 100) + "%<span class='meter'><i style='width:" +
            Math.round(p.confidence * 100) + "%'></i></span>";
      return "<tr><td>" + esc(s) + "</td><td class='num'>" + Number(p.qty || 0).toFixed(4) +
        "</td><td class='num'>" + money(p.avg) + "</td><td class='num'>" + money(stop) +
        "</td><td class='num'><span class='muted'>" + dist + "</span></td>" +
        "<td class='num'>" + conf + "</td><td class='num'>" + money(p.last) +
        "</td><td class='num " + pnlClass(p.unrealized) + "'>" + signed(p.unrealized) + "</td></tr>";
    }).join('');
  }

  function drawRegime(reg) {
    var body = document.getElementById('regime-body');
    if (!body) return;
    var syms = Object.keys(reg || {}).sort();
    if (!syms.length) { body.innerHTML = "<tr><td colspan='6' class='muted'>No regime reading in the heartbeat.</td></tr>"; return; }
    body.innerHTML = syms.map(function (s) {
      var r = reg[s] || {}, cls = ({trend_up:'up', trend_down:'down', high_vol:'warn', range:'flat'})[r.label] || 'flat';
      function n(x, d) { return (typeof x === 'number' && isFinite(x)) ? x.toFixed(d) : '\\u2014'; }
      return "<tr><td>" + esc(s) + "</td><td><span class='badge " + cls + "'>" + esc(r.label || '?') +
        "</span></td><td>" + esc(r.direction || '\\u2014') + "</td><td class='num'>" + n(r.trend_score, 3) +
        "</td><td class='num'>" + n(r.vol_score, 2) + "</td><td class='num'>" + (r.stressed ? 'yes' : 'no') + "</td></tr>";
    }).join('');
  }

  function render(s) {
    setText('kpi-equity', money(s.equity));
    setText('kpi-daypnl', signed(s.day_pnl));
    setText('kpi-daypct', (s.day_pct >= 0 ? '+' : '') + Number(s.day_pct || 0).toFixed(2) + '%');
    setText('kpi-positions', String(Object.keys(s.positions || {}).length));
    setText('kpi-halts', String(s.halts || 0));
    var fe = document.getElementById('kpi-equity'), dp = document.getElementById('kpi-daypnl');
    if (fe) fe.className = 'kpi-value ' + pnlClass(s.equity - s.first_equity);
    if (dp) dp.className = 'kpi-value ' + pnlClass(s.day_pnl);
    drawCurve(s.curve); drawPositions(s.positions); drawRegime(s.regime);
    if (note) note.textContent = 'live · updated ' + new Date().toISOString().slice(11, 19) + ' UTC';
  }

  function tick() {
    fetch('api/state', {cache: 'no-store'})
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(render)
      .catch(function () { if (note) note.textContent = 'live poll failed — retrying'; });
  }
  tick();
  setInterval(tick, POLL);
})();
</script>
""".replace("__REFRESH_MS__", str(int(refresh_ms)))


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------
def render_html(a, generated_ts=None, refresh_ms=DEFAULT_REFRESH_MS):
    generated_ts = generated_ts or a["last_ts"] or 0
    gen = dt.datetime.fromtimestamp(generated_ts, dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC") if generated_ts else "—"

    feeds = pr.feed_rows(a["feeds"])
    feeds_ok = sum(1 for f in feeds if not f["abstain"])

    kpis = "".join([
        _kpi("Equity", pr.fmt_money(a["last_equity"]),
             pr.pnl_class(a["last_equity"] - a["first_equity"]), "kpi-equity"),
        _kpi("Today's P&L", pr.fmt_signed(a["day_pnl"]),
             pr.pnl_class(a["day_pnl"]), "kpi-daypnl"),
        _kpi("Today %", pr.fmt_pct(a["day_pct"]), pr.pnl_class(a["day_pct"]),
             "kpi-daypct"),
        _kpi("Open positions", str(len(a["positions"])), "", "kpi-positions"),
        _kpi("Kill-switch trips", str(a["halts"]), "", "kpi-halts"),
        _kpi("Feeds OK", f"{feeds_ok}/{len(feeds)}"),
    ])

    pos_html = positions_table_html(a["positions"])
    regime_html = regime_table_html(a.get("regime") or {})

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

    lim = pr.ENGINE_LIMITS

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
  .meter {{ display:inline-block; width:38px; height:5px; margin-left:7px;
            vertical-align:middle; background:#242b36; border-radius:3px;
            overflow:hidden; }}
  .meter > i {{ display:block; height:100%; background:var(--accent); }}
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
    · journal <code>{html.escape(a['journal'])}</code>
    · <span id="live-note">static snapshot</span></div>
</header>
<main>
  <section><h2>Headline</h2><div class="kpis">{kpis}</div></section>
  <section><h2>Equity curve</h2><div class="card">{_svg_curve(a['curve'])}</div></section>
  <section><h2>Open positions</h2><div class="card">{pos_html}</div></section>
  <section><h2>Regime status</h2><div class="card">{regime_html}</div></section>
  <section><h2>Feed health</h2><div class="card">{feed_html}</div></section>
  <section><h2>Recent fills</h2><div class="card">{fills_html}</div></section>
  <section><h2>Risk</h2><div class="card">
    <table><tbody>
      <tr><td>Per-trade risk budget</td><td class="num">{lim['per_trade_pct']:.1f}% of equity</td></tr>
      <tr><td>Per-position stop</td><td class="num">{lim['stop_loss_pct'] * 100:.1f}%</td></tr>
      <tr><td>Daily kill-switch</td><td class="num">{lim['daily_kill_pct']:.1f}%</td></tr>
      <tr><td>Weekly kill-switch</td><td class="num">{lim['weekly_kill_pct']:.1f}%</td></tr>
      <tr><td>Kill-switch trips (lifetime)</td><td class="num">{a['halts']}</td></tr>
      <tr><td>ABSTAIN events</td><td class="num">{a['abstains']}</td></tr>
      <tr><td>Risk-gate refusals</td><td class="num">{a['refused']}</td></tr>
      <tr><td>Signals journalled</td><td class="num">{a['signals']}</td></tr>
      <tr><td>Records</td><td class="num">{a['records']}</td></tr>
    </tbody></table>
    <p class="muted" style="margin:10px 0 0">Limits are enforced in the C++
    engine; this page only reports them.</p></div></section>
</main>
<footer>{html.escape(DISCLAIMER)}</footer>
{_live_script(refresh_ms)}
</body>
</html>
"""


def state_json(a):
    """The payload the live poller reads. Mirrors render_html()'s inputs."""
    return {
        "generated_ts": a["last_ts"],
        "day": a["day"],
        "equity": a["last_equity"],
        "first_equity": a["first_equity"],
        "day_pnl": a["day_pnl"],
        "day_pct": a["day_pct"],
        "curve": a["curve"],
        "positions": a["positions"],
        "regime": a.get("regime") or {},
        "feeds": a["feeds"],
        "halts": a["halts"],
        "abstains": a["abstains"],
        "refused": a["refused"],
        "signals": a["signals"],
        "records": a["records"],
    }


def render_state_json(a):
    return json.dumps(state_json(a), sort_keys=True, default=str)


def main() -> int:
    ap = argparse.ArgumentParser(description="OmniSeed single-file paper dashboard")
    ap.add_argument("--journal", default="state/paper_journal.csv")
    ap.add_argument("--status", default="state/paper_status.json")
    ap.add_argument("--out", default="state/dashboard.html")
    ap.add_argument("--refresh-ms", type=int, default=DEFAULT_REFRESH_MS,
                    help="live poll interval when served over HTTP")
    a = ap.parse_args()

    analysis = pr.analyze(a.journal, a.status)
    doc = render_html(analysis, refresh_ms=a.refresh_ms)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[dashboard] wrote {a.out} ({len(doc)} bytes, "
          f"equity {pr.fmt_money(analysis['last_equity'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
