#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/nightly_report.py
#  M5 — the nightly paper-trading report.
#
#  Reads the append-only journal (+ the loop's status heartbeat) and writes a
#  Markdown summary: equity, today's P&L, open positions, today's fills,
#  per-symbol contribution, kill-switch trips, and feed health.
#
#  Email is OPT-IN and never faked: it is sent only when every OMNISEED_SMTP_*
#  variable is present; otherwise the tool says so plainly and still writes the
#  report file.
#
#  Usage:
#    python tools/nightly_report.py --out state/nightly_report.md
#    python tools/nightly_report.py --html state/nightly_report.html --email
# =============================================================================
import argparse
import os
import smtplib
import sys
from email.mime.text import MIMEText

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import paper_report as pr  # noqa: E402

DISCLAIMER = ("Educational analytics. Losses are minimized by discipline, never "
              "eliminated. Past performance does not guarantee future results. "
              "Paper trading only — no real orders are ever placed.")


def build_markdown(a, now_ts=None):
    if now_ts is None:
        now_ts = a["last_ts"]
    lines = []
    lines.append(f"# OmniSeed — nightly paper report ({a['day']} UTC)")
    lines.append("")
    lines.append(f"- Journal: `{a['journal']}` ({a['records']} records, "
                 f"{a['fills']} fills, {a['signals']} signals)")
    if a["status_path"]:
        lines.append(f"- Status: `{a['status_path']}`")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **Equity**: {pr.fmt_money(a['last_equity'])}")
    lines.append(f"- **Today's P&L**: {pr.fmt_signed(a['day_pnl'])} "
                 f"({pr.fmt_pct(a['day_pct'])})")
    lines.append(f"- **Open positions**: {len(a['positions'])}")
    lines.append(f"- **Kill-switch trips (lifetime)**: {a['halts']}")
    lines.append(f"- **ABSTAIN events**: {a['abstains']} · "
                 f"**risk-gate refusals**: {a['refused']}")
    lines.append("")

    lines.append("## Open positions")
    lines.append("")
    if a["positions"]:
        lines.append("| symbol | qty | avg | last | unrealized |")
        lines.append("|---|---:|---:|---:|---:|")
        for sym in sorted(a["positions"]):
            p = a["positions"][sym]
            lines.append(f"| {sym} | {p['qty']:.4f} | {p['avg']:.2f} | "
                         f"{p['last']:.2f} | {pr.fmt_signed(p['unrealized'])} |")
    else:
        lines.append("_none — the book is flat._")
    lines.append("")

    lines.append("## Today's fills")
    lines.append("")
    if a["today_fills"]:
        lines.append("| ts (UTC) | symbol | qty | price | pnl | reason |")
        lines.append("|---|---|---:|---:|---:|---|")
        for r in a["today_fills"]:
            when = pr.dt.datetime.fromtimestamp(
                r["ts"], pr.dt.timezone.utc).strftime("%H:%M:%S")
            lines.append(f"| {when} | {r['ticker']} | {r['qty']:.4f} | "
                         f"{r['price']:.2f} | {pr.fmt_signed(r['pnl'])} | "
                         f"{r['reason']} |")
    else:
        lines.append("_no fills today._")
    lines.append("")

    lines.append("## Per-symbol (session)")
    lines.append("")
    if a["per_symbol"]:
        lines.append("| symbol | entries | exits | realized |")
        lines.append("|---|---:|---:|---:|")
        for sym in sorted(a["per_symbol"]):
            d = a["per_symbol"][sym]
            lines.append(f"| {sym} | {d['entries']} | {d['exits']} | "
                         f"{pr.fmt_signed(d['realized'])} |")
    else:
        lines.append("_no fills recorded._")
    lines.append("")

    lines.append("## Feed health")
    lines.append("")
    rows = pr.feed_rows(a["feeds"])
    if rows:
        lines.append("| symbol | asset | provider | state | 24h | volume |")
        lines.append("|---|---|---|---|---:|---:|")
        for f in rows:
            flag = "ABSTAIN" if f["abstain"] else f["state"]
            lines.append(f"| {f['symbol']} | {f['asset']} | {f['provider']} | "
                         f"{flag} | {pr.fmt_pct(f['change_pct_24h'])} | "
                         f"{f['last']:.4g} |")
    else:
        lines.append("_no status heartbeat found — run tools/paper_loop.py._")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"> {DISCLAIMER}")
    lines.append("")
    return "\n".join(lines)


def send_email(subject, body, cfg):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]
    with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as s:
        s.starttls()
        s.login(cfg["user"], cfg["password"])
        s.sendmail(cfg["user"], [t.strip() for t in cfg["to"].split(",")],
                   msg.as_string())


def email_config_from_env(env):
    keys = ["OMNISEED_SMTP_HOST", "OMNISEED_SMTP_USER", "OMNISEED_SMTP_PASS",
            "OMNISEED_REPORT_TO"]
    if not all(env.get(k) for k in keys):
        return None
    return {"host": env["OMNISEED_SMTP_HOST"],
            "port": int(env.get("OMNISEED_SMTP_PORT", "587")),
            "user": env["OMNISEED_SMTP_USER"],
            "password": env["OMNISEED_SMTP_PASS"],
            "to": env["OMNISEED_REPORT_TO"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="OmniSeed nightly paper report")
    ap.add_argument("--journal", default="state/paper_journal.csv")
    ap.add_argument("--status", default="state/paper_status.json")
    ap.add_argument("--out", default="state/nightly_report.md")
    ap.add_argument("--html", default="",
                    help="also write an HTML copy via paper_dashboard")
    ap.add_argument("--email", action="store_true",
                    help="send by SMTP when OMNISEED_SMTP_* are set")
    a = ap.parse_args()

    analysis = pr.analyze(a.journal, a.status)
    md = build_markdown(analysis)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[report] wrote {a.out} ({analysis['records']} records, "
          f"equity {pr.fmt_money(analysis['last_equity'])}, "
          f"today {pr.fmt_signed(analysis['day_pnl'])})")

    if a.html:
        import paper_dashboard as pd  # local import: optional dependency
        html = pd.render_html(analysis)
        os.makedirs(os.path.dirname(a.html) or ".", exist_ok=True)
        with open(a.html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[report] wrote {a.html}")

    if a.email:
        cfg = email_config_from_env(os.environ)
        if cfg is None:
            print("[report] email NOT sent: OMNISEED_SMTP_HOST/USER/PASS and "
                  "OMNISEED_REPORT_TO must all be set. Refusing to fake a send.")
        else:
            try:
                send_email(f"OmniSeed paper report {analysis['day']}", md, cfg)
                print(f"[report] emailed to {cfg['to']}")
            except Exception as e:                       # noqa: BLE001
                print(f"[report] email FAILED: {e}")
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
