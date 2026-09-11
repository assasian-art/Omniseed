#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/fetch_news.py
#
#  Downloads recent market-news headlines into models/news/ for the C++ news
#  pipeline (NewsFeed::ingest_rss consumes this format directly).
#
#  Sources (no API keys):
#    * Yahoo Finance RSS  — https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US
#    * MarketWatch top    — https://feeds.content.dowjones.io/public/rss/mw_topstories
#    * CNBC markets       — https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258
#  When the network is unavailable, writes a tiny offline sample so the
#  pipeline stays testable (items marked [SAMPLE]).
#
#  Usage:
#    python tools/fetch_news.py --out models/news/headlines.xml
#    python tools/fetch_news.py --tickers AAPL,MSFT --per-ticker
# =============================================================================
import argparse
import os
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (omniseed-news/1.0)"}

FEEDS = {
    "yahoo":     "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "marketwatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "cnbc":      "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
}

SAMPLE = """<rss version="2.0"><channel><title>OmniSeed offline sample</title>
<item><guid>sample-1</guid><title>Stocks rally as tech shares surge on earnings beats</title>
<pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate></item>
<item><guid>sample-2</guid><title>AAPL upgraded to buy rating, revenue growth strong</title>
<pubDate>Mon, 01 Jan 2024 12:30:00 GMT</pubDate></item>
<item><guid>sample-3</guid><title>Chipmaker warns demand will slow, shares slump</title>
<pubDate>Mon, 01 Jan 2024 13:00:00 GMT</pubDate></item>
</channel></rss>
"""

def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", "replace")

def main() -> int:
    ap = argparse.ArgumentParser(description="OmniSeed news fetcher (RSS)")
    ap.add_argument("--out", default=os.path.join("models", "news", "headlines.xml"))
    ap.add_argument("--source", default="yahoo", choices=list(FEEDS))
    ap.add_argument("--ticker", default="AAPL", help="ticker for yahoo source")
    ap.add_argument("--tickers", default="", help="comma list; writes one file per ticker with --per-ticker")
    ap.add_argument("--per-ticker", action="store_true")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    if a.per_ticker and a.tickers:
        out_dir = os.path.dirname(a.out) or "."
        ok = 0
        for t in [x.strip().upper() for x in a.tickers.split(",") if x.strip()]:
            url = FEEDS["yahoo"].format(ticker=t)
            path = os.path.join(out_dir, f"{t}.xml")
            try:
                text = fetch(url)
                src = "yahoo"
            except Exception as e:                       # noqa: BLE001
                print(f"[news] {t}: fetch failed ({e}); writing sample", file=sys.stderr)
                text, src = SAMPLE, "sample"
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[news] {src} -> {path}")
            ok += 1
        return 0 if ok else 1

    url = FEEDS[a.source].format(ticker=a.ticker)
    try:
        text = fetch(url)
        src = a.source
    except Exception as e:                              # noqa: BLE001
        print(f"[news] fetch failed ({e}); writing offline sample", file=sys.stderr)
        text, src = SAMPLE, "sample"
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[news] {src} -> {a.out} ({len(text)} bytes)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
