#!/usr/bin/env python3
"""Minimal Markdown -> HTML converter for the OmniSeed master spec.

Supports the subset used by docs/OMNISEED_MASTER_SPEC.md: headings, fenced
code blocks, pipe tables, bullet lists, bold/inline-code, hr, paragraphs.
"""
import html
import sys


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    for a, b in (("**", "<strong>"), ("`", "<code>")):
        parts = s.split(a)
        # pairwise join
        buf = []
        for i, p in enumerate(parts):
            if i % 2 == 0:
                buf.append(p)
            else:
                buf.append(b + p + (b if i + 1 < len(parts) else ""))
        s = "".join(buf)
    return s


def convert(md: str) -> str:
    lines = md.splitlines()
    out = []
    i = 0
    in_code = False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            in_code = not in_code
            out.append("<pre><code>" if in_code else "</code></pre>")
            i += 1
            continue
        if in_code:
            out.append(html.escape(ln))
            i += 1
            continue
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("### "):
            out.append("<h3>" + inline(ln[4:]) + "</h3>")
        elif ln.startswith("## "):
            out.append("<h2>" + inline(ln[3:]) + "</h2>")
        elif ln.startswith("# "):
            out.append("<h1>" + inline(ln[2:]) + "</h1>")
        elif ln.strip() == "---":
            out.append("<hr>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1
            out.append("<table>")
            for r, row in enumerate(rows):
                tag = "th" if r == 0 else "td"
                out.append("<tr>" + "".join(
                    f"<{tag}>{inline(c)}</{tag}>" for c in row) + "</tr>")
            out.append("</table>")
            continue
        elif ln.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append("<li>" + inline(lines[i][2:]) + "</li>")
                i += 1
            out.append("</ul>")
            continue
        else:
            out.append("<p>" + inline(ln) + "</p>")
        i += 1
    return "\n".join(out)


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "docs/OMNISEED_MASTER_SPEC.md"
    dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".html"
    md = open(src, encoding="utf-8").read()
    body = convert(md)
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniSeed — Master Specification</title>
<style>
body{{font-family:Georgia,serif;max-width:960px;margin:2rem auto;padding:0 1rem;
     line-height:1.55;color:#222;background:#fdfdfb}}
h1,h2,h3{{font-family:'Segoe UI',system-ui,sans-serif;color:#1a3a5c}}
h1{{border-bottom:3px solid #1a3a5c;padding-bottom:.3rem}}
h2{{border-bottom:1px solid #ccd;padding-bottom:.2rem;margin-top:2rem}}
table{{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.92em}}
th,td{{border:1px solid #bbb;padding:.4rem .6rem;text-align:left;vertical-align:top}}
th{{background:#eef2f6;font-family:'Segoe UI',system-ui,sans-serif}}
tr:nth-child(even){{background:#f6f8fa}}
code{{background:#f0ede4;padding:.1em .35em;border-radius:3px;font-size:.9em}}
pre code{{display:block;padding:.8rem;overflow-x:auto}}
pre{{background:#2d2d2d;color:#f8f8f2;border-radius:6px}}
pre code{{background:none;color:inherit}}
hr{{border:none;border-top:1px solid #ccd;margin:1.5rem 0}}
strong{{color:#1a3a5c}}
</style></head><body>
{body}
</body></html>"""
    open(dst, "w", encoding="utf-8", newline="\n").write(page)
    print(f"wrote {dst} ({len(page)} bytes)")


if __name__ == "__main__":
    main()
