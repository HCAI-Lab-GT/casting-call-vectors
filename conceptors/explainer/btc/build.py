"""Assemble the Beyond the Cap explainer from its parts.

usage: python build.py DATA_JSON OUT_HTML PREVIEW_HTML JS_OUT
OUT_HTML is the artifact body (the publish step adds the doctype skeleton). PREVIEW_HTML wraps the same content in a
minimal skeleton for local browser checks. JS_OUT is the bare script, for `node --check`.
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
HEAD = """<title>Beyond the Cap</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap">
"""


def main() -> None:
    data, out, preview, js_out = (Path(a) for a in sys.argv[1:5])
    lib = (HERE / "lib.js").read_text().replace('"use strict";\n', "", 1)
    js = '"use strict";\nconst DATA=' + data.read_text().strip() + ";\n" + lib + "\n" + "\n".join(
        (HERE / f).read_text() for f in ("p_ran.js", "p_moves.js", "p_more.js"))
    page = (HEAD + "<style>\n" + (HERE / "style.css").read_text() + "</style>\n\n" + (HERE / "body.html").read_text()
            + "\n<script>\n" + js + "</script>\n")
    out.write_text(page)
    js_out.write_text(js)
    preview.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                       '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
                       '<style>body{margin:0}</style></head><body>\n' + page + "</body></html>\n")
    print(f"page {len(page.encode()):,} bytes; script {len(js.encode()):,} bytes")


if __name__ == "__main__":
    main()
