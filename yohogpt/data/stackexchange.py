"""Convert an extracted Stack Exchange ``Posts.xml`` into Q&A text documents.

    7z x stackoverflow.com-Posts.7z -odata/raw/_url/se/stackoverflow
    python -m yohogpt.data.stackexchange data/raw/_url/se/stackoverflow/Posts.xml \
        --out data/raw/pretrain/stackexchange --min-score 2

Each question becomes one document: title, body, tags, then the accepted answer
and the top-voted answers. HTML is converted to light markdown (code blocks kept).
"""

from __future__ import annotations

import argparse
import html
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

from .filters import redact
from .io import ShardWriter

_CODE_RE = re.compile(r"<pre[^>]*><code[^>]*>(.*?)</code></pre>", re.S)
_INLINE_RE = re.compile(r"<code>(.*?)</code>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def html_to_markdown(body: str) -> str:
    body = _CODE_RE.sub(lambda m: "\n```\n" + html.unescape(m.group(1)).strip("\n") + "\n```\n", body)
    body = _INLINE_RE.sub(lambda m: "`" + html.unescape(m.group(1)) + "`", body)
    body = re.sub(r"<li>", "- ", body)
    body = re.sub(r"</p>|<br\s*/?>|</li>", "\n", body)
    body = _TAG_RE.sub("", body)
    return re.sub(r"\n{3,}", "\n\n", html.unescape(body)).strip()


def convert(posts_xml: str, out: str, min_score: int = 1, max_answers: int = 3) -> int:
    questions: dict[str, dict] = {}
    answers: dict[str, list[dict]] = defaultdict(list)
    for _, el in ET.iterparse(posts_xml, events=("end",)):
        if el.tag != "row":
            continue
        a = el.attrib
        if a.get("PostTypeId") == "1" and int(a.get("Score", 0)) >= min_score:
            questions[a["Id"]] = {
                "title": a.get("Title", ""), "body": a.get("Body", ""), "tags": a.get("Tags", ""),
                "accepted": a.get("AcceptedAnswerId"), "score": int(a.get("Score", 0)),
            }
        elif a.get("PostTypeId") == "2" and int(a.get("Score", 0)) >= 0:
            answers[a.get("ParentId", "")].append({"id": a["Id"], "body": a.get("Body", ""), "score": int(a.get("Score", 0))})
        el.clear()

    n = 0
    with ShardWriter(out) as w:
        for qid, q in questions.items():
            ans = sorted(answers.get(qid, []), key=lambda x: (x["id"] != q["accepted"], -x["score"]))[:max_answers]
            if not ans:
                continue
            tags = " ".join(t for t in re.split(r"[<>|]", q["tags"]) if t)
            parts = [f"# {html.unescape(q['title'])}", f"Tags: {tags}", "", html_to_markdown(q["body"])]
            for x in ans:
                label = "Accepted answer" if x["id"] == q["accepted"] else f"Answer (score {x['score']})"
                parts += ["", f"## {label}", "", html_to_markdown(x["body"])]
            w.write({"text": redact("\n".join(parts)), "meta": {"source": "stackexchange", "qid": qid}})
            n += 1
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("posts_xml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=int, default=1)
    ap.add_argument("--max-answers", type=int, default=3)
    args = ap.parse_args(argv)
    print(f"wrote {convert(args.posts_xml, args.out, args.min_score, args.max_answers)} documents")


if __name__ == "__main__":
    main()
