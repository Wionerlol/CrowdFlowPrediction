"""Render the bilingual internship project reports to HTML and PDF.

The Markdown reports are the source of truth. This renderer performs no model
training or metric recomputation; it validates local figure references and
creates submission-friendly A4 documents.
"""

from __future__ import annotations

import re
from pathlib import Path

from markdown import markdown
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"

REPORTS = {
    "cn": {
        "source": REPORT_DIR / "internship_project_report_cn.md",
        "title": "基于多源时空数据融合的城市人流预测与异常预警系统",
        "lang": "zh-CN",
    },
    "en": {
        "source": REPORT_DIR / "internship_project_report_en.md",
        "title": "Urban Crowd-Flow Forecasting and Anomaly Alerting",
        "lang": "en",
    },
}

CSS = r"""
@page {
  size: A4;
  margin: 17mm 16mm 18mm 16mm;
  @bottom-center { content: counter(page); font-size: 8pt; color: #68768a; }
}
@font-face {
  font-family: ReportSans;
  src: local("Noto Sans CJK SC"), local("Noto Sans CJK JP"), local("Microsoft YaHei"), local("Arial");
}
html { font-family: ReportSans, "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; color: #202b3a; font-size: 9.6pt; line-height: 1.58; }
body { margin: 0; }
article { max-width: 100%; }
h1 { color: #173a5e; font-size: 21pt; line-height: 1.25; margin: 8mm 0 4mm; padding-bottom: 3mm; border-bottom: 2px solid #2874a6; page-break-before: always; page-break-after: avoid; }
article > h1:first-child { page-break-before: avoid; font-size: 25pt; margin-top: 20mm; }
h2 { color: #1f5f8f; font-size: 14.5pt; margin: 7mm 0 2.5mm; padding-bottom: 1mm; border-bottom: 1px solid #d8e3ed; page-break-after: avoid; }
h3 { color: #305f80; font-size: 11.5pt; margin: 5mm 0 2mm; page-break-after: avoid; }
p { margin: 2mm 0; text-align: justify; orphans: 3; widows: 3; }
ul, ol { margin: 2mm 0 3mm 6mm; padding-left: 5mm; }
li { margin: 1mm 0; }
table { width: 100%; border-collapse: collapse; margin: 3mm 0 5mm; font-size: 8.3pt; break-inside: avoid; }
th { background: #eaf2f8; color: #173a5e; font-weight: 700; }
th, td { border: 1px solid #cbd7e2; padding: 1.6mm 1.7mm; vertical-align: top; }
tr:nth-child(even) td { background: #fafcfd; }
img { display: block; max-width: 96%; max-height: 162mm; width: auto; height: auto; margin: 4mm auto 1mm; break-inside: avoid; }
figure { margin: 4mm 0 5mm; break-inside: avoid; page-break-inside: avoid; }
figure img { margin: 0 auto 1.5mm; }
figcaption { color: #59697a; font-size: 8.4pt; line-height: 1.45; text-align: left; }
pre { background: #f4f6f8; border: 1px solid #dce3e9; padding: 3mm; white-space: pre-wrap; font-size: 8.3pt; break-inside: avoid; }
code { background: #f1f4f7; border-radius: 2px; padding: 0.15mm 0.7mm; font-size: 8.4pt; }
pre code { background: transparent; padding: 0; }
strong { color: #132f4b; }
em { color: #59697a; }
hr { border: 0; border-top: 1px solid #d8e3ed; margin: 6mm 0; }
.toc { background: #f7fafc; border: 1px solid #d8e3ed; padding: 4mm 6mm; margin: 6mm 0; break-inside: avoid; }
.toc::before { content: "Contents / 目录"; display: block; color: #173a5e; font-size: 13pt; font-weight: 700; margin-bottom: 2mm; }
.toc ul { margin: 0 0 0 4mm; }
.toc a { color: #315f85; text-decoration: none; }
a { color: #2563a6; }
"""


def validate_images(source: Path, text: str) -> None:
    missing: list[str] = []
    for relative in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        target = (source.parent / relative).resolve()
        if not target.exists():
            missing.append(relative)
    if missing:
        raise FileNotFoundError(f"Missing report images in {source.name}: {missing}")


def render(key: str, spec: dict[str, object]) -> tuple[Path, Path]:
    source = Path(spec["source"])
    text = source.read_text(encoding="utf-8")
    validate_images(source, text)
    body = markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        extension_configs={"toc": {"permalink": False}},
    )
    # Markdown emits an image paragraph followed by an italic caption
    # paragraph. Wrap them so a page break cannot separate the two.
    body = re.sub(
        r"<p>(<img\s+[^>]+\s*/>)</p>\s*<p><em>(.*?)</em></p>",
        r"<figure>\1<figcaption>\2</figcaption></figure>",
        body,
        flags=re.DOTALL,
    )
    document = f"""<!doctype html>
<html lang="{spec['lang']}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{spec['title']}</title>
  <style>{CSS}</style>
</head>
<body><article>{body}</article></body>
</html>
"""
    html_path = source.with_suffix(".html")
    pdf_path = source.with_suffix(".pdf")
    html_path.write_text(document, encoding="utf-8")
    HTML(string=document, base_url=str(source.parent)).write_pdf(pdf_path)
    return html_path, pdf_path


def main() -> None:
    for key, spec in REPORTS.items():
        html_path, pdf_path = render(key, spec)
        print(f"{key}: {html_path.relative_to(ROOT)}")
        print(f"{key}: {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
