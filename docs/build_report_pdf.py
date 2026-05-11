"""
Convertit docs/RAPPORT_PROJET_ML_WEB.md en HTML puis en PDF via Chrome ou Edge (headless).
Usage (depuis la racine du repo) : python docs/build_report_pdf.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    print("Installez : pip install markdown")
    sys.exit(1)

DOCS = Path(__file__).resolve().parent
MD_NAME = "RAPPORT_PROJET_ML_WEB.md"
OUT_HTML = DOCS / "RAPPORT_PROJET_ML_WEB_print.html"
OUT_PDF = DOCS / "RAPPORT_PROJET_ML_WEB.pdf"


def find_browser() -> Path | None:
    candidates = [
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
        / "Microsoft/Edge/Application/msedge.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> None:
    md_path = DOCS / MD_NAME
    if not md_path.exists():
        print("Fichier manquant:", md_path)
        sys.exit(1)

    text = md_path.read_text(encoding="utf-8")
    # Retirer le commentaire HTML en tête du MD pour la conversion
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("<!--"):
        end = 0
        for i, line in enumerate(lines):
            if "-->" in line:
                end = i + 1
                break
        text = "\n".join(lines[end:])
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
        extension_configs={"tables": {}},
    )

    # Chemins images : relatifs docs/images -> fichier absolu file://
    html_body = html_body.replace('src="images/', f'src="{DOCS.as_uri()}/images/')
    html_body = html_body.replace("src='images/", f"src='{DOCS.as_uri()}/images/")

    wrapper = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <title>Rapport Medical ML</title>
  <style>
    @page {{ margin: 18mm; size: A4; }}
    body {{
      font-family: "Segoe UI", system-ui, sans-serif;
      font-size: 11pt;
      line-height: 1.45;
      color: #111;
      max-width: 190mm;
      margin: 0 auto;
      padding: 8px;
    }}
    h1 {{ font-size: 1.55rem; border-bottom: 2px solid #0369a1; padding-bottom: 0.35rem; }}
    h2 {{ font-size: 1.15rem; margin-top: 1.4rem; color: #0f172a; }}
    h3 {{ font-size: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0.8rem 0; font-size: 10pt; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; }}
    code, pre {{ font-family: Consolas, monospace; font-size: 9pt; background: #f8fafc; }}
    pre {{ padding: 10px; overflow-x: auto; border: 1px solid #e2e8f0; }}
    img {{ max-width: 100%; height: auto; page-break-inside: avoid; border: 1px solid #e2e8f0; }}
    hr {{ border: none; border-top: 1px solid #e2e8f0; margin: 1.2rem 0; }}
    em {{ color: #475569; }}
    a {{ color: #0369a1; }}
  </style>
</head>
<body>
{html_body}
</body>
</html>"""

    OUT_HTML.write_text(wrapper, encoding="utf-8")
    print("HTML généré :", OUT_HTML)

    browser = find_browser()
    if not browser:
        print("Chrome / Edge introuvable. Ouvrez le HTML et faites Fichier > Imprimer > PDF.")
        sys.exit(0)

    html_url = OUT_HTML.as_uri()
    pdf_str = str(OUT_PDF.resolve())
    cmd = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        f"--print-to-pdf={pdf_str}",
        "--no-pdf-header-footer",
        html_url,
    ]
    print("Commande :", " ".join(cmd[:4]), "...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not OUT_PDF.exists():
        print("Échec headless :", r.stderr or r.stdout)
        print("Ouvrez manuellement :", OUT_HTML)
        sys.exit(1)
    print("PDF généré :", OUT_PDF)


if __name__ == "__main__":
    main()
