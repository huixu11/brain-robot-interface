from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "THOUGHTLINK_TECH_REPORT.md"
PDF_PATH = ROOT / "THOUGHTLINK_TECH_REPORT.pdf"


def _pdf_escape(text: str) -> str:
    # Escape backslashes and parentheses for PDF string literals.
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _load_lines(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in raw:
        # Keep this generator simple: strip markdown headings but preserve content.
        line = re.sub(r"^#{1,6}\\s+", "", line).rstrip()
        out.append(line)
    # Trim trailing blank lines.
    while out and out[-1].strip() == "":
        out.pop()
    return out


def _build_pdf(lines: list[str], out_path: Path) -> None:
    # Minimal, dependency-free 1-page PDF generator using built-in Helvetica fonts.
    # Page size: Letter (612x792 points).
    w, h = 612, 792
    margin_x = 54  # 0.75in
    y0 = 760

    # Preformat content stream: Title in bold, then body in regular.
    # Use a fixed line height; the source markdown is pre-wrapped.
    content: list[str] = []
    content.append("BT")
    content.append("/F2 16 Tf")
    content.append(f"{margin_x} {y0} Td")
    if lines:
        content.append(f"({_pdf_escape(lines[0])}) Tj")
    content.append("/F1 10 Tf")
    content.append("0 -22 Td")
    content.append("12 TL")
    for line in lines[1:]:
        # Keep empty lines: they act like paragraph spacing.
        if line.strip() == "":
            content.append("T*")
            continue
        content.append(f"({_pdf_escape(line)}) Tj")
        content.append("T*")
    content.append("ET")

    stream_data = ("\n".join(content) + "\n").encode("ascii", errors="replace")

    objects: list[bytes] = []

    def add_obj(b: bytes) -> int:
        objects.append(b)
        return len(objects)

    # 1) Catalog
    add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2) Pages
    add_obj(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    # 3) Page (links fonts + content stream)
    add_obj(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] "
        f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> "
        f"/Contents 4 0 R >>".encode("ascii")
    )
    # 4) Contents
    add_obj(
        b"<< /Length " + str(len(stream_data)).encode("ascii") + b" >>\nstream\n" + stream_data + b"endstream"
    )
    # 5) Font F1 Helvetica
    add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    # 6) Font F2 Helvetica-Bold
    add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    # Write PDF with xref.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(b"%PDF-1.4\n")
        offsets: list[int] = [0]
        for i, obj in enumerate(objects, start=1):
            offsets.append(f.tell())
            f.write(f"{i} 0 obj\n".encode("ascii"))
            f.write(obj)
            f.write(b"\nendobj\n")

        xref_pos = f.tell()
        f.write(b"xref\n")
        f.write(f"0 {len(objects) + 1}\n".encode("ascii"))
        f.write(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            f.write(f"{off:010d} 00000 n \n".encode("ascii"))
        f.write(b"trailer\n")
        f.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii"))
        f.write(b"startxref\n")
        f.write(f"{xref_pos}\n".encode("ascii"))
        f.write(b"%%EOF\n")


def main() -> None:
    if not MD_PATH.exists():
        raise SystemExit(f"Missing source markdown: {MD_PATH}")
    lines = _load_lines(MD_PATH)
    _build_pdf(lines, PDF_PATH)
    print(f"[tech_report] wrote {PDF_PATH}")


if __name__ == "__main__":
    main()

