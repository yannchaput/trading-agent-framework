"""Tiny in-memory zips and PDFs for the Clerk client tests."""

from __future__ import annotations

import io
import zipfile

from pypdf import PdfWriter


def make_zip(year: int, xml: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{year}FD.xml", xml)
        archive.writestr(f"{year}FD.txt", "tab separated copy of the index")
    return buffer.getvalue()


def make_text_pdf(text: str) -> bytes:
    """A one-page PDF whose text layer is `text` (one line; no parentheses or backslashes)."""
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def make_image_only_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=600, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
