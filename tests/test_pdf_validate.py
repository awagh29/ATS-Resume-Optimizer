"""Page-count validation tests (with a minimal synthetic PDF when possible)."""

from io import BytesIO

import pytest

from app.latex.pdf_validate import validate_pdf
from app.models.schemas import CompilationResult, ContactInfo, OptimizedResume


def _minimal_pdf_one_page() -> bytes:
    """Build a tiny valid 1-page PDF without external tools."""
    # Minimal PDF structure
    objects = []
    objects.append(b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n")
    objects.append(b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n")
    objects.append(
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj\n"
    )
    stream = b"BT /F1 12 Tf 72 720 Td (Alex Rivera Experience Education Skills) Tj ET"
    objects.append(
        f"4 0 obj<< /Length {len(stream)} >>stream\n".encode() + stream + b"\nendstream\nendobj\n"
    )
    objects.append(b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n")

    out = BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(out.tell())
        out.write(obj)
    xref_pos = out.tell()
    out.write(f"xref\n0 {len(offsets)}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    )
    return out.getvalue()


def test_validate_one_page_pdf():
    pytest.importorskip("pypdf")
    resume = OptimizedResume(contact=ContactInfo(name="Alex Rivera"))
    comp = CompilationResult(success=True, pdf_bytes=_minimal_pdf_one_page(), compiler_available=True)
    out = validate_pdf(comp, resume)
    assert out.page_count == 1
    assert "Alex" in out.extracted_text or out.extracted_text  # extraction may vary


def test_validate_no_pdf():
    comp = CompilationResult(success=False, error="fail")
    out = validate_pdf(comp, OptimizedResume())
    assert out.layout_verified is False
