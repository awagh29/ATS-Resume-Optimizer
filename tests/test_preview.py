"""A real single-page preview renders without a browser PDF plugin."""

from io import BytesIO

import pytest
from PIL import Image
from pypdf import PdfWriter

from app.latex.preview import render_pdf_preview


def pdf(pages=1, width=612):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=width, height=792)
    result = BytesIO()
    writer.write(result)
    return result.getvalue()


def test_render_real_one_page_png():
    result = render_pdf_preview(pdf())
    assert result.startswith(b"\x89PNG")
    with Image.open(BytesIO(result)) as image:
        assert image.size == (918, 1188)


@pytest.mark.parametrize("source", [b"", b"not a PDF", pdf(2), pdf(width=99999)])
def test_reject_invalid_or_unbounded_preview(source):
    with pytest.raises(Exception):
        render_pdf_preview(source)
