"""Render accepted PDF bytes into a private, bounded browser preview."""

from io import BytesIO
from threading import Lock

import pypdfium2 as pdfium

# PDFium is not thread-safe, even across different documents. Waitress runs
# request threads, so guard the complete native render/cleanup lifecycle.
_RENDER_LOCK = Lock()


def render_pdf_preview(pdf_bytes: bytes) -> bytes:
    if not pdf_bytes or len(pdf_bytes) > 5_000_000:
        raise ValueError("Preview PDF is empty or too large.")
    with _RENDER_LOCK:
        document = pdfium.PdfDocument(pdf_bytes)
        try:
            if len(document) != 1:
                raise ValueError("Only verified one-page PDFs can be previewed.")
            page = document[0]
            try:
                width, height = page.get_size()
                if not (300 <= width <= 850 and 300 <= height <= 1200):
                    raise ValueError("PDF dimensions exceed preview limits.")
                bitmap = page.render(scale=1.5)
                try:
                    rendered = bitmap.to_pil()
                    try:
                        buffer = BytesIO()
                        rendered.save(buffer, format="PNG")
                        return buffer.getvalue()
                    finally:
                        rendered.close()
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            document.close()
