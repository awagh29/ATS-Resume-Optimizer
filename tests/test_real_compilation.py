"""Exercise the real compiler whenever the environment provides one."""

from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfReader

from app.latex.compiler import find_compiler
from app.llm.adapter import LLMClient
from app.pipeline.orchestrator import run_optimization


@pytest.mark.skipif(find_compiler() is None, reason="A real LaTeX compiler is required")
@pytest.mark.parametrize("page_size, expected_width", [("letter", 612), ("a4", 595.28)])
def test_real_sample_is_verified_one_page(page_size, expected_width):
    samples = Path(__file__).resolve().parents[1] / "samples"
    result = run_optimization(
        (samples / "sample_resume.tex").read_text(encoding="utf-8"),
        (samples / "sample_job_description.txt").read_text(encoding="utf-8"),
        page_size=page_size, llm=LLMClient(api_key=""),
    )
    assert result.compilation.success, result.compilation.error
    assert result.compilation.layout_verified and result.compilation.page_count == 1
    reader = PdfReader(BytesIO(result.compilation.pdf_bytes))
    assert len(reader.pages) == 1
    assert float(reader.pages[0].mediabox.width) == pytest.approx(expected_width, abs=1)
    text = reader.pages[0].extract_text()
    assert "Alex Rivera" in text and "BrightPath Labs" in text and "Harbor Retail" in text
