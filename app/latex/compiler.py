"""Compile the application's restricted LaTeX dialect with bounded local tools.

The allowlist is defence in depth, not an operating-system sandbox. Public
deployments should additionally isolate the compiler in a dedicated container.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from flask import current_app, has_app_context

from app.config import Config
from app.models.schemas import CompilationResult

MAX_TEX_BYTES = 500_000
MAX_PDF_BYTES = 5_000_000
MAX_LOG_BYTES = 2_000_000

# Edits to generated wording and spacing work; arbitrary uploaded TeX is never
# accepted as a program to execute.
ALLOWED_COMMANDS = frozenset({
    "documentclass", "usepackage", "hypersetup", "pagestyle", "setlength",
    "parindent", "parskip", "emergencystretch", "setlist", "titleformat",
    "section", "large", "bfseries", "uppercase", "titlerule", "titlespacing",
    "renewcommand", "baselinestretch", "begin", "end", "LARGE",
    "normalsize", "vspace", "textbf", "textit", "hfill", "item", "cdot",
    "textbackslash", "textasciitilde", "textasciicircum", "textbar",
})
ALLOWED_PACKAGES = frozenset({
    "geometry", "fontenc", "inputenc", "lmodern", "enumitem", "hyperref",
    "titlesec", "setspace", "microtype", "xurl",
})
ALLOWED_ENVIRONMENTS = frozenset({"document", "center", "itemize"})


def _setting(name: str, default=None):
    return current_app.config.get(name, default) if has_app_context() else getattr(Config, name, default)


def validate_latex_source(tex_source: str) -> str:
    """Return a user-facing rejection reason, or an empty string if permitted."""
    if not tex_source or len(tex_source.encode("utf-8")) > MAX_TEX_BYTES:
        return "LaTeX source is empty or exceeds the size limit."
    if "^^" in tex_source or "\x00" in tex_source:
        return "Encoded TeX commands and null characters are not permitted."
    # An escaped percent is text, whereas an unescaped percent begins a comment.
    source = re.sub(r"(?<!\\)%[^\n]*", "", tex_source)
    if len(re.findall(r"\\begin\s*\{document\}", source)) != 1 or len(
        re.findall(r"\\end\s*\{document\}", source)
    ) != 1:
        return "Use exactly one document environment from the generated template."
    if re.search(r"\\end\s*\{document\}\s*\S", source):
        return "Content after the document end is not permitted."
    classes = re.findall(r"\\documentclass\s*(?:\[([^\]]*)\])?\s*\{([^}]*)\}", source)
    if len(classes) != 1 or classes[0][1] != "article":
        return "Only the generated article document class is supported."
    if any(x.strip() not in {"letterpaper", "a4paper", "10pt", "11pt", "12pt"}
           for x in classes[0][0].split(",") if x.strip()):
        return "Use Letter or A4 paper and a readable 10–12 point font."
    for match in re.finditer(r"\\([A-Za-z@]+)|\\([^A-Za-z])", source):
        command, symbol = match.groups()
        if command and command not in ALLOWED_COMMANDS:
            return f"LaTeX command \\{command} is outside the supported resume template."
        if symbol and symbol not in {"\\", "&", "%", "$", "#", "_", "{", "}", " ", ",", ";", ":", "!"}:
            return "This LaTeX control symbol is outside the supported resume template."
    packages_list = re.findall(r"\\usepackage\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", source)
    if len(packages_list) != len(re.findall(r"\\usepackage\b", source)):
        return "Package names must use the generated literal package declarations."
    for packages in packages_list:
        if any(p.strip() not in ALLOWED_PACKAGES for p in packages.split(",")):
            return "Only packages used by the controlled resume template are permitted."
    environments = re.findall(r"\\(?:begin|end)\s*\{([^}]*)\}", source)
    if len(environments) != len(re.findall(r"\\(?:begin|end)\b", source)) or any(
        name not in ALLOWED_ENVIRONMENTS for name in environments
    ):
        return "Only document, center, and itemize environments are supported."
    # These commands can redefine state; restrict their targets and values.
    renewals = re.findall(r"\\renewcommand\s*\{\\([^}]+)\}\s*\{([^}]*)\}", source)
    if len(renewals) != len(re.findall(r"\\renewcommand\b", source)) or any(
        name != "baselinestretch" or not re.fullmatch(r"\s*(?:0\.(?:9[5-9])|1(?:\.0[0-9]?)?)\s*", value)
        for name, value in renewals
    ):
        return "Only the template's bounded line-spacing setting may be redefined."
    lengths = re.findall(r"\\setlength\s*\{\\([^}]+)\}\s*\{([^}]*)\}", source)
    if len(lengths) != len(re.findall(r"\\setlength\b", source)) or any(
        name not in {"parindent", "parskip", "emergencystretch"}
        or not re.fullmatch(r"\s*\d+(?:\.\d+)?(?:pt|em)\s*", value)
        for name, value in lengths
    ):
        return "Only the template's paragraph spacing lengths may be edited."
    for options, package in re.findall(
        r"\\usepackage\s*(?:\[([^\]]*)\])?\s*\{([^}]*)\}", source
    ):
        if package == "geometry":
            margin = re.fullmatch(r"margin\s*=\s*(\d+(?:\.\d+)?)in", options.strip())
            if not margin or not 0.5 <= float(margin.group(1)) <= 1.2:
                return "Resume margins must be between 0.5 and 1.2 inches."
        elif options and (package, options) not in {("fontenc", "T1"), ("inputenc", "utf8")}:
            return "Custom package options are outside the supported resume template."
    # Negative or extreme vertical motion could hide text while retaining its
    # extractable PDF content. Keep the same bounded spacing dialect as rendering.
    vspaces = re.findall(r"\\vspace\s*\{([^}]*)\}", source)
    if len(vspaces) != len(re.findall(r"\\vspace\b", source)):
        return "Use literal numeric vertical spacing from the generated template."
    for value in vspaces:
        amount = re.fullmatch(r"\s*(\d+(?:\.\d+)?)(em|pt)\s*", value)
        if not amount or float(amount.group(1)) > (2 if amount.group(2) == "em" else 24):
            return "Vertical spacing must be nonnegative and no greater than two lines."
    if re.search(r"\\\\\s*\[\s*-", source):
        return "Negative line spacing is not permitted."
    title_formats = re.findall(r"\\titleformat[^\n]*", source)
    if len(title_formats) != len(re.findall(r"\\titleformat\b", source)) or any(
        re.sub(r"\s+", "", value) != r"\titleformat{\section}{\large\bfseries\uppercase}{}{0em}{}[\titlerule]"
        for value in title_formats
    ):
        return "Use the generated section heading format."
    title_spacings = re.findall(
        r"\\titlespacing\*\{\\section\}\{0pt\}\{\s*(\d+(?:\.\d+)?)em\}\{\s*(\d+(?:\.\d+)?)em\}", source
    )
    if len(title_spacings) != len(re.findall(r"\\titlespacing\b", source)) or any(
        float(top) > 2 or float(bottom) > 2 for top, bottom in title_spacings
    ):
        return "Section heading spacing must remain within the readable template bounds."
    setups = re.findall(r"\\hypersetup\s*\{([^\n]*)\}", source)
    if len(setups) != len(re.findall(r"\\hypersetup\b", source)) or any(
        setup.strip() != "colorlinks=false,pdfborder={0 0 0}" for setup in setups
    ):
        return "Use the generated PDF link settings."
    lists = re.findall(r"\\setlist\[itemize\]\{([^}]*)\}", source)
    if len(lists) != len(re.findall(r"\\setlist\b", source)) or any(
        not re.fullmatch(r"leftmargin=\*,itemsep=0\.\d+em,topsep=0\.1em,parsep=0pt,partopsep=0pt", value)
        for value in lists
    ):
        return "Use the generated bullet-list spacing settings."
    return ""


def find_compiler(preferred: str | None = None) -> str | None:
    preferred = (preferred or _setting("LATEX_COMPILER", "") or "").strip()
    # An explicit setting should fail clearly rather than silently use a different engine.
    if preferred:
        return shutil.which(preferred)
    for name in ("pdflatex", "tectonic", "xelatex", "lualatex"):
        path = shutil.which(name)
        if path:
            return path
    portable = Path(__file__).resolve().parents[2] / ".tools" / "tectonic.exe"
    return str(portable) if portable.is_file() else None


def _read_log_tail(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read(limit).decode("utf-8", errors="replace")


def compile_latex(tex_source: str, *, timeout: int | None = None) -> CompilationResult:
    """Compile permitted source; PDF acceptance is performed by validate_pdf."""
    result = CompilationResult()
    rejection = validate_latex_source(tex_source)
    if rejection:
        result.error = rejection
        return result
    compiler = find_compiler()
    result.compiler_available = compiler is not None
    if not compiler:
        result.error = (
            "No LaTeX compiler found. Install TeX Live, MiKTeX, or Tectonic, "
            "or set LATEX_COMPILER to its executable. LaTeX source is available; "
            "a verified one-page PDF has not been produced."
        )
        return result

    timeout = timeout if timeout is not None else _setting("LATEX_TIMEOUT_SECONDS", 30)
    try:
        with tempfile.TemporaryDirectory(prefix="resumeopt_") as tmpdir:
            tex_path = Path(tmpdir) / "resume.tex"
            tex_path.write_text(tex_source, encoding="utf-8")
            tectonic = Path(compiler).stem.lower() == "tectonic"
            if tectonic:
                cmd = [compiler, "--untrusted", "--keep-logs", "--outdir", tmpdir, str(tex_path)]
            else:
                cmd = [compiler, "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape",
                       "-file-line-error", "-output-directory", tmpdir, str(tex_path)]
            # Discard stdout: it can contain private resume text and be unbounded.
            # Engine logs are bounded after completion and never returned verbatim.
            proc = subprocess.run(
                cmd, cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=timeout, check=False, env=_restricted_env(),
            )
            pdf_path = Path(tmpdir) / "resume.pdf"
            log_path = Path(tmpdir) / "resume.log"
            if log_path.exists() and log_path.stat().st_size > MAX_LOG_BYTES:
                result.error = "Compiler output exceeded the log size limit."
                return result
            log = _read_log_tail(log_path, MAX_LOG_BYTES)
            if proc.returncode != 0 or not pdf_path.exists():
                result.error = "LaTeX compilation failed. Check syntax in the generated template and installed packages."
                if "not found" in log.lower() or "can't find file" in log.lower():
                    result.error += " A required TeX package or font is unavailable."
                return result
            if pdf_path.stat().st_size > MAX_PDF_BYTES:
                result.error = "Compiled PDF exceeds the size limit."
                return result
            # A page count alone cannot detect text extending beyond its box.
            overfull = re.findall(r"Overfull \\[hv]box \((\d+(?:\.\d+)?)pt too (wide|high)\)", log)
            if any(float(amount) > 1.0 for amount, _ in overfull):
                result.warnings.append("Text overflows the printable layout; revise long lines or reduce content.")
            result.success = True
            result.pdf_bytes = pdf_path.read_bytes()
            return result
    except subprocess.TimeoutExpired:
        result.error = f"LaTeX compilation timed out after {timeout}s."
    except OSError:
        result.error = "The configured LaTeX compiler could not be executed."
    return result


def _restricted_env() -> dict[str, str]:
    """Pass only runtime paths, omitting all provider credentials."""
    keep = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update({"openin_any": "p", "openout_any": "p", "shell_escape": "f",
                "TECTONIC_UNTRUSTED_MODE": "1"})
    return env
