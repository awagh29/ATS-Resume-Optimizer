# Resume Optimizer

A local resume workspace that reverse engineers a complete job description, retrieves your confirmed experience from a persistent knowledge base, and targets a 95+ internal match estimate using Claude Sonnet 5.5. It keeps optimization history and releases a PDF after verifying exactly one readable page. An employer's ATS score cannot be guaranteed, and missing qualifications are never invented to reach the target.

## Start on Windows

Python 3.12 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If .env already exists, edit it instead of copying over it. Set:

```dotenv
ANTHROPIC_API_KEY=your-anthropic-api-key
CLAUDE_MODEL=claude-sonnet-5-5
FLASK_SECRET_KEY=replace-with-a-long-random-secret
```

Generate a secret with: python -c "import secrets; print(secrets.token_hex(32))".
Get an API key from [Anthropic](https://platform.claude.com/). A Claude chat subscription is separate from an API key.

Install [MiKTeX](https://miktex.org/download), [TeX Live](https://www.tug.org/texlive/), or [Tectonic](https://tectonic-typesetting.github.io/en-US/install.html). The compiler needs geometry, fontenc, inputenc, lmodern, enumitem, hyperref, titlesec, and setspace. Tectonic downloads packages on its first run, requiring network/cache access; temporarily increase LATEX_TIMEOUT_SECONDS for initial setup if necessary.

The Windows executable .tools/tectonic.exe is detected automatically. Otherwise set LATEX_COMPILER to the executable's full path or add it to PATH.

```powershell
.\.venv\Scripts\python.exe run.py
```

Open [the app](http://127.0.0.1:5000). It uses Waitress on localhost by default. The home page shows Claude and compiler readiness. [Health diagnostics](http://127.0.0.1:5000/health) report configuration and executable discovery; they do not spend credits or compile a probe.

Without a key, the app explicitly enters local demo mode. A failed live Claude request never silently becomes demo output. Restart after changing .env. On macOS/Linux use .venv/bin/python.

## Docker

The image includes Python and TeX packages:

```sh
docker compose up --build
```

Compose reads ANTHROPIC_API_KEY, CLAUDE_MODEL, and FLASK_SECRET_KEY from .env, binds localhost port 5000, and uses a non-root user, read-only filesystem, bounded temporary storage, and resource limits. The named `resume-data` volume preserves the knowledge database across container restarts. The deployment recipe is supplied; validate its image build on your Docker host.

## Usage

1. Open **Knowledge Base** and save confirmed skills, achievements, metrics, or project details. Include the employer/project in each achievement's text; these sources can go beyond the current resume.
2. Paste LaTeX or upload a UTF-8 .tex file.
3. Add the complete job description and optional company/role.
4. Optionally add confirmed facts for this run and choose whether to include relevant saved knowledge. Unscoped skills may enter the Skills section.
5. Choose Letter or A4 and select **Optimize with Claude** (or **Try local demo** without a key).
6. Review the hiring blueprint, evidence, gaps, selected knowledge, token usage, and final resume.
7. Preview/download the verified one-page PDF, or edit/download LaTeX and recompile. **History** restores past analysis and editable resumes without any model request; recompile locally to recover a PDF.

Parsing supports standard sections, multiline bullets, common resumeSubheading/resumeItem/resumeProjectHeading macros, and common moderncv cventry/cvitem entries. Arbitrary custom LaTeX is not a complete parsing target. Review extraction warnings and use the sample format when needed.

The editor accepts the controlled template's commands/packages. Wording and bounded spacing edits work; arbitrary TeX programs, imports, custom packages, file/shell commands, and negative spacing are rejected.

The browser preview is rendered from the accepted PDF by PDFium, so it also works in browsers without an embedded PDF viewer. Downloads contain the original compiled PDF.

## Configuration

- ANTHROPIC_API_KEY: live optimization, backend only.
- CLAUDE_MODEL: default claude-sonnet-5-5; choose a model available to your account.
- ANTHROPIC_API_BASE_URL: default https://api.anthropic.com/v1; HTTPS only.
- CLAUDE_MAX_TOKENS: hard maximum per call, default 8192.
- CLAUDE_PLAN_MAX_TOKENS / CLAUDE_REWRITE_MAX_TOKENS / CLAUDE_REPAIR_MAX_TOKENS: stage caps, default 6000 / 4000 / 4000. The smaller of this setting and the hard maximum applies. Increase the plan cap for unusually dense descriptions if the provider reports truncation.
- CLAUDE_PROMPT_CACHING: 1 by default; marks stable system instructions for provider caching. Short prompts may fall below the provider's cache minimum. Usage reports actual cache reads/writes, never assumed savings.
- CLAUDE_EFFORT: medium by default for response time/cost balance; low or high are available. Leave empty for models that do not support this parameter.
- LLM_TIMEOUT_SECONDS: response timeout per call, default 90.
- LLM_MAX_RETRIES: transient failure retries, default 2, bounded to 3.
- LATEX_COMPILER: executable name/path; empty means discovery.
- LATEX_TIMEOUT_SECONDS: compiler timeout per layout attempt, default 30.
- FLASK_SECRET_KEY: stable random session secret.
- HOST / PORT: default 127.0.0.1 / 5000.
- SESSION_COOKIE_SECURE: set to 1 with HTTPS.
- FLASK_DEBUG: development only, default 0.
- MAX_RESUME_CHARS / MAX_JD_CHARS / MAX_SUPPLEMENTAL_CHARS: input limits.
- KNOWLEDGE_DB_PATH: SQLite database, defaults to `data/knowledge.sqlite3` in the project. Back up this database with the app stopped to preserve knowledge and history.

## Model efficiency and output contracts

The workflow uses a small plan/execute/refine state machine:

1. **Plan:** one Claude request reverse engineers the entire description and maps requirements to source facts together, replacing two separate analysis/mapping calls. Local extraction preserves requirements the model omits.
2. **Execute:** one request writes the complete resume from the plan and evidence. Local source checks, formatting, compilation, and scoring require no model tokens.
3. **Refine:** at most one additional request when the score is below 95 and supported content or wording can improve. An independently audited and fitted candidate replaces the first only when its score increases, mandatory gaps do not increase, and it preserves successful compilation. A refinement error retains the first result.

Design patterns: bounded orchestration, retrieval-augmented generation, exact-response memoization, prompt-prefix caching, native structured-output contracts, and local evaluator feedback. Compact JSON removes whitespace and duplicate schemas from requests; only the most relevant complete knowledge entries are retrieved (up to 12 entries and 6,000 characters), with stable provenance. Raw history is never appended to prompts or treated as candidate evidence.

Exact stage responses are cached in SQLite using versioned hashes of the profile, model, endpoint, effort, schema, prompts, source facts, and effective output budget. Identical stages reuse validated output without a provider request; changing the JD, resume, or retrieved facts invalidates affected stages. Deleting knowledge clears cached responses. The cache is bounded to 256 entries / 16 MB.

The adapter uses [Anthropic Messages](https://platform.claude.com/docs/en/api/overview), [native structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), and [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching). Pydantic validates responses locally. [Medium effort](https://platform.claude.com/docs/en/build-with-claude/effort) is explicit for predictable cost/latency. Invalid keys, refusals, malformed output, and truncation produce actionable errors. Errors omit credentials and provider response bodies. No live token or quality benchmark was run for this update; savings depend on input, cache reuse, and provider behavior.

## One-page contract

1. Parse source facts with stable identifiers.
2. Analyze the role and map requirements to cited facts.
3. Ask Claude for concise relevant bullets while preserving official identities.
4. Restore original contact details, employers, titles, dates, and degrees; audit source IDs, metrics, and known tools.
5. Escape text into the single-column template.
6. Compile in a temporary directory with shell execution disabled and a restricted environment.
7. Verify actual page count, extractable text, intended content, and overflow warnings.
8. Tighten spacing if needed, then omit complete lower-priority content with an explanation.

Fitting preserves 10-point body text, margins of at least 0.5 inches, and each employment entry. It never crops, hides text, shrinks the PDF, or cuts words. If a readable page cannot be produced within bounded attempts, editable LaTeX and an error are returned; PDF preview/download remain disabled. Recompilation uses the same validation. Failed runs clear any previous PDF.

Structural and common factual checks cannot prove every semantic paraphrase. Review final claims. Compiler restrictions provide defense in depth, not an OS-level sandbox.

## Match estimate

The heuristic score is an internal estimate, not an employer ATS score or hiring prediction:

- Required qualifications: 35 points.
- Responsibilities: 25.
- Skills and terminology: 15.
- Accomplishment evidence: 15.
- Document structure: 10.

Mandatory gaps stay visible and prevent declaring the target reached. Keywords cannot establish missing qualifications. Final scores and recommendations are recalculated after layout omissions. The bounded refinement never changes scoring weights or invents qualifications to raise a score.

## Privacy and deployment scope

SQLite persists explicitly confirmed knowledge, job descriptions, result snapshots, score breakdowns, selected evidence provenance, editable LaTeX, usage counters, and validated model-response cache entries on this computer. Generated history never becomes confirmed knowledge automatically. The database is unencrypted; keep it in a private local directory. Knowledge and history can be deleted in the app; deleting knowledge excludes it from future runs, while existing history retains its source snapshot until deleted separately. No vector service, embedding API, or external database is needed.

Cookies contain only an opaque session ID and CSRF token. PDF downloads live in a thread-safe per-process memory store with a one-hour expiry, 64-entry limit, and 32 MB limit. Expired entries are removed on store access; shutdown clears PDF artifacts, while history retains editable output. Downloads are session-scoped; private responses use Cache-Control: no-store. The local knowledge/history library is shared by this app's local user and is not an authenticated multi-user service.

The browser holds submitted text and displayed results. Live runs send resume facts, relevant retrieved knowledge, confirmed supplemental facts, and job information to Anthropic; review its policies for your use case.

The local server uses one process/four threads. Before scaling across workers, add shared artifact storage and configure a stable secret. Authentication, public-service rate limits, durable background jobs, and centralized storage are outside this local/private deployment scope.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Coverage includes Claude protocol/schema handling, errors, preservation of live rewrites, identity/provenance, unsupported claims, scoring, parser macros, strict PDF rejection, bounded fitting, compiler restrictions, CSRF, uploads, private downloads, expiry, and stale artifacts.

Claude responses are mocked; an automatic test fixture clears real credentials and blocks Python network connections and HTTP requests. Tests never spend model tokens, including when a real key exists in .env. Compiler integration tests use the local compiler when installed. New coverage verifies persistence, retrieval bounds, source/history separation, knowledge opt-out, cache invalidation, usage accounting, CSRF, restored history, and acceptance/rejection of bounded refinements. The CI workflow installs TeX and runs the suite on Python 3.12.

