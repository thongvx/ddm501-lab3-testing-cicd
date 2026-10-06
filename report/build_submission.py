"""Build the LMS submission PDF source (report/submission.html) from repo docs and screenshots."""

from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/thongvx/ddm501-lab3-testing-cicd"

strategy = markdown.markdown(
    (ROOT / "docs" / "TESTING_STRATEGY.md").read_text(encoding="utf-8"),
    extensions=["tables", "fenced_code"],
)
coverage = markdown.markdown(
    (ROOT / "docs" / "COVERAGE.md").read_text(encoding="utf-8"),
    extensions=["tables", "fenced_code"],
)
SHOTS = [
    ("01_repo_readme_badges.png", "Repository with passing CI / CD / Model Validation badges"),
    ("02_actions_all_workflows.png", "All workflow runs green (main, develop, tag v1.0.0)"),
    (
        "03_ci_pipeline_run.png",
        "CI Pipeline: lint, type check, tests on Python 3.10 & 3.11, Docker smoke test",
    ),
    (
        "04_cd_pipeline_run.png",
        "CD Pipeline on tag v1.0.0: CI gate, build & push, staging, release, production",
    ),
    ("05_model_validation_run.png", "Model Validation: training + RMSE/MAE quality gate"),
    ("06_release_v1.0.0.png", "GitHub Release v1.0.0 created by the CD pipeline"),
    ("07_coverage_report_100pct.png", "HTML coverage report (CI artifact coverage-report): 100%"),
]
shots = "".join(
    f'<figure><img src="../docs/screenshots/{f}"><figcaption>Figure {i}. {c}</figcaption></figure>'
    for i, (f, c) in enumerate(SHOTS, 1)
)

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>DDM501 Lab 3 Submission</title>
<style>
@page {{ size: A4; margin: 16mm 15mm 17mm 15mm;
  @bottom-right {{ content: "Page " counter(page) " of " counter(pages); font: 9pt Helvetica; color:#666; }}
  @bottom-left {{ content: "DDM501 Lab 3 – Testing & CI/CD – Vương Xuân Thong (25MS13306)"; font: 9pt Helvetica; color:#666; }} }}
@page :first {{ @bottom-right {{ content: none; }} @bottom-left {{ content: none; }} }}
html {{ font: 10pt/1.45 "Helvetica Neue", Helvetica, Arial, sans-serif; color:#1d2433; }}
h1 {{ font-size: 24pt; color:#0f2747; margin: 0 0 4mm; }}
h2 {{ font-size: 14pt; color:#0f2747; border-bottom: 2px solid #1f5fa8; padding-bottom: 3pt; margin: 16pt 0 7pt; break-after: avoid; }}
h3 {{ font-size: 11.5pt; color:#0f2747; margin: 11pt 0 4pt; break-after: avoid; }}
table {{ width:100%; border-collapse: collapse; font-size: 8.8pt; margin: 4pt 0 9pt; }}
th, td {{ border:1px solid #d5dae3; padding: 3pt 5pt; text-align:left; vertical-align: top; }}
th {{ background:#eef3fa; }} tr {{ break-inside: avoid; }}
code {{ font: 8.4pt Menlo, monospace; background:#f3f5f8; padding: 0 2pt; }}
pre {{ background:#f7f8fb; border:1px solid #d9dee8; padding: 6pt; font: 7.8pt/1.35 Menlo, monospace; white-space: pre-wrap; break-inside: avoid; }}
pre code {{ background:none; padding:0; }}
figure {{ margin: 6pt 0 12pt; break-inside: avoid; text-align:center; }}
figure img {{ max-width:100%; max-height: 225mm; border:1px solid #d5dae3; }}
figcaption {{ font-size: 8.8pt; color:#5b6475; margin-top: 3pt; }}
.cover {{ height: 255mm; display:flex; flex-direction:column; justify-content:center; border-top: 6px solid #1f5fa8; }}
.cover td:first-child {{ color:#5b6475; width: 30%; }} .cover table {{ font-size: 10.5pt; width: 85%; }}
.pb {{ break-before: page; }} .ok {{ color:#15803d; font-weight:700; }}
</style></head><body>
<section class="cover">
  <div style="color:#1f5fa8;font-weight:600;letter-spacing:.08em">DDM501 · AI IN PRODUCTION: FROM MODELS TO SYSTEMS</div>
  <h1 style="margin-top:6mm">Lab 3 – Testing &amp; CI/CD for ML Systems</h1>
  <div style="font-size:13pt;color:#5b6475;margin-bottom:10mm">Movie Rating Prediction API (FastAPI + SVD, MovieLens 100K)</div>
  <table>
    <tr><td>Student</td><td>Vương Xuân Thong</td></tr>
    <tr><td>Student ID</td><td>25MS13306</td></tr>
    <tr><td>Repository</td><td><a href="{REPO}">{REPO}</a></td></tr>
    <tr><td>CI status</td><td><a href="{REPO}/actions">{REPO}/actions</a> – all workflows passing</td></tr>
    <tr><td>Release</td><td><a href="{REPO}/releases/tag/v1.0.0">v1.0.0</a> · image <code>ghcr.io/thongvx/movie-rating-api:1.0.0</code></td></tr>
    <tr><td>Submission date</td><td>October 2026</td></tr>
  </table>
</section>

<h2 class="pb">1. Deliverables vs. grading rubric</h2>
<table>
<tr><th style="width:24%">Criterion</th><th>Evidence</th><th style="width:9%">Status</th></tr>
<tr><td>Unit tests (10%)</td><td>63 tests: model wrapper (real + isolated stub model), schemas, config, helpers – <code>tests/unit/</code></td><td class="ok">✔</td></tr>
<tr><td>Integration tests (8%)</td><td>44 tests: all endpoints, 404/405/422/500/503, batch, CORS, latency – <code>tests/integration/</code></td><td class="ok">✔</td></tr>
<tr><td>Data tests (6%)</td><td>32 tests: sample fixture + real MovieLens 100K data contract – <code>tests/data/</code></td><td class="ok">✔</td></tr>
<tr><td>Model behavioural tests (6%)</td><td>23 tests: invariance, directional, minimum functionality, performance, robustness – <code>tests/model/</code></td><td class="ok">✔</td></tr>
<tr><td>CI workflow works (12%)</td><td><code>ci.yml</code>: lint (pre-commit) · mypy → tests on Python 3.10 &amp; 3.11 → Docker build + smoke test (Fig. 2–3)</td><td class="ok">✔</td></tr>
<tr><td>All checks pass (10%)</td><td>Every run green on <code>main</code>, <code>develop</code> and tag <code>v1.0.0</code>; badges "passing" (Fig. 1–2)</td><td class="ok">✔</td></tr>
<tr><td>CD workflow configured (8%)</td><td><code>cd.yml</code>: CI gate → push to GHCR → staging smoke test → GitHub Release → production; plus <code>model-validation.yml</code> (Fig. 4–6)</td><td class="ok">✔</td></tr>
<tr><td>Pre-commit hooks (8%)</td><td>hygiene hooks, black, isort, flake8, mypy, local pytest hook; same hooks run in CI</td><td class="ok">✔</td></tr>
<tr><td>Linting passes (6%)</td><td>black, isort, flake8 – 0 issues (CI job "Lint")</td><td class="ok">✔</td></tr>
<tr><td>Type hints (6%)</td><td>all of <code>app/</code> and <code>scripts/</code> typed; <code>mypy --disallow-untyped-defs</code> 0 errors</td><td class="ok">✔</td></tr>
<tr><td>Testing strategy doc (10%)</td><td><code>docs/TESTING_STRATEGY.md</code> (section 3 of this document)</td><td class="ok">✔</td></tr>
<tr><td>README updated (5%)</td><td>badges, structure, how to run/test/release, CI/CD table, screenshots, changes to starter</td><td class="ok">✔</td></tr>
<tr><td>Coverage report (5%)</td><td>100% line coverage (minimum 80% enforced), HTML report artifact, <code>docs/COVERAGE.md</code> (Fig. 7)</td><td class="ok">✔</td></tr>
</table>

<h2>2. Screenshots of passing CI workflows</h2>
{shots}

<h2 class="pb">3. Testing strategy</h2>
{strategy}

<h2 class="pb">4. Coverage report</h2>
{coverage}
</body></html>"""

out = ROOT / "report" / "submission.html"
out.write_text(html, encoding="utf-8")
print(out)
