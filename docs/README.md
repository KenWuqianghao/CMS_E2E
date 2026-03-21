# Documentation

## PDF report (Tasks 2b + 2c)

- **Built PDF:** [GSoC26_CMS_E2E_Report.pdf](GSoC26_CMS_E2E_Report.pdf)
- **LaTeX source:** [GSoC26_CMS_E2E_Report.tex](GSoC26_CMS_E2E_Report.tex)

Rebuild (requires a LaTeX distribution with `pdflatex`):

```bash
cd docs
pdflatex -interaction=nonstopmode GSoC26_CMS_E2E_Report.tex
pdflatex -interaction=nonstopmode GSoC26_CMS_E2E_Report.tex
```

Intermediate files (`*.aux`, `*.log`, `*.out`, `*.toc`) are ignored by git when listed in the repo root `.gitignore`.
