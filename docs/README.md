# Documentation (Task 2b only)

## PDF report

- **Built PDF:** [GSoC26_Task2b_Report.pdf](GSoC26_Task2b_Report.pdf)
- **LaTeX source:** [GSoC26_Task2b_Report.tex](GSoC26_Task2b_Report.tex)

Rebuild (requires `pdflatex`):

```bash
cd docs
pdflatex -interaction=nonstopmode GSoC26_Task2b_Report.tex
pdflatex -interaction=nonstopmode GSoC26_Task2b_Report.tex
```

Intermediate LaTeX build files under `docs/` are gitignored via `docs/.gitignore`.
