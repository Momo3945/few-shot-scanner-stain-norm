# Paper draft (IEEE conference format)

`main.tex` + `sections/*.tex` + `references.bib`; `IEEEtran.cls` / `IEEEtran.bst` are extracted unmodified
from `conference-latex-template.zip` and `IEEEtranBST2.zip`. Figures in `figures/` are copies of
`docs/report/figures/` (regenerate there, then copy).

Build with Docker (from the repo root):

```
docker run --rm -v "D:/Research/Mitos/report:/work" -w /work texlive/texlive:latest-medium \
  latexmk -pdf -interaction=nonstopmode main.tex
```

(In Git Bash prefix with `MSYS_NO_PATHCONV=1`.) The medium TeX Live scheme has no `multirow`, so the
draft avoids it. Every number in Table I is checked against `docs/results/` by a cross-check script.
