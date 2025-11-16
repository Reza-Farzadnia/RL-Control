# RL – Research Workspace

Organized project for DRTO/RL-DRTO research on a PI-controlled CSTR.

Directory layout
- `tex/` – LaTeX sources
  - `build/` – LaTeX auxiliary files (auto-managed by `latexmkrc`)
- `docs/models/` – Built PDFs (compiled from `tex/`)
- `docs/figures/` – Figures used in LaTeX and reports
- `data/` – CSV and other result data
- `src/` – Python source code (planned)
- `tests/` – Unit tests (planned)
- `scripts/` – Helper scripts (e.g., build docs)

Build docs
- Use `make docs` or run `scripts/build_docs.sh`.
- Aux files go to `tex/build`; PDFs go to `docs/models`.

Notes
- LaTeX builds are configured via `latexmkrc` at the repo root.
- Figures are resolved via `\graphicspath` to `docs/figures/`.
