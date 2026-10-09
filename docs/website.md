# RePert project website deployment

The public website is maintained as a minimal, static GitHub Pages site.

- Landing page: `index.html`
- CSS/JS/favicon: `assets/site/`
- Original scientific Figure 1: `figures/data/figure_1/figure1_authors.pdf`
- README-compatible PNG: `assets/repert_figure1.png`, generated from the unmodified PDF by `scripts/render_public_figure.py`.
- Build/deploy workflow: `.github/workflows/pages.yml` (stages **only** the public website files, not all analysis data).

**Repository admin step, if Pages has not already been enabled:** Settings → Pages → Build and deployment → Source: **GitHub Actions**. A public project site is then expected at `https://zhao-weijing.github.io/RePert-code/` after a successful deployment on `main`. Do not describe the URL as live before a successful workflow run and an HTTP check.

The render workflow (`.github/workflows/render-figure.yml`) converts the authors' supplied PDF into a raster preview; the PDF remains the authoritative figure. The release's scientific `metadata/FILE_MANIFEST.csv` should continue to be updated whenever a tracked documentation file changes. New site files need to be added to the manifest if included in package-integrity verification.

Local static preview: `python -m http.server 8000` from the repository root and visit `http://localhost:8000/`. This previews the site but is **not** a GitHub Pages deployment.
