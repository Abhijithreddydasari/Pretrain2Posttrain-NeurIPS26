# Workshop manuscript source

**What Changes During SVG Fine-Tuning? Separating Visual Discrimination from Executable Generation**

This directory contains the current source associated with the NeurIPS 2026 workshop submission. It uses the bundled `neurips_2026.sty`, the `dblblindworkshop` option, and the workshop title *Transitioning from Pre-Training to Post-Training*. Anonymous author metadata is retained.

## Build

Upload this directory to Overleaf and select **`main.tex`** as the main document, or compile locally with a LaTeX installation that provides `latexmk` and BibTeX:

```bash
cd paper/neurips2026
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The manuscript references PDF figures under `figures/`, all included in Git. No training artifacts are needed to compile it. A compiled paper PDF is not currently tracked.

Inspect the resulting PDF for figure legibility, references, and layout before distributing a new version. Compare it with the actual submitted PDF before labeling it an exact submission snapshot; no byte-identical submission archive is recorded here.

## Contents

- [main.tex](main.tex) — manuscript and appendices.
- [references.bib](references.bib) — bibliographic entries.
- [neurips_2026.sty](neurips_2026.sty) — bundled style.
- [figures/](figures/) — current manuscript figure assets.

See the [paper README](../README.md) for findings, figure provenance, and the distinction between this manuscript and the obsolete Markdown skeleton.
