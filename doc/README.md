# MTF-JEPA-MAE-VICReg: research and development brief

`representation-encoder.pdf` is the one-page outreach brief. Editable sources are
`representation-encoder.tex` and `references.bib`.

The format follows the supplied Cryptographic Electronics white paper: the
unmodified IEEEtran conference class, US Letter paper, default body size and
margins, two columns, numbered sections, and IEEE references. The author/contact
line follows that reference. This is a research and development brief, not a claim
of IEEE publication or endorsement.

The manuscript uses the encoder's established name, MTF-JEPA-MAE-VICReg, and
describes the implemented sensor-history encoder. It concludes with the
readiness of the working core, the capability demonstrated by the implementation,
and the support sought to refine it through completion. The encoder name is the
title, with the world-model description as a smaller subtitle. A numbered Source
Code section before the references provides the supplied repository link and a
note on the implementation and workflow. The input rank and dimensions are
defined individually; MTF, JEPA, MAE, and VICReg are spelled out. Embeddings,
tokens, learning objectives, and the dynamics equation are explained in plain
language. Action-conditioned
dynamics and robotics evaluation remain proposed
work; the brief does not claim measured robotics performance or real-time deployment.

## Build

Reuse the existing environment; the paper builder does not create containers:

```powershell
& C:\Work\documents\cv.ps1 start
.\build.ps1 -Render
```

This uses the installed latexmk/pdfLaTeX, Poppler and qpdf in `documents-latex`.
Sources are copied to a unique container `/tmp` snapshot; the final PDF and
build/preview files are copied back here. The script requires a one-page result.
No LaTeX packages are added to the embedding runtime container.

## Sources

Implementation claims were checked against `../include/embedding/` and the
project's documented validation results. The three bibliography entries link to
the original JEPA, masked-autoencoder and VICReg papers as methodological
background; their image benchmarks are not results for this encoder.

See `vendor/PROVENANCE.md` for the formatting dependency.
