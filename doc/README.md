# MTF-JEPA-MAE-VICReg: research and development brief

For container setup, named task sessions and coordination, see
[the development environment](ENVIRONMENT.md).

The [embedding evaluation policy](EMBEDDING_EVALUATION_POLICY.md) is the shared
reference for evaluating any encoder, separating correctness, optimization,
representation evidence, and acceptance decisions.
The [implementation review](EVALUATION_IMPLEMENTATION_REVIEW.md) records current
policy coverage and gaps. The separate [evaluation integration](../code/evaluation/README.md)
documents shared provider/card contracts, the `embedding_evaluate` command,
minimum `feature_harness` controls and encoder registration. Its controlled
protocol v2 produces development evidence; historical baseline reports preserve
their original protocol and measurements.

Use the [results reporting standard](RESULTS_REPORTING_STANDARD.md) for compact
quality/training tables and plain explanations. The
[embedding version registry](EMBEDDING_VERSIONS.md) gives stable tags and short
descriptions. **RPB-v4 — Learned global bottleneck** is the active experimental
version; its [next advance plan](../code/encoders/raw_patch_bottleneck_mae/NEXT_ADVANCE.md)
uses native encoder outputs and standalone raw/PCA baselines. The
[completed native baseline comparison](../code/encoders/raw_patch_bottleneck_mae/NATIVE_BASELINE_COMPARISON.md)
fills the missing controls without retraining the encoder or opening TEST inputs.
The separate [completed native curve](../code/encoders/raw_patch_bottleneck_mae/NATIVE_CURVE_ADVANCE.md)
trains fresh RPB-v4 instances on verified CUDA, selects 512 updates on native
linear VALIDATION, and reports fresh synthetic TEST plus fixed-readout stress.
It introduces no post-encoder PCA or classifier tuning. Phase 3 is the next
encoder mechanism experiment, as recorded in the advance plan.

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

Implementation claims were checked against `../code/encoders/mtf_jepa_mae_vicreg/include/embedding/encoders/mtf_jepa_mae_vicreg/` and the
project's documented validation results. The three bibliography entries link to
the original JEPA, masked-autoencoder and VICReg papers as methodological
background; their image benchmarks are not results for this encoder.

See `vendor/PROVENANCE.md` for the formatting dependency.

The implementation is organized under [code](../code/encoders/mtf_jepa_mae_vicreg/README.md),
with [shared components](../code/shared/README.md) and the
[RPB-MAE encoder](../code/encoders/raw_patch_bottleneck_mae/README.md).
Its separate [implementation guidelines](../code/encoders/raw_patch_bottleneck_mae/RPB_MAE_implementation_guidelines.md)
define architecture contracts, reuse boundaries and evaluation gates.
See [environment coordination](ENVIRONMENT.md) for development commands from
the repository root. Paper build commands above run from this `doc/` directory.

The [balanced-view RPB-v8 diagnostic](../code/encoders/raw_patch_bottleneck_mae/CONTEXT_BALANCED_VALIDATION_DIAGNOSTIC.md)
and [durable result summary](results/context_balanced_validation_v1.json) record
the latest completed and independently audited TRAIN/known-VALIDATION comparison.
RPB-v4 remains active; no TEST/stress or promotion occurred.
