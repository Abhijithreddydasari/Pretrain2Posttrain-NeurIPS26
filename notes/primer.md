# Project Primer — Math & Computation

Short track only. Read alongside code; do not restart a literature marathon.

## 1. Training objective

Image tokens and a short prompt condition the model. Loss is **autoregressive cross-entropy on target SVG tokens only** (prompt/image positions masked):

\[
\mathcal{L} = -\sum_{t \in \mathcal{T}_{\text{svg}}} \log p_\theta(y_t \mid y_{<t}, x_{\text{image}}, x_{\text{prompt}})
\]

Base weights stay frozen under LoRA. Adapters update \(\Delta W = \frac{\alpha}{r}BA\). QLoRA stores base weights in 4-bit form; adapters/optimizer states stay higher precision — needed for 8GB E2B smoke tests.

**Base vs IT:** starting from `gemma-4-E4B` (no `-it`) supports “post-training from pretraining.” Instruct checkpoints must be framed as *continued* post-training.

## 2. Emergence statistics

For metric \(m\), let \(m_0\) be base and \(m_1\) final. Normalized gain at checkpoint \(c\):

\[
g_m(c) = \frac{m(c) - m_0}{m_1 - m_0 + \varepsilon}
\]

- \(t_{50}\): first checkpoint with \(g_m \ge 0.5\)
- \(t_{90}\): first with \(g_m \ge 0.9\)
- Area between syntax and structure curves over checkpoints summarizes lag

Bootstrap over **examples** for CIs. Single-seed training uncertainty ≠ example bootstrap.

## 3. Structure metrics

- **Entity F1 / relation F1:** precision/recall on typed nodes and typed edges after matching predicted scene graph to gold (label + type; greedy or Hungarian on small graphs).
- **Reachability (workflows):** fraction of gold directed paths of length ≤ \(k\) that exist in the prediction.
- **Geometry relations:** accuracy on incidence / containment / left-right / above-below predicates recovered from SVG geometry.
- **OOD gap:** \(\Delta = m_{\text{ID}} - m_{\text{OOD}}\).

Exact SVG string match is inappropriate: many programs render the same diagram.

## 4. SVG computation (minimum)

- `viewBox` defines the user coordinate system; we lock `0 0 512 512`.
- Drawable primitives: `rect`, `circle`, `line`, `path`, `polygon`, `text`, …
- Forbidden: scripts, external images, base64 rasters (see `canonical_svg.md`).
- Rasterization must be deterministic for eval hashes and DINO inputs.

## 5. Papers — what to skim

| Source | Steal |
|--------|-------|
| Gemma 4 HF docs/blog | Processor, chat template, multimodal forward |
| LoRA / QLoRA papers or primer | \(\Delta W=(\alpha/r)BA\), frozen base |
| FlowGen | Graph params; Strict/Relaxed F1; triplet semantics |
| StarVector (data+eval sections) | Inverse rendering; why DINO ≠ topology |
| LIMA / Chu (already read) | Claim style; ID vs OOD |

## 6. Controls

Score the same targets under **correct**, **shuffled**, and **blank** images. If shuffled ≈ correct, the model is not using vision.
