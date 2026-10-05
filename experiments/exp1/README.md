# Experiment 1: can text tell a deformable attention where to look?

A self-contained spec of the first experiment in this repo: what the model computes, why each step exists, how it is
trained and evaluated, what it costs, and what is still undecided. It supersedes the earlier plan in the repo-root
`README.md` for this experiment. That plan treated the plain CLIP dual encoder as "E1" and re-ranked a top-K
shortlist; neither applies here.

---

## 1. Task and question

**Task: text-to-image vehicle re-identification on T2I-VeRW.** Given a free-text description of a vehicle (the
*query*), rank every image in a *gallery* so that images of the described vehicle come first. Train and test
vehicle IDs are disjoint, so the model can't memorise IDs. It has to learn a matching function that generalises to
unseen vehicles.

**Hypothesis.** A caption like *"…two silver inspection stickers in the upper right corner of the windshield…"*
says *where* the discriminative evidence is. If the caption, after a look at the image, can predict **where each
image location should sample features from**, the image representation should capture caption-relevant details
better than CLIP's single global vector.

---

## 2. Data and evaluation protocol

| | T2I-VeRW paper (arXiv 2605.06012) | Our annotation file |
|---|---|---|
| file | (official split, not available here) | `annotations/reid_with_mask_prompt_and_boxes_filepath_prefixid_idsplit_70_30.json` |
| total images | 14,668 | 14,668 |
| train | 10,265 pairs / 1,251 IDs | 10,264 pairs / 1,797 IDs |
| test ("val") | 4,403 pairs / 545 IDs | 4,404 pairs / 769 IDs |
| captions per image | 1 | 1 |

- Our split has no ID overlap between train and val (verified), but its ID counts don't match the paper's. Treat the
  comparison with the paper as approximate until the official split is used.
- **Query set:** the 4,404 val captions, one per val image. **Gallery:** the 4,404 val images. A gallery image is a
  correct match for a query if it has the same vehicle ID. That includes the image the caption was written for,
  which is the CUHK-PEDES protocol: no same-image or same-camera removal. Each val ID has 3–8 images (mean 5.7).
- **Metrics:** Rank-1/5/10, mAP and mINP over the full 4,404-image ranking of every query. Implemented in
  `utils/metrics.py:t2i_metrics` and checked against a hand-computed toy example.
- **Numbers to beat** (paper Table IV, T2I-VeRW): IRRA 47.3 R1 / 22.8 mAP; PFCVR (best) 55.2 R1 / 82.8 R5 /
  90.3 R10 / **26.2 mAP**. The paper used 384×384 images; we use 224×224. (The "29.2" sometimes quoted is PFCVR's
  Rank-1 on a different dataset, T2I-VeRI.)

---

## 3. The model, step by step

Notation: B = batch size, d = 512 (CLIP ViT-B/16 joint embedding width), 196 = 14×14 patch grid (224×224 input,
16-px patches), L = 77 text tokens, H = 8 heads, P = 4 sampling points per head.

### Per image and per caption (computed once each)

**Step 1: CLIP encoders** (`model/models.py`: `encode_image_tokens`, `encode_text_tokens`; fine-tuned).
- Image *j*: `V_j` [197, d] = CLS token `cls_j` [d] plus 196 patch tokens `P_j` [196, d], row-major (index
  q = row·14 + col).
- Caption *i*: tokens `T_i` [77, d], padding mask `m_i` [77] (True up to and including the end-of-text token), and the
  global text vector `t_i` = `T_i` at the EOT position [d].

### Per (caption *i*, image *j*) pair

**Step 2: enrich the text with the image** (`CrossAttentionTransformer`, depth 4, 8 heads, pre-LN, residual).
`T̃_ij = CrossAttnTransformer(query = T_i, context = P_j)` → [77, d].
Words attend to image patches, so "windshield" can carry where the windshield is *in this image*. This is
one-directional: the image tokens are context only and are **not** updated by the text in Experiment 1.

**Step 3: learned spatial tokens read the enriched text** (`SpatialTokens` + a second `CrossAttentionTransformer`,
depth 2).
`C` [196, d] is a learned parameter with one token per patch location, initialised to a 2-D sin-cos positional
embedding plus small noise. `C̃_ij = CrossAttnTransformer(query = C, context = T̃_ij, context_mask = m_i)` → [196, d].
Token q is the "image query" at grid cell q. After this step it holds what the caption (grounded in image *j*) implies
about where to look from that cell.

**Step 4: offsets and weights** (`OffsetHead`, applied to `LayerNorm(C̃_ij)`).
- `Linear(d → H·P·2)` → offsets [196, H, P, 2] in patch units. `Linear(d → H·P)` + softmax over P → weights
  [196, H, P].
- Reference point of query q is its patch centre `((col+0.5)/14, (row+0.5)/14)`. Sampling locations =
  reference + offset/14, normalised (x, y) in [0, 1].
- Init: the offset weight is zero and its bias is a fixed "star" (head h points in direction 2πh/8; point p sits
  p+1 patches out). The weight layer is all zero, giving uniform weights. At step 0 every query samples the same
  star around itself, regardless of the text.
- Each query gets H·P = 32 points; each pair gets 196 × 32 = 6,272.
- Steps 2–4 are `TextGuidedOffsets.forward(T_i, m_i, V_j) -> (locations, weights)`, already in the repo.

**Step 5: deformable attention on the image** (`model/text_guided.py:DeformableAttention`).
- `value = Linear(P_j)` [196, d], reshaped to a 14×14 map with H heads of d/H = 64 channels.
- For every query q, head h and point p: bilinearly sample head h's 64-channel map at location (q, h, p) with
  `F.grid_sample(value, 2·loc − 1, mode='bilinear', padding_mode='zeros', align_corners=False)`. With
  `align_corners=False`, location (i+0.5)/14 hits patch i's centre exactly.
- Weighted sum over the P points, concatenate the H heads, then `Linear(d → d)`. The result is the deformable
  features `D_ij` [196, d].
- The *queries* are the spatial tokens (they decide where to look). The *values* are image patch features (what is
  read). So the content comes from the image and the geometry comes from the text path.

**Step 6: one image vector per pair** (`TextGuidedDeformable.score_pairs`; config `model.pool`).
`g_ij` [d] = a reduction of `D_ij` [196, d] to one vector. Candidates:
(a) `pool: mean` → `g_ij = mean_q D_ij`;
(b) `pool: cls_residual` (**default**) → `g_ij = cls_j + γ · mean_q D_ij` with a learned scalar γ initialised to 0,
so training starts exactly at CLIP's global image vector.

**Step 7: score.** `s_ij = cos(t_i, g_ij)`, a single number, using CLIP's EOT vector `t_i` (not the enriched text).

### Why steps 6–7 are necessary

Retrieval means sorting 4,404 gallery images for each query, and sorting needs **one number per (query, image)
pair**. Step 5 outputs 196 vectors per pair, which can't be sorted. Steps 6–7 are the minimal reduction to a number.
They do what plain CLIP does with its CLS vector, applied to the deformable features instead. Token-level matching
(compare each of the 196 features with the caption's word tokens and sum the best matches) is another valid
reduction; it isn't the default only because it's more complex.

The score also drives learning. The offsets in step 4 get gradient only through the loss on `s_ij`. When the
sampled locations make the correct image score higher than the wrong ones, the loss falls, and the gradient flows
back through `grid_sample`, which is differentiable with respect to the sampling locations, into the offsets.
Without a score there's no gradient, and the offsets would stay at their initial star.

**Gradient flow at initialisation (expected, not a bug).** Two zero inits gate the gradient for the first steps:
- With `cls_residual`, γ = 0 means the deformable path doesn't affect the score, so at step 0 only γ gets gradient.
- `OffsetHead`'s weight matrices start at zero, so the offsets don't depend on `C̃` at step 0. The spatial tokens,
  spatial decoder and text-enrichment stack get gradient only after the first update makes those weights non-zero.

Both open up after one optimizer step; Adam normalises per parameter, so small initial gradients still give
LR-sized updates. This is the same scheme as Deformable-DETR and zero-init residual branches.

---

## 4. Why every caption must be paired with every image

`g_ij` depends on **both** *i* and *j*. The text decides where image *j* is sampled (steps 2–4 use `T_i`), and the
image enriches the text (step 2 uses `P_j`). So image *j* has no single embedding: it gets a different `g_ij` for
every caption it's compared against. This is a *cross-encoder*. A CLIP-style *dual encoder*, by contrast, embeds
each image once.

Example: with 3 captions and 3 images in a batch, the loss needs all 9 scores `s_11 … s_33`. CLIP (step 1) runs 3
times for images and 3 for text, but steps 2–7 run **9** times.

- **Training:** B² pairs per step.
- **Evaluation:** every query against every gallery image, 4,404² ≈ 19.4M pairs. CLIP features are computed once
  and cached; only steps 2–7 run per pair.

---

## 5. Training

- **Objective:** `L = L_SDM + L_ID`.
  - **SDM** (Similarity Distribution Matching, IRRA CVPR'23) on the B×B score matrix `S` with `S[i, j] = s_ij`,
    temperature τ = 0.02. For caption *i*: `p_i = softmax_j(S[i, :] / τ)`; target `q_i` = uniform over the batch
    images with the same vehicle ID as caption *i* (0 elsewhere). Loss = mean_i KL(p_i ‖ q_i), plus the same in the
    image→text direction (softmax over each column). Implemented in `loss/losses.py:sdm_loss`; it currently takes
    feature vectors and computes the cosine matrix itself, so the pairwise version needs to accept a precomputed
    `S`.
  - **ID loss:** a shared `Linear(d → 1797)` classifier on `t_i` and on the matched-pair image vector `g_ii`,
    cross-entropy, averaged. Training-only; discarded at test (test IDs are unseen).
- **Batch size: B = 80** → 6,400 pairs per step, 128 steps per epoch (24 samples dropped by `drop_last`). The largest
  batch that fits a 192 GiB MI300X (measured, §6).
- **Optimiser:** AdamW, β = (0.9, 0.999), weight decay 4e-5 (none on 1-D params). LR 1e-5 for CLIP, 1e-4 for all
  new modules (steps 2–6 and the ID classifier). This is the batch-64 IRRA recipe; square-root scaling to 80 would
  be ×1.12.
- **Schedule:** 200 epochs = 25,600 steps. Linear warmup for 10 epochs (1,280 steps, 5%) from 0.1× LR, then cosine
  annealing to 0 over the remaining 190 epochs. bf16
  autocast, fp32 weights (`clip.load` returns fp16 weights on GPU, so the model is cast with `.float()`). Gradient
  clipping at a total norm of 1.0.
- **Augmentation:** 224×224 bicubic resize, pad 10 + random crop, random erasing p = 0.5. **No horizontal flip**:
  captions name sides, and flipping would make the geometry the text describes wrong.
- **Logging (W&B project `t2i-reid`):**
  - every 10 steps, window-averaged: total, SDM and ID losses, total gradient norm (before clipping), ID accuracy
    for images and text, in-batch text→image Rank-1, mean matched / unmatched cosine, LR per group, samples/s, peak
    GPU memory, plus three Experiment 1 diagnostics:
    - `train/offset_disp`: mean distance (in patches) of a sampling point from its query's patch centre; the init
      star averages about 3.0
    - `train/offset_text_std`: for a fixed image, how much a sampling point moves across the batch's different
      captions (in patches). It is exactly 0 at init. **It must become clearly > 0 for the hypothesis to hold**:
      if it stays near 0, the text isn't steering the sampling
    - `train/gamma`: the weight of the deformable evidence in the image vector (`cls_residual` only)
  - every 20 epochs and at the end: `eval/R1, R5, R10, R50, mAP, mINP` from the full pairwise ranking, plus
    `eval/dual_*`, the plain CLIP ranking (text EOT vs image CLS) of the same fine-tuned backbone. The gap between
    the two is the contribution of the text-guided path
  - no step-0 evaluation (`eval_at_start: false`). With `cls_residual` it would equal zero-shot CLIP, since γ = 0,
    which is already measured on this split: R1 3.11 / R5 8.42 / R10 13.10 / mAP 2.71 / mINP 0.74

---

## 6. Measured cost

**On one AMD MI300X (192 GiB)**, with the real model, real captions, bf16 autocast, and a full training step
(forward, backward, gradient clipping, AdamW):

| B | pairs/step | peak allocated | peak reserved | s/step | steps/epoch | min/epoch |
|---|---|---|---|---|---|---|
| 64 | 4,096 | 98.3 GiB | 110.3 GiB | 1.10 | 160 | 2.9 |
| 72 | 5,184 | 122.9 GiB | 138.1 GiB | 1.35 | 142 | 3.2 |
| **80** | **6,400** | **150.4 GiB** | **168.9 GiB** | **1.65** | **128** | **3.5** |
| 88 | 7,744 | out of memory | | | | |

**Evaluation**, 500 real val queries × all 4,404 gallery images (2.2M pairs), including CLIP encoding of the whole
query and gallery sets:

| pairs per forward | time | peak allocated |
|---|---|---|
| 8,192 | 168.9 s | 33.0 GiB |
| 16,384 | 162.5 s | 61.8 GiB |
| 32,768 | 162.1 s | 119.4 GiB |

Evaluation is compute-bound (about 14.7K pairs/s), so chunk size barely matters; the config uses 8,192. The full
evaluation (19.4M pairs) **extrapolates to about 20 minutes**. A full run was started but cancelled before it
finished, so that figure is not directly measured.

**Total for the final config:** 200 epochs × 3.5 min ≈ 11.7 h training + 10 evaluations × ~20 min ≈ 3.5 h → about
15 h, inside the 24 h Slurm limit. `slurm/train.sh` resumes from the last epoch checkpoint if resubmitted.

Earlier CPU-based estimates (18.8 MiB saved activations and 5.06 GFLOP forward per pair; 142 GB predicted for
B = 72) were close on memory (123 GiB measured) but about 2× optimistic on evaluation time.

---

## 7. Open decisions

1. **Step 6 reduction** (`model.pool`): default `cls_residual` (`cls + γ·mean(D)`) starts at CLIP and adds
   deformable evidence, but can learn to ignore it (watch `train/gamma`). `mean` (`mean(D)`) is the purest test of
   the hypothesis, since the image vector comes entirely from text-guided sampling, but it starts weaker than CLIP.
   Token-level matching is not implemented.
2. **Text vector in step 7:** implemented with CLIP's `t_i`, which doesn't depend on the image. Using the enriched
   `T̃_ij` at the EOT position would need `TextGuidedOffsets.forward` (in `nbs/01_model.ipynb`) to also return `T̃`.
3. **Using the official T2I-VeRW split**, if obtainable, for an exact comparison with the paper.

---

## 8. Implementation status

| Piece | Where | Status |
|---|---|---|
| CLIP token encoders | `model/models.py` | done |
| Steps 2–4 (`CrossAttentionTransformer`, `SpatialTokens`, `OffsetHead`, `TextGuidedOffsets`) | `model/models.py` (generated from `nbs/01_model.ipynb`) | done |
| Step 5 deformable attention | `model/text_guided.py:DeformableAttention` | done |
| Steps 6–7 scoring, B×B pairwise forward | `model/text_guided.py:TextGuidedDeformable` | done |
| SDM on a precomputed score matrix | `loss/losses.py:sdm_loss_from_sim` | done |
| Pairwise full-gallery evaluation (chunked, cached CLIP tokens and value maps) | `train.py:evaluate_pairwise` | done |
| Training loop, W&B logging, checkpoint/resume, metrics | `train.py`, `utils/metrics.py` | done |
| Config | `configs/exp1_baseline.yaml` (`model.arch: text_guided_deformable`) | done |

Verified on CPU with the real CLIP weights:
- the B×B score matrix equals scoring each pair individually
- with γ = 0 the scores equal plain CLIP cosine
- gradients reach the offset layers, spatial tokens, text enricher, deformable layers, CLIP and the classifier
  (after the first-step gating described in §3)
- chunked pairwise evaluation gives the same metrics as a brute-force loop

GPU memory and evaluation time have not been measured on the GPU yet. The smoke test
(`slurm/smoke.sh`, which evaluates only the first 200 queries) will report them.

**Known issues in the existing model code:**
- ~~`CrossAttention.scale = embed_dim ** -0.5`~~ **fixed**: now `(embed_dim // num_heads) ** -0.5` (1/√64), in
  `nbs/01_model.ipynb` and regenerated into `model/models.py`. Before the fix, attention logits were √8 too small.
- `CrossAttentionBlock` calls `cross_attn(..., return_attention=True)`, but `CrossAttention.forward`'s argument is
  `return_attn`, so requesting attention maps raises a `TypeError`.
- `OffsetHead`'s own default is `num_points=64`. `TextGuidedOffsets` always passes 4, but building `OffsetHead()`
  directly would initialise points up to 64 patches away on a 14-patch grid, i.e. outside the image.
