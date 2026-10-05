# Experiment 2: text-conditioned patch queries

Branch `exp2-image-enrich` (this experiment) and `exp2-control-no-text` (its control). Everything not listed here
is identical to Experiment 1 (`experiments/exp1/README.md`): data, split, protocol, metrics, losses, optimiser.

## 1. Question

Experiment 1 predicted each patch query's sampling offsets from a learned spatial token C_q that read the
(image-enriched) caption. Experiment 2 asks the image patch itself: **patch q cross-attends to the caption, and the
text-conditioned patch predicts where q should sample.** The deformable sampling still reads the original CLIP patch
features, so the caption can only act through *where* the model looks.

The control is the same network with the caption replaced by one learned token. Its offsets depend on the image
only. **Experiment 2 minus the control is the effect of text on the sampling geometry.**

## 2. Model (steps that differ from Experiment 1)

Notation as in Experiment 1: d = 512, 196 = 14×14 patches, L = 77 text tokens, H = 8 heads, **P = 16 points per head**.

| Step | Experiment 1 | Experiment 2 |
|---|---|---|
| 1. CLIP | image tokens `[B,197,d]`, text tokens `[B,77,d]` | same |
| 2. Enrich | text ← image patches (4 blocks) → `[B,77,d]` | **patches + fixed 2-D sin-cos position ← caption tokens** (4 blocks, padding masked) → `[B,196,d]` |
| 3. Spatial tokens | 196 learned tokens C ← enriched text (2 blocks) | removed; each enriched patch is its own query |
| 4. Offsets | OffsetHead(C̃) → `[B,196,8,4,2]` + weights | OffsetHead(LN(enriched patch)) → `[B,196,8,16,2]` + weights `[B,196,8,16]` (softmax over P) |
| 5. Sample | bilinear sampling of `Linear(CLIP patches)` | same, 128 points per query instead of 32 |
| 6–7. Pool, score | `g = CLS + γ·mean_q(D)`, cosine with the caption's EOT | same |

- **Blocks:** each block is pre-LN cross-attention (8 heads), then an FFN (4× width), with residuals: the existing
  `CrossAttentionTransformer`.
- **Position:** `sincos_2d(14, 512)` is added to the patch tokens before step 2, scaled by a learnable scalar
  initialised to 0.25. That gives norm ~4, against ~7.4 for CLIP patch tokens.
- **Point initialisation:** the P points of head h lie along direction 2πh/H at radii 0.25, 0.5, …, 4 patches.
  Experiment 1's init (radius p+1) would put P=16 points up to 16 patches out, mostly off the 14×14 map, where zero
  padding gives no value and no gradient. Offset/weight layer weights start at zero, and γ = 0. So step 0 is plain
  CLIP, as in Experiment 1.
- **New parameters:** 13.3M (Experiment 1: see its README).

**Control (`text_conditioning: false`).**
- **Context:** the cross-attention reads a single learned token (init N(0, 0.02²)) instead of the caption. With one
  key, attention is constant, so each block reduces to a learned bias plus FFN, and the offsets are a function of
  the image alone.
- **Scoring:** g doesn't depend on the caption, so the control is a dual encoder. Training computes B image
  vectors and evaluation embeds each gallery image once. That is the same function as scoring every pair
  (checked against the pairwise path), only cheaper.

## 3. Training and evaluation

| | Experiment 2 | Control | Experiment 1 |
|---|---|---|---|
| Batch | 80, pairs scored in checkpointed chunks of 1,600 (60 GiB, 2.1 s/step) | 80 (12 GiB, 0.23 s/step) | 80 |
| Epochs | 100, warmup 5 then cosine | same | 200, warmup 10 |
| LR | 1e-5 CLIP, 1e-4 new modules | same | same |
| Eval | every 10 epochs, full pairwise (Q×G) | every 10 epochs, dual | every 20 epochs |

Time: Experiment 2 ~4.4 min/epoch → 7.4 h of training + 10 evaluations × ~25 min ≈ 11.5 h. Control: ~0.5 min/epoch,
seconds per evaluation, under 1 h in total.

Two implementation changes keep this affordable. Both give the same results as the straightforward version (checked
on CPU):
- **Dense bilinear sampling** (`model.deform_impl: dense`). On a 14×14 map, sampling is a matmul with a 196×196
  matrix built from separable tent weights. `grid_sample`'s backward scatters 128 points × 196 queries per pair with
  atomic adds: 3.5× slower here (B=32: 1.28 → 0.36 s/step).
- **Activation checkpointing over pair chunks** (`model.checkpoint_pairs: 1600`). Steps 2–7 hold ~33 MB of
  activations per pair, about 210 GB for B=80's 6,400 pairs. Backward recomputes each 1,600-pair chunk instead.

Both runs report `dual_*`: the same fine-tuned backbone ranked by CLS vs EOT, with the deformable branch off.

## 4. What to compare

1. **Experiment 2 vs control**, both at their best and their final epoch: R1/R5/R10/mAP/mINP. The text effect on
   sampling geometry.
2. **`train/offset_text_std`** (Experiment 2 only; the control is 0 by construction): how far, in patches, a
   sampling point moves when only the caption changes. Experiment 1 kept this near 0 (≤ 0.04) for its first 60
   epochs and 0.11 at its best epoch.
3. **Caption-swap evaluation** (planned, as run on Experiment 1): rank with offsets computed from a different
   caption. A drop vs normal evaluation means the caption-specific geometry is used.
