# T2I-ReID: can text tell attention where to look?

Text-to-image re-identification on **T2I-VeRW** (vehicles), built on CLIP ViT-B/16.

> nbdev note: `nbdev_readme` / `nbdev_prepare` regenerate this file from `nbs/index.ipynb`.
> Move this content into `index.ipynb` (or don't run those commands) or it will be overwritten.

---

## 0. What's in the repo today

| File | What it gives us |
|---|---|
| `model/models.py` | `make_model(cfg)` → CLIP ViT-B/16. `encode_image_tokens(model, img)` → `[B, 197, 512]` (token 0 = CLS, then 14×14 patches). `encode_text_tokens(model, txt)` → `[B, 77, 512]` (global feature = EOT token at `txt.argmax(-1)`) |
| `data/dataset.py` | `T2IVeRW()` → `train` (pid, img, caption) / `query` (pid, caption) / `gallery` (pid, img). `build_dataloaders` → train (B=64, shuffled), query and gallery loaders (B=256) |
| `loss/` | empty: losses to be written |
| `nbs/data_vis.ipynb` | shows the 24×24 vehicle mask and part masks (windows, lights, wheels, mirrors, roof, doors) attached to every record |

Dataset statistics (val split = test split):

```
train:   10264 captions | 1797 ids   (1 caption per image → 10264 images)
query:    4404 captions |  769 ids
gallery:  4404 images   |  769 ids   (≈5.7 images per id)
```

Train and test IDs are **disjoint**. That single fact explains most of how ReID training and evaluation work (§2).

---

## 1. The idea

CLIP-style T2I-ReID compares one global text vector with one global image vector. The image vector is computed
**without knowing what the text cares about**. A caption like *"…two silver inspection stickers in the upper right
corner of the windshield…"* is precise about *where* the discriminative evidence is.

**Hypothesis:** text (grounded by a quick look at the image) contains enough geometric information to predict
*sampling offsets* for a deformable attention over the image. The image representation then gathers evidence from
the places the caption talks about.

Flow, in one line:

> text ⟵ cross-attn ⟵ image  ⇒  enriched text  ⇒  spatial tokens **C** ⟵ cross-attn ⟵ enriched text
> ⇒  Linear  ⇒  per-patch sampling offsets  ⇒  deformable attention over the image  ⇒  text-conditioned image embedding  ⇒  match score

---

## 2. ReID training vs inference, explained

### 2.1 Why it's confusing

In classification, the classes at test time are the same classes seen in training. In ReID they are **not**:

- **Training** sees vehicle IDs `0..1796`.
- **Testing** sees 769 *different* vehicles that were never seen in training.

So a ReID model can't be "a classifier that outputs the vehicle ID". It must learn an **embedding space** where
same-identity text and images are close and different identities are far apart. Testing is a **retrieval
problem**: for each text query, rank every gallery image by similarity, and check whether the images of the
correct (never-seen) vehicle come out on top.

The ID-classification head we train (`Linear(512 → 1797)`) is only a training signal. **It is thrown away at
test time**, because its 1,797 classes don't exist in the test set.

### 2.2 Training loop (what one step does)

The unit of training is a **(pid, image, caption)** triple. One epoch = 10,264 triples = 160 batches of 64.

```
for epoch in 1..60:
  for batch in train_loader:                          # {'pid': [64], 'image': [64,3,224,224], 'caption': [64,77]}
      V = encode_image_tokens(clip, image)            # [64,197,512]
      T = encode_text_tokens(clip, caption)           # [64, 77,512]
      v = V[:,0];  t = T[range(64), caption.argmax(-1)]   # global vectors [64,512]

      S = cos(t, v)                                   # [64,64] text×image similarity matrix
      L_sdm = SDM(S, pid)                             # pull same-pid pairs together, push others apart
      L_id  = CE(W·v, pid) + CE(W·t, pid)             # ID classification, train-time only

      pairs = positives + hard negatives (mined from S)   # 3×64 (text, image) pairs
      L_itm = BCE(head(T_i, V_j), match_ij)           # the proposed text-guided module (§3.3)

      (L_sdm + L_id + L_itm).backward(); opt.step()
```

**Toy example of the label matrix.** Take a batch of 4 with pids `[7, 7, 12, 30]`. Samples 0 and 1 are
*different images* of the same vehicle 7. The "who should match whom" matrix is

```
            img0 img1 img2 img3
text0 (7) [  1    1    0    0 ]     → SDM target row = [0.5, 0.5, 0, 0]
text1 (7) [  1    1    0    0 ]
text2 (12)[  0    0    1    0 ]     → [0, 0, 1, 0]
text3 (30)[  0    0    0    1 ]
```

A plain CLIP contrastive loss would treat (text0, img1) as a *negative*, even though it is the same vehicle.
**SDM** (IRRA, CVPR'23) fixes this. It normalises each row of the identity matrix into a target distribution `q`,
softmaxes the similarity row `S_i/τ` into `p`, and minimises `KL(p ‖ q)` in both directions (text→image and
image→text).

**Hard negatives for ITM.** For text0 (pid 7), the valid negative images are img2 and img3 only. img1 is the same
vehicle, so it is masked out. We sample one with probability ∝ `softmax(S[0, {2,3}])`, which favours the
negative that *currently looks most similar*. Do the same for a negative text per image. Each step this gives
64 positive pairs, 64 (text, hard-neg image) pairs and 64 (hard-neg text, image) pairs, with labels 1/0/0.

### 2.3 Inference loop (what evaluation does)

The unit of evaluation is a **query caption**. Each of the 4,404 captions is used once as a query against **all**
4,404 gallery images. The correct answers are every gallery image with the same pid. That includes the image the
caption was written for, which, unlike image-ReID, is *not* removed (standard CUHK-PEDES protocol, followed
by this dataset).

**Stage 1: dual encoder (cheap, full gallery).**

```
gallery: encode all 4404 images once → cache V_g [4404,197,512] (~0.9 GB fp16), v_g [4404,512]
queries: encode all 4404 captions once → cache T_q [4404, 77,512] (~0.35 GB fp16), t_q [4404,512]
S_dual = norm(t_q) @ norm(v_g)ᵀ                       # [4404,4404], one matmul
rank each row → metrics (stage-1 numbers)
```

**Stage 2: re-rank with the text-guided module (§3.3).**

The proposed module's image feature depends on the text, so a score needs one forward pass **per (query, image)
pair**. All pairs would be 4404² ≈ 19.4M passes. Instead, re-score only each query's top-K:

```
for each query i:
    top = top-K gallery indices of S_dual[i]          # K = 50
    s_itm[i, top] = head(T_q[i], V_g[top])            # runs on cached tokens; batch ~1024 pairs at a time
    new order = top re-sorted by s_itm, then the remaining 4354 items in stage-1 order
```

That is 4,404 × 50 ≈ 220K head passes on cached CLIP tokens: a few minutes on one GPU.
**Ceiling:** stage 2 can only reorder what stage 1 put in the top-K, so also report stage-1 Recall@50.

### 2.4 Metrics, with a worked example

Gallery of 5 images with pids `g1:A g2:A g3:B g4:B g5:C`. Three queries:

| Query (pid) | similarity to g1..g5 | ranking | match vector |
|---|---|---|---|
| q1 (A) | .62 .30 .71 .10 .25 | g3 g1 g2 g5 g4 | 0 1 1 0 0 |
| q2 (B) | .20 .10 .80 .50 .40 | g3 g4 g5 g1 g2 | 1 1 0 0 0 |
| q3 (C) | .30 .60 .20 .10 .50 | g2 g5 g1 g3 g4 | 0 1 0 0 0 |

- **Rank-k**: is there at least one correct match in the top-k? R1: q1 ✗, q2 ✓, q3 ✗ → **33.3%**. R5: all ✓ → **100%**.
- **AP** (average precision): the mean of the precision at each correct hit.
  q1: hits at ranks 2 and 3 → (1/2 + 2/3)/2 = **0.583**; q2: (1/1 + 2/2)/2 = **1.0**; q3: 1/2 = **0.5**.
  **mAP** = (0.583 + 1.0 + 0.5)/3 = **0.694**.
- **INP** = #correct / rank of the *last* correct hit, i.e. how deep you must dig to find *all* of them.
  q1: 2/3, q2: 2/2, q3: 1/2 → **mINP = 0.722**.

**Re-ranking example.** Suppose q1's stage-1 top-3 is `[g3, g1, g2]` and the head scores them `[0.2, 0.9, 0.7]`.
The new order is `g1 g2 g3 | g5 g4` and the match vector becomes `1 1 0 0 0`. R1 flips to ✓ and AP rises from 0.583 to 1.0.

We report **R1, R5, R10, mAP, mINP** for stage 1 and stage 2.

---

## 3. Architecture

Notation: B = batch, L = 77 text tokens, N = 196 patch tokens (14×14 grid, 16-px patches on 224×224),
d = 512 (CLIP joint-embedding width, the output of `encode_*_tokens`), H = 8 heads, P = 4 sampling points per head.

### 3.1 Backbone (shared)

```
V = encode_image_tokens(clip, image)   # [B,197,512];  v_cls = V[:,0];  V_p = V[:,1:]  → [B,196,512] = 14×14 map
T = encode_text_tokens(clip, caption)  # [B, 77,512];  t_eot = T[b, eot_b];  pad mask m = (pos ≤ eot)
```

The whole backbone is fine-tuned.

### 3.2 Baseline: CLIP dual encoder (stage 1)

`s_dual(text, image) = cos(t_eot, v_cls)`, trained with **SDM + ID loss**. This is the IRRA recipe without its
MLM head. It is (a) the baseline everything is compared against and (b) the stage-1 retriever.

### 3.3 Proposed: text-guided deformable head (stage 2)

This head operates on one (text i, image j) pair: `T_i [77,512]` and `V_p,j [196,512]`.

```
                          ┌───────────────────────────────────────────────────┐
 V_p (B,196,512) ────────►│(1) T̃ = T + XAttn(q=T, kv=V_p); T̃ += FFN(T̃)          │ text enriched by image
 T   (B, 77,512) ────────►│                                                   │
                          │(2) C̃ = C + XAttn(q=C, kv=T̃, key_mask=m); += FFN    │ C: 196 learned spatial tokens
                          │                                                   │
                          │(3) Δ = Linear_off(C̃)  → (B,196,H,P,2)  offsets     │
                          │    A = softmax_P(Linear_att(C̃)) → (B,196,H,P)     │
                          │                                                   │
                          │(4) loc = p_ref + Δ/14            (normalised xy)   │ deformable attention
                          │    Out_q = Σ_h,p A·grid_sample(Value(V_p), loc)   │ over the 14×14 map
                          │                                                   │
                          │(5) Ṽ_p = V_p + γ·Proj(Out);  Ṽ_p += FFN(Ṽ_p)       │ text-conditioned patches
                          │                                                   │
                          │(6) z = mean_q Ṽ_p                                  │
                          │    s_itm = MLP([z ; t_eot ; z⊙t_eot]) → scalar     │ match logit
                          └───────────────────────────────────────────────────┘
```

Step by step:

1. **Text enrichment.** Text tokens (queries) cross-attend to the 196 patch tokens (keys/values): one pre-LN
   cross-attention block (8 heads) + FFN, residual. Output `T̃ [B,77,512]`. Each word now knows what the image
   it's paired with looks like ("windshield" can now carry *where* the windshield is in this image).
2. **Spatial tokens C.** `C ∈ ℝ^{196×512}` is a learned parameter, one token per grid cell, initialised as a 2-D
   sin-cos positional embedding plus small noise, so token q "knows" it lives at cell q. C (queries) cross-attends to
   `T̃` (keys/values, padding masked by m), then FFN, residual. Output `C̃ [B,196,512]`: for each image location,
   "what the text says about looking from here".
3. **Offsets & weights.** `Linear_off: 512 → H·P·2 = 64`, reshaped to **`(B, 196, H, P, 2)`**, one (Δx, Δy) per
   sampling point per head per query. (Your sketch `(B, num_queries, 2·num_samples, d)` is read this way: the last
   axis of an *offset* is 2 (x, y). A d-sized vector per sample is what gets *sampled*, not the offset.)
   `Linear_att: 512 → H·P`, softmax over P, gives attention weights `A (B,196,H,P)`. Both are driven by C̃, so
   *where* and *how much* come from the text path. The content being gathered comes from the image.
4. **Deformable sampling.** The reference point of query q is the centre of its patch,
   `p_ref = ((col+0.5)/14, (row+0.5)/14)`. Offsets are in patch units: `loc = p_ref + Δ/14`.
   `Value = Linear(V_p)`, reshaped to `(B·H, 64, 14, 14)`. Use `F.grid_sample(value, 2·loc−1, bilinear,
   padding='zeros', align_corners=False)` to sample `(B·H, 64, 196, P)`, weight by A, sum over P, concatenate heads, then `Proj`.
   This is pure PyTorch, single scale, and needs no custom CUDA op.
5. **Update.** Residual into the patch tokens with a LayerScale `γ` initialised to 0, then FFN.
6. **Score.** Mean-pool the 196 updated tokens into `z` and fuse with the text global feature in a 2-layer MLP. The
   output is a scalar match logit `s_itm`.

**Initialisation (important, since we start from a good CLIP):**
- `Linear_off.weight = 0`; `bias` uses the Deformable-DETR ring init: head h points in direction `2πh/H`, point p at
  radius `p+1` patches. Every query starts by looking at a fixed star pattern, and training learns to bend it.
- `Linear_att.weight = 0, bias = 0` gives uniform weights at the start.
- `γ = 0`, so the head is an identity on `V_p` at step 0 and can't damage CLIP features early.

**Size/cost:** about 6M new parameters. Per pair: 77×196 + 196×77 attention plus 196·8·4 bilinear samples, which is cheap
compared with the backbone. The backbone runs once per image/caption; only the head runs per pair.

### 3.4 Why the head is a cross-encoder (and what that implies)

Step (1) mixes image *j* into text *i*, and steps (3–5) build image *j*'s feature from text *i*. So the
score is a joint function `s(i, j)`; neither side has a standalone embedding. Consequences:
- **Training:** a contrastive loss over all B² pairs would need B² head passes. Instead, we train the head with **ITM
  (binary match) on hard negatives**, the ALBEF/BLIP recipe also used by RaSa and APTM for T2I-ReID.
- **Inference:** we can't precompute gallery embeddings for the head, hence **retrieve-then-re-rank** (§2.3).

---

## 4. Training details

| Item | Setting |
|---|---|
| Data | `T2IVeRW()`; `build_dataloaders(dataset, tokenizer, batch_size=64)` |
| Tokenizer | `clip.tokenize(c, truncate=True)[0]` (captions up to ~130 words get cut at 77 tokens) |
| Images | `build_transform()`: 224×224 bicubic resize, CLIP mean/std. **No horizontal flip**, because captions name sides ("upper left of the windshield"), and flipping would make the geometry the text describes wrong |
| Precision | `clip.load` on CUDA returns **fp16 weights**. Call `model.float()` before training and use `torch.autocast(bfloat16)` |
| Losses | `L = L_sdm + L_id + λ·L_itm`, λ = 1 |
| SDM | on `cos(t_eot, v_cls)`, τ = 0.02, both directions, identity-aware targets (§2.2) |
| ID | one shared `Linear(512 → 1797)` on `v_cls` and `t_eot`, CE. Discarded at test |
| ITM | 64 pos + 64 (text, hard-neg image) + 64 (hard-neg text, image) per step; negatives sampled ∝ softmax of the *detached* `S_dual` row, same-pid masked; BCE on `s_itm` |
| Optimiser | AdamW, β = (0.9, 0.999), weight decay 4e-5 (none on bias/LN/C/γ) |
| LR | backbone 1e-5, new modules (head, ID classifier) 1e-4 |
| Schedule | 60 epochs × 160 iters = 9,600 iters; linear warmup 5 epochs (from 0.1×), cosine to 0 |
| Gradients | ITM gradients flow into the backbone (joint training); grad-clip 1.0 |
| Eval | every 5 epochs: stage 1 (full gallery) + stage 2 (top-K = 50) |
| Model selection | T2I-VeRW has no separate val split (val = test), so **report the final epoch** and don't pick the best epoch on test. If tuning is needed later, hold out ~10% of train IDs |
| Seeds | 3 seeds for the final comparison (mean ± std); 1 seed while debugging |
| Hardware | one GPU; B=64 with a ViT-B/16 backbone and 192 head pairs fits comfortably in 24 GB with bf16 |

---

## 5. Experiments (first round)

The minimum set to answer the question. Every run uses identical data, schedule and seeds.

| ID | Model | What it tells us |
|---|---|---|
| E0 | Zero-shot CLIP (no training), stage 1 only | sanity floor |
| E1 | **Baseline**: dual encoder, SDM + ID | the number to beat |
| E2 | E1 + head with **static offsets**: skip step (2), `C̃ = C`, so offsets are learned but the same for every pair | the gain from re-ranking and learned deformable sampling *without* text geometry |
| E3 | **Proposed**: full head (§3.3) | text-conditioned geometry |

**The hypothesis is supported if E3 > E2 on stage-2 R1/mAP** (E2 > E1 alone would only show that re-ranking helps).

Follow-up ablations, once E1–E3 work:
- **Zero offsets** (`Δ ≡ 0`): each query looks only at its own cell, so there is no spatial gathering.
- **Text-only geometry**: skip step (1), so C̃ attends to raw `T`. This tests whether the *text alone* carries the geometry,
  or whether it's leaking in from the image through the enrichment.
- **Image-only geometry**: C̃ attends to `V_p` instead of `T̃`, so there is no text at all.
- P ∈ {4, 8}, K ∈ {20, 50, 100}, combined score `s_dual + β·σ(s_itm)` vs `s_itm` alone.

---

## 6. Diagnostics: does text actually move the sampling points?

Metrics alone can't answer the hypothesis. These checks can:

1. **Offset visualisation.** For a gallery image, overlay the 196×H×P sampling locations (sized by A) for 2–3
   *different* captions. If text drives geometry, the patterns should differ.
2. **Text sensitivity.** For a fixed image, measure the mean ‖Δ(caption a) − Δ(caption b)‖ across captions. In E2 this is 0 by
   construction; in E3 it should be clearly > 0, and not collapse to 0 during training.
3. **Part-mask grounding (uses data we already have).** Every T2I-VeRW record has 24×24 part masks (windows,
   lights, wheels, mirrors, roof, doors). For captions mentioning a part word ("sunroof" → roof, "sticker"/"windshield" →
   windows, "wheels", "headlights"…), compute the share of A-weighted sample mass that lands inside that part's mask.
   Compare captions that mention the part with those that don't, and E3 with E2. A positive gap is direct evidence that the
   text is steering attention toward the regions it describes.
4. **Vehicle-mask coverage.** The fraction of sample mass that lands on the vehicle mask vs the background.

---

## 7. Planned code layout (not implemented yet)

| Notebook → module | Contents |
|---|---|
| `nbs/03_loss.ipynb` → `loss/losses.py` | `sdm_loss`, `id_loss`, `itm_loss`, `mine_hard_negatives` |
| `nbs/04_head.ipynb` → `model/text_guided.py` | `TextGuidedDeformableHead` (steps 1–6), `deformable_sample` (grid_sample core), ring init |
| `nbs/05_train.ipynb` → `train.py` | config (OmegaConf), optimiser param groups, train loop, checkpointing |
| `nbs/06_eval.ipynb` → `eval.py` | token caching, stage-1 ranking, stage-2 top-K re-rank, R1/R5/R10/mAP/mINP |

Reuse: `make_model`, `encode_image_tokens`, `encode_text_tokens` (`model/models.py`); `T2IVeRW`,
`build_dataloaders`, `build_transform` (`data/dataset.py`). `pyproject.toml` still has to list its dependencies
(torch, torchvision, clip, omegaconf).

---

## 8. References

- IRRA: Jiang & Ye, *Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval*, CVPR 2023 (SDM loss, CLIP baseline).
- RaSa: Bai et al., IJCAI 2023, and APTM: Yang et al., ACM MM 2023 (ITM with hard negatives + top-K re-ranking for T2I-ReID).
- ALBEF: Li et al., NeurIPS 2021 (ITM hard-negative mining).
- Deformable DETR: Zhu et al., ICLR 2021 (deformable attention, offset initialisation).
