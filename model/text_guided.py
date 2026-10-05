"""Experiment 1: text-guided deformable sampling (plain module, not nbdev-generated). Spec: experiments/exp1/README.md."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .models import encode_image_tokens, encode_text_tokens, TextGuidedOffsets

__all__ = ['DeformableAttention', 'TextGuidedDeformable']


class DeformableAttention(nn.Module):

    """Step 5: each of the Q patch queries reads its H×P sampling locations from the 14×14 patch-value map (one
    64-channel slice per head), weights the P samples, concatenates the heads and projects."""

    def __init__(self, dim=512, grid_size=14, num_heads=8):

        super().__init__()

        assert dim % num_heads == 0, f'num_heads must divide dim, got dim {dim}, num_heads {num_heads}'
        self.grid_size, self.h = grid_size, num_heads
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def project_values(self, patches):

        """[N, 196, D] -> value maps [N, H, D/H, 14, 14]. Depends only on the image, so compute it once per image."""

        return rearrange(self.value_proj(patches), 'n (y x) (h d) -> n h d y x', y=self.grid_size, h=self.h)

    def forward(self, values, locations, weights):

        """values [N, H, d, 14, 14]; locations [N, Q, H, P, 2], (x, y) in [0, 1]; weights [N, Q, H, P] -> [N, Q, D]."""

        n = values.shape[0]
        grid = rearrange(2 * locations - 1, 'n q h p xy -> (n h) q p xy')                   # grid_sample wants [-1, 1]
        # align_corners=False: location (i + 0.5) / 14 lands exactly on patch i's centre.
        sampled = F.grid_sample(rearrange(values, 'n h d y x -> (n h) d y x'), grid.to(values.dtype),
                                mode='bilinear', padding_mode='zeros', align_corners=False)   # [(N H), d, Q, P]
        out = (sampled * rearrange(weights, 'n q h p -> (n h) 1 q p').to(sampled.dtype)).sum(-1)
        return self.out_proj(rearrange(out, '(n h) d q -> n q (h d)', n=n))


class TextGuidedDeformable(nn.Module):

    """CLIP encoders + steps 2-7 of Experiment 1. The image vector of a pair depends on the caption, so every
    (caption, image) pair is scored separately: B^2 pairs per training batch, Q x G pairs at evaluation."""

    pairwise = True

    def __init__(self, clip_model, num_classes, embed_dim=512, grid_size=14, num_heads=8, num_points=4,
                 text_depth=4, spatial_depth=2, dropout=0.0, pool='cls_residual'):

        super().__init__()

        assert pool in ('cls_residual', 'mean'), f'unknown pool {pool!r}'
        self.clip = clip_model.float()                    # clip.load on CUDA returns fp16 weights; train in fp32 + autocast
        self.offsets = TextGuidedOffsets(embed_dim, grid_size, num_heads, num_points, text_depth, spatial_depth, dropout)
        self.deform = DeformableAttention(embed_dim, grid_size, num_heads)
        self.pool = pool
        # cls_residual: g = cls + gamma * mean(D) with gamma = 0 at init, so training starts exactly at CLIP's image vector.
        self.gamma = nn.Parameter(torch.zeros(1)) if pool == 'cls_residual' else None
        self.classifier = nn.Linear(embed_dim, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.zeros_(self.classifier.bias)

    def encode_image(self, image):

        """Step 1 (image): all CLIP visual tokens [B, 197, D]; token 0 is CLS, then the 14×14 patches row-major."""

        return encode_image_tokens(self.clip, image)

    def encode_text(self, caption):

        """Step 1 (text): tokens [B, 77, D], padding mask [B, 77] (True up to and including EOT), EOT vector [B, D]."""

        tokens = encode_text_tokens(self.clip, caption)
        eot = caption.argmax(dim=-1)
        mask = torch.arange(caption.shape[1], device=caption.device)[None] <= eot[:, None]
        return tokens, mask, tokens[torch.arange(len(caption), device=caption.device), eot]

    def score_pairs(self, text, image_tokens, ti, ij, values=None):

        """Steps 2-7 for the pairs (caption ti[k], image ij[k]).

        text: the output of encode_text; image_tokens: the output of encode_image; ti, ij: [N] indices into them.
        values: optional precomputed deform.project_values(image_tokens[:, 1:]) (it depends on the image only).
        Returns 'score' [N] (cosine), 'image_feat' [N, D] (the caption-conditioned image vector) and 'locations'.
        """

        tokens, mask, t = text
        V = image_tokens[ij]                                                                # [N, 197, D]
        locations, weights = self.offsets(tokens[ti], mask[ti], V)                         # steps 2-4
        if values is None:
            values = self.deform.project_values(image_tokens[:, 1:])
        D = self.deform(values[ij], locations, weights)                                    # step 5, [N, 196, D]
        g = D.mean(dim=1)                                                                   # step 6
        if self.pool == 'cls_residual':
            g = V[:, 0] + self.gamma * g
        score = F.cosine_similarity(t[ti].float(), g.float(), dim=-1)                       # step 7
        return {'score': score, 'image_feat': g, 'locations': locations}

    def forward(self, image, caption):

        """Training forward over a batch: scores all B×B (caption, image) pairs."""

        image_tokens, text = self.encode_image(image), self.encode_text(caption)
        B = len(image)
        idx = torch.arange(B, device=image.device)
        out = self.score_pairs(text, image_tokens, idx.repeat_interleave(B), idx.repeat(B))   # pair k = (k // B, k % B)
        matched = out['image_feat'][idx * B + idx]                                          # g_ii, [B, D]
        t = text[2]
        return {'sim': out['score'].view(B, B),                                             # [i, j] = s(caption i, image j)
                'image_feat': matched, 'text_feat': t,
                'image_logits': self.classifier(matched), 'text_logits': self.classifier(t),
                'locations': out['locations'].view(B, B, *out['locations'].shape[1:])}
