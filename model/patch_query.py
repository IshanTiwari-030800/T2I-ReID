"""Experiment 2: text-conditioned patch queries (plain module, not nbdev-generated). Spec: experiments/exp2/README.md."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import CrossAttentionTransformer, OffsetHead
from .pos_embeddings import sincos_2d
from .text_guided import TextGuidedDeformable

__all__ = ['PatchQueryOffsets', 'PatchQueryDeformable']


class PatchQueryOffsets(nn.Module):

    """Steps 2-4 of Experiment 2. Each of the 196 CLIP patch tokens (plus a fixed 2-D position) cross-attends to the
    caption tokens; the text-conditioned patch then predicts its own H×P sampling offsets and weights.

    text_conditioning=False (control): the caption is replaced by one learned token, so the network is identical but
    the offsets depend on the image only.
    """

    def __init__(self, dim=512, grid_size=14, num_heads=8, num_points=16, depth=4, dropout=0.0,
                 text_conditioning=True, init_radius=4.0, pos_scale=0.25):

        super().__init__()

        self.grid_size, self.text_conditioning = grid_size, text_conditioning
        # CLIP patch tokens have norm ~7.4 and sincos_2d rows norm 16, so the position starts at norm 4 (learnable).
        self.register_buffer('pos', sincos_2d(grid_size, dim), persistent=False)
        self.pos_scale = nn.Parameter(torch.tensor(float(pos_scale)))
        self.null_context = None if text_conditioning else nn.Parameter(0.02 * torch.randn(1, 1, dim))
        self.enricher = CrossAttentionTransformer(query_dim=dim, context_dim=dim, embed_dim=dim, dropout=dropout,
                                                  depth=depth, num_heads=num_heads)
        self.norm = nn.LayerNorm(dim)
        self.offset_head = OffsetHead(dim, num_heads, num_points)
        self._init_offsets(init_radius)

        ys, xs = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
        ref = torch.stack([(xs + 0.5) / grid_size, (ys + 0.5) / grid_size], dim=-1).reshape(-1, 2)
        self.register_buffer('reference_points', ref, persistent=False)                # [196, 2], (x, y) in [0, 1]

    def _init_offsets(self, init_radius):

        """OffsetHead puts point p of head h at p+1 patches along direction h, which leaves most of P=16 points
        off the 14×14 map (zero padding: no value, no gradient). Space the P points evenly out to init_radius."""

        h, p = self.offset_head.h, self.offset_head.p
        theta = torch.arange(h) * (2 * math.pi / h)
        ring = torch.stack([theta.cos(), theta.sin()], dim=-1)
        ring = ring / ring.abs().max(dim=-1, keepdim=True).values                       # onto the unit square
        radii = torch.arange(1, p + 1) * (init_radius / p)
        with torch.no_grad():
            self.offset_head.offsets.bias.copy_((ring[:, None] * radii[None, :, None]).flatten())   # 'h p xy'

    def forward(self, text_tokens, text_mask, image_tokens):

        """text_tokens [N, 77, D], text_mask [N, 77] (ignored by the control), image_tokens [N, 197, D]
        -> locations [N, 196, H, P, 2] in [0, 1], weights [N, 196, H, P] (softmax over P)."""

        x = image_tokens[:, 1:] + self.pos_scale * self.pos.to(image_tokens.dtype)         # [N, 196, D], drop CLS
        if self.text_conditioning:
            context, mask = text_tokens, text_mask
        else:
            context, mask = self.null_context.expand(len(x), -1, -1).to(x.dtype), None
        x = self.enricher(x, context, context_mask=mask)                                    # text-conditioned patches
        offsets, weights = self.offset_head(self.norm(x))
        locations = self.reference_points[None, :, None, None] + offsets / self.grid_size
        return locations, weights


class PatchQueryDeformable(TextGuidedDeformable):

    """Experiment 2: Experiment 1 with steps 2-4 replaced by PatchQueryOffsets. Deformable sampling of the original
    CLIP patch values, pooling and scoring are unchanged.

    With text conditioning the image vector depends on the caption (cross-encoder, pairwise scoring like
    Experiment 1). Without it (control) the image vector depends on the image only, so the model is a dual encoder:
    B image vectors per batch and one embedding per gallery image at evaluation. The scores are the same function
    either way; this only skips recomputing a caption-independent vector for every caption.
    """

    def __init__(self, clip_model, num_classes, embed_dim=512, grid_size=14, num_heads=8, num_points=16, depth=4,
                 dropout=0.0, pool='cls_residual', text_conditioning=True, init_radius=4.0, deform_impl='dense'):

        offsets = PatchQueryOffsets(embed_dim, grid_size, num_heads, num_points, depth, dropout,
                                    text_conditioning, init_radius)
        super().__init__(clip_model, num_classes, embed_dim, grid_size, num_heads, num_points, dropout=dropout,
                         pool=pool, offsets=offsets, deform_impl=deform_impl)
        self.pairwise = text_conditioning

    def image_only_feat(self, image_tokens):

        """Control only: the caption-independent image vector g [N, D] and its sampling locations."""

        assert not self.pairwise
        locations, weights = self.offsets(None, None, image_tokens)
        D = self.deform(self.deform.project_values(image_tokens[:, 1:]), locations, weights)
        g = D.mean(dim=1)
        if self.pool == 'cls_residual':
            g = image_tokens[:, 0] + self.gamma * g
        return g, locations

    def embed_image(self, image):

        """Control only (dual evaluation): (g [N, D], CLIP CLS [N, D])."""

        image_tokens = self.encode_image(image)
        return self.image_only_feat(image_tokens)[0], image_tokens[:, 0]

    def embed_text(self, caption):

        return self.encode_text(caption)[2]

    def forward(self, image, caption):

        if self.pairwise:
            return super().forward(image, caption)

        image_tokens, t = self.encode_image(image), self.encode_text(caption)[2]
        g, locations = self.image_only_feat(image_tokens)
        sim = F.normalize(t.float(), dim=-1) @ F.normalize(g.float(), dim=-1).t()            # [i, j] = s(caption i, image j)
        return {'sim': sim, 'image_feat': g, 'text_feat': t,
                'image_logits': self.classifier(g), 'text_logits': self.classifier(t),
                'locations': locations[None]}                                              # [1, B, ...]: one per image
