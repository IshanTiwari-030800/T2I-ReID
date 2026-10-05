"""Experiment 1 baseline model (plain module, not nbdev-generated)."""

import torch
import torch.nn as nn

from .models import encode_image_tokens, encode_text_tokens

__all__ = ['DualEncoder']


class DualEncoder(nn.Module):
    """CLIP image and text encoders compared through their global features (image CLS, text EOT), plus a shared
    ID classifier used only as a training signal. Image and text are encoded independently, so the gallery can be
    embedded once and every query ranked with a single matmul."""

    def __init__(self, clip_model, num_classes, embed_dim=512):
        super().__init__()
        self.clip = clip_model.float()                    # clip.load on CUDA returns fp16 weights; train in fp32 + autocast
        self.classifier = nn.Linear(embed_dim, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.zeros_(self.classifier.bias)

    def encode_image(self, image):
        return encode_image_tokens(self.clip, image)[:, 0]                                  # [B, D]

    def encode_text(self, caption):
        tokens = encode_text_tokens(self.clip, caption)
        return tokens[torch.arange(len(caption), device=caption.device), caption.argmax(dim=-1)]   # EOT, [B, D]

    def forward(self, image, caption):
        image_feat, text_feat = self.encode_image(image), self.encode_text(caption)
        return {'image_feat': image_feat, 'text_feat': text_feat,
                'image_logits': self.classifier(image_feat), 'text_logits': self.classifier(text_feat)}
