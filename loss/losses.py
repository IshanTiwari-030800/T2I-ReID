"""Training objectives for T2I-ReID (plain module, not nbdev-generated)."""

import torch
import torch.nn.functional as F

__all__ = ['sdm_loss', 'sdm_loss_from_sim', 'id_loss']


def sdm_loss(image_feat, text_feat, pids, temperature=0.02, eps=1e-8):
    """Similarity Distribution Matching (IRRA, CVPR'23).

    Softmax over the in-batch cosine similarities (text→image and image→text) is pulled, via KL, towards a target
    that spreads the mass uniformly over every same-pid sample, so other images of the same vehicle are not negatives.
    image_feat, text_feat: [B, D] (unnormalised); pids: [B].
    """
    sim = F.normalize(text_feat.float(), dim=-1) @ F.normalize(image_feat.float(), dim=-1).t()
    return sdm_loss_from_sim(sim, pids, temperature, eps)


def sdm_loss_from_sim(sim, pids, temperature=0.02, eps=1e-8):
    """SDM on a precomputed [B, B] cosine-similarity matrix, sim[i, j] = s(caption i, image j). Rows give the
    text→image direction, columns the image→text one. Used when the score of a pair can't be factorised into two
    independent embeddings (e.g. the text-guided model, whose image vector depends on the caption)."""
    labels = (pids[:, None] == pids[None, :]).float()
    target = labels / labels.sum(dim=1, keepdim=True)                    # symmetric, so it serves both directions

    t2i = sim.float() / temperature
    kl = lambda logits: (logits.softmax(1) * (logits.log_softmax(1) - torch.log(target + eps))).sum(1).mean()
    return kl(t2i) + kl(t2i.t())


def id_loss(image_logits, text_logits, pids):
    """ID classification on both modalities through a shared classifier (train-time only), averaged as in IRRA."""
    return (F.cross_entropy(image_logits.float(), pids) + F.cross_entropy(text_logits.float(), pids)) / 2
