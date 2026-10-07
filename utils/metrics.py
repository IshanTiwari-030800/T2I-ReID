"""Text-to-image retrieval metrics (plain module, not nbdev-generated)."""

import torch

__all__ = ['t2i_metrics']


@torch.no_grad()
def t2i_metrics(sim, q_pids, g_pids, ranks=(1, 5, 10, 50)):
    """Rank-k, mAP and mINP (in %) for a [num_query, num_gallery] similarity matrix.

    Every gallery image with the query's pid is a correct match, including the image the caption was written for
    (CUHK-PEDES protocol: no same-camera / same-image filtering, unlike image ReID).
    """
    num_gallery = sim.shape[1]
    order = sim.argsort(dim=1, descending=True)
    matches = g_pids[order] == q_pids[:, None]                            # [Q, G] bool, in ranked order
    num_rel = matches.sum(dim=1)
    assert (num_rel > 0).all(), 'every query pid must appear in the gallery'

    hits = matches.cumsum(dim=1)
    out = {f'R{k}': (hits[:, k - 1] > 0).float().mean().item() * 100 for k in ranks if k <= num_gallery}

    precision = hits / torch.arange(1, num_gallery + 1, device=sim.device)
    ap = (precision * matches).sum(dim=1) / num_rel
    last_hit = num_gallery - matches.flip(dims=[1]).int().argmax(dim=1)   # 1-based rank of the last correct match
    out['mAP'] = ap.mean().item() * 100
    out['mINP'] = (num_rel / last_hit).mean().item() * 100
    return out
