"""Train and evaluate text-to-image ReID models on T2I-VeRW, logging to Weights & Biases.

    python train.py --config configs/exp1_baseline.yaml [key=value ...]

Overrides use OmegaConf dot-list syntax, e.g. `optim.epochs=1 log.max_iters_per_epoch=20 wandb.mode=offline`.
Training resumes from <output_dir>/checkpoint_latest.pth when `resume=true` and that file exists.
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from t2i_reid.data import T2IVeRW, ImageTextDataset, ImageDataset, TextDataset, build_transform
from t2i_reid.data.dataset import CLIP_MEAN, CLIP_STD
from t2i_reid.loss import sdm_loss_from_sim, id_loss
from t2i_reid.model import make_model, DualEncoder, TextGuidedDeformable, PatchQueryDeformable
from t2i_reid.utils import t2i_metrics

CKPT_NAME = 'checkpoint_latest.pth'
logger = logging.getLogger('t2i_reid')


# ----------------------------------------------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------------------------------------------

def load_config():
    parser = argparse.ArgumentParser(description='T2I-ReID training')
    parser.add_argument('--config', required=True, help='path to a YAML config')
    parser.add_argument('overrides', nargs='*', help='key=value overrides (OmegaConf dot-list)')
    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    OmegaConf.set_readonly(cfg, True)
    return cfg, args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir, resume):
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(os.path.join(output_dir, 'train_log.txt'), mode='a' if resume else 'w')):
        handler.setFormatter(fmt)
        logger.addHandler(handler)


def build_train_transform(cfg):

    size, aug = tuple(cfg.data.image_size), cfg.data.aug
    tfs = [T.Resize(size, interpolation=T.InterpolationMode.BICUBIC)]
    if aug.hflip:
        tfs.append(T.RandomHorizontalFlip())
    if aug.pad_crop:
        tfs += [T.Pad(aug.pad_crop), T.RandomCrop(size)]
    tfs += [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
    if aug.random_erasing:
        tfs.append(T.RandomErasing(p=aug.random_erasing, scale=(0.02, 0.4), value=0))  # 0 == the mean after Normalize
    return T.Compose(tfs)


def build_loaders(cfg, dataset):

    """Train loader with augmentation; query and gallery loaders with the plain resize transform."""

    tokenizer = lambda caption: clip.tokenize(caption, truncate=True)[0]
    test_transform = build_transform(tuple(cfg.data.image_size))
    common = dict(num_workers=cfg.data.num_workers, pin_memory=True, persistent_workers=cfg.data.num_workers > 0)
    train = DataLoader(ImageTextDataset(dataset.train, build_train_transform(cfg), tokenizer),
                       batch_size=cfg.data.batch_size, shuffle=True, drop_last=True, **common)
    query = DataLoader(TextDataset(dataset.query, tokenizer), batch_size=cfg.data.test_batch_size, **common)
    gallery = DataLoader(ImageDataset(dataset.gallery, test_transform), batch_size=cfg.data.test_batch_size, **common)
    return train, query, gallery


def build_model(cfg, num_classes):

    """cfg.model.arch: 'text_guided_deformable' (Experiment 1), 'patch_query_deformable' (Experiment 2 and its
    control, model.text_conditioning) or 'dual_encoder' (plain CLIP, CLS vs EOT)."""

    clip_model, _ = make_model(cfg)
    m = cfg.model
    if m.arch == 'dual_encoder':
        return DualEncoder(clip_model, num_classes, m.embed_dim)
    if m.arch == 'text_guided_deformable':
        return TextGuidedDeformable(clip_model, num_classes, embed_dim=m.embed_dim, grid_size=m.grid_size,
                                    num_heads=m.num_heads, num_points=m.num_points, text_depth=m.text_depth,
                                    spatial_depth=m.spatial_depth, dropout=m.dropout, pool=m.pool)
    if m.arch == 'patch_query_deformable':
        model = PatchQueryDeformable(clip_model, num_classes, embed_dim=m.embed_dim, grid_size=m.grid_size,
                                     num_heads=m.num_heads, num_points=m.num_points, depth=m.depth, dropout=m.dropout,
                                     pool=m.pool, text_conditioning=m.text_conditioning, init_radius=m.init_radius,
                                     deform_impl=m.get('deform_impl', 'dense'))
        model.checkpoint_pairs = m.get('checkpoint_pairs', 0) or 0
        return model
    raise ValueError(f'unknown model.arch {m.arch!r}')


def build_optimizer(cfg, model):

    """AdamW with two LR groups (CLIP backbone vs new modules); no weight decay on 1-D params (biases, norms, ...)
    or on learned tokens (Experiment 1's spatial tokens, the control's null caption token)."""

    groups = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr_key = 'backbone' if name.startswith('clip.') else 'head'
        decay_key = 'no_decay' if param.ndim <= 1 or 'spatial_tokens' in name or 'null_context' in name else 'decay'
        groups[(lr_key, decay_key)].append(param)
    lrs = {'backbone': cfg.optim.lr, 'head': cfg.optim.head_lr}
    param_groups = [{'params': params, 'name': f'{lr_key}/{decay_key}', 'lr': lrs[lr_key],
                     'weight_decay': cfg.optim.weight_decay if decay_key == 'decay' else 0.0}
                    for (lr_key, decay_key), params in sorted(groups.items())]
    return torch.optim.AdamW(param_groups, betas=tuple(cfg.optim.betas))


def build_scheduler(cfg, optimizer, total_steps, steps_per_epoch):

    """Per-step LR multiplier: linear warmup from warmup_factor to 1, then cosine decay to 0."""

    warmup_steps = cfg.optim.warmup_epochs * steps_per_epoch
    factor = cfg.optim.warmup_factor

    def lr_lambda(step):
        if step < warmup_steps:
            return factor + (1 - factor) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path, **state):

    """Written atomically, so a crash mid-write can't corrupt the last good checkpoint."""

    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def wandb_log(data, step):

    """wandb.log at an explicit step, skipping steps a resumed run already holds (they were logged before the
    interruption, and the resumed job replays them from the last epoch checkpoint)."""

    if wandb.run is not None and step >= wandb.run.starting_step:
        wandb.log(data, step=step)


# ----------------------------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------------------------

def evaluate(model, query_loader, gallery_loader, device, autocast, cfg):

    if getattr(model, 'pairwise', False):
        return evaluate_pairwise(model, query_loader, gallery_loader, device, autocast,
                                 cfg.log.eval_pairs_per_chunk, cfg.log.eval_max_queries)
    return evaluate_dual(model, query_loader, gallery_loader, device, autocast)


@torch.no_grad()
def evaluate_pairwise(model, query_loader, gallery_loader, device, autocast, pairs_per_chunk, max_queries=None):

    """Score every (query caption, gallery image) pair with the full text-guided model and rank each query's row.

    CLIP runs once per caption and per image; steps 2-7 run on Q x G pairs (19.4M for T2I-VeRW val) in chunks of
    pairs_per_chunk. Also reports the plain CLIP ranking (text EOT vs image CLS) of the same backbone as dual_*.
    max_queries (smoke tests only) keeps the first max_queries queries; metrics are then not comparable.
    """

    model.eval()
    tokens, masks, t_eot, q_pids, image_tokens, g_pids = [], [], [], [], [], []

    with autocast():
        for batch in query_loader:
            tok, mask, t = model.encode_text(batch['caption'].to(device, non_blocking=True))
            tokens.append(tok), masks.append(mask), t_eot.append(t), q_pids.append(batch['pid'])
        for batch in gallery_loader:
            image_tokens.append(model.encode_image(batch['image'].to(device, non_blocking=True)))
            g_pids.append(batch['pid'])

    text = (torch.cat(tokens), torch.cat(masks), torch.cat(t_eot))
    q_pids = torch.cat(q_pids).to(device)
    if max_queries:
        text, q_pids = tuple(x[:max_queries] for x in text), q_pids[:max_queries]
    image_tokens, g_pids = torch.cat(image_tokens), torch.cat(g_pids).to(device)
    num_q, num_g = len(q_pids), len(g_pids)

    sim = torch.empty(num_q * num_g, device=device)
    start_time = time.time()
    with autocast():
        values = model.deform.project_values(image_tokens[:, 1:])                         # once per gallery image
        for start in range(0, num_q * num_g, pairs_per_chunk):
            k = torch.arange(start, min(start + pairs_per_chunk, num_q * num_g), device=device)
            sim[start:start + len(k)] = model.score_pairs(text, image_tokens, k // num_g, k % num_g, values)['score'].float()
    logger.info(f'Pairwise eval: {num_q} x {num_g} = {num_q * num_g} pairs in {time.time() - start_time:.0f}s')

    metrics = t2i_metrics(sim.view(num_q, num_g), q_pids, g_pids)
    dual = F.normalize(text[2].float(), dim=-1) @ F.normalize(image_tokens[:, 0].float(), dim=-1).t()
    metrics.update({f'dual_{k}': v for k, v in t2i_metrics(dual, q_pids, g_pids).items()})
    model.train()

    return metrics


@torch.no_grad()
def evaluate_dual(model, query_loader, gallery_loader, device, autocast):

    """Embed every query caption and gallery image once, rank the gallery for each query by cosine similarity.

    Models with a deformable branch but no caption dependence (Experiment 2 control) provide embed_image /
    embed_text; their embed_image also returns the CLIP CLS, reported as dual_* (the backbone alone, CLS vs EOT).
    """

    model.eval()
    embed_image = getattr(model, 'embed_image', model.encode_image)
    embed_text = getattr(model, 'embed_text', model.encode_text)
    q_feats, q_pids, g_feats, g_cls, g_pids = [], [], [], [], []

    with autocast():
        for batch in query_loader:
            q_feats.append(embed_text(batch['caption'].to(device, non_blocking=True)).float())
            q_pids.append(batch['pid'])
        for batch in gallery_loader:
            feats = embed_image(batch['image'].to(device, non_blocking=True))
            feats = feats if isinstance(feats, tuple) else (feats,)
            g_feats.append(feats[0].float())
            g_cls.extend(feats[1:])
            g_pids.append(batch['pid'])

    q_pids, g_pids = torch.cat(q_pids).to(device), torch.cat(g_pids).to(device)
    q = F.normalize(torch.cat(q_feats), dim=-1)
    metrics = t2i_metrics(q @ F.normalize(torch.cat(g_feats), dim=-1).t(), q_pids, g_pids)  # [Q, G]
    if g_cls:
        dual = q @ F.normalize(torch.cat(g_cls).float(), dim=-1).t()
        metrics.update({f'dual_{k}': v for k, v in t2i_metrics(dual, q_pids, g_pids).items()})
    model.train()

    return metrics


def log_eval(metrics, epoch, step):

    logger.info('Eval @ epoch {} (step {}): '.format(epoch, step) +
                ' | '.join(f'{k} {v:.2f}' for k, v in metrics.items()))

    wandb_log({**{f'eval/{k}': v for k, v in metrics.items()}, 'epoch': epoch}, step=step)

# ----------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------

def offset_stats(model, locations):

    """Where the text-guided model samples, in patch units. locations [B (captions), B (images), Q, H, P, 2].
    offset_disp: mean distance of a sampling point from its query's patch centre (the init star averages ~3.0).
    offset_text_std: for a fixed image, std of a sampling point's position across the B captions, averaged; 0 means
    the text has no influence on where the image is sampled (the hypothesis needs this to be clearly > 0)."""

    grid = model.offsets.grid_size
    loc = locations.float()
    ref = getattr(model.offsets, 'reference_points', None)
    if ref is None:
        ref = model.offsets.spatial_tokens.reference_points
    ref = ref[None, None, :, None, None]
    # A caption-independent model passes one set of locations per image ([1, B, ...]): no spread across captions.
    text_std = (loc.std(dim=0) * grid).norm(dim=-1).mean().item() if loc.shape[0] > 1 else 0.0
    stats = {'offset_disp': ((loc - ref) * grid).norm(dim=-1).mean().item(), 'offset_text_std': text_std}
    if getattr(model, 'gamma', None) is not None:
        stats['gamma'] = model.gamma.item()
    return stats


def train_step(cfg, model, batch, device, autocast):

    """Forward + losses. Returns the total loss and a dict of scalars to log (losses and batch diagnostics)."""

    image = batch['image'].to(device, non_blocking=True)
    caption = batch['caption'].to(device, non_blocking=True)
    pids = batch['pid'].to(device, non_blocking=True)

    with autocast():
        out = model(image, caption)

    # sim[i, j] = s(caption i, image j). The text-guided model scores all B×B pairs itself; for the dual encoder it is
    # the cosine of the two independent embeddings.
    sim = out.get('sim')
    if sim is None:
        sim = F.normalize(out['text_feat'].float(), dim=-1) @ F.normalize(out['image_feat'].float(), dim=-1).t()
    l_sdm = sdm_loss_from_sim(sim, pids, cfg.loss.temperature)
    l_id = id_loss(out['image_logits'], out['text_logits'], pids)
    loss = cfg.loss.sdm_weight * l_sdm + cfg.loss.id_weight * l_id

    with torch.no_grad():

        sim = sim.detach().float()
        same = pids[:, None] == pids[None, :]

        stats = {
            'loss': loss.item(),
            'loss_sdm': l_sdm.item(),
            'loss_id': l_id.item(),
            'id_acc_image': (out['image_logits'].argmax(1) == pids).float().mean().item(),
            'id_acc_text': (out['text_logits'].argmax(1) == pids).float().mean().item(),
            'batch_t2i_r1': same[torch.arange(len(pids)), sim.argmax(1)].float().mean().item(),
            'sim_pos': sim[same].mean().item(),
            'sim_neg': sim[~same].mean().item() if (~same).any() else 0.0,
        }

        if 'locations' in out:
            stats.update(offset_stats(model, out['locations']))

    return loss, stats


def main():
    cfg, args = load_config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.output_dir, CKPT_NAME)
    resuming = cfg.resume and os.path.exists(ckpt_path)
    setup_logger(cfg.output_dir, resuming)
    set_seed(cfg.seed)
    logger.info('Config:\n' + OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.device)

    use_amp = cfg.optim.amp == 'bf16' and device.type == 'cuda'
    autocast = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if use_amp else nullcontext

    # --- data ---
    dataset = T2IVeRW(root=cfg.data.root, ann_file=cfg.data.ann_file, test_split=cfg.data.test_split)
    logger.info(repr(dataset))
    train_loader, query_loader, gallery_loader = build_loaders(cfg, dataset)
    steps_per_epoch = len(train_loader)
    if cfg.log.max_iters_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.log.max_iters_per_epoch)
    total_steps = cfg.optim.epochs * steps_per_epoch

    # --- model / optimisation ---
    model = build_model(cfg, dataset.num_train_pids).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, total_steps, steps_per_epoch)
    params = [p for p in model.parameters() if p.requires_grad]
    max_norm = cfg.optim.grad_clip if cfg.optim.grad_clip is not None else float('inf')
    logger.info('Param groups: ' + ', '.join(f"{g['name']} ({sum(p.numel() for p in g['params']) / 1e6:.2f}M, "
                                             f"lr {g['lr']:.0e})" for g in optimizer.param_groups))

    start_epoch, global_step, best = 1, 0, {'mAP': -1.0, 'epoch': 0}
    wandb_run_id = None
    if resuming:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        torch.set_rng_state(ckpt['rng']['torch'])
        torch.cuda.set_rng_state_all(ckpt['rng']['cuda'])
        np.random.set_state(ckpt['rng']['numpy'])
        random.setstate(ckpt['rng']['python'])
        start_epoch, global_step, best, wandb_run_id = ckpt['epoch'] + 1, ckpt['global_step'], ckpt['best'], ckpt['wandb_run_id']
        logger.info(f"Resumed from {ckpt_path} (finished epoch {ckpt['epoch']}, step {global_step})")
        del ckpt

    # --- wandb ---
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        run_config = OmegaConf.to_container(cfg, resolve=True)
        run_config['dataset_stats'] = {'train_samples': len(dataset.train), 'num_train_pids': dataset.num_train_pids,
                                       'num_query': len(dataset.query), 'num_gallery': len(dataset.gallery),
                                       'num_test_pids': len({pid for pid, _ in dataset.gallery})}
        run_config['steps_per_epoch'], run_config['total_steps'] = steps_per_epoch, total_steps
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name or os.path.basename(os.path.normpath(cfg.output_dir)),
                   config=run_config, tags=list(cfg.wandb.tags), mode=cfg.wandb.mode, dir=cfg.output_dir,
                   id=wandb_run_id, resume='allow')
        wandb_run_id = wandb.run.id
        wandb.run.summary['paper/IRRA_mAP'], wandb.run.summary['paper/PFCVR_mAP'] = 22.8, 26.2

    if cfg.log.eval_at_start and global_step == 0:
        log_eval(evaluate(model, query_loader, gallery_loader, device, autocast, cfg), epoch=0, step=0)

    # --- train ---
    model.train()
    for epoch in range(start_epoch, cfg.optim.epochs + 1):
        window, window_start, window_steps = defaultdict(float), time.time(), 0
        for it, batch in enumerate(train_loader):
            if it >= steps_per_epoch:
                break
            loss, stats = train_step(cfg, model, batch, device, autocast)
            if not math.isfinite(stats['loss']):
                raise FloatingPointError(f'non-finite loss at step {global_step}: {stats}')

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Total L2 norm of all gradients, measured before clipping (returned by clip_grad_norm_).
            stats['grad_norm'] = torch.nn.utils.clip_grad_norm_(params, max_norm).item()
            optimizer.step()
            scheduler.step()
            global_step += 1

            for k, v in stats.items():
                window[k] += v
            window_steps += 1

            if global_step % cfg.log.every == 0 or it + 1 == steps_per_epoch:
                elapsed = time.time() - window_start
                avg = {k: v / window_steps for k, v in window.items()}
                lrs = {g['name'].split('/')[0]: g['lr'] for g in optimizer.param_groups}
                logger.info(f'Epoch {epoch} [{it + 1}/{steps_per_epoch}] step {global_step} | '
                            f"loss {avg['loss']:.4f} (sdm {avg['loss_sdm']:.4f}, id {avg['loss_id']:.4f}) | "
                            f"grad_norm {avg['grad_norm']:.3f} | batch_r1 {avg['batch_t2i_r1']:.3f} | "
                            + (f"offset disp {avg['offset_disp']:.2f} text_std {avg['offset_text_std']:.3f} | "
                               if 'offset_disp' in avg else '')
                            + (f"gamma {avg['gamma']:.4f} | " if 'gamma' in avg else '') +
                            f"lr {lrs['backbone']:.2e} | {window_steps * cfg.data.batch_size / elapsed:.1f} samples/s | "
                            f"peak mem {torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0:.1f} GB")
                wandb_log({**{f'train/{k}': v for k, v in avg.items()},
                           **{f'train/lr_{k}': v for k, v in lrs.items()},
                           'train/samples_per_sec': window_steps * cfg.data.batch_size / elapsed,
                           'train/gpu_mem_gb': torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0,
                           'epoch': global_step / steps_per_epoch}, step=global_step)
                window, window_start, window_steps = defaultdict(float), time.time(), 0

        if epoch % cfg.log.eval_every == 0 or epoch == cfg.optim.epochs:
            metrics = evaluate(model, query_loader, gallery_loader, device, autocast, cfg)
            log_eval(metrics, epoch, global_step)
            if metrics['mAP'] > best['mAP']:
                best = {**metrics, 'epoch': epoch}
                torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'model_best.pth'))
                logger.info(f"New best mAP {metrics['mAP']:.2f} at epoch {epoch} -> model_best.pth")
            if wandb.run is not None:
                wandb.run.summary.update({**{f'best/{k}': v for k, v in best.items()},
                                          **{f'last/{k}': v for k, v in metrics.items()}, 'last/epoch': epoch})

        save_checkpoint(ckpt_path, epoch=epoch, global_step=global_step, best=best, wandb_run_id=wandb_run_id,
                        model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        rng={'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                             'numpy': np.random.get_state(), 'python': random.getstate()})

    torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'model_final.pth'))
    logger.info(f"Done. Best mAP {best['mAP']:.2f} at epoch {best['epoch']}")
    if wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
