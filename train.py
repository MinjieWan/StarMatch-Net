import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.loader import DataLoader as GeoDataLoader

os.environ["CONFIG_FILE"] = r"path.yaml"

from config import cfg
from dataset import PrecomputedStarGraphDataset, StarGraphDataset, make_dataloaders
from model import DualViewIDNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lrs(optimizer) -> list:
    return [pg["lr"] for pg in optimizer.param_groups]


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


@torch.no_grad()
def evaluate_on_val(model, dl_val, device, ce_loss):
    if dl_val is None:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    model.eval()

    total_nodes = 0
    sum_loss = 0.0

    correct_fused = 0
    correct_g = 0
    correct_t = 0

    total_graphs = 0
    success_graphs5 = 0

    for batch in dl_val:
        batch = batch.to(device)

        logits_g, logits_t, logits_f = model(batch)

        y = batch.y_id_mapped
        mask_true = (y != -100)

        if mask_true.any():
            y_true = y[mask_true]
            lf = logits_f[mask_true]
            lg = logits_g[mask_true]
            lt = logits_t[mask_true]

            loss = ce_loss(lf, y_true)
            sum_loss += loss.item() * y_true.numel()
            total_nodes += y_true.numel()

            correct_fused += (lf.argmax(-1) == y_true).sum().item()
            correct_g += (lg.argmax(-1) == y_true).sum().item()
            correct_t += (lt.argmax(-1) == y_true).sum().item()

        graph_idx = batch.batch if hasattr(batch, "batch") else torch.zeros_like(y)
        pred_all = logits_f.argmax(-1)

        num_graphs = int(graph_idx.max().item()) + 1 if graph_idx.numel() > 0 else 0
        for g in range(num_graphs):
            mask_g = (graph_idx == g) & mask_true
            total_graphs += 1
            if not mask_g.any():
                continue
            if (pred_all[mask_g] == y[mask_g]).sum().item() >= 5:
                success_graphs5 += 1

    if total_nodes == 0:
        return float("nan"), 0.0, 0.0, 0.0, success_graphs5 / max(total_graphs, 1)

    return (
        sum_loss / total_nodes,
        correct_fused / total_nodes,
        correct_g / total_nodes,
        correct_t / total_nodes,
        success_graphs5 / max(total_graphs, 1),
    )


def build_dataloaders(args):
    use_cuda = torch.cuda.is_available()

    if os.path.isdir(args.train_csv) and os.path.exists(os.path.join(args.train_csv, "graph_000000.pt")):
        ds_train = PrecomputedStarGraphDataset(root=args.train_csv)
        num_real_ids = ds_train.num_real_ids
        if num_real_ids <= 0:
            raise RuntimeError("No valid real-star classes found in the precomputed training set (num_real_ids<=0).")

        dl_train = GeoDataLoader(
            ds_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
        )

        ds_val = None
        dl_val = None

        if args.val_csv is not None:
            if os.path.isdir(args.val_csv) and os.path.exists(os.path.join(args.val_csv, "graph_000000.pt")):
                ds_val = PrecomputedStarGraphDataset(root=args.val_csv)
                dl_val = GeoDataLoader(
                    ds_val,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=max(1, args.num_workers // 2),
                    pin_memory=use_cuda,
                )
            else:
                ds_val = StarGraphDataset(
                    args.val_csv,
                    k=args.k,
                    id_map=ds_train.id_map,
                    verbose=1,
                )
                dl_val = GeoDataLoader(
                    ds_val,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=max(1, args.num_workers // 2),
                    pin_memory=use_cuda,
                )

        return ds_train, ds_val, dl_train, dl_val, num_real_ids

    ds_train, ds_val, dl_train, dl_val = make_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        k=args.k,
        num_workers=args.num_workers,
        shuffle_train=True,
        verbose=1,
    )

    num_real_ids = ds_train.num_real_ids
    if num_real_ids <= 0:
        raise RuntimeError("No valid real-star classes found in the training set (star_id>0 required).")

    return ds_train, ds_val, dl_train, dl_val, num_real_ids


def build_optimizer_and_scheduler(model, dl_train):
    gnn_params = list(model.gnn.parameters())

    tf_params = (
        list(model.tf.parameters())
        + list(model.head_t.parameters())
        + list(model.g_inject.parameters())
        + list(model.tf_in_proj.parameters())
    )

    taken = {id(p) for p in (gnn_params + tf_params)}
    rest_params = [p for p in model.parameters() if id(p) not in taken]

    param_groups = [
        {"params": gnn_params, "lr": cfg.lr_gnn, "weight_decay": cfg.weight_decay},
        {"params": tf_params, "lr": cfg.lr_tf, "weight_decay": cfg.weight_decay},
    ]
    max_lrs = [cfg.lr_gnn * 3.0, cfg.lr_tf]

    if rest_params:
        param_groups.append({"params": rest_params, "lr": cfg.lr_other})
        max_lrs.append(cfg.lr_other)

    optimizer = optim.AdamW(param_groups)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        epochs=cfg.epochs,
        steps_per_epoch=len(dl_train),
        pct_start=cfg.onecycle_pct_start,
        div_factor=cfg.onecycle_div_factor,
        final_div_factor=cfg.onecycle_final_div,
    )

    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=cfg.train_source)
    parser.add_argument("--val_csv", default=cfg.val_main_source)
    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--num_workers", type=int, default=getattr(cfg, "num_workers", 0))
    parser.add_argument("--k", type=int, default=cfg.k)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--save", default=cfg.save_path)
    parser.add_argument("--log_txt", default=r"train_Log.txt", help="Path to a training log file (append mode).")
    args = parser.parse_args()

    ensure_dir_for_file(args.log_txt)
    set_seed(args.seed)

    device = torch.device(args.device)
    log_f = open(args.log_txt, "a", encoding="utf-8")

    def log_print(s: str) -> None:
        print(s)
        log_f.write(s + "\n")
        log_f.flush()

    log_print("")
    log_print(f"==================== TRAIN START {now_str()} ====================")
    log_print("[Ablation] MNM removed (no node mask, no x20 reconstruction)")
    log_print(f"train_csv={args.train_csv}")
    log_print(f"val_csv={args.val_csv}")
    log_print(f"device={args.device} epochs={args.epochs} batch_size={args.batch_size} k={args.k} seed={args.seed}")
    log_print(f"save_ckpt={args.save}")
    log_print(f"log_txt={args.log_txt}")
    log_print("---------------------------------------------------------------------")

    last_safe = None
    ds_train = None

    try:
        # Data
        ds_train, ds_val, dl_train, dl_val, num_real_ids = build_dataloaders(args)
        log_print(f"[Data] train_graphs={len(ds_train)} classes={num_real_ids}")
        if ds_val is not None:
            log_print(f"[Data] val_graphs={len(ds_val)} classes={num_real_ids}")
        else:
            log_print("[Data] val_loader=None")

        # Model + loss
        model = DualViewIDNet(num_real_ids=num_real_ids).to(device)
        ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

        # Optimizer + scheduler
        optimizer, scheduler = build_optimizer_and_scheduler(model, dl_train)

        # Auxiliary loss weights
        w_g = float(getattr(cfg, "lambda_aux_g", 0.3))
        w_t = float(getattr(cfg, "lambda_aux_t", 0.1))

        best_val = -1.0
        last_safe = copy.deepcopy(model.state_dict())

        log_print(f"[INFO] aux weights: w_g={w_g:.3f} w_t={w_t:.3f}")
        log_print("---------------------------------------------------------------------")

        # Training loop
        for epoch in range(1, int(cfg.epochs) + 1):
            model.train()

            id_correct_f = 0
            id_correct_g = 0
            id_correct_t = 0
            id_total = 0

            running_loss = 0.0
            lr_start = get_lrs(optimizer)

            for batch in tqdm(dl_train, desc=f"Epoch {epoch}"):
                batch = batch.to(device)

                logits_g, logits_t, logits_f = model(batch)

                y = batch.y_id_mapped
                m = (y != -100)

                loss = torch.tensor(0.0, device=device)
                if m.any():
                    loss_f = ce_loss(logits_f[m], y[m])
                    loss_g = ce_loss(logits_g[m], y[m])
                    loss_t = ce_loss(logits_t[m], y[m])
                    loss = loss_f + w_g * loss_g + w_t * loss_t

                if not torch.isfinite(loss):
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()

                if m.any():
                    id_correct_f += (logits_f[m].argmax(-1) == y[m]).sum().item()
                    id_correct_g += (logits_g[m].argmax(-1) == y[m]).sum().item()
                    id_correct_t += (logits_t[m].argmax(-1) == y[m]).sum().item()
                    id_total += int(m.sum().item())

                running_loss += float(loss.item()) * max(int(m.sum().item()), 1)
                last_safe = copy.deepcopy(model.state_dict())

            train_loss = running_loss / max(id_total, 1)
            acc_f = id_correct_f / max(id_total, 1)
            acc_g = id_correct_g / max(id_total, 1)
            acc_t = id_correct_t / max(id_total, 1)

            # Validation
            val_loss, vaf, vag, vat, vgraph5 = evaluate_on_val(model, dl_val, device, ce_loss)
            lr_end = get_lrs(optimizer)

            log_line = (
                f"[{now_str()}] "
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} "
                f"acc_fused={acc_f:.4f} acc_g={acc_g:.4f} acc_t={acc_t:.4f} | "
                f"w_g={w_g:.3f} w_t={w_t:.3f} | "
                f"lr_gnn={lr_start[0]:.6f}->{lr_end[0]:.6f} "
                f"lr_tf={lr_start[1]:.6f}->{lr_end[1]:.6f} | "
                f"val_loss={val_loss:.4f} "
                f"val_acc_fused={vaf:.4f} val_acc_g={vag:.4f} val_acc_t={vat:.4f} "
                f"val_acc_graph5={vgraph5:.4f}"
            )

            saved = False
            if vaf >= best_val:
                best_val = vaf
                torch.save({"model_state": model.state_dict(), "id_map": ds_train.id_map, "cfg": vars(cfg)}, args.save)
                saved = True

            if saved:
                log_line += "  [*saved*]"

            log_print(log_line)

        log_print(f"Training finished. Checkpoint: {args.save}")
        log_print(f"===================== TRAIN END {now_str()} =====================")

    except KeyboardInterrupt:
        log_print("[WARN] KeyboardInterrupt received. Saving last_safe checkpoint...")
        if last_safe is not None and ds_train is not None:
            torch.save(
                {"model_state": last_safe, "id_map": ds_train.id_map, "cfg": vars(cfg)},
                args.save + ".last_safe.pt",
            )
            log_print(f"[WARN] last_safe saved: {args.save}.last_safe.pt")
        raise

    finally:
        try:
            log_f.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
