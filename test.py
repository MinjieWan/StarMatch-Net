import argparse
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from config import cfg as default_cfg
from dataset import StarGraphDataset
from model import DualViewIDNet


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_idmap(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "id_map" not in ckpt:
        raise RuntimeError(f"Invalid checkpoint format (expect model_state & id_map): {ckpt_path}")

    id_map = ckpt["id_map"]
    inv_id_map = {idx: sid for sid, idx in id_map.items()}
    num_real_ids = len(id_map)

    model = DualViewIDNet(num_real_ids=num_real_ids).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, id_map, inv_id_map


def build_val_loader(val_source: str, k: int, id_map: dict, batch_size: int, num_workers: int):
    ds_val = StarGraphDataset(source=val_source, k=k, id_map=id_map, verbose=1)
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return ds_val, dl_val


def shuffle_batch_nodes(batch):
    if not hasattr(batch, "batch"):
        return batch

    batch_ids = batch.batch
    n = int(batch_ids.size(0))
    if n <= 1:
        return batch

    num_graphs = int(batch_ids.max().item()) + 1
    perm_parts = []
    for g in range(num_graphs):
        idx_g = (batch_ids == g).nonzero(as_tuple=True)[0]
        if idx_g.numel() == 0:
            continue
        perm_parts.append(idx_g[torch.randperm(idx_g.numel(), device=idx_g.device)])

    perm = torch.cat(perm_parts, dim=0)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)

    for key, value in batch:
        if torch.is_tensor(value) and value.dim() >= 1 and value.size(0) == n:
            setattr(batch, key, value[perm])

    if hasattr(batch, "edge_index") and batch.edge_index is not None:
        batch.edge_index = inv_perm[batch.edge_index]

    return batch


def infer_on_val(model: torch.nn.Module, dl_val, device: torch.device):
    model.eval()

    total_nodes = 0
    correct_nodes = 0

    total_graphs = 0
    success_graphs3 = 0
    success_graphs5 = 0

    has_vcm = hasattr(model, "vcm") and (getattr(model, "vcm") is not None)
    vcm_abs_sum = {"gg": 0.0, "bg": 0.0, "gt": 0.0, "bt": 0.0, "gf": 0.0, "bf": 0.0}
    vcm_cnt = {"gg": 0, "bg": 0, "gt": 0, "bt": 0, "gf": 0, "bf": 0}
    vcm_first = None

    with torch.no_grad():
        for batch in dl_val:
            batch = batch.to(device)
            batch = shuffle_batch_nodes(batch)

            if has_vcm:
                try:
                    gg, bg, gt, bt, gf, bf = model.vcm(batch)
                    vcm_abs_sum["gg"] += float(torch.abs(gg - 1.0).sum().item())
                    vcm_cnt["gg"] += int(gg.numel())
                    vcm_abs_sum["bg"] += float(torch.abs(bg).sum().item())
                    vcm_cnt["bg"] += int(bg.numel())

                    vcm_abs_sum["gt"] += float(torch.abs(gt - 1.0).sum().item())
                    vcm_cnt["gt"] += int(gt.numel())
                    vcm_abs_sum["bt"] += float(torch.abs(bt).sum().item())
                    vcm_cnt["bt"] += int(bt.numel())

                    vcm_abs_sum["gf"] += float(torch.abs(gf - 1.0).sum().item())
                    vcm_cnt["gf"] += int(gf.numel())
                    vcm_abs_sum["bf"] += float(torch.abs(bf).sum().item())
                    vcm_cnt["bf"] += int(bf.numel())

                    if vcm_first is None:
                        vcm_first = {
                            "G": int(gg.size(0)),
                            "gg_min": float(gg.min().item()),
                            "gg_max": float(gg.max().item()),
                            "gt_min": float(gt.min().item()),
                            "gt_max": float(gt.max().item()),
                            "gf_min": float(gf.min().item()),
                            "gf_max": float(gf.max().item()),
                            "mabs_gg": float(torch.abs(gg - 1.0).mean().item()),
                            "mabs_bg": float(torch.abs(bg).mean().item()),
                            "mabs_gt": float(torch.abs(gt - 1.0).mean().item()),
                            "mabs_bt": float(torch.abs(bt).mean().item()),
                            "mabs_gf": float(torch.abs(gf - 1.0).mean().item()),
                            "mabs_bf": float(torch.abs(bf).mean().item()),
                        }
                except Exception:
                    has_vcm = False

            logits_f = model.forward_infer(batch)

            y_map = batch.y_id_mapped
            mask_true = (y_map != -100)
            pred_all = logits_f.argmax(dim=-1)

            if mask_true.any():
                correct_nodes += (pred_all[mask_true] == y_map[mask_true]).sum().item()
                total_nodes += mask_true.sum().item()

            graph_idx = batch.batch if hasattr(batch, "batch") else torch.zeros_like(y_map)
            num_graphs = int(graph_idx.max().item()) + 1 if graph_idx.numel() > 0 else 0

            for g in range(num_graphs):
                mask_g_true = (graph_idx == g) & mask_true
                total_graphs += 1
                if not mask_g_true.any():
                    continue

                correct_cnt = (pred_all[mask_g_true] == y_map[mask_g_true]).sum().item()
                if correct_cnt >= 3:
                    success_graphs3 += 1
                if correct_cnt >= 5:
                    success_graphs5 += 1

    acc_node = correct_nodes / total_nodes if total_nodes > 0 else float("nan")
    acc_graph3 = success_graphs3 / total_graphs if total_graphs > 0 else float("nan")
    acc_graph5 = success_graphs5 / total_graphs if total_graphs > 0 else float("nan")

    print("========== Validation Inference ==========")
    print(f"True-star nodes        : {total_nodes}")
    print(f"Graphs                 : {total_graphs}")
    print(f"Node accuracy          : {acc_node:.4f}")
    print(f"Graph accuracy (>=3)   : {acc_graph3:.4f}")
    print(f"Graph accuracy (>=5)   : {acc_graph5:.4f}")

    if has_vcm and vcm_cnt["gg"] > 0:
        mabs_gg = vcm_abs_sum["gg"] / max(1, vcm_cnt["gg"])
        mabs_bg = vcm_abs_sum["bg"] / max(1, vcm_cnt["bg"])
        mabs_gt = vcm_abs_sum["gt"] / max(1, vcm_cnt["gt"])
        mabs_bt = vcm_abs_sum["bt"] / max(1, vcm_cnt["bt"])
        mabs_gf = vcm_abs_sum["gf"] / max(1, vcm_cnt["gf"])
        mabs_bf = vcm_abs_sum["bf"] / max(1, vcm_cnt["bf"])

        print("\n========== VCM Modulation Stats (VAL) ==========")
        print("Mean(|gamma-1|) and Mean(|beta|) over validation:")
        print(f"  gdown : |gg-1|={mabs_gg:.6f}  |bg|={mabs_bg:.6f}")
        print(f"  tf_in : |gt-1|={mabs_gt:.6f}  |bt|={mabs_bt:.6f}")
        print(f"  fused : |gf-1|={mabs_gf:.6f}  |bf|={mabs_bf:.6f}")

        if vcm_first is not None:
            print("First-batch snapshot:")
            print(
                f"  G={vcm_first['G']} | "
                f"gg[{vcm_first['gg_min']:.3f},{vcm_first['gg_max']:.3f}] "
                f"gt[{vcm_first['gt_min']:.3f},{vcm_first['gt_max']:.3f}] "
                f"gf[{vcm_first['gf_min']:.3f},{vcm_first['gf_max']:.3f}]"
            )
            print(
                f"  mean abs: gg={vcm_first['mabs_gg']:.6f} bg={vcm_first['mabs_bg']:.6f} | "
                f"gt={vcm_first['mabs_gt']:.6f} bt={vcm_first['mabs_bt']:.6f} | "
                f"gf={vcm_first['mabs_gf']:.6f} bf={vcm_first['mabs_bf']:.6f}"
            )

    return {
        "acc_node": acc_node,
        "acc_graph3": acc_graph3,
        "acc_graph5": acc_graph5,
        "total_nodes": total_nodes,
        "total_graphs": total_graphs,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=default_cfg.save_path)
    p.add_argument("--batch_size", type=int, default=default_cfg.batch_size)
    p.add_argument("--k", type=int, default=default_cfg.k)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=default_cfg.device)
    p.add_argument("--seed", type=int, default=default_cfg.seed)
    p.add_argument("--val_idx", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    model, id_map, _inv_id_map = load_model_and_idmap(args.ckpt, device)

    val_sources = getattr(default_cfg, "val_mag_sources", None)
    if val_sources is None or not isinstance(val_sources, (list, tuple)) or len(val_sources) == 0:
        raise RuntimeError("default_cfg.val_mag_sources is missing or empty.")

    val_idx = max(0, min(int(args.val_idx), len(val_sources) - 1))
    val_source = val_sources[val_idx]

    ds_val, dl_val = build_val_loader(
        val_source=val_source,
        k=args.k,
        id_map=id_map,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"[INFO] Validation graphs: {len(ds_val)}, classes: {len(id_map)}, val_source_idx: {val_idx}")
    infer_on_val(model, dl_val, device)


if __name__ == "__main__":
    main()
