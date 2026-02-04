from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import os
import json

try:
    import yaml

    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def _expand_vars(s: str, ctx: Dict[str, Any]) -> str:
    out = s
    for _ in range(5):
        if "${" not in out:
            break
        start = out.find("${")
        end = out.find("}", start + 2)
        if start < 0 or end < 0:
            break

        key = out[start + 2 : end].strip()
        cur: Any = ctx
        ok = True
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if not ok:
            break

        out = out[:start] + str(cur) + out[end + 1 :]
    return out


def _load_cfg_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suf = p.suffix.lower()
    if suf in (".yml", ".yaml"):
        if not _HAS_YAML:
            raise RuntimeError("pyyaml is not installed. Install it via `pip install pyyaml`, or use a .json config.")
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    elif suf == ".json":
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise ValueError("Only .yaml/.yml or .json config files are supported.")

    def walk(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str):
            return _expand_vars(x, raw)
        return x

    return walk(raw)


@dataclass
class Config:
    config_file: Optional[str] = None

    data_root: str = r"E:/GroundTruth"
    train_source: str = r"E:/GroundTruth/precomputed_train_v2"
    val_main_source: str = r"E:/GroundTruth/val_main/processed_truth_*.txt"

    levels: int = 20
    val_split_dir: str = "val1_20"
    test_split_dir: str = "test"

    val_mag_sources: Dict[int, str] = field(default_factory=dict)
    val_pos_sources: Dict[int, str] = field(default_factory=dict)
    val_spur_sources: Dict[int, str] = field(default_factory=dict)
    val_mixed_source: str = ""

    test_mag_sources: Dict[int, str] = field(default_factory=dict)
    test_pos_sources: Dict[int, str] = field(default_factory=dict)
    test_spur_sources: Dict[int, str] = field(default_factory=dict)
    test_mixed_source: str = ""

    save_path: str = "checkpoint.pt"
    log_dir: str = "runs/gatv2_idnet"

    heads: int = 18
    in_proj_out: int = 100
    dropout: float = 0.4
    gnn_layers: int = 2

    gnn_feat_k: int = 8
    use_pixel_angle: bool = False

    gnn_bottleneck_dim: int = 256
    gnn_bottleneck_heads: int = 8

    trans_in_dim: int = 20
    trans_hidden_dim: int = 1024
    trans_layers: int = 1
    trans_heads: int = 16
    trans_dropout: float = 0.2

    gnn_inject_dim: int = 256

    msm_mask_p: float = 0.15
    lambda_msm: float = 0.1

    sensor_w: int = 1280
    sensor_h: int = 1024
    ring_d_pixels: int = 82
    ring_bins: int = 20

    k: int = 18
    batch_size: int = 5
    num_workers: int = 0

    epochs: int = 150
    lr_gnn: float = 7e-4
    lr_tf: float = 2.5e-5
    lr_other: float = 2.5e-5
    weight_decay: float = 3.5e-4

    onecycle_maxlr_mul: float = 3.0
    onecycle_pct_start: float = 0.20
    onecycle_div_factor: float = 2.0
    onecycle_final_div: float = 50.0

    tau: float = 0.7
    lambda_kl_max: float = 0.5
    lambda_feat_max: float = 0.1
    ramp_epochs: int = 15

    drop_one_view_p: float = 0.0

    grad_norm_gnn_max: float = 5.0
    grad_norm_tf_max: float = 10.0
    rollback_on_bad_grad: bool = True
    rollback_lr_shrink: float = 0.5

    fuse_bottleneck_dim: int = 256
    lambda_aux_g: float = 0.3
    lambda_aux_t: float = 0.1

    vcm_dim: int = 128
    vcm_scale: float = 0.5
    vcm_use_edge_stats: bool = False

    seed: int = 42
    device: str = "cuda"
    topk: int = 5

    def _init_paths(self) -> None:
        if not self.train_source:
            self.train_source = str(Path(self.data_root) / "train")
        if not self.val_main_source:
            self.val_main_source = str(Path(self.data_root) / "precomputed_val")

        val_root = Path(self.data_root) / self.val_split_dir
        test_root = Path(self.data_root) / self.test_split_dir

        self.val_mixed_source = str(val_root / "mixed" / "processed_truth_*.txt")
        self.test_mixed_source = str(test_root / "mixed" / "processed_truth_*.txt")

        self.val_mag_sources = {
            i: str(val_root / "mag" / f"mag_{i}" / "processed_truth_*.txt")
            for i in range(1, self.levels + 1)
        }
        self.val_pos_sources = {
            i: str(val_root / "pos" / f"shift_{i}" / "processed_truth_*.txt")
            for i in range(1, self.levels + 1)
        }
        self.val_spur_sources = {
            i: str(val_root / "spur" / f"spur_{i}" / "processed_truth_*.txt")
            for i in range(1, self.levels + 1)
        }

        self.test_mag_sources = {
            i: str(test_root / "mag" / f"mag_{i}" / "processed_truth_*.txt")
            for i in range(1, self.levels + 1)
        }
        self.test_pos_sources = {
            i: str(test_root / "pos" / f"shift_{i}" / "processed_truth_*.txt")
            for i in range(1, self.levels + 1)
        }
        self.test_spur_sources = {
            i: str(test_root / "spur" / f"spur_{i}" / "processed_truth_*.txt")
            for i in range(1, self.levels + 1)
        }

    def load_from_file(self, path: str) -> None:
        raw = _load_cfg_file(path)

        def get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
            cur: Any = d
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur

        data_root = get(raw, "data", "root")
        if data_root is not None:
            self.data_root = str(data_root)

        train_source = get(raw, "data", "train_source")
        if train_source is not None:
            self.train_source = str(train_source)

        val_main = get(raw, "data", "val_main_source")
        if val_main is not None:
            self.val_main_source = str(val_main)

        levels = get(raw, "data", "levels")
        if levels is not None:
            self.levels = int(levels)

        val_split_dir = get(raw, "data", "val_split_dir")
        if val_split_dir is not None:
            self.val_split_dir = str(val_split_dir)

        test_split_dir = get(raw, "data", "test_split_dir")
        if test_split_dir is not None:
            self.test_split_dir = str(test_split_dir)

        save_path = get(raw, "output", "save_path")
        if save_path is not None:
            self.save_path = str(save_path)

        log_dir = get(raw, "output", "log_dir")
        if log_dir is not None:
            self.log_dir = str(log_dir)

        device = get(raw, "device")
        if device is not None:
            self.device = str(device)

        seed = get(raw, "seed")
        if seed is not None:
            self.seed = int(seed)

        self._init_paths()

    def ensure_dirs(self) -> None:
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        sp = Path(self.save_path)
        if str(sp.parent) not in (".", ""):
            sp.parent.mkdir(parents=True, exist_ok=True)


cfg = Config()
cfg._init_paths()

_default_candidates = [
    "path.yaml",
    "path.yml",
    os.path.join("configs", "path.yaml"),
    os.path.join("configs", "path.yml"),
    "path.json",
    os.path.join("configs", "path.json"),
]

for _p in _default_candidates:
    if os.path.exists(_p):
        cfg.load_from_file(_p)
        break
