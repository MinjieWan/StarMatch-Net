import torch
import torch.nn as nn

from config import cfg
from transformer import TabTransformer
from GATv2 import GATv2Block


def _build_x_gnn_from_batch(batch, k_feat: int) -> torch.Tensor:
    pos = getattr(batch, "pos", None)
    if pos is None:
        raise RuntimeError("batch.pos is missing; cannot build x_gnn (please provide pos in the dataset).")

    n = pos.size(0)
    if k_feat <= 0:
        return pos

    out = torch.zeros((n, k_feat), dtype=pos.dtype, device=pos.device)

    edge_index = getattr(batch, "edge_index", None)
    edge_attr = getattr(batch, "edge_attr", None)
    if edge_index is None or edge_attr is None or edge_index.numel() == 0:
        return torch.cat([pos, out], dim=1)

    src = edge_index[0]
    d = edge_attr.view(-1)

    for i in range(n):
        m = (src == i)
        if not m.any():
            continue
        di = d[m]
        di, _ = torch.sort(di)
        di = di[:k_feat]
        out[i, : di.numel()] = di

    return torch.cat([pos, out], dim=1)


def _num_graphs_from_batch(batch) -> int:
    b = getattr(batch, "batch", None)
    if b is None or b.numel() == 0:
        return 1
    return int(b.max().item()) + 1


def _bincount(idx: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros((n,), device=idx.device, dtype=torch.long)
    ones = torch.ones_like(idx, dtype=torch.long)
    out.index_add_(0, idx, ones)
    return out


def _scatter_sum(x: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    if x.dim() == 1:
        out = torch.zeros((n,), device=x.device, dtype=x.dtype)
        out.index_add_(0, idx, x)
        return out
    out = torch.zeros((n, x.size(-1)), device=x.device, dtype=x.dtype)
    out.index_add_(0, idx, x)
    return out


def _scatter_mean(x: torch.Tensor, idx: torch.Tensor, n: int, eps: float = 1e-8) -> torch.Tensor:
    s = _scatter_sum(x, idx, n)
    c = _bincount(idx, n).to(s.dtype)
    if s.dim() == 1:
        return s / (c + eps)
    return s / (c.unsqueeze(-1) + eps)


def _scatter_mean_std_1d(x: torch.Tensor, idx: torch.Tensor, n: int, eps: float = 1e-8):
    mean = _scatter_mean(x, idx, n, eps=eps)
    mean2 = _scatter_mean(x * x, idx, n, eps=eps)
    var = torch.clamp(mean2 - mean * mean, min=0.0)
    std = torch.sqrt(var + eps)
    return mean, std


class VCM(nn.Module):
    def __init__(self, x20_dim: int, inj_dim: int, tf_dim: int, fuse_in_dim: int):
        super().__init__()
        self.x20_dim = int(x20_dim)
        self.inj_dim = int(inj_dim)
        self.tf_dim = int(tf_dim)
        self.fuse_in_dim = int(fuse_in_dim)

        self.c_dim = int(getattr(cfg, "vcm_dim", 128))
        self.scale = float(getattr(cfg, "vcm_scale", 0.5))
        self.use_edge_stats = bool(getattr(cfg, "vcm_use_edge_stats", True))

        cin = self.x20_dim + 1 + (2 if self.use_edge_stats else 0)

        self.cond = nn.Sequential(
            nn.Linear(cin, self.c_dim),
            nn.GELU(),
            nn.Linear(self.c_dim, self.c_dim),
            nn.GELU(),
        )

        self.to_g = nn.Linear(self.c_dim, 2 * self.inj_dim)
        self.to_t = nn.Linear(self.c_dim, 2 * self.tf_dim)
        self.to_f = nn.Linear(self.c_dim, 2 * self.fuse_in_dim)

        nn.init.zeros_(self.to_g.weight)
        nn.init.zeros_(self.to_g.bias)
        nn.init.zeros_(self.to_t.weight)
        nn.init.zeros_(self.to_t.bias)
        nn.init.zeros_(self.to_f.weight)
        nn.init.zeros_(self.to_f.bias)

    def _build_condition(self, batch) -> torch.Tensor:
        device = batch.x20.device
        graph_idx = getattr(batch, "batch", None)
        if graph_idx is None:
            graph_idx = torch.zeros((batch.x20.size(0),), device=device, dtype=torch.long)

        g = _num_graphs_from_batch(batch)

        x20_mean = _scatter_mean(batch.x20, graph_idx, g)
        cnt_nodes = _bincount(graph_idx, g).to(x20_mean.dtype)
        logn = torch.log1p(cnt_nodes).unsqueeze(-1)

        feats = [x20_mean, logn]

        if self.use_edge_stats:
            edge_attr = getattr(batch, "edge_attr", None)
            edge_index = getattr(batch, "edge_index", None)

            if edge_attr is None or edge_index is None or edge_index.numel() == 0:
                em = torch.zeros((g,), device=device, dtype=x20_mean.dtype)
                es = torch.zeros((g,), device=device, dtype=x20_mean.dtype)
            else:
                src = edge_index[0]
                edge_g = graph_idx[src]
                d = edge_attr.view(-1).to(x20_mean.dtype)
                em, es = _scatter_mean_std_1d(d, edge_g, g)

            feats.append(em.unsqueeze(-1))
            feats.append(es.unsqueeze(-1))

        return torch.cat(feats, dim=-1)

    def forward(self, batch):
        c_in = self._build_condition(batch)
        h = self.cond(c_in)

        def split_film(out: torch.Tensor, d: int):
            g_raw, b_raw = out[:, :d], out[:, d:]
            gamma = 1.0 + self.scale * torch.tanh(g_raw)
            beta = self.scale * torch.tanh(b_raw)
            return gamma, beta

        gg, bg = split_film(self.to_g(h), self.inj_dim)
        gt, bt = split_film(self.to_t(h), self.tf_dim)
        gf, bf = split_film(self.to_f(h), self.fuse_in_dim)
        return gg, bg, gt, bt, gf, bf


class GATv2IDNet(nn.Module):
    def __init__(self, in_proj_out, heads, dropout, num_real_ids, num_layers: int = 1):
        super().__init__()
        self.k_feat = int(getattr(cfg, "gnn_feat_k", 0))
        self.in_dim = 2 + self.k_feat

        self.in_proj = nn.Linear(self.in_dim, in_proj_out)
        self._out_dim = in_proj_out * heads

        self.gat1 = GATv2Block(in_proj_out, self._out_dim, heads=heads, edge_dim=1)

        self.num_layers = int(num_layers)
        if self.num_layers >= 2:
            self.bottleneck_dim = int(getattr(cfg, "gnn_bottleneck_dim", 256))
            self.bottleneck_heads = int(getattr(cfg, "gnn_bottleneck_heads", 8))
            if self.bottleneck_dim % self.bottleneck_heads != 0:
                raise ValueError(
                    f"gnn_bottleneck_dim ({self.bottleneck_dim}) must be divisible by "
                    f"gnn_bottleneck_heads ({self.bottleneck_heads})."
                )

            self.down = nn.Linear(self._out_dim, self.bottleneck_dim, bias=False)
            self.down_ln = nn.LayerNorm(self.bottleneck_dim)

            self.gat2 = GATv2Block(
                self.bottleneck_dim,
                self.bottleneck_dim,
                heads=self.bottleneck_heads,
                edge_dim=1,
            )

            self.up = nn.Linear(self.bottleneck_dim, self._out_dim, bias=False)
            self.up_drop = nn.Dropout(dropout)
            self.up_ln = nn.LayerNorm(self._out_dim)
        else:
            self.gat2 = None

        self.classifier = nn.Linear(self._out_dim, num_real_ids)

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, batch, return_logits: bool = True, return_h1: bool = False):
        if hasattr(batch, "x_gnn") and batch.x_gnn is not None:
            x_in = batch.x_gnn
        else:
            x_in = _build_x_gnn_from_batch(batch, self.k_feat)

        h0 = self.in_proj(x_in)
        h1 = self.gat1(h0, batch.edge_index, getattr(batch, "edge_attr", None))

        if self.gat2 is not None:
            z = self.down_ln(self.down(h1))
            z = self.gat2(z, batch.edge_index, getattr(batch, "edge_attr", None))
            h2 = self.up(z)
            h = self.up_ln(h1 + self.up_drop(h2))
        else:
            h = h1

        logits = self.classifier(h) if return_logits else None
        if return_h1:
            return logits, h, h1
        return logits, h


class DualViewIDNet(nn.Module):
    def __init__(self, num_real_ids):
        super().__init__()

        self.gnn = GATv2IDNet(
            cfg.in_proj_out,
            cfg.heads,
            cfg.dropout,
            num_real_ids,
            num_layers=getattr(cfg, "gnn_layers", 1),
        )

        self.inj_dim = int(getattr(cfg, "gnn_inject_dim", 256))
        self.g_inject = nn.Sequential(
            nn.Linear(self.gnn.out_dim, self.inj_dim, bias=False),
            nn.LayerNorm(self.inj_dim),
        )

        self.tf_in_proj = nn.Linear(cfg.trans_in_dim + self.inj_dim, cfg.trans_hidden_dim)

        self.tf = TabTransformer(
            in_dim=cfg.trans_hidden_dim,
            d_model=cfg.trans_hidden_dim,
            num_layers=cfg.trans_layers,
            nhead=cfg.trans_heads,
            dropout=cfg.trans_dropout,
            use_embed=False,
        )

        self.head_t = nn.Linear(cfg.trans_hidden_dim, num_real_ids)

        self.fuse_bottleneck_dim = int(getattr(cfg, "fuse_bottleneck_dim", 512))
        self.fuse_in_dim = self.gnn.out_dim + cfg.trans_hidden_dim

        self.fuse_ln = nn.LayerNorm(self.fuse_in_dim)
        self.fuse_fc = nn.Linear(self.fuse_in_dim, self.fuse_bottleneck_dim)
        self.fuse_act = nn.GELU()
        self.fuse_drop = nn.Dropout(cfg.trans_dropout)
        self.fuse_head = nn.Linear(self.fuse_bottleneck_dim, num_real_ids)

        self.fuse_alpha_logit = nn.Parameter(torch.tensor(2.0))

        self.vcm = VCM(
            x20_dim=cfg.trans_in_dim,
            inj_dim=self.inj_dim,
            tf_dim=cfg.trans_hidden_dim,
            fuse_in_dim=self.fuse_in_dim,
        )

    def _apply_fused_skip(self, logits_f_raw: torch.Tensor, logits_g: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.fuse_alpha_logit)
        return logits_f_raw + alpha * logits_g

    def forward(self, batch):
        logits_g, h_g, h1 = self.gnn(batch, return_logits=True, return_h1=True)

        gg, bg, gt, bt, gf, bf = self.vcm(batch)
        graph_idx = getattr(batch, "batch", None)
        if graph_idx is None:
            graph_idx = torch.zeros((h_g.size(0),), device=h_g.device, dtype=torch.long)

        gdown = self.g_inject(h1)
        gdown = gg[graph_idx] * gdown + bg[graph_idx]

        x_cat = torch.cat([batch.x20, gdown], dim=-1)
        hq = self.tf_in_proj(x_cat)
        hq = gt[graph_idx] * hq + bt[graph_idx]

        h_t = self.tf(hq, graph_idx, kv_tokens=None, node_mask=None)
        logits_t = self.head_t(h_t)

        fuse_in = torch.cat([h_g, h_t], dim=-1)
        u = self.fuse_ln(fuse_in)
        u = gf[graph_idx] * u + bf[graph_idx]

        h_f = self.fuse_drop(self.fuse_act(self.fuse_fc(u)))
        logits_f_raw = self.fuse_head(h_f)
        logits_f = self._apply_fused_skip(logits_f_raw, logits_g)

        return logits_g, logits_t, logits_f

    @torch.no_grad()
    def forward_infer(self, batch):
        logits_g, h_g, h1 = self.gnn(batch, return_logits=True, return_h1=True)

        gg, bg, gt, bt, gf, bf = self.vcm(batch)
        graph_idx = getattr(batch, "batch", None)
        if graph_idx is None:
            graph_idx = torch.zeros((h_g.size(0),), device=h_g.device, dtype=torch.long)

        gdown = self.g_inject(h1)
        gdown = gg[graph_idx] * gdown + bg[graph_idx]

        hq = self.tf_in_proj(torch.cat([batch.x20, gdown], dim=-1))
        hq = gt[graph_idx] * hq + bt[graph_idx]

        h_t = self.tf(hq, graph_idx, kv_tokens=None, node_mask=None)

        fuse_in = torch.cat([h_g, h_t], dim=-1)
        u = self.fuse_ln(fuse_in)
        u = gf[graph_idx] * u + bf[graph_idx]

        h_f = self.fuse_drop(self.fuse_act(self.fuse_fc(u)))
        logits_f_raw = self.fuse_head(h_f)

        return self._apply_fused_skip(logits_f_raw, logits_g)
