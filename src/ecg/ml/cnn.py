"""Model D: compact 1D-CNN (optional, needs PyTorch, CPU only).

Input per beat:
* the causally filtered waveform from 0.25 s before to 0.25 s after R
  (124 samples at 250 Hz), normalised per beat (median removed, divided by std),
* four causal RR inputs from the feature extractor (rr_prev_s, rr_prev2_s,
  rr_local_mean_s, rr_ratio). Supraventricular beats are mostly a TIMING
  abnormality, and a waveform-only network would barely see them.

Architecture (configurable): Conv-BN-ReLU-MaxPool blocks, global average
pooling, then concatenation with the RR inputs, one hidden layer and a 4-class
output. The 32-unit hidden layer is the "embedding" used by the hybrid model.

Explanations use integrated gradients (Sundararajan et al., 2017) with a flat
waveform and median RR inputs as the baseline. Saliency maps are exploratory:
they show where the network's output was sensitive, not that its reasoning is
physiologically valid.
"""

from __future__ import annotations

import io
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..features import FEATURE_NAMES

RR_INPUTS: tuple[str, ...] = ("rr_prev_s", "rr_prev2_s", "rr_local_mean_s", "rr_ratio")
RR_INDEX = [FEATURE_NAMES.index(n) for n in RR_INPUTS]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class CNNConfig:
    channels: tuple[int, ...] = (16, 32, 32)
    kernels: tuple[int, ...] = (7, 5, 3)
    hidden: int = 32
    dropout: float = 0.2
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4  # early stopping on validation macro-F1


def normalise_waves(waves: np.ndarray) -> np.ndarray:
    w = np.asarray(waves, dtype=np.float32)
    w = w - np.median(w, axis=1, keepdims=True)
    return w / (w.std(axis=1, keepdims=True) + 1e-6)


def _build(cfg: CNNConfig, n_classes: int, n_rr: int):
    import torch
    from torch import nn

    class BeatCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            c_in = 1
            for c_out, k in zip(cfg.channels, cfg.kernels):
                layers += [nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2)]
                c_in = c_out
            self.conv = nn.Sequential(*layers)
            self.embed = nn.Sequential(nn.Linear(c_in + n_rr, cfg.hidden), nn.ReLU())
            self.drop = nn.Dropout(cfg.dropout)
            self.out = nn.Linear(cfg.hidden, n_classes)

        def embedding(self, wave: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
            pooled = self.conv(wave.unsqueeze(1)).mean(dim=2)
            return self.embed(torch.cat([pooled, rr], dim=1))

        def forward(self, wave: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
            return self.out(self.drop(self.embedding(wave, rr)))

    return BeatCNN()


@dataclass
class CNNModel:
    """Trained network plus everything needed to reproduce its inputs."""

    state: dict[str, Any]
    config: CNNConfig
    classes: tuple[str, ...]
    rr_mean: np.ndarray
    rr_std: np.ndarray
    rr_median: np.ndarray  # integrated-gradients baseline for RR inputs (scaled units)
    history: list[dict] = field(default_factory=list)
    _net: Any = field(default=None, repr=False)

    @property
    def classes_(self) -> np.ndarray:
        return np.array(self.classes)

    def net(self):
        if self._net is None:
            import torch

            self._net = _build(self.config, len(self.classes), len(RR_INPUTS))
            self._net.load_state_dict({k: torch.as_tensor(v) for k, v in self.state.items()})
            self._net.eval()
        return self._net

    def _inputs(self, waves: np.ndarray, X: np.ndarray):
        import torch

        rr = (X[:, RR_INDEX] - self.rr_mean) / self.rr_std
        return torch.as_tensor(normalise_waves(waves)), torch.as_tensor(rr, dtype=torch.float32)

    def predict_proba(self, waves: np.ndarray, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            w, rr = self._inputs(waves, X)
            return torch.softmax(self.net()(w, rr), dim=1).numpy()

    def embed(self, waves: np.ndarray, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            w, rr = self._inputs(waves, X)
            return self.net().embedding(w, rr).numpy()

    def integrated_gradients(self, wave: np.ndarray, x: np.ndarray, target: int, steps: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Attributions for one beat: (per waveform sample, per RR input), in logit units."""
        import torch

        w, rr = self._inputs(wave.reshape(1, -1), x.reshape(1, -1))
        w0 = torch.zeros_like(w)  # flat line after normalisation
        rr0 = torch.as_tensor(self.rr_median.reshape(1, -1), dtype=torch.float32)
        alphas = torch.linspace(0, 1, steps).reshape(-1, 1)
        wi = (w0 + alphas * (w - w0)).requires_grad_(True)
        ri = (rr0 + alphas * (rr - rr0)).requires_grad_(True)
        logits = self.net()(wi, ri)[:, target].sum()
        gw, gr = torch.autograd.grad(logits, [wi, ri])
        att_w = ((w - w0) * gw.mean(dim=0, keepdim=True)).detach().numpy()[0]
        att_r = ((rr - rr0) * gr.mean(dim=0, keepdim=True)).detach().numpy()[0]
        return att_w, att_r


def train_cnn(
    train_waves: np.ndarray,
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    classes: tuple[str, ...],
    *,
    cfg: CNNConfig = CNNConfig(),
    seed: int = 42,
    val_records: int = 4,
) -> CNNModel:
    """Train on DS1, early-stopping on a record-wise validation split taken FROM DS1."""
    import torch
    from sklearn.metrics import f1_score
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    records = np.unique(train_groups)
    val_set = set(rng.choice(records, size=val_records, replace=False))
    is_val = np.isin(train_groups, list(val_set))

    rr = train_X[:, RR_INDEX]
    rr_mean, rr_std = rr[~is_val].mean(axis=0), rr[~is_val].std(axis=0) + 1e-6
    index = {c: i for i, c in enumerate(classes)}
    y_int = np.array([index[v] for v in train_y])

    def tensors(mask):
        return (torch.as_tensor(normalise_waves(train_waves[mask])),
                torch.as_tensor((rr[mask] - rr_mean) / rr_std, dtype=torch.float32),
                torch.as_tensor(y_int[mask]))

    tw, tr, ty = tensors(~is_val)
    vw, vr, vy = tensors(is_val)
    counts = np.bincount(ty.numpy(), minlength=len(classes)).astype(float)
    weights = torch.as_tensor(np.sqrt(counts.sum() / np.maximum(counts, 1)), dtype=torch.float32)
    net = _build(cfg, len(classes), len(RR_INPUTS))
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best = (-1.0, None)
    history: list[dict] = []
    stale = 0
    for epoch in range(cfg.epochs):
        net.train()
        order = torch.randperm(tw.shape[0])
        total = 0.0
        t0 = time.perf_counter()
        for i in range(0, order.numel(), cfg.batch_size):
            b = order[i : i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(net(tw[b], tr[b]), ty[b])
            loss.backward()
            opt.step()
            total += float(loss) * b.numel()
        net.eval()
        with torch.no_grad():
            pred = net(vw, vr).argmax(dim=1).numpy()
        val_f1 = float(f1_score(vy.numpy(), pred, labels=list(range(len(classes))), average="macro", zero_division=0))
        history.append({"epoch": epoch + 1, "train_loss": total / tw.shape[0], "val_macro_f1": val_f1,
                        "seconds": time.perf_counter() - t0})
        if val_f1 > best[0]:
            best = (val_f1, {k: v.detach().clone() for k, v in net.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    state = {k: v.numpy() for k, v in best[1].items()}
    return CNNModel(
        state=state, config=cfg, classes=classes, rr_mean=rr_mean, rr_std=rr_std,
        rr_median=(np.median(rr[~is_val], axis=0) - rr_mean) / rr_std, history=history,
    )


def state_size_bytes(model: CNNModel) -> int:
    buf = io.BytesIO()
    np.savez_compressed(buf, **model.state)
    return buf.tell()


def describe(cfg: CNNConfig) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()}
