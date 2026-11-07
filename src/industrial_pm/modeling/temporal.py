from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


def set_deterministic_seed(seed: int, threads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True)
class SequenceBatch:
    features: np.ndarray
    targets: np.ndarray
    unit_ids: np.ndarray
    cycles: np.ndarray


def subset_sequences(batch: SequenceBatch, mask: np.ndarray) -> SequenceBatch:
    selected = np.asarray(mask, dtype=bool)
    if len(selected) != len(batch.targets):
        raise ValueError("Sequence mask length does not match batch")
    return SequenceBatch(
        features=batch.features[selected],
        targets=batch.targets[selected],
        unit_ids=batch.unit_ids[selected],
        cycles=batch.cycles[selected],
    )


def build_sequences(
    frame: pd.DataFrame,
    scaled_features: np.ndarray,
    sequence_length: int = 30,
    stride: int = 1,
    final_only: bool = False,
) -> SequenceBatch:
    """Build causal, left-padded sequences without crossing engine boundaries."""

    if len(frame) != len(scaled_features):
        raise ValueError("Frame and scaled feature rows must align")
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    unit_ids: list[int] = []
    cycles: list[int] = []
    for unit_id, positions in frame.groupby("unit_id", sort=False).indices.items():
        indexes = np.asarray(positions, dtype=int)
        endpoints = [len(indexes) - 1] if final_only else list(range(0, len(indexes), stride))
        if not final_only and endpoints[-1] != len(indexes) - 1:
            endpoints.append(len(indexes) - 1)
        engine_features = scaled_features[indexes]
        engine_frame = frame.iloc[indexes]
        for endpoint in endpoints:
            start = max(0, endpoint - sequence_length + 1)
            sequence = engine_features[start : endpoint + 1]
            if len(sequence) < sequence_length:
                padding = np.repeat(sequence[[0]], sequence_length - len(sequence), axis=0)
                sequence = np.concatenate([padding, sequence], axis=0)
            sequences.append(sequence.astype(np.float32, copy=False))
            targets.append(float(engine_frame.iloc[endpoint]["rul"]))
            unit_ids.append(int(cast(float, unit_id)))
            cycles.append(int(engine_frame.iloc[endpoint]["cycle"]))
    return SequenceBatch(
        features=np.stack(sequences),
        targets=np.asarray(targets, dtype=np.float32),
        unit_ids=np.asarray(unit_ids, dtype=np.int32),
        cycles=np.asarray(cycles, dtype=np.int32),
    )


class TemporalLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 48,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return cast(torch.Tensor, self.head(output[:, -1, :]).squeeze(-1))


@dataclass
class TemporalTrainingResult:
    model: TemporalLSTM
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_rmse: float


def train_temporal_model(
    train: SequenceBatch,
    validation: SequenceBatch,
    *,
    hidden_size: int = 48,
    num_layers: int = 1,
    dropout: float = 0.0,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    max_epochs: int = 18,
    patience: int = 4,
    seed: int = 42,
    threads: int = 4,
) -> TemporalTrainingResult:
    set_deterministic_seed(seed, threads)
    model = TemporalLSTM(
        input_size=train.features.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=15.0)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train.features), torch.from_numpy(train.targets)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_x = torch.from_numpy(validation.features)
    validation_y = torch.from_numpy(validation.targets)
    history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    best_rmse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_x)
            rmse = float(torch.sqrt(torch.mean((validation_prediction - validation_y) ** 2)))
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "val_rmse": rmse})
        print(
            f"[train:lstm] epoch={epoch} train_loss={np.mean(losses):.4f} "
            f"validation_rmse={rmse:.4f}",
            flush=True,
        )
        if rmse < best_rmse - 1e-4:
            best_rmse = rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Temporal training did not produce a model state")
    model.load_state_dict(best_state)
    return TemporalTrainingResult(model, history, best_epoch, best_rmse)


def predict_temporal(model: TemporalLSTM, batch: SequenceBatch, batch_size: int = 512) -> np.ndarray:
    model.eval()
    dataset = cast(Dataset[tuple[torch.Tensor]], TensorDataset(torch.from_numpy(batch.features)))
    loader: DataLoader[tuple[torch.Tensor]] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for (features,) in loader:
            predictions.append(model(features).cpu().numpy())
    return np.concatenate(predictions).clip(min=0.0)
