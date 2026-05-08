
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

notebook_start = time.perf_counter()

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "final_dataframes" / "container.csv"

TARGETS = [
    "target_p95_waiting_1w",
    "target_p95_waiting_2w",
]

TRAIN_END = "2024-06-30"
VAL_END = "2024-12-31"
GAP_DAYS = 14

PORTS_TO_KEEP = [
    "HOUSTON",
    "LOS_ANGELES_LONG_BEACH",
    "NEW_YORK_NEW_JERSEY",
    "PORT_OF_VIRGINIA",
]

SEQ_LEN = 28
HIDDEN_SIZE = 64
NUM_LAYERS = 1
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
LR = 1e-3
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


FEAT_M1 = [
    "feat_p95_waiting_hours_lag1w",
    "feat_p95_waiting_hours_lag2w",
    "feat_p95_waiting_hours_lag4w",
    "feat_p95_waiting_hours_roll4w_mean",
    "feat_p95_waiting_hours_roll4w_std",
    "feat_p95_waiting_hours_roll8w_mean",
    "feat_p95_waiting_hours_roll8w_std",
    "feat_p95_waiting_change_1w",
]

FEAT_MACRO = [
    "macro_gscpi",
    "macro_ism_pmi",
    "macro_us_imports_goods",
    "macro_us_trade_balance",
]


def load_data(path: Path, ports_to_keep: list[str]) -> pl.DataFrame:
    """Load container CSV, parse dates, filter ports."""
    df = pl.read_csv(path, try_parse_dates=True)
    df = df.filter(pl.col("port_name").is_in(ports_to_keep))
    df = df.sort(["port_name", "ref_date"])
    return df


df = load_data(DATA_PATH, PORTS_TO_KEEP)
ports = sorted(df["port_name"].unique().to_list())

feat_all = [c for c in df.columns if c.startswith("feat_")]
port_ohe_cols = [f"port_{p}" for p in ports]
df = df.with_columns([
    (pl.col("port_name") == p).cast(pl.Int32).alias(f"port_{p}")
    for p in ports
])

FEAT_M1_FULL = FEAT_M1 + port_ohe_cols
FEAT_M3 = feat_all + port_ohe_cols
FEAT_M4 = feat_all + FEAT_MACRO + port_ohe_cols

print(f"Loaded {df.shape[0]} rows, {len(ports)} ports: {ports}")
print(f"Feature counts — M1: {len(FEAT_M1_FULL)}, M3: {len(FEAT_M3)}, M4: {len(FEAT_M4)}")
print(f"Device: {DEVICE}")


def build_sequences(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    ports: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding-window sequences per port.

    Args:
        df: Input DataFrame (sorted by port, date).
        feature_cols: Feature column names.
        target_col: Target column name.
        seq_len: Number of timesteps per sequence.
        ports: Port names.

    Returns:
        X (n_samples, seq_len, n_features), y (n_samples,),
        port_indices (n_samples,) for per-port evaluation.
    """
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    port_list: list[int] = []

    for port_idx, port in enumerate(ports):
        port_df = df.filter(pl.col("port_name") == port).sort("ref_date")
        features = port_df.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)
        target = port_df[target_col].fill_null(0).to_numpy().astype(np.float32)

        n = len(features)
        if n <= seq_len:
            continue

        for i in range(seq_len, n):
            X_list.append(features[i - seq_len : i])
            y_list.append(float(target[i]))
            port_list.append(port_idx)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    port_indices = np.array(port_list, dtype=np.int32)
    return X, y, port_indices


def compute_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std from training sequences."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_scaler(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Standardize features using precomputed mean/std."""
    return (X - mean) / std


class CongestionGRU(nn.Module):
    """Simple GRU for congestion severity regression."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns non-negative predictions."""
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return torch.relu(out).squeeze(-1)


def train_gru(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_size: int,
    model_name: str,
) -> CongestionGRU:
    """Train GRU with early stopping on validation loss.

    Args:
        X_train: Training sequences (n, seq_len, features).
        y_train: Training targets.
        X_val: Validation sequences.
        y_val: Validation targets.
        input_size: Number of input features.
        model_name: Label for logging.

    Returns:
        Best model (by validation loss).
    """
    torch.manual_seed(RANDOM_STATE)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CongestionGRU(input_size, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(y_batch)
        val_loss /= len(val_ds)

        if epoch % 10 == 0:
            print(f"  [{model_name}] Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [{model_name}] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"  [{model_name}] Best val loss: {best_val_loss:.4f}")
    return model


def predict_gru(model: CongestionGRU, X: np.ndarray) -> np.ndarray:
    """Generate predictions from a trained GRU."""
    model.eval()
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    preds_list: list[np.ndarray] = []
    with torch.no_grad():
        for (X_batch,) in dl:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch)
            preds_list.append(preds.cpu().numpy())
    return np.concatenate(preds_list)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "n": 0}

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": len(y_true)}


def evaluate_gru_per_port(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    port_indices: np.ndarray,
    model_name: str,
    ports: list[str],
) -> list[dict[str, object]]:
    """Compute global + per-port metrics for GRU predictions."""
    results: list[dict[str, object]] = []

    metrics = evaluate(y_true, y_pred)
    results.append({"model": model_name, "port": "ALL", **metrics})

    for port_idx, port in enumerate(ports):
        mask = port_indices == port_idx
        if mask.sum() == 0:
            continue
        metrics = evaluate(y_true[mask], y_pred[mask])
        results.append({"model": model_name, "port": port, **metrics})

    return results


def print_results(results: list[dict[str, object]]) -> None:
    """Pretty-print results table."""
    header = f"{'Model':<12} {'Port':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        r2_val = r["r2"]
        r2_str = f"{r2_val:>8.3f}" if isinstance(r2_val, float) and not np.isnan(r2_val) else "     N/A"
        print(
            f"{r['model']:<12} {r['port']:<30} "
            f"{r['rmse']:>8.2f} {r['mae']:>8.2f} {r2_str} {r['n']:>6}"
        )


def run_gru_spec(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
    target: str,
    feature_cols: list[str],
    model_name: str,
) -> list[dict[str, object]]:
    """Build sequences, standardize, train GRU, and evaluate.

    Args:
        df_train: Training split.
        df_val: Validation split.
        df_test: Test split.
        target: Target column name.
        feature_cols: Feature columns for this spec.
        model_name: Label for logging/results.

    Returns:
        Results list with per-port metrics.
    """
    t0 = time.perf_counter()
    print(f"\n=== {model_name}: {len(feature_cols)} features ===")

    X_train_raw, y_train, _ = build_sequences(df_train, feature_cols, target, SEQ_LEN, ports)
    X_val_raw, y_val, _ = build_sequences(df_val, feature_cols, target, SEQ_LEN, ports)
    X_test_raw, y_test, port_idx_test = build_sequences(df_test, feature_cols, target, SEQ_LEN, ports)

    print(f"  Sequences — train: {X_train_raw.shape}, val: {X_val_raw.shape}, test: {X_test_raw.shape}")

    mean, std = compute_scaler(X_train_raw)
    X_train = apply_scaler(X_train_raw, mean, std)
    X_val = apply_scaler(X_val_raw, mean, std)
    X_test = apply_scaler(X_test_raw, mean, std)

    model = train_gru(X_train, y_train, X_val, y_val, len(feature_cols), model_name)

    preds = predict_gru(model, X_test)
    results = evaluate_gru_per_port(y_test, preds, port_idx_test, model_name, ports)

    elapsed = time.perf_counter() - t0
    print(f"  [{model_name}] Total time: {elapsed:.1f}s")
    print_results(results)

    return results


_train_end = pl.lit(TRAIN_END).str.to_date("%Y-%m-%d")
_val_end = pl.lit(VAL_END).str.to_date("%Y-%m-%d")

df_train_all = df.filter(pl.col("ref_date") <= _train_end)
df_val_all = df.filter(
    (pl.col("ref_date") > _train_end.dt.offset_by(f"{GAP_DAYS}d"))
    & (pl.col("ref_date") <= _val_end)
)
df_test_all = df.filter(pl.col("ref_date") > _val_end.dt.offset_by(f"{GAP_DAYS}d"))

print(f"Train: {df_train_all.shape[0]} | Val: {df_val_all.shape[0]} | Test: {df_test_all.shape[0]}")
print(f"(Gap: {GAP_DAYS} days excluded between train/val and val/test)")

grand_results: list[dict[str, object]] = []

for target in TARGETS:
    horizon_start = time.perf_counter()
    horizon_label = target.replace("target_p95_waiting_", "")

    df_train = df_train_all.filter(pl.col(target).is_not_null())
    df_val = df_val_all.filter(pl.col(target).is_not_null())
    df_test = df_test_all.filter(pl.col(target).is_not_null())

    print("\n" + "#" * 80)
    print(f"# HORIZON: {horizon_label} — target = {target}")
    print(f"# Train: {df_train.shape[0]} | Val: {df_val.shape[0]} | Test: {df_test.shape[0]}")
    print("#" * 80)

    m1_results = run_gru_spec(df_train, df_val, df_test, target, FEAT_M1_FULL, "M1-GRU")

    m3_results = run_gru_spec(df_train, df_val, df_test, target, FEAT_M3, "M3-GRU")

    m4_results = run_gru_spec(df_train, df_val, df_test, target, FEAT_M4, "M4-GRU")

    horizon_results = m1_results + m3_results + m4_results
    print(f"\n{'=' * 80}")
    print(f"GRU SUMMARY — {horizon_label}")
    print(f"{'=' * 80}")
    print_results(horizon_results)
    print(f"Horizon time: {time.perf_counter() - horizon_start:.1f}s")

    for r in horizon_results:
        grand_results.append({**r, "horizon": horizon_label})


print("\n" + "=" * 90)
print("GRU GRAND SUMMARY — ALL HORIZONS")
print("=" * 90)
header = f"{'Horizon':<8} {'Model':<12} {'Port':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}"
print(header)
print("-" * len(header))
for r in grand_results:
    r2_val = r["r2"]
    r2_str = f"{r2_val:>8.3f}" if isinstance(r2_val, float) and not np.isnan(r2_val) else "     N/A"
    print(
        f"{r['horizon']:<8} {r['model']:<12} {r['port']:<30} "
        f"{r['rmse']:>8.2f} {r['mae']:>8.2f} {r2_str} {r['n']:>6}"
    )

total_time = time.perf_counter() - notebook_start
print(f"\nTotal notebook runtime: {total_time:.1f}s ({total_time / 60:.1f}m)")
