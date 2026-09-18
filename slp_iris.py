"""
slp_iris.py
===========
Single Layer Perceptron (SLP) untuk klasifikasi biner data Iris.

    Iris-setosa      -> target 0
    Iris-versicolor  -> target 1

Kode ini adalah replikasi 1:1 dari spreadsheet SLP (Google Sheets / Excel).
Setiap rumus di kode ini memiliki pasangan langsung di spreadsheet:

    kolom N  z          = bias + teta1*X1 + teta2*X2 + teta3*X3 + teta4*X4
    kolom O  g(z)       = 1 / (1 + EXP(-z))                       (sigmoid)
    kolom P  prediksi   = IF(g(z) > 0.5, 1, 0)
    kolom Q  error      = g(z) - target
    kolom R  SSE        = error^2                                 (loss per sampel)
    kolom T  dbias      = 2*(g-y)*(1-g)*g
    kolom U  dteta1     = 2*(g-y)*(1-g)*g * X1     (dst. untuk X2, X3, X4)
    kolom I  bias baru  = bias lama - learning_rate * dbias        (dst. untuk teta)

Catatan penting agar hasilnya sama persis dengan spreadsheet:

  1. Update bobot dilakukan per SAMPEL (stochastic gradient descent), bukan per
     batch. Setiap baris spreadsheet memakai bobot hasil update baris di atasnya.
  2. Loss dan accuracy training dihitung "online", yaitu dari nilai g(z) pada
     setiap baris epoch tersebut (bobot masih berubah di tengah epoch). Ini
     sama dengan AVERAGE(R5:R84) dan AVERAGE(S5:S84) di spreadsheet.
  3. Pada validasi bobot TIDAK di-update. Bobot diambil dari hasil training pada
     akhir epoch yang bersangkutan, lalu dipakai konstan untuk 20 sampel.
  4. Urutan sampel training tetap sama di setiap epoch (tidak diacak), persis
     seperti urutan baris di spreadsheet.

Pemakaian
    python slp_iris.py                    # jalankan dengan konfigurasi default
    python slp_iris.py --epochs 10 --lr 0.05

Keluaran
    slp_history.csv             rekap loss & accuracy per epoch
    chart_accuracy_python.png   grafik accuracy training vs validation
    chart_loss_python.png       grafik loss training vs validation

Dependensi: numpy, matplotlib
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")           # backend non-interaktif, aman dijalankan di server
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Konfigurasi default - harus sama dengan spreadsheet
# --------------------------------------------------------------------------- #
DEFAULT_EPOCHS = 5
DEFAULT_LR = 0.1            # sel K2 (training) dan K4 (validation) di spreadsheet
DEFAULT_INIT_WEIGHT = 0.5   # bias dan teta1..teta4 pada baris pertama

# Pembagian data mengikuti spreadsheet (indeks baris file iris.csv, mulai dari 1):
#   training   : 1-40  (setosa) + 51-90  (versicolor) = 80 sampel
#   validation : 41-50 (setosa) + 91-100 (versicolor) = 20 sampel
TRAIN_ROWS = list(range(1, 41)) + list(range(51, 91))
VAL_ROWS = list(range(41, 51)) + list(range(91, 101))

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_iris(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Baca iris.csv -> matriks fitur X (n, 4) dan vektor target y (n,)."""
    X, y = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            X.append([
                float(row["sepal_length"]),
                float(row["sepal_width"]),
                float(row["petal_length"]),
                float(row["petal_width"]),
            ])
            y.append(0.0 if "setosa" in row["species"].strip().lower() else 1.0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def split_data(X, y):
    tr = [i - 1 for i in TRAIN_ROWS]     # csv baris 1 -> indeks 0
    va = [i - 1 for i in VAL_ROWS]
    return X[tr], y[tr], X[va], y[va]


# --------------------------------------------------------------------------- #
# Komponen SLP
# --------------------------------------------------------------------------- #
def sigmoid(z):
    """Fungsi aktivasi g(z) = 1 / (1 + e^-z)  -- kolom O spreadsheet."""
    return 1.0 / (1.0 + np.exp(-z))


def forward(w, x):
    """z = bias + teta . x, lalu g(z).  Bobot w = [bias, teta1..teta4]."""
    z = w[0] + np.dot(w[1:], x)
    return z, sigmoid(z)


def gradient(g, target, x):
    """Turunan loss (g - y)^2 terhadap [bias, teta1..teta4].

    d/dbias  = 2*(g-y) * g*(1-g)          -> kolom T
    d/dteta_i= 2*(g-y) * g*(1-g) * x_i    -> kolom U..X
    """
    common = 2.0 * (g - target) * (1.0 - g) * g
    return np.concatenate(([common], common * x))


def evaluate(w, X, y):
    """Hitung loss (MSE) dan accuracy dengan bobot tetap -- dipakai untuk validasi."""
    z = w[0] + X @ w[1:]
    g = sigmoid(z)
    pred = (g > 0.5).astype(float)
    loss = float(np.mean((g - y) ** 2))
    acc = float(np.mean(pred == y))
    return loss, acc


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(X_tr, y_tr, X_va, y_va, epochs, lr, init_w):
    w = np.full(5, init_w, dtype=float)      # [bias, teta1, teta2, teta3, teta4]
    history = []

    for epoch in range(1, epochs + 1):
        sse_list, correct_list = [], []

        # ---- satu epoch: telusuri seluruh sampel training satu per satu ------
        for i in range(len(X_tr)):
            x, target = X_tr[i], y_tr[i]

            z, g = forward(w, x)                      # kolom N dan O
            pred = 1.0 if g > 0.5 else 0.0            # kolom P
            error = g - target                        # kolom Q

            sse_list.append(error ** 2)               # kolom R
            correct_list.append(1.0 if pred == target else 0.0)   # kolom S

            w = w - lr * gradient(g, target, x)       # kolom I..M baris berikutnya

        train_loss = float(np.mean(sse_list))
        train_acc = float(np.mean(correct_list))

        # ---- validasi memakai bobot akhir epoch, tanpa update bobot ----------
        val_loss, val_acc = evaluate(w, X_va, y_va)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "weights": w.copy(),
        })

    return w, history


# --------------------------------------------------------------------------- #
# Keluaran
# --------------------------------------------------------------------------- #
def print_report(history, final_w):
    print()
    print("=" * 78)
    print("SINGLE LAYER PERCEPTRON - IRIS (Setosa = 0, Versicolor = 1)")
    print("=" * 78)
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>11} "
          f"{'Val Loss':>12} {'Val Acc':>10}")
    print("-" * 78)
    for h in history:
        print(f"{h['epoch']:>6} {h['train_loss']:>12.6f} {h['train_acc']:>10.2%} "
              f"{h['val_loss']:>12.6f} {h['val_acc']:>9.2%}")
    print("-" * 78)
    names = ["bias", "teta1", "teta2", "teta3", "teta4"]
    print("Bobot akhir : " + ", ".join(f"{n}={v:.6f}" for n, v in zip(names, final_w)))
    print("=" * 78)


def save_history_csv(history, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                    "bias", "teta1", "teta2", "teta3", "teta4"])
        for h in history:
            w.writerow([h["epoch"],
                        f"{h['train_loss']:.10f}", f"{h['train_acc']:.10f}",
                        f"{h['val_loss']:.10f}", f"{h['val_acc']:.10f}",
                        *[f"{v:.10f}" for v in h["weights"]]])
    print(f"Tersimpan: {path}")


def _style(ax, title, xlabel, ylabel, epochs):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(frameon=False, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_accuracy(history, path):
    ep = [h["epoch"] for h in history]
    tr = [h["train_acc"] * 100 for h in history]
    va = [h["val_acc"] * 100 for h in history]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    ax.plot(ep, tr, marker="o", linewidth=2.2, color="#1a73e8", label="Training")
    ax.plot(ep, va, marker="s", linewidth=2.2, color="#e8710a", label="Validation")
    for x, y in zip(ep, tr):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="#1a73e8")
    for x, y in zip(ep, va):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="#e8710a")
    ax.set_ylim(0, 108)
    _style(ax, "Accuracy Chart - Training vs Validation (Python)",
           "Epoch", "Accuracy (%)", ep)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Tersimpan: {path}")


def plot_loss(history, path):
    ep = [h["epoch"] for h in history]
    tr = [h["train_loss"] for h in history]
    va = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    ax.plot(ep, tr, marker="o", linewidth=2.2, color="#d93025", label="Training")
    ax.plot(ep, va, marker="s", linewidth=2.2, color="#188038", label="Validation")
    for x, y in zip(ep, tr):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="#d93025")
    for x, y in zip(ep, va):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="#188038")
    ax.set_ylim(bottom=0)
    _style(ax, "Loss / Error Chart - Training vs Validation (Python)",
           "Epoch", "Loss (Mean Squared Error)", ep)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Tersimpan: {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="SLP Iris - Setosa vs Versicolor")
    ap.add_argument("--data", default=os.path.join(HERE, "iris.csv"))
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--init-weight", type=float, default=DEFAULT_INIT_WEIGHT)
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()

    X, y = load_iris(args.data)
    X_tr, y_tr, X_va, y_va = split_data(X, y)

    print(f"Data          : {len(X)} sampel")
    print(f"Training      : {len(X_tr)} sampel "
          f"({int((y_tr == 0).sum())} setosa, {int((y_tr == 1).sum())} versicolor)")
    print(f"Validation    : {len(X_va)} sampel "
          f"({int((y_va == 0).sum())} setosa, {int((y_va == 1).sum())} versicolor)")
    print(f"Learning rate : {args.lr}   Epoch: {args.epochs}   "
          f"Bobot awal: {args.init_weight}")

    final_w, history = train(X_tr, y_tr, X_va, y_va,
                             args.epochs, args.lr, args.init_weight)

    print_report(history, final_w)
    os.makedirs(args.outdir, exist_ok=True)
    save_history_csv(history, os.path.join(args.outdir, "slp_history.csv"))
    plot_accuracy(history, os.path.join(args.outdir, "chart_accuracy_python.png"))
    plot_loss(history, os.path.join(args.outdir, "chart_loss_python.png"))


if __name__ == "__main__":
    main()
