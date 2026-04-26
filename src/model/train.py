"""
src/model/train.py
------------------
Full training loop for the two-tower recommendation model.

Pipeline:
    1. Load train/val datasets via InteractionDataset
    2. Train with BCEWithLogitsLoss + temperature scaling + Adam
    3. Each epoch: print train_loss, val_loss, pos/neg score separation
    4. Save best model to data/models/best_model.pt
    5. Save training log to data/models/training_log.json
    6. Early stopping if val loss doesn't improve for 3 epochs
    7. Final test set evaluation

Loss function choice -- BCEWithLogitsLoss with temperature:
    Our model outputs cosine similarity in [-1, +1].
    BCEWithLogitsLoss internally does sigmoid(input) then BCE.
    Problem: sigmoid([-1, +1]) = [0.27, 0.73] -- too compressed.
    Fix: divide scores by temperature (0.1) first:
        logits = score / 0.1  -->  maps [-1, +1] to [-10, +10]
        sigmoid([-10, +10]) = [0.00005, 0.99995]  -->  full range
    This is standard in contrastive learning (CLIP, SimCLR).

    Why not plain BCELoss(sigmoid(scores), labels)?
        Numerically unstable: log(sigmoid(x)) can underflow.
        BCEWithLogitsLoss fuses sigmoid + log for stability.

Run with:
    python src/model/train.py           (2-epoch self-test, then full 20 epochs)
    python src/model/train.py --test    (2-epoch self-test only)
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Windows console fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import our modules from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from dataset import InteractionDataset
from towers import TwoTowerModel

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE    = 512     # examples per mini-batch
LEARNING_RATE = 1e-3    # Adam step size
WEIGHT_DECAY  = 1e-5    # L2 regularization on weights
TEMPERATURE   = 0.1     # scales cosine scores before sigmoid
EPOCHS        = 20      # max training epochs
PATIENCE      = 3       # early stopping: stop after this many epochs with no improvement


def train_one_epoch(model, loader, optimizer, loss_fn, temperature):
    """
    One full pass through training data, updating weights each batch.

    Returns:
        avg_loss       : float -- mean loss across all batches
        avg_pos_score  : float -- mean raw cosine score for label=1 pairs
        avg_neg_score  : float -- mean raw cosine score for label=0 pairs
    """
    model.train()  # enable dropout + batchnorm training mode

    total_loss = 0.0
    n_batches = 0
    pos_scores_sum = 0.0  # accumulate positive pair scores
    neg_scores_sum = 0.0  # accumulate negative pair scores
    pos_count = 0
    neg_count = 0

    for user_feat, item_feat, labels in loader:
        # 1. Zero gradients from previous batch
        optimizer.zero_grad()

        # 2. Forward pass: cosine similarity in [-1, +1]
        scores = model(user_feat, item_feat)

        # 3. Temperature scaling: stretch [-1,+1] to [-10,+10]
        logits = scores / temperature

        # 4. Compute loss (BCEWithLogitsLoss = sigmoid + BCE, fused)
        loss = loss_fn(logits, labels)

        # 5. Backward pass: compute gradients
        loss.backward()

        # 6. Update weights
        optimizer.step()

        # Track metrics (detach scores from computation graph first)
        total_loss += loss.item()
        n_batches += 1

        with torch.no_grad():
            pos_mask = labels == 1
            neg_mask = labels == 0
            if pos_mask.sum() > 0:
                pos_scores_sum += scores[pos_mask].sum().item()
                pos_count += pos_mask.sum().item()
            if neg_mask.sum() > 0:
                neg_scores_sum += scores[neg_mask].sum().item()
                neg_count += neg_mask.sum().item()

    avg_loss = total_loss / n_batches
    avg_pos = pos_scores_sum / max(pos_count, 1)
    avg_neg = neg_scores_sum / max(neg_count, 1)

    return avg_loss, avg_pos, avg_neg


@torch.no_grad()
def evaluate(model, loader, loss_fn, temperature):
    """
    Evaluate model on val/test set WITHOUT updating weights.

    Returns:
        avg_loss, avg_pos_score, avg_neg_score : floats
    """
    model.eval()  # disable dropout, use running stats for batchnorm

    total_loss = 0.0
    n_batches = 0
    pos_scores_sum = 0.0
    neg_scores_sum = 0.0
    pos_count = 0
    neg_count = 0

    for user_feat, item_feat, labels in loader:
        scores = model(user_feat, item_feat)
        logits = scores / temperature
        loss = loss_fn(logits, labels)

        total_loss += loss.item()
        n_batches += 1

        pos_mask = labels == 1
        neg_mask = labels == 0
        if pos_mask.sum() > 0:
            pos_scores_sum += scores[pos_mask].sum().item()
            pos_count += pos_mask.sum().item()
        if neg_mask.sum() > 0:
            neg_scores_sum += scores[neg_mask].sum().item()
            neg_count += neg_mask.sum().item()

    avg_loss = total_loss / n_batches
    avg_pos = pos_scores_sum / max(pos_count, 1)
    avg_neg = neg_scores_sum / max(neg_count, 1)

    return avg_loss, avg_pos, avg_neg


def run_training(max_epochs, description="Training"):
    """
    Full training pipeline. Used for both the 2-epoch self-test
    and the full 20-epoch training run.

    Parameters:
        max_epochs  : int -- how many epochs to train
        description : str -- label for print output

    Returns:
        model       : trained TwoTowerModel
        log         : list of dicts (one per epoch)
    """
    print("=" * 70)
    print(f"  Two-Tower Model -- {description} ({max_epochs} epochs)")
    print("=" * 70)

    # ── 1. Load datasets ──────────────────────────────────────────────
    print("\n[1] Loading datasets...")
    train_dataset = InteractionDataset(split="train")
    val_dataset   = InteractionDataset(split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"    Train: {len(train_dataset):,} examples, "
          f"{len(train_loader)} batches/epoch")
    print(f"    Val:   {len(val_dataset):,} examples, "
          f"{len(val_loader)} batches/epoch")

    # ── 2. Create model ───────────────────────────────────────────────
    print("\n[2] Creating model...")
    model = TwoTowerModel(
        user_input_dim=11, item_input_dim=384,
        hidden_dim=256, output_dim=128, dropout=0.2,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params:,}")

    # ── 3. Loss + optimizer + scheduler ───────────────────────────────
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Halve LR if val loss plateaus for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    print(f"    Loss:        BCEWithLogitsLoss (temp={TEMPERATURE})")
    print(f"    Optimizer:   Adam (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"    Early stop:  patience={PATIENCE}")

    # ── 4. Training loop ──────────────────────────────────────────────
    print(f"\n[3] Training for up to {max_epochs} epochs...")
    header = (f"{'Ep':>3} | {'Tr Loss':>7} | {'Val Loss':>8} | "
              f"{'Pos Scr':>7} | {'Neg Scr':>7} | {'Sep':>6} | "
              f"{'LR':>8} | {'Time':>5}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    os.makedirs(MODELS_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    training_log = []
    training_start = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Train one epoch
        tr_loss, tr_pos, tr_neg = train_one_epoch(
            model, train_loader, optimizer, loss_fn, TEMPERATURE
        )

        # Evaluate on validation set
        val_loss, val_pos, val_neg = evaluate(
            model, val_loader, loss_fn, TEMPERATURE
        )

        # Separation = how far apart positive and negative scores are
        separation = val_pos - val_neg

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # Log this epoch
        epoch_log = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_pos_score": round(val_pos, 6),
            "val_neg_score": round(val_neg, 6),
            "separation": round(separation, 6),
            "lr": current_lr,
            "time_sec": round(epoch_time, 1),
        }
        training_log.append(epoch_log)

        # Print with * marking best epoch
        marker = " *" if val_loss < best_val_loss else ""
        print(f"{epoch:>3} | {tr_loss:>7.4f} | {val_loss:>8.4f} | "
              f"{val_pos:>+7.4f} | {val_neg:>+7.4f} | {separation:>6.4f} | "
              f"{current_lr:>8.6f} | {epoch_time:>4.1f}s{marker}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            model_path = os.path.join(MODELS_DIR, "best_model.pt")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n    Early stopping at epoch {epoch} "
                      f"(no val_loss improvement for {PATIENCE} epochs)")
                break

    total_time = time.time() - training_start
    print("-" * len(header))
    print(f"    Done in {total_time:.1f}s ({total_time / 60:.1f} min). "
          f"Best val_loss: {best_val_loss:.4f}")

    # Save training log
    log_path = os.path.join(MODELS_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"    Log saved to: {log_path}")

    return model, training_log


def test_evaluation(model):
    """Load best model and evaluate on the held-out test set."""
    print("\n[4] Test set evaluation...")

    model_path = os.path.join(MODELS_DIR, "best_model.pt")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"    Loaded best model from: {model_path}")

    loss_fn = nn.BCEWithLogitsLoss()
    test_dataset = InteractionDataset(split="test")
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_loss, test_pos, test_neg = evaluate(
        model, test_loader, loss_fn, TEMPERATURE
    )
    separation = test_pos - test_neg

    print(f"    Test Loss:      {test_loss:.4f}")
    print(f"    Test Pos Score: {test_pos:+.4f}")
    print(f"    Test Neg Score: {test_neg:+.4f}")
    print(f"    Test Separation: {separation:.4f}")

    return test_loss, test_pos, test_neg


def main():
    # ── Phase 1: 2-epoch self-test ────────────────────────────────────
    print("PHASE 1: Quick self-test (2 epochs)\n")
    _, test_log = run_training(max_epochs=2, description="Self-Test")

    # Verify loss is decreasing
    loss_1 = test_log[0]["train_loss"]
    loss_2 = test_log[1]["train_loss"]
    sep_1  = test_log[0]["separation"]
    sep_2  = test_log[1]["separation"]

    print(f"\n    Self-test results:")
    print(f"      Train loss: {loss_1:.4f} -> {loss_2:.4f} "
          f"({'decreasing' if loss_2 < loss_1 else 'NOT decreasing'})")
    print(f"      Separation: {sep_1:.4f} -> {sep_2:.4f} "
          f"({'increasing' if sep_2 > sep_1 else 'NOT increasing'})")

    if loss_2 >= loss_1:
        print("\n    WARNING: Loss did not decrease. Check data/model.")
        print("    Continuing anyway (loss sometimes needs more epochs)...")

    # ── Check for --test flag ─────────────────────────────────────────
    if "--test" in sys.argv:
        print("\n    --test flag: skipping full training.")
        print("=" * 70)
        return

    # ── Phase 2: Full training ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PHASE 2: Full training (up to 20 epochs)\n")
    model, full_log = run_training(max_epochs=EPOCHS, description="Full Training")

    # ── Phase 3: Test evaluation ──────────────────────────────────────
    test_loss, test_pos, test_neg = test_evaluation(model)

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Training pipeline complete!")
    print(f"  Model:  {os.path.join(MODELS_DIR, 'best_model.pt')}")
    print(f"  Log:    {os.path.join(MODELS_DIR, 'training_log.json')}")
    print(f"  Test:   loss={test_loss:.4f}  pos={test_pos:+.4f}  "
          f"neg={test_neg:+.4f}  sep={test_pos - test_neg:.4f}")
    print()
    print("  Next step: python src/retrieval/faiss_index.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
