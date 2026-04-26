"""
src/pipeline/features.py
─────────────────────────
Reads raw user data from SQLite, runs SQL feature engineering,
normalizes all features to [0, 1], and saves train/val/test splits
ready for the model.

This file is the bridge between raw data and the neural network.
Everything the model sees comes from here.

Run with:
    python src/pipeline/features.py
Inputs:
    data/processed/users.db
    data/catalog/workflows.json
Outputs:
    data/processed/features_train.csv
    data/processed/features_val.csv
    data/processed/features_test.csv
    data/processed/scaler_stats.json   ← min/max values for un-normalizing later
    data/processed/interactions.csv    ← (user, workflow) pairs for training
"""

import sqlite3
import json
import csv
import os
import random
import math


# ── Which features go into the user tower ──────────────────────────────────────
# Order matters — the model sees them in this exact order every time.
# Changing order after training = wrong predictions. Lock this in now.

USER_FEATURES = [
    "issue_count",
    "pr_count",
    "push_count",
    "repo_diversity",
    "collab_density",
    "label_diversity",
    "sprint_velocity",
    "recency_weight",
]

# ── Category → workflow index mapping ──────────────────────────────────────────
CATEGORIES = [
    "Agile/Scrum", "Kanban", "Bug Triage", "Release Management",
    "Design Sprint", "DevOps/CI-CD", "Open Source", "Research",
    "Incident Response", "Customer Support",
]


def load_users_from_db(db_path):
    """
    Pull all users from SQLite and return as a list of dicts.
    This is the SQL feature engineering step.
    In production at Atlassian, this query would run on Spark/BigQuery
    over millions of rows. Here it runs on 10k rows locally.
    """
    conn   = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row   # makes rows accessible as dicts
    cursor = conn.cursor()

    # This SQL query IS your feature engineering.
    # Each computed column is one feature going into the user tower.
    # Note: CAST(...) converts integer division to float in SQLite.
    cursor.execute("""
        SELECT
            user_id,
            archetype,
            preferred_categories,

            -- Raw counts (will be log-scaled then normalized below)
            issue_count,
            pr_count,
            push_count,
            repo_diversity,
            label_diversity,
            sprint_velocity,

            -- Already a ratio [0,1], normalize directly
            collab_density,
            recency_weight,

            -- Derived features (computed from raw columns)
            -- pr_ratio: how code-review-heavy is this developer?
            CAST(pr_count AS REAL) / (issue_count + pr_count + push_count + 1) AS pr_ratio,

            -- issue_ratio: how ticket-driven is this developer?
            CAST(issue_count AS REAL) / (issue_count + pr_count + push_count + 1) AS issue_ratio,

            -- activity_score: overall volume of GitHub activity
            (issue_count + pr_count + push_count) AS activity_score

        FROM users
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def log_scale(value, epsilon=1.0):
    """
    Apply log(1 + x) transformation to count features.

    WHY log scaling?
    Count features (issues, PRs, pushes) are right-skewed:
    most developers have small counts, but a few have enormous counts.
    Without log scaling, those outliers dominate normalization.

    log(1 + x) compresses large values while preserving order:
        log(1+0)  = 0
        log(1+10) = 2.4
        log(1+100)= 4.6     (not 10x larger, just ~2x)
        log(1+1000)= 6.9    (not 100x larger, just ~3x)

    This is standard practice in RecSys feature engineering.
    Google, Netflix, and Spotify all do this on count features.
    """
    return math.log(1.0 + value + epsilon)


def compute_min_max(users, feature_names):
    """
    Compute min and max for each feature across all users.
    These values are used for min-max normalization.

    Min-max normalization formula:
        normalized = (x - min) / (max - min)
    Result is always in [0, 1].

    We save these stats to scaler_stats.json so we can apply
    the SAME normalization at inference time (when a new user arrives).
    Applying different normalization at inference = wrong predictions.
    """
    stats = {}
    for feat in feature_names:
        values = [u[feat] for u in users]
        stats[feat] = {
            "min": min(values),
            "max": max(values),
        }
    return stats


def normalize(value, feat_stats):
    """Apply min-max normalization. Clamps to [0,1] to handle edge cases."""
    min_val = feat_stats["min"]
    max_val = feat_stats["max"]
    if max_val == min_val:
        return 0.0   # constant feature — all values identical
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))   # clamp


def generate_interactions(users, workflows, n_interactions=50_000):
    """
    Generate (user_id, workflow_id, label) training pairs.

    label = 1  →  this workflow is RELEVANT to this user
    label = 0  →  this workflow is NOT relevant (negative sample)

    How we decide relevance:
      Each user archetype has preferred_categories.
      Workflows in those categories → positive (label=1)
      Workflows in other categories → negative (label=0)

    This simulates what "click" or "adoption" data would tell us
    in a real system — users implicitly signal relevance by
    adopting workflows that match their working style.

    Ratio: 1 positive : 4 negatives
    Why? In real recommender systems, negatives vastly outnumber
    positives. Training on 1:1 makes the model overconfident.
    1:4 is a standard ratio in industry RecSys papers.
    """
    random.seed(42)

    # Build category → workflow lookup
    cat_to_workflows = {cat: [] for cat in CATEGORIES}
    for wf in workflows:
        cat_to_workflows[wf["category"]].append(wf["workflow_id"])

    interactions = []

    for user in users:
        preferred = set(user["preferred_categories"].split(","))
        other_cats = [c for c in CATEGORIES if c not in preferred]

        # How many interactions to generate per user
        # Active users get more interactions (proportional to activity)
        activity = user["issue_count"] + user["pr_count"]
        n_user   = max(3, min(10, activity // 20))

        for _ in range(n_user):
            # ── Positive: pick workflow from preferred category ──────────────
            pos_cat  = random.choice(list(preferred))
            pos_wfs  = cat_to_workflows.get(pos_cat, [])
            if pos_wfs:
                pos_wf = random.choice(pos_wfs)
                interactions.append({
                    "user_id":     user["user_id"],
                    "workflow_id": pos_wf,
                    "label":       1,
                })

            # ── Negatives: 4 workflows from non-preferred categories ─────────
            for _ in range(4):
                if other_cats:
                    neg_cat = random.choice(other_cats)
                    neg_wfs = cat_to_workflows.get(neg_cat, [])
                    if neg_wfs:
                        neg_wf = random.choice(neg_wfs)
                        interactions.append({
                            "user_id":     user["user_id"],
                            "workflow_id": neg_wf,
                            "label":       0,
                        })

    # Shuffle so positives and negatives are interleaved
    random.shuffle(interactions)
    return interactions


def main():
    random.seed(42)

    DB_PATH       = "data/processed/users.db"
    CATALOG_PATH  = "data/catalog/workflows.json"
    OUTPUT_DIR    = "data/processed"

    print("=" * 52)
    print("  Feature Engineering Pipeline")
    print("=" * 52)

    # ── Step 1: Load raw data ───────────────────────────────────────────────────
    print("\n[1/5] Loading users from SQLite...")
    users = load_users_from_db(DB_PATH)
    print(f"      Loaded {len(users):,} users")

    print("\n[2/5] Loading workflow catalog...")
    with open(CATALOG_PATH, encoding="utf-8") as f:
        workflows = json.load(f)
    print(f"      Loaded {len(workflows):,} workflows")

    # ── Step 2: Log-scale count features ───────────────────────────────────────
    print("\n[3/5] Applying log scaling to count features...")

    COUNT_FEATURES = [
        "issue_count", "pr_count", "push_count",
        "repo_diversity", "label_diversity", "sprint_velocity",
        "activity_score",
    ]
    RATIO_FEATURES = [
        "collab_density", "recency_weight", "pr_ratio", "issue_ratio",
    ]

    for user in users:
        for feat in COUNT_FEATURES:
            if feat in user:
                user[f"{feat}_log"] = log_scale(user[feat])

    # Final feature set going into the model
    # We use log-scaled versions of count features + raw ratios
    MODEL_FEATURES = (
        [f"{f}_log" for f in COUNT_FEATURES] +
        RATIO_FEATURES
    )
    print(f"      {len(MODEL_FEATURES)} features total: {MODEL_FEATURES}")

    # ── Step 3: Compute normalization stats on ALL data ─────────────────────────
    print("\n[4/5] Computing normalization statistics...")
    stats = compute_min_max(users, MODEL_FEATURES)

    # Apply normalization to every user
    for user in users:
        for feat in MODEL_FEATURES:
            raw_val       = user[feat]
            user[f"{feat}_norm"] = normalize(raw_val, stats[feat])

    NORM_FEATURES = [f"{f}_norm" for f in MODEL_FEATURES]

    # Show what normalization did to a few features
    sample = users[0]
    print(f"\n      Example normalization (first user: {sample['user_id']}):")
    print(f"      {'Feature':<28} {'Raw':>10}  {'Log':>8}  {'Norm':>8}")
    print(f"      {'─'*58}")
    for feat in COUNT_FEATURES[:3]:
        raw  = sample[feat]
        log  = sample[f"{feat}_log"]
        norm = sample[f"{feat}_log_norm"]
        print(f"      {feat:<28} {raw:>10.1f}  {log:>8.3f}  {norm:>8.3f}")

    # ── Step 4: Train / val / test split ────────────────────────────────────────
    # 80% train, 10% val, 10% test — industry standard split
    random.shuffle(users)
    n      = len(users)
    n_train= int(n * 0.80)
    n_val  = int(n * 0.10)

    train_users = users[:n_train]
    val_users   = users[n_train : n_train + n_val]
    test_users  = users[n_train + n_val:]

    print(f"\n      Split: {len(train_users):,} train / {len(val_users):,} val / {len(test_users):,} test")

    # ── Step 5: Save everything ─────────────────────────────────────────────────
    print("\n[5/5] Saving outputs...")

    def save_csv(data, path, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in data:
                writer.writerow({k: row[k] for k in fields})

    base_fields = ["user_id", "archetype", "preferred_categories"] + NORM_FEATURES

    save_csv(train_users, f"{OUTPUT_DIR}/features_train.csv", base_fields)
    save_csv(val_users,   f"{OUTPUT_DIR}/features_val.csv",   base_fields)
    save_csv(test_users,  f"{OUTPUT_DIR}/features_test.csv",  base_fields)
    print(f"      Saved train/val/test CSVs")

    # Save scaler stats — CRITICAL for inference later
    scaler_path = f"{OUTPUT_DIR}/scaler_stats.json"
    with open(scaler_path, "w") as f:
        json.dump({
            "model_features":  MODEL_FEATURES,
            "norm_features":   NORM_FEATURES,
            "stats":           stats,
        }, f, indent=2)
    print(f"      Saved scaler stats → {scaler_path}")

    # ── Step 6: Generate interaction pairs ──────────────────────────────────────
    print("\n      Generating (user, workflow, label) interaction pairs...")
    interactions = generate_interactions(users, workflows)

    # Split interactions by user split
    train_ids = {u["user_id"] for u in train_users}
    val_ids   = {u["user_id"] for u in val_users}
    test_ids  = {u["user_id"] for u in test_users}

    train_ints = [i for i in interactions if i["user_id"] in train_ids]
    val_ints   = [i for i in interactions if i["user_id"] in val_ids]
    test_ints  = [i for i in interactions if i["user_id"] in test_ids]

    int_fields = ["user_id", "workflow_id", "label"]
    save_csv(train_ints, f"{OUTPUT_DIR}/interactions_train.csv", int_fields)
    save_csv(val_ints,   f"{OUTPUT_DIR}/interactions_val.csv",   int_fields)
    save_csv(test_ints,  f"{OUTPUT_DIR}/interactions_test.csv",  int_fields)

    # Count positives vs negatives
    pos = sum(1 for i in interactions if i["label"] == 1)
    neg = sum(1 for i in interactions if i["label"] == 0)
    print(f"      Total interactions: {len(interactions):,}")
    print(f"      Positives: {pos:,} ({pos/len(interactions)*100:.1f}%)")
    print(f"      Negatives: {neg:,} ({neg/len(interactions)*100:.1f}%)")
    print(f"      Ratio: 1:{neg//max(pos,1)} (pos:neg)")

    print()
    print("=" * 52)
    print("  Pipeline complete. Files saved to data/processed/")
    print()
    print("  Your data/processed/ folder now contains:")
    print("    features_train.csv       → model training input")
    print("    features_val.csv         → tune hyperparameters")
    print("    features_test.csv        → final evaluation only")
    print("    scaler_stats.json        → normalization params")
    print("    interactions_train.csv   → (user, workflow, label)")
    print("    interactions_val.csv")
    print("    interactions_test.csv")
    print()
    print("  Next step: python src/model/towers.py")
    print("=" * 52)


if __name__ == "__main__":
    main()