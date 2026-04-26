"""
src/evaluation/metrics.py
-------------------------
Computes official evaluation metrics for the two-tower recommendation model:

    1. nDCG@10  -- position-sensitive ranking quality (primary metric)
    2. Recall@50 -- retrieval coverage
    3. A/B simulation -- model vs random baseline comparison

What is nDCG@10?
    nDCG = Normalized Discounted Cumulative Gain

    Imagine you recommended 10 workflows to a user. Some are relevant,
    some are not. nDCG answers: "How good is this ranked list?"

    It rewards TWO things:
      - Finding relevant items at all (cumulative gain)
      - Finding them EARLY in the list (discounting)

    A relevant item at rank 1 is worth 1/log2(2) = 1.0
    A relevant item at rank 2 is worth 1/log2(3) = 0.63
    A relevant item at rank 10 is worth 1/log2(11) = 0.29

    DCG@k  = sum of (relevance / log2(rank + 1)) for top-k items
    IDCG@k = DCG if we had a PERFECT ranking (all relevant items first)
    nDCG@k = DCG / IDCG  (normalized to [0, 1])

    nDCG = 1.0 --> perfect ranking (all relevant items at the top)
    nDCG = 0.0 --> no relevant items in top-k at all

What is Recall@50?
    Of all workflows that are relevant to this user,
    what fraction did we find in our top-50 recommendations?

    Recall@50 = |relevant found in top-50| / |all relevant|

    Recall = 1.0 --> found every relevant item
    Recall = 0.0 --> missed every relevant item

Run with:
    python src/evaluation/metrics.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import faiss

# Windows console fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import our modules
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, MODEL_DIR)
from towers import TwoTowerModel

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH        = os.path.join(PROJECT_ROOT, "data", "models", "best_model.pt")
FAISS_INDEX_PATH  = os.path.join(PROJECT_ROOT, "data", "embeddings", "faiss_index.bin")
ITEM_EMB_PATH     = os.path.join(PROJECT_ROOT, "data", "embeddings", "item_embeddings_128.npy")
WORKFLOW_IDS_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings", "workflow_ids.json")
CATALOG_PATH      = os.path.join(PROJECT_ROOT, "data", "catalog", "workflows.json")
FEATURES_TEST     = os.path.join(PROJECT_ROOT, "data", "processed", "features_test.csv")
INTERACTIONS_TEST = os.path.join(PROJECT_ROOT, "data", "processed", "interactions_test.csv")
RESULTS_PATH      = os.path.join(PROJECT_ROOT, "data", "models", "evaluation_results.json")

FEATURE_COLS = [
    "issue_count_log_norm", "pr_count_log_norm", "push_count_log_norm",
    "repo_diversity_log_norm", "label_diversity_log_norm",
    "sprint_velocity_log_norm", "activity_score_log_norm",
    "collab_density_norm", "recency_weight_norm",
    "pr_ratio_norm", "issue_ratio_norm",
]


# ══════════════════════════════════════════════════════════════════════
#  Metric Functions
# ══════════════════════════════════════════════════════════════════════

def compute_ndcg_at_k(recommended_ids, relevant_ids, k=10):
    """
    Compute Normalized Discounted Cumulative Gain at rank k.

    Parameters:
        recommended_ids : list of str -- ordered workflow IDs (model's ranking)
        relevant_ids    : set of str  -- workflow IDs that are actually relevant
        k               : int         -- only consider top-k of the ranked list

    Returns:
        float in [0, 1] -- 1.0 = perfect ranking, 0.0 = no relevant in top-k
    """
    # Truncate to top-k
    top_k = recommended_ids[:k]

    # DCG: sum of (relevance / log2(rank + 1))
    # relevance is binary: 1 if item is relevant, 0 otherwise
    dcg = 0.0
    for i, wid in enumerate(top_k):
        if wid in relevant_ids:
            # rank is 1-indexed, so position i has rank (i+1)
            # discount factor: 1 / log2(rank + 1)
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed

    # IDCG: best possible DCG if all relevant items were ranked first
    # Take min(k, num_relevant) because we can't have more hits than
    # either the number of relevant items or k
    n_relevant_in_k = min(len(relevant_ids), k)
    idcg = 0.0
    for i in range(n_relevant_in_k):
        idcg += 1.0 / np.log2(i + 2)

    # Avoid division by zero (no relevant items exist)
    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_recall_at_k(recommended_ids, relevant_ids, k=50):
    """
    Compute Recall at rank k.

    What fraction of all relevant items appear in the top-k recommendations?

    Parameters:
        recommended_ids : list of str -- ordered workflow IDs
        relevant_ids    : set of str  -- actually relevant workflow IDs
        k               : int         -- cutoff

    Returns:
        float in [0, 1] -- 1.0 = found all relevant items, 0.0 = found none
    """
    if len(relevant_ids) == 0:
        return 0.0

    top_k = set(recommended_ids[:k])
    found = top_k.intersection(relevant_ids)

    return len(found) / len(relevant_ids)


# ══════════════════════════════════════════════════════════════════════
#  Ground Truth: what is "relevant" for a user?
# ══════════════════════════════════════════════════════════════════════

def get_relevant_workflows(user_id, interactions_df, user_preferred_cats, catalog):
    """
    Determine which workflows are relevant for a given user.

    A workflow is relevant if:
      1. It appears in the user's test interactions with label=1  (explicit signal)
      2. Its category matches any of the user's preferred_categories  (category signal)

    We use both signals because individual users may have few positive
    interactions in the test set (sparse data problem).

    Parameters:
        user_id           : str, e.g. "u02725"
        interactions_df   : DataFrame with columns [user_id, workflow_id, label]
        user_preferred_cats : set of str, e.g. {"Agile/Scrum", "Release Management"}
        catalog           : dict mapping workflow_id -> metadata dict

    Returns:
        set of str -- relevant workflow IDs
    """
    relevant = set()

    # Source 1: explicit positive interactions in test set
    user_interactions = interactions_df[interactions_df["user_id"] == user_id]
    positive_ids = set(user_interactions[user_interactions["label"] == 1]["workflow_id"])
    relevant.update(positive_ids)

    # Source 2: workflows whose category matches user's preferred categories
    for wid, wf in catalog.items():
        if wf["category"] in user_preferred_cats:
            relevant.add(wid)

    return relevant


# ══════════════════════════════════════════════════════════════════════
#  Model Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_model(model, index, workflow_ids, test_users_df,
                   interactions_df, catalog):
    """
    Evaluate the model on all test users.

    For each test user:
        1. Encode their features through UserTower
        2. FAISS search for top-50 nearest workflows
        3. Compute nDCG@10 and Recall@50 against ground truth

    Returns:
        mean_ndcg   : float -- average nDCG@10 across all test users
        mean_recall : float -- average Recall@50 across all test users
        per_user    : list of dicts with per-user results
    """
    model.eval()
    ndcg_scores = []
    recall_scores = []
    per_user = []

    for _, row in test_users_df.iterrows():
        uid = row["user_id"]
        preferred_cats = set(c.strip() for c in row["preferred_categories"].split(","))

        # Get ground truth relevant workflows
        relevant = get_relevant_workflows(uid, interactions_df, preferred_cats, catalog)

        if len(relevant) == 0:
            continue  # skip users with no relevant items

        # Encode user features
        user_feat = row[FEATURE_COLS].values.astype(np.float32)
        with torch.no_grad():
            user_tensor = torch.tensor(user_feat, dtype=torch.float32).unsqueeze(0)
            user_emb = model.get_user_embedding(user_tensor).numpy().astype(np.float32)

        # FAISS search: top-50
        scores, indices = index.search(user_emb, 50)
        recommended = [workflow_ids[idx] for idx in indices[0]]

        # Compute metrics
        ndcg = compute_ndcg_at_k(recommended, relevant, k=10)
        recall = compute_recall_at_k(recommended, relevant, k=50)

        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
        per_user.append({
            "user_id": uid,
            "archetype": row["archetype"],
            "n_relevant": len(relevant),
            "ndcg_at_10": round(ndcg, 4),
            "recall_at_50": round(recall, 4),
        })

    mean_ndcg = np.mean(ndcg_scores)
    mean_recall = np.mean(recall_scores)

    return mean_ndcg, mean_recall, per_user


def random_baseline(workflow_ids, test_users_df, interactions_df, catalog, n_trials=5):
    """
    Compute metrics for a random recommender (control group).

    Randomly shuffles all 500 workflows and uses that as the ranking.
    Averages over n_trials to reduce variance.

    Returns:
        mean_ndcg, mean_recall : floats
    """
    rng = np.random.RandomState(42)
    all_ndcg = []
    all_recall = []

    for trial in range(n_trials):
        for _, row in test_users_df.iterrows():
            uid = row["user_id"]
            preferred_cats = set(c.strip() for c in row["preferred_categories"].split(","))
            relevant = get_relevant_workflows(uid, interactions_df, preferred_cats, catalog)

            if len(relevant) == 0:
                continue

            # Random ranking: shuffle all workflow IDs
            shuffled = list(workflow_ids)
            rng.shuffle(shuffled)

            ndcg = compute_ndcg_at_k(shuffled, relevant, k=10)
            recall = compute_recall_at_k(shuffled, relevant, k=50)

            all_ndcg.append(ndcg)
            all_recall.append(recall)

    return np.mean(all_ndcg), np.mean(all_recall)


def run_ab_simulation(model, index, workflow_ids, test_users_df,
                      interactions_df, catalog):
    """
    A/B test simulation: model (treatment) vs random (control).

    Returns:
        dict with control and treatment metrics + improvement ratios
    """
    print("    Running treatment group (two-tower model)...")
    t_ndcg, t_recall, per_user = evaluate_model(
        model, index, workflow_ids, test_users_df, interactions_df, catalog
    )

    print("    Running control group (random baseline, 5 trials)...")
    c_ndcg, c_recall = random_baseline(
        workflow_ids, test_users_df, interactions_df, catalog
    )

    results = {
        "control": {
            "method": "random_shuffle",
            "ndcg_at_10": round(c_ndcg, 4),
            "recall_at_50": round(c_recall, 4),
        },
        "treatment": {
            "method": "two_tower_model",
            "ndcg_at_10": round(t_ndcg, 4),
            "recall_at_50": round(t_recall, 4),
        },
        "improvement": {
            "ndcg_ratio": round(t_ndcg / max(c_ndcg, 1e-6), 2),
            "recall_ratio": round(t_recall / max(c_recall, 1e-6), 2),
        },
        "per_user_results": per_user,
    }

    return results


# ══════════════════════════════════════════════════════════════════════
#  Self-Test
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Evaluation Metrics -- nDCG@10, Recall@50, A/B Simulation")
    print("=" * 70)

    # ── 1. Load everything ────────────────────────────────────────────
    print("\n[1] Loading resources...")

    # Model
    model = TwoTowerModel(
        user_input_dim=11, item_input_dim=384,
        hidden_dim=256, output_dim=128, dropout=0.2,
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    print(f"    Model loaded from: {MODEL_PATH}")

    # FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"    FAISS index loaded: {index.ntotal} vectors")

    # Workflow IDs (maps FAISS row -> workflow_id)
    with open(WORKFLOW_IDS_PATH, "r") as f:
        workflow_ids = json.load(f)

    # Catalog (workflow metadata)
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        workflows_list = json.load(f)
    catalog = {wf["workflow_id"]: wf for wf in workflows_list}
    print(f"    Catalog loaded: {len(catalog)} workflows")

    # Test users
    test_users_df = pd.read_csv(FEATURES_TEST)
    print(f"    Test users loaded: {len(test_users_df)} users")

    # Test interactions
    interactions_df = pd.read_csv(INTERACTIONS_TEST)
    print(f"    Test interactions loaded: {len(interactions_df):,} rows")

    # Quick stats on test interactions
    n_pos = (interactions_df["label"] == 1).sum()
    n_neg = (interactions_df["label"] == 0).sum()
    print(f"    Positive: {n_pos:,}  Negative: {n_neg:,}  "
          f"Ratio: 1:{n_neg // max(n_pos, 1)}")

    # ── 2. Run full evaluation ────────────────────────────────────────
    print("\n[2] Evaluating model on test set...")
    start = time.time()

    mean_ndcg, mean_recall, per_user = evaluate_model(
        model, index, workflow_ids, test_users_df, interactions_df, catalog
    )

    eval_time = time.time() - start
    print(f"    Evaluated {len(per_user)} users in {eval_time:.1f}s")
    print(f"    Mean nDCG@10:   {mean_ndcg:.4f}")
    print(f"    Mean Recall@50: {mean_recall:.4f}")

    # ── 3. Per-archetype breakdown ────────────────────────────────────
    print("\n[3] Per-archetype breakdown:")
    print(f"    {'Archetype':<25} | {'nDCG@10':>8} | {'Recall@50':>9} | {'Users':>5}")
    print(f"    {'-'*25}-+-{'-'*8}-+-{'-'*9}-+-{'-'*5}")

    # Group per_user results by archetype
    archetype_groups = {}
    for pu in per_user:
        arch = pu["archetype"]
        if arch not in archetype_groups:
            archetype_groups[arch] = {"ndcg": [], "recall": []}
        archetype_groups[arch]["ndcg"].append(pu["ndcg_at_10"])
        archetype_groups[arch]["recall"].append(pu["recall_at_50"])

    for arch in sorted(archetype_groups.keys()):
        g = archetype_groups[arch]
        print(f"    {arch:<25} | {np.mean(g['ndcg']):>8.4f} | "
              f"{np.mean(g['recall']):>9.4f} | {len(g['ndcg']):>5}")

    # ── 4. A/B simulation ─────────────────────────────────────────────
    print("\n[4] A/B Simulation (model vs random)...")
    ab_results = run_ab_simulation(
        model, index, workflow_ids, test_users_df, interactions_df, catalog
    )

    ctrl = ab_results["control"]
    treat = ab_results["treatment"]
    imp = ab_results["improvement"]

    print(f"\n    {'Metric':<15} | {'Random':>8} | {'Model':>8} | {'Ratio':>6}")
    print(f"    {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    print(f"    {'nDCG@10':<15} | {ctrl['ndcg_at_10']:>8.4f} | "
          f"{treat['ndcg_at_10']:>8.4f} | {imp['ndcg_ratio']:>5.1f}x")
    print(f"    {'Recall@50':<15} | {ctrl['recall_at_50']:>8.4f} | "
          f"{treat['recall_at_50']:>8.4f} | {imp['recall_ratio']:>5.1f}x")

    # ── 5. Target check ──────────────────────────────────────────────
    print(f"\n[5] Target check:")
    ndcg_pass = treat["ndcg_at_10"] >= 0.35
    recall_pass = treat["recall_at_50"] >= 0.55
    print(f"    nDCG@10  >= 0.35:  {treat['ndcg_at_10']:.4f}  "
          f"{'PASS' if ndcg_pass else 'BELOW TARGET (but check ratio)'}")
    print(f"    Recall@50 >= 0.55: {treat['recall_at_50']:.4f}  "
          f"{'PASS' if recall_pass else 'BELOW TARGET (but check ratio)'}")
    print(f"    Improvement over random: "
          f"nDCG {imp['ndcg_ratio']:.1f}x, Recall {imp['recall_ratio']:.1f}x")

    # ── 6. Save results ──────────────────────────────────────────────
    print(f"\n[6] Saving results...")

    # Remove per_user from saved results to keep file small
    save_data = {
        "model_metrics": {
            "ndcg_at_10": round(mean_ndcg, 4),
            "recall_at_50": round(mean_recall, 4),
        },
        "ab_simulation": {
            "control": ctrl,
            "treatment": treat,
            "improvement": imp,
        },
        "per_archetype": {
            arch: {
                "ndcg_at_10": round(np.mean(g["ndcg"]), 4),
                "recall_at_50": round(np.mean(g["recall"]), 4),
                "n_users": len(g["ndcg"]),
            }
            for arch, g in archetype_groups.items()
        },
        "config": {
            "n_test_users": len(per_user),
            "n_workflows": index.ntotal,
            "embedding_dim": 128,
        },
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"    Saved to: {RESULTS_PATH}")

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Evaluation complete!")
    print(f"  nDCG@10:   {mean_ndcg:.4f}  (random: {ctrl['ndcg_at_10']:.4f}, "
          f"{imp['ndcg_ratio']:.1f}x better)")
    print(f"  Recall@50: {mean_recall:.4f}  (random: {ctrl['recall_at_50']:.4f}, "
          f"{imp['recall_ratio']:.1f}x better)")
    print()
    print("  Next step: python app/streamlit_app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
