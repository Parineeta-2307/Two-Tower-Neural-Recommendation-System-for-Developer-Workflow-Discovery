"""
src/retrieval/faiss_index.py
----------------------------
Builds a FAISS nearest-neighbor index over workflow embeddings and
provides a query function for real-time recommendations.

Pipeline:
    1. Load the trained TwoTowerModel from data/models/best_model.pt
    2. Load raw 384-dim workflow embeddings (from sentence-transformer)
    3. Pass them through the trained ItemTower -> 500 x 128-dim embeddings
    4. Build a FAISS IndexFlatIP (inner product = cosine similarity)
    5. Save the index + 128-dim embeddings to disk
    6. Query: user features -> top-K most similar workflows

Why FAISS?
    FAISS (Facebook AI Similarity Search) is a library for fast
    nearest-neighbor search in high-dimensional spaces.
    
    IndexFlatIP = exact inner product search (brute force).
    For 500 items this is instant (< 1ms). For millions of items
    you'd use approximate indexes like IndexIVFFlat or IndexHNSW.
    
    "IP" = Inner Product. Since our embeddings are L2-normalized
    (unit vectors), inner product = cosine similarity.

Run with:
    python src/retrieval/faiss_index.py
"""

import os
import sys
import json
import time
import numpy as np
import torch
import faiss          # Facebook AI Similarity Search
import pandas as pd   # for loading user features CSV

# Windows console fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add model directory to path so we can import towers.py
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, MODEL_DIR)
from towers import TwoTowerModel

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Input paths
MODEL_PATH       = os.path.join(PROJECT_ROOT, "data", "models", "best_model.pt")
RAW_EMBED_PATH   = os.path.join(PROJECT_ROOT, "data", "embeddings", "workflow_embeddings.npy")
WORKFLOW_IDS_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings", "workflow_ids.json")
CATALOG_PATH      = os.path.join(PROJECT_ROOT, "data", "catalog", "workflows.json")
FEATURES_TEST     = os.path.join(PROJECT_ROOT, "data", "processed", "features_test.csv")

# Output paths
FAISS_INDEX_PATH  = os.path.join(PROJECT_ROOT, "data", "embeddings", "faiss_index.bin")
ITEM_EMB_128_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings", "item_embeddings_128.npy")

# The 11 feature columns that the UserTower expects (same order as dataset.py)
FEATURE_COLS = [
    "issue_count_log_norm", "pr_count_log_norm", "push_count_log_norm",
    "repo_diversity_log_norm", "label_diversity_log_norm",
    "sprint_velocity_log_norm", "activity_score_log_norm",
    "collab_density_norm", "recency_weight_norm",
    "pr_ratio_norm", "issue_ratio_norm",
]


def build_item_embeddings(model):
    """
    Pass all 500 workflow embeddings (384-dim) through the trained
    ItemTower to get 128-dim embeddings in the shared space.

    Returns:
        item_embeddings : np.ndarray, shape (500, 128), float32
        workflow_ids    : list of str, e.g. ["w0000", "w0001", ...]
    """
    # Load raw sentence-transformer embeddings (500, 384)
    raw_embeddings = np.load(RAW_EMBED_PATH)
    with open(WORKFLOW_IDS_PATH, "r") as f:
        workflow_ids = json.load(f)

    print(f"    Raw embeddings loaded: {raw_embeddings.shape}")

    # Convert to torch tensor and pass through ItemTower
    model.eval()
    with torch.no_grad():
        raw_tensor = torch.tensor(raw_embeddings, dtype=torch.float32)
        # get_item_embedding() runs the ItemTower and L2-normalizes
        item_emb = model.get_item_embedding(raw_tensor)  # (500, 128)

    # Convert back to numpy (FAISS works with numpy arrays)
    item_embeddings = item_emb.numpy().astype(np.float32)

    print(f"    Item embeddings: {item_embeddings.shape} "
          f"(passed through trained ItemTower)")

    # Verify L2 normalization
    norms = np.linalg.norm(item_embeddings, axis=1)
    print(f"    L2 norms: min={norms.min():.6f}, max={norms.max():.6f} "
          f"(should be ~1.0)")

    return item_embeddings, workflow_ids


def build_faiss_index(item_embeddings):
    """
    Build a FAISS IndexFlatIP from the 128-dim item embeddings.

    IndexFlatIP = exact brute-force inner product search.
    For 500 items, this is instant. The index stores all vectors
    and compares every query against every stored vector.

    Since our vectors are L2-normalized:
        inner_product(u, v) = cosine_similarity(u, v)

    Parameters:
        item_embeddings : np.ndarray, shape (N, 128), float32

    Returns:
        index : faiss.IndexFlatIP
    """
    dim = item_embeddings.shape[1]  # 128

    # Create the index: "IP" = Inner Product
    index = faiss.IndexFlatIP(dim)

    # Add all item embeddings to the index
    # FAISS requires float32, contiguous C-order arrays
    index.add(np.ascontiguousarray(item_embeddings, dtype=np.float32))

    print(f"    FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def query_top_k(model, index, user_features, workflow_ids, catalog, k=10,
                preferred_categories=None):
    """
    Top-K retrieval with deduplication and forced category diversity.
    If preferred_categories provided, guarantees at least one result
    per preferred category in the top-K.
    """
    # ── 1. Encode user ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor(user_features, dtype=torch.float32).unsqueeze(0)
        user_emb    = model.get_user_embedding(user_tensor)
        user_emb_np = user_emb.numpy().astype(np.float32)

    # ── 2. FAISS search — fetch ALL 500 so we can always find diversity ─
    start = time.perf_counter()
    scores, indices = index.search(user_emb_np, len(workflow_ids))
    latency_ms = (time.perf_counter() - start) * 1000

    # ── 3. Build full ranked list, deduplicated by name ─────────────────
    seen_names = set()
    all_ranked  = []

    for i in range(len(indices[0])):
        idx   = indices[0][i]
        score = scores[0][i]
        wid   = workflow_ids[idx]
        wf    = catalog[wid]
        name  = wf["name"]

        if name in seen_names:
            continue
        seen_names.add(name)

        all_ranked.append({
            "workflow_id": wid,
            "name":        name,
            "category":    wf["category"],
            "description": wf["description"],
            "tooling_tags":wf.get("tooling_tags", []),
            "score":       float(score),
        })

    # ── 4. Forced diversity — guarantee slots per preferred category ────
    # Strategy:
    #   - Reserve 1 guaranteed slot per preferred category
    #   - Fill remaining slots with highest-scoring non-duplicate results
    #
    # Example: k=10, preferred=["Agile/Scrum", "Release Management"]
    #   → guaranteed: 1 Agile/Scrum + 1 Release Management
    #   → remaining 8 slots: best available regardless of category

    results = []
    used_ids = set()

    if preferred_categories and len(preferred_categories) > 0:
        # Guarantee one best result per preferred category
        for cat in preferred_categories:
            for item in all_ranked:
                if item["workflow_id"] in used_ids:
                    continue
                if item["category"] == cat:
                    results.append(item)
                    used_ids.add(item["workflow_id"])
                    break   # one per category guaranteed

    # Fill remaining slots with highest-scoring unused items
    for item in all_ranked:
        if len(results) >= k:
            break
        if item["workflow_id"] in used_ids:
            continue
        results.append(item)
        used_ids.add(item["workflow_id"])

    # Sort final list by score descending, re-rank
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(results):
        item["rank"] = i + 1

    return results[:k], latency_ms


def load_catalog():
    """Load workflows.json into a dict keyed by workflow_id."""
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        workflows = json.load(f)
    return {wf["workflow_id"]: wf for wf in workflows}


def main():
    print("=" * 70)
    print("  FAISS Index Builder + Retrieval Test")
    print("=" * 70)

    # ── 1. Load trained model ─────────────────────────────────────────
    print("\n[1] Loading trained model...")
    model = TwoTowerModel(
        user_input_dim=11, item_input_dim=384,
        hidden_dim=256, output_dim=128, dropout=0.2,
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    print(f"    Loaded from: {MODEL_PATH}")

    # ── 2. Build 128-dim item embeddings ──────────────────────────────
    print("\n[2] Building 128-dim item embeddings (ItemTower forward pass)...")
    item_embeddings, workflow_ids = build_item_embeddings(model)

    # Save the 128-dim embeddings (used later by metrics.py and app)
    np.save(ITEM_EMB_128_PATH, item_embeddings)
    print(f"    Saved: {ITEM_EMB_128_PATH}")

    # ── 3. Build FAISS index ──────────────────────────────────────────
    print("\n[3] Building FAISS IndexFlatIP...")
    index = build_faiss_index(item_embeddings)

    # Save the FAISS index to disk
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"    Saved: {FAISS_INDEX_PATH}")

    # ── 4. Load catalog for result display ────────────────────────────
    print("\n[4] Loading workflow catalog...")
    catalog = load_catalog()
    print(f"    {len(catalog)} workflows loaded")

    # ── 5. Latency benchmark: 100 random queries ─────────────────────
    print("\n[5] Latency benchmark (100 random queries, top-10)...")

    # Generate 100 random user feature vectors for benchmarking
    rng = np.random.RandomState(42)
    latencies = []

    for _ in range(100):
        fake_user = rng.rand(11).astype(np.float32)  # random features in [0, 1]
        _, latency = query_top_k(
            model, index, fake_user, workflow_ids, catalog, k=10
        )
        latencies.append(latency)

    latencies = np.array(latencies)
    print(f"    Average latency: {latencies.mean():.3f} ms")
    print(f"    Min latency:     {latencies.min():.3f} ms")
    print(f"    Max latency:     {latencies.max():.3f} ms")
    print(f"    Median latency:  {np.median(latencies):.3f} ms")
    print(f"    Target: < 10ms  -> "
          f"{'PASS' if latencies.max() < 10 else 'FAIL'}")

    # ── 6. Sample query: real user from test set ──────────────────────
    print("\n[6] Sample query: first user from features_test.csv...")

    # Load test features to get a real user
    test_df = pd.read_csv(FEATURES_TEST)
    first_user = test_df.iloc[0]

    user_id = first_user["user_id"]
    archetype = first_user["archetype"]
    preferred = first_user["preferred_categories"]

    print(f"    User:       {user_id}")
    print(f"    Archetype:  {archetype}")
    print(f"    Preferred:  {preferred}")

    # Extract the 11 feature values
    user_features = first_user[FEATURE_COLS].values.astype(np.float32)

    # Query top-10
    # Query top-10
    preferred_list = [c.strip() for c in preferred.split(",")]

    results, latency = query_top_k(
        model, index, user_features, workflow_ids, catalog,
        k=10,
        preferred_categories=preferred_list,
    )

    print(f"\n    Top-10 recommended workflows ({latency:.3f} ms):")
    print(f"    {'Rank':>4} | {'Score':>6} | {'Category':<25} | Name")
    print(f"    {'-'*4}-+-{'-'*6}-+-{'-'*25}-+-{'-'*30}")
    for r in results:
        print(f"    {r['rank']:>4} | {r['score']:>6.4f} | "
              f"{r['category']:<25} | {r['name']}")

    # ── 7. Category match analysis ────────────────────────────────────
    print(f"\n[7] Category match analysis...")

    # Parse preferred categories (stored as comma-separated string)
    preferred_set = set(c.strip() for c in preferred.split(","))
    recommended_cats = [r["category"] for r in results]
    matches = sum(1 for c in recommended_cats if c in preferred_set)

    print(f"    Preferred categories: {preferred_set}")
    print(f"    Recommended categories: {recommended_cats}")
    print(f"    Matches in top-10: {matches}/10 "
          f"({matches * 10}% category precision)")

    # Also check top-50 for Recall@50 proxy
    results_50, _ = query_top_k(
        model, index, user_features, workflow_ids, catalog, k=50
    )
    matches_50 = sum(1 for r in results_50 if r["category"] in preferred_set)
    print(f"    Matches in top-50: {matches_50}/50 "
          f"({matches_50 * 2}% category precision)")

    # ── Summary ───────────────────────────────────────────────────────
    idx_size = os.path.getsize(FAISS_INDEX_PATH)
    emb_size = os.path.getsize(ITEM_EMB_128_PATH)

    print()
    print("=" * 70)
    print("  FAISS index built and tested!")
    print(f"  Index:      {FAISS_INDEX_PATH} ({idx_size / 1024:.1f} KB)")
    print(f"  Embeddings: {ITEM_EMB_128_PATH} ({emb_size / 1024:.1f} KB)")
    print(f"  Latency:    {latencies.mean():.3f} ms avg (target < 10ms)")
    print()
    print("  Next step: python src/evaluation/metrics.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
