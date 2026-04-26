"""
src/model/dataset.py
────────────────────
PyTorch Dataset that serves (user_features, item_features, label) triples
for training the two-tower recommendation model.

Data flow:
    interactions_train.csv       features_train.csv       item_embeddings.npy
    ┌──────────────────────┐    ┌───────────────────┐    ┌──────────────────┐
    │ user_id  wf_id label │    │ user_id  11 feats │    │ wf_id  384 dims  │
    │ u03771   w0341  0    │    │ u03771   0.81 ... │    │ w0341  [0.1 ...] │
    │ u00281   w0475  1    │    │ u00281   0.22 ... │    │ w0475  [0.3 ...] │
    └──────────┬───────────┘    └────────┬──────────┘    └────────┬─────────┘
               │                         │                        │
               └─────────── __getitem__(idx) ─────────────────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
              user_tensor     item_tensor       label_tensor
              (11 floats)    (384 floats)       (1 float: 0 or 1)

Why a Dataset class?
    - PyTorch's DataLoader expects a Dataset to handle batching + shuffling
    - Dataset.__getitem__(i) returns ONE training example by index
    - Dataset.__len__() returns total number of examples
    - DataLoader then groups examples into mini-batches automatically

Run with:
    python src/model/dataset.py   (runs a self-test)
"""

import os                          # for building file paths
import json                        # for reading workflows.json
import numpy as np                 # for fast numerical arrays
import pandas as pd                # for reading CSV files
import torch                       # main PyTorch library
from torch.utils.data import Dataset, DataLoader  # Dataset base class + DataLoader

# ── Path setup ─────────────────────────────────────────────────────────────────
# os.path.abspath(__file__) gives the absolute path of THIS script.
# We go up 3 levels: dataset.py → model/ → src/ → project root
# This way, paths work no matter where you run the script from.

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Where the processed data lives
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Where item embeddings will be stored (once encode.py creates them)
EMBED_DIR = os.path.join(PROJECT_ROOT, "data", "embeddings")

# Where the workflow catalog is
CATALOG_PATH = os.path.join(PROJECT_ROOT, "data", "catalog", "workflows.json")


class InteractionDataset(Dataset):
    """
    A PyTorch Dataset that joins three data sources:

        1. interactions CSV → tells us which (user, item, label) triples to serve
        2. user features CSV → 11-dim normalized feature vector per user
        3. item embeddings   → 384-dim sentence-transformer vector per workflow

    How PyTorch uses this:
        dataset = InteractionDataset("train")
        loader  = DataLoader(dataset, batch_size=512, shuffle=True)

        for user_feat, item_feat, labels in loader:
            scores = model(user_feat, item_feat)   # forward pass
            loss   = loss_fn(scores, labels)        # compute loss
            ...

    The DataLoader calls __getitem__() 512 times, stacks the results into
    tensors, and hands you a (user_batch, item_batch, label_batch) tuple.
    """

    def __init__(self, split="train"):
        """
        Load all data into memory for fast __getitem__ lookups.

        Parameters:
            split : str, one of "train", "val", "test"
                    determines which interaction + feature CSV to load
        """
        # ── 1. Load interactions ──────────────────────────────────────────
        # Each row is: user_id, workflow_id, label (0 or 1)
        interactions_path = os.path.join(DATA_DIR, f"interactions_{split}.csv")
        self.interactions = pd.read_csv(interactions_path)

        # Store the columns as numpy arrays for faster indexing later.
        # .values converts a pandas Series → numpy array.
        self.user_ids    = self.interactions["user_id"].values     # array of strings
        self.workflow_ids = self.interactions["workflow_id"].values # array of strings
        self.labels      = self.interactions["label"].values       # array of 0/1 ints

        # ── 2. Load user features ─────────────────────────────────────────
        # Each row has: user_id, archetype, preferred_categories, + 11 feature columns
        features_path = os.path.join(DATA_DIR, f"features_{split}.csv")
        features_df = pd.read_csv(features_path)

        # These are the 11 numeric columns the UserTower expects
        feature_cols = [
            "issue_count_log_norm",
            "pr_count_log_norm",
            "push_count_log_norm",
            "repo_diversity_log_norm",
            "label_diversity_log_norm",
            "sprint_velocity_log_norm",
            "activity_score_log_norm",
            "collab_density_norm",
            "recency_weight_norm",
            "pr_ratio_norm",
            "issue_ratio_norm",
        ]

        # Build a dictionary: user_id → numpy array of 11 floats
        # We convert to float32 because PyTorch defaults to float32 and
        # mixing float64 with float32 causes errors.
        #
        # .set_index("user_id")  → makes user_id the row label
        # [feature_cols]         → keep only the 11 feature columns
        # .T                     → transpose so .to_dict() gives {user_id: {col: val}}
        #
        # Instead, we do it row by row for clarity:
        self.user_features = {}
        for _, row in features_df.iterrows():
            uid = row["user_id"]
            # Extract just the 11 feature values as a float32 numpy array
            self.user_features[uid] = row[feature_cols].values.astype(np.float32)

        # ── 3. Load item (workflow) embeddings ────────────────────────────
        # These are 384-dim vectors from sentence-transformers.
        # encode.py will save them as a .npy file + a JSON id map.
        #
        # File layout after encode.py runs:
        #   data/embeddings/workflow_embeddings.npy   → (500, 384) float32 array
        #   data/embeddings/workflow_ids.json          → ["w0000", "w0001", ...]

        embeddings_path = os.path.join(EMBED_DIR, "workflow_embeddings.npy")
        ids_path        = os.path.join(EMBED_DIR, "workflow_ids.json")

        if os.path.exists(embeddings_path) and os.path.exists(ids_path):
            # ── Real embeddings exist (encode.py has been run) ──
            # np.load reads a .npy file back into a numpy array
            all_embeddings = np.load(embeddings_path)  # shape: (500, 384)

            with open(ids_path, "r") as f:
                workflow_id_list = json.load(f)        # ["w0000", "w0001", ...]

            # Build dict: workflow_id → 384-dim numpy array
            self.item_embeddings = {
                wid: all_embeddings[i].astype(np.float32)
                for i, wid in enumerate(workflow_id_list)
            }
            print(f"  Loaded real embeddings for {len(self.item_embeddings)} workflows")

        else:
            # ── Fallback: generate random embeddings for testing ──
            # encode.py hasn't been built yet, so we create temporary
            # random 384-dim vectors. The model will still train (just
            # won't learn meaningful item representations yet).
            print(f"  WARNING: No embeddings found at {embeddings_path}")
            print(f"           Using random 384-dim vectors as placeholder.")
            print(f"           Run src/catalog/encode.py to generate real embeddings.")

            # Get all unique workflow IDs from the interactions
            unique_wids = set(self.workflow_ids)

            # np.random.RandomState with a fixed seed → reproducible random numbers
            rng = np.random.RandomState(42)
            self.item_embeddings = {
                wid: rng.randn(384).astype(np.float32)
                for wid in unique_wids
            }
            print(f"  Generated random embeddings for {len(self.item_embeddings)} workflows")

        # ── Summary ───────────────────────────────────────────────────────
        print(f"  [{split}] Loaded {len(self.interactions):,} interactions, "
              f"{len(self.user_features):,} users, "
              f"{len(self.item_embeddings):,} items")

    def __len__(self):
        """
        Return the total number of (user, item, label) examples.

        The DataLoader calls this to know how many batches make up one epoch.
        For example: 156,645 interactions / batch_size 512 = ~306 batches per epoch.
        """
        return len(self.interactions)

    def __getitem__(self, idx):
        """
        Return one training example by its index.

        Parameters:
            idx : int — index into the interactions table (0 to len-1)

        Returns:
            user_feat : torch.Tensor of shape (11,)   — normalized user features
            item_feat : torch.Tensor of shape (384,)   — workflow embedding
            label     : torch.Tensor of shape ()        — scalar, 0.0 or 1.0

        How this gets used:
            The DataLoader calls __getitem__ many times and STACKS the results:
                single call:  (11,), (384,), ()
                batch of 512: (512, 11), (512, 384), (512,)
            This stacking happens automatically — you don't need to do it.
        """
        # Look up this interaction's user_id, workflow_id, and label
        uid   = self.user_ids[idx]       # e.g., "u03771"
        wid   = self.workflow_ids[idx]   # e.g., "w0341"
        label = self.labels[idx]         # 0 or 1

        # Look up the pre-loaded feature vectors
        user_feat = self.user_features[uid]      # numpy array, shape (11,)
        item_feat = self.item_embeddings[wid]    # numpy array, shape (384,)

        # Convert numpy → PyTorch tensors
        # torch.tensor() copies the data; torch.from_numpy() shares memory
        # We use from_numpy for speed (no copy), it's safe because we don't
        # modify these arrays after loading.
        user_tensor = torch.from_numpy(user_feat)         # shape: (11,)
        item_tensor = torch.from_numpy(item_feat)         # shape: (384,)
        label_tensor = torch.tensor(label, dtype=torch.float32)  # shape: ()

        return user_tensor, item_tensor, label_tensor


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("  InteractionDataset — Self-Test")
    print("=" * 58)

    # ── 1. Create the training dataset ────────────────────────────────
    print("\n[1] Loading training dataset...")
    train_dataset = InteractionDataset(split="train")

    # ── 2. Check the size ─────────────────────────────────────────────
    print(f"\n[2] Dataset size: {len(train_dataset):,} examples")

    # ── 3. Fetch a single example ─────────────────────────────────────
    user_feat, item_feat, label = train_dataset[0]
    print(f"\n[3] Single example (index 0):")
    print(f"    User features shape: {user_feat.shape}   dtype: {user_feat.dtype}")
    print(f"    Item features shape: {item_feat.shape}  dtype: {item_feat.dtype}")
    print(f"    Label: {label.item()}  (0=negative, 1=positive)")
    print(f"    User features (first 5): {user_feat[:5].tolist()}")

    # ── 4. Test with DataLoader ───────────────────────────────────────
    print(f"\n[4] DataLoader test (batch_size=512, shuffle=True)...")

    # DataLoader wraps the Dataset and handles batching + shuffling
    # num_workers=0 means loading happens in the main process (safest on Windows)
    # pin_memory=False because we're on CPU (pin_memory is for GPU transfers)
    loader = DataLoader(
        train_dataset,      # our Dataset
        batch_size=512,     # how many examples per batch
        shuffle=True,       # randomize order each epoch (important for training)
        num_workers=0,      # no multiprocessing (Windows compatibility)
        drop_last=False,    # keep the last partial batch (don't waste data)
    )

    # Grab the first batch
    for batch_user, batch_item, batch_label in loader:
        print(f"    Batch user features: {batch_user.shape}")   # (512, 11)
        print(f"    Batch item features: {batch_item.shape}")   # (512, 384)
        print(f"    Batch labels:        {batch_label.shape}")  # (512,)
        print(f"    Label distribution:  "
              f"pos={batch_label.sum().item():.0f}, "
              f"neg={(batch_label == 0).sum().item():.0f}")
        print(f"    Positive ratio:      "
              f"{batch_label.mean().item():.3f} (expect ~0.20 = 1:4 ratio)")
        break  # only check the first batch

    # ── 5. Quick timing test ──────────────────────────────────────────
    import time

    print(f"\n[5] Timing: iterating one full epoch...")
    start = time.time()
    n_batches = 0
    for _ in loader:
        n_batches += 1
    elapsed = time.time() - start

    print(f"    {n_batches} batches processed in {elapsed:.2f}s "
          f"({elapsed / n_batches * 1000:.1f} ms/batch)")

    # ── 6. Verify val split loads too ─────────────────────────────────
    print(f"\n[6] Loading validation dataset...")
    val_dataset = InteractionDataset(split="val")
    print(f"    Val size: {len(val_dataset):,} examples")

    print()
    print("=" * 58)
    print("  All checks passed. Dataset is ready for training.")
    print("  Next step: python src/model/train.py")
    print("=" * 58)
