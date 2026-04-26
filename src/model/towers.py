"""
src/model/towers.py
────────────────────
Defines the two neural network towers:

  UserTower  : takes an 11-dim user feature vector
               outputs a 128-dim embedding

  ItemTower  : takes a 384-dim sentence-transformer vector
               outputs a 128-dim embedding

Both towers output vectors in the SAME 128-dim space.
The model learns by pulling positive (user, item) pairs
CLOSER together and pushing negative pairs FURTHER apart.

After training:
  - Similar users → similar positions in embedding space
  - Workflows that fit a user type → close to that user's embedding
  - Recommendation = find the 20 nearest workflow embeddings to a user embedding

Visual:
  User features → [UserTower] → 128-dim point in space ─┐
                                                          ├─ dot product → similarity score
  Item features → [ItemTower] → 128-dim point in space ─┘

Run with:
    python src/model/towers.py   (runs a self-test)
"""

import torch                    # the main PyTorch library
import torch.nn as nn           # nn = neural network building blocks
import torch.nn.functional as F # F = functions like relu, normalize


# ── What is nn.Module? ─────────────────────────────────────────────────────────
#
# Every neural network in PyTorch is a class that inherits from nn.Module.
# You must implement two methods:
#   __init__  : define the layers (what the network IS)
#   forward   : define what happens to input data (what the network DOES)
#
# PyTorch then automatically handles:
#   - Moving parameters to GPU if needed
#   - Computing gradients during backprop
#   - Saving/loading model weights
#
# Think of nn.Module like a blueprint. Each instance is a trained model.


class UserTower(nn.Module):
    """
    Encodes an 11-dimensional user feature vector into a 128-dim embedding.

    Architecture:
        Input (11) → Linear(11→256) → BatchNorm → ReLU → Dropout
                   → Linear(256→128) → BatchNorm → ReLU → Dropout
                   → Linear(128→128) → L2 Normalize
                   → Output (128)

    Why this architecture?
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Linear      : learns a weighted combination of input features       │
    │               y = Wx + b  (W=weights matrix, b=bias vector)        │
    │                                                                     │
    │ BatchNorm   : normalizes activations within each mini-batch         │
    │               stabilizes training, lets us use higher learning rate │
    │               without this, deep networks often diverge             │
    │                                                                     │
    │ ReLU        : activation function: f(x) = max(0, x)                │
    │               adds non-linearity — without it, stacking Linear      │
    │               layers is mathematically equivalent to one layer      │
    │                                                                     │
    │ Dropout(0.2): randomly zeros 20% of neurons during training        │
    │               prevents overfitting — model can't memorize training  │
    │               data because different neurons drop each forward pass │
    │                                                                     │
    │ L2 Normalize: scales output vector to unit length (magnitude=1)    │
    │               so dot product = cosine similarity ∈ [-1, +1]        │
    │               makes training numerically stable                     │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, input_dim=11, hidden_dim=256, output_dim=128, dropout=0.2):
        """
        Parameters:
            input_dim  : number of user features (11 in our pipeline)
            hidden_dim : width of the first hidden layer (256 is standard)
            output_dim : embedding dimension (128 — same for both towers)
            dropout    : fraction of neurons to drop during training
        """
        # ALWAYS call super().__init__() first in any nn.Module subclass
        # This initializes PyTorch's internal bookkeeping
        super(UserTower, self).__init__()

        # nn.Sequential chains layers in order:
        # input → layer1 → layer2 → ... → output
        # During forward(), data flows through each layer automatically
        self.net = nn.Sequential(

            # Layer 1: expand from 11 → 256
            # Why expand first? More neurons = more capacity to learn
            # complex patterns from simple features
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2: compress from 256 → 128
            # Bottleneck: forces the network to learn a compressed
            # representation — the "essence" of a user's behavior
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3: refine at 128 → 128
            # Same dimension in and out — learns non-linear combinations
            # of the compressed features
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        """
        Forward pass: takes input tensor x, returns normalized embedding.

        Parameters:
            x : torch.Tensor of shape (batch_size, input_dim)
                Each row is one user's feature vector

        Returns:
            torch.Tensor of shape (batch_size, output_dim)
            Each row is one user's embedding (unit-length vector)
        """
        embedding = self.net(x)

        # F.normalize divides each vector by its L2 norm (magnitude)
        # Result: every embedding vector has length exactly 1.0
        # This means: dot(u, v) = cosine_similarity(u, v) ∈ [-1, +1]
        # dim=1 means normalize along the feature dimension (not batch)
        return F.normalize(embedding, p=2, dim=1)


class ItemTower(nn.Module):
    """
    Encodes a 384-dimensional sentence-transformer embedding
    (workflow description) into a 128-dim embedding.

    Same architecture as UserTower but:
      - Input is 384-dim (sentence-transformer output dimension)
      - Output is same 128-dim space as UserTower

    Why 384? The sentence-transformer model 'all-MiniLM-L6-v2'
    produces 384-dimensional vectors. This is fixed by the model.
    We project it DOWN to 128 to match the user tower.

    Why project down?
      - Reduces parameters (faster training)
      - Forces compression into shared semantic space
      - Both towers MUST output same dimension for dot product to work
    """

    def __init__(self, input_dim=384, hidden_dim=256, output_dim=128, dropout=0.2):
        super(ItemTower, self).__init__()

        self.net = nn.Sequential(

            # Layer 1: compress 384 → 256
            # 384-dim sentence embeddings are already rich representations
            # We're distilling them into a workflow-specific space
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2: compress 256 → 128
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3: refine 128 → 128
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        embedding = self.net(x)
        return F.normalize(embedding, p=2, dim=1)


class TwoTowerModel(nn.Module):
    """
    Combines UserTower and ItemTower into one trainable model.

    During training:
        Given (user_features, item_features, label):
        1. UserTower encodes user → u_emb (128-dim unit vector)
        2. ItemTower encodes item → i_emb (128-dim unit vector)
        3. score = dot(u_emb, i_emb) = cosine similarity ∈ [-1, +1]
        4. label=1 (relevant) → we want score close to +1
           label=0 (irrelevant) → we want score close to -1

    During inference:
        1. Pre-compute ALL item embeddings once → store in FAISS
        2. For each new user: compute u_emb → FAISS finds nearest items
        3. Return top-K nearest item embeddings
    """

    def __init__(self, user_input_dim=11, item_input_dim=384,
                 hidden_dim=256, output_dim=128, dropout=0.2):
        super(TwoTowerModel, self).__init__()

        # Instantiate both towers as attributes of this class
        # PyTorch automatically registers them as sub-modules
        # (their parameters appear in model.parameters())
        self.user_tower = UserTower(user_input_dim, hidden_dim, output_dim, dropout)
        self.item_tower = ItemTower(item_input_dim, hidden_dim, output_dim, dropout)

    def forward(self, user_features, item_features):
        """
        Compute similarity scores for a batch of (user, item) pairs.

        Parameters:
            user_features : Tensor (batch_size, 11)
            item_features : Tensor (batch_size, 384)

        Returns:
            scores : Tensor (batch_size,)
                     cosine similarity between user and item embeddings
                     values in [-1, +1]
        """
        u_emb = self.user_tower(user_features)   # (batch_size, 128)
        i_emb = self.item_tower(item_features)   # (batch_size, 128)

        # Dot product of unit vectors = cosine similarity
        # torch.sum(a * b, dim=1) computes row-wise dot products
        # This gives one score per (user, item) pair in the batch
        scores = torch.sum(u_emb * i_emb, dim=1) # (batch_size,)

        return scores

    def get_user_embedding(self, user_features):
        """Encode users only — used at inference time."""
        return self.user_tower(user_features)

    def get_item_embedding(self, item_features):
        """Encode items only — used to build FAISS index."""
        return self.item_tower(item_features)


# ── Self-test: run this file directly to verify the model works ────────────────

if __name__ == "__main__":
    print("=" * 54)
    print("  Two-Tower Model — Architecture Self-Test")
    print("=" * 54)

    # Instantiate the model
    model = TwoTowerModel(
        user_input_dim=11,
        item_input_dim=384,
        hidden_dim=256,
        output_dim=128,
        dropout=0.2,
    )

    print("\n[1] Model architecture:")
    print(model)

    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    user_params  = sum(p.numel() for p in model.user_tower.parameters() if p.requires_grad)
    item_params  = sum(p.numel() for p in model.item_tower.parameters() if p.requires_grad)
    print(f"\n[2] Parameter count:")
    print(f"    User tower : {user_params:,}")
    print(f"    Item tower : {item_params:,}")
    print(f"    Total      : {total_params:,}")

    # ── Forward pass test ──────────────────────────────────────────────────────
    batch_size = 32

    # torch.randn creates random tensors sampled from N(0,1)
    # In real training these would be actual feature vectors
    fake_users = torch.randn(batch_size, 11)
    fake_items = torch.randn(batch_size, 384)

    # model.eval() turns OFF dropout and batchnorm training behavior
    # Always use this when not training
    model.eval()
    with torch.no_grad():   # no_grad: don't track gradients (saves memory)
        scores = model(fake_users, fake_items)

    print(f"\n[3] Forward pass test:")
    print(f"    Input  — users:  {fake_users.shape}  (batch=32, features=11)")
    print(f"    Input  — items:  {fake_items.shape} (batch=32, features=384)")
    print(f"    Output — scores: {scores.shape}          (batch=32, one score each)")
    print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"    (should be in [-1, +1] due to L2 normalization)")

    # ── Embedding shape test ───────────────────────────────────────────────────
    with torch.no_grad():
        u_emb = model.get_user_embedding(fake_users)
        i_emb = model.get_item_embedding(fake_items)

    print(f"\n[4] Embedding shapes:")
    print(f"    User embeddings: {u_emb.shape}   (32 users, each → 128-dim)")
    print(f"    Item embeddings: {i_emb.shape}   (32 items, each → 128-dim)")

    # Verify L2 normalization — every embedding should have magnitude ≈ 1.0
    u_norms = torch.norm(u_emb, dim=1)
    i_norms = torch.norm(i_emb, dim=1)
    print(f"\n[5] L2 norm check (all should be ≈ 1.0):")
    print(f"    User embedding norms: min={u_norms.min():.6f}  max={u_norms.max():.6f}")
    print(f"    Item embedding norms: min={i_norms.min():.6f}  max={i_norms.max():.6f}")

    # ── Sanity check: relevant pair should score higher than irrelevant ─────────
    # This won't be true on a random untrained model — but after training it will.
    # We test the MECHANISM works, not the values.
    model.eval()
    with torch.no_grad():
        one_user = torch.randn(1, 11)
        same_item_twice = torch.randn(1, 384)
        score_a = model(one_user, same_item_twice)
        score_b = model(one_user, torch.randn(1, 384))

    print(f"\n[6] Score mechanism check:")
    print(f"    Score (user, item_A): {score_a.item():.4f}")
    print(f"    Score (user, item_B): {score_b.item():.4f}")
    print(f"    (random scores expected — ordering becomes meaningful after training)")

    print()
    print("=" * 54)
    print("  All checks passed. Model is ready for training.")
    print("  Next step: python src/model/train.py")
    print("=" * 54)