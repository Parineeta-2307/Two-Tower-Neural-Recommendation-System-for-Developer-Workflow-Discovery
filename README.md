# Two-Tower Neural Recommendation System for Developer Workflow Discovery

**Jira for your Jira -- recommends the right workflow before you know you need it.**

A production-grade recommendation system that analyzes developer activity patterns (issues, PRs, commits, repo diversity) and recommends optimal team workflows using a two-tower neural architecture with FAISS retrieval.

**[GitHub](https://github.com/Parineeta-2307/Two-Tower-Neural-Recommendation-System-for-Developer-Workflow-Discovery)

---

## Results

Evaluated on 1,000 held-out test users across 8 developer archetypes:

| Metric | Random Baseline | Our Model | Improvement |
|---|---|---|---|
| **nDCG@10** | 0.2639 | **0.9711** | **3.7x** |
| **Recall@50** | 0.1000 | **0.3815** | **3.8x** |
| **Retrieval Latency** | -- | **0.11 ms** | (target: <10ms) |
| **Training Time** | -- | **1.2 min** | (CPU only) |

Per-archetype nDCG@10: all 8 archetypes score above 0.94, with `ml_researcher` and `solo_hacker` achieving perfect 1.0.

---

## Architecture

```
                        TWO-TOWER ARCHITECTURE
    
    USER SIDE                                    ITEM SIDE
    
    GitHub/Jira Activity                    Workflow Templates (500)
          |                                        |
    Feature Extraction                     Sentence-Transformer
    (11 behavioral features)              (all-MiniLM-L6-v2, 384-dim)
          |                                        |
    +-------------+                       +-------------+
    |  UserTower  |                       |  ItemTower  |
    |  11 -> 256  |                       | 384 -> 256  |
    | 256 -> 128  |                       | 256 -> 128  |
    | L2-normalize|                       | L2-normalize|
    +-------------+                       +-------------+
          |                                        |
          v                                        v
    user_embedding (128-dim)             item_embedding (128-dim)
          |                                        |
          +---------> cosine similarity <----------+
                            |
                    ranking / retrieval
                            |
                      FAISS IndexFlatIP
                     (sub-millisecond)
```

**Training:** BCEWithLogitsLoss with temperature scaling (T=0.1) on 125K user-workflow interaction pairs. Positive pairs from behavioral matching; negatives sampled at 4:1 ratio.

**Inference:** Encode user features once through UserTower, search pre-indexed item embeddings via FAISS. Total latency < 1ms for 500 items.

---

## Why Two-Tower?

1. **Decoupled encoding enables real-time serving.** Item embeddings are precomputed and indexed. At query time, only the user-side forward pass runs (11-dim input, two linear layers). This is the same architecture YouTube, Google, and Spotify use for candidate generation at scale.

2. **Dot-product retrieval scales to millions of items.** Because both towers produce L2-normalized embeddings, cosine similarity reduces to a dot product -- which FAISS computes over millions of vectors in milliseconds using approximate nearest neighbor search.

3. **Cold-start resilient.** New users get recommendations from their activity features alone (no interaction history required). New workflows get embeddings from their text description via the sentence-transformer -- no retraining needed.

---

## Project Structure

```
two_tower_recsys/
|
|-- app/
|   |-- streamlit_app.py          # Interactive demo UI (two modes)
|
|-- src/
|   |-- pipeline/
|   |   |-- download.py           # GitHub Archive data downloader
|   |   |-- generate_users.py     # Synthetic user + interaction generator
|   |   |-- features.py           # Feature extraction + normalization
|   |
|   |-- catalog/
|   |   |-- generate.py           # 500 workflow template generator
|   |   |-- encode.py             # Sentence-transformer encoding (384-dim)
|   |
|   |-- model/
|   |   |-- towers.py             # UserTower, ItemTower, TwoTowerModel
|   |   |-- dataset.py            # PyTorch Dataset + DataLoader
|   |   |-- train.py              # Training loop (BCE + Adam + early stopping)
|   |
|   |-- retrieval/
|   |   |-- faiss_index.py        # FAISS index builder + query function
|   |
|   |-- evaluation/
|       |-- metrics.py            # nDCG@10, Recall@50, A/B simulation
|
|-- data/
|   |-- catalog/workflows.json    # 500 workflow templates
|   |-- processed/                # User features, interactions (train/val/test)
|   |-- embeddings/               # Workflow embeddings + FAISS index
|   |-- models/                   # Trained model + evaluation results
|
|-- requirements.txt
|-- README.md
```

---

## Setup and Run

### Prerequisites

- Python 3.10+ (tested on 3.13)
- No GPU required -- runs entirely on CPU

### Installation

```bash
git clone https://github.com/Parineeta-2307/Two-Tower-Neural-Recommendation-System-for-Developer-Workflow-Discovery.git
cd Two-Tower-Neural-Recommendation-System-for-Developer-Workflow-Discovery

pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# 1. Generate synthetic data (already included in repo)
python src/pipeline/generate_users.py      # 10,000 users + interactions
python src/pipeline/features.py            # Feature extraction + normalization
python src/catalog/generate.py             # 500 workflow templates

# 2. Encode workflows
python src/catalog/encode.py               # Sentence-transformer embeddings

# 3. Train the model
python src/model/train.py                  # ~1.2 min on CPU

# 4. Build retrieval index
python src/retrieval/faiss_index.py        # FAISS index from trained embeddings

# 5. Evaluate
python src/evaluation/metrics.py           # nDCG@10, Recall@50, A/B test

# 6. Launch demo
python -m streamlit run app/streamlit_app.py
```

The demo opens at `http://localhost:8501`.

### Quick Start (data already included)

If you just want to see the demo without retraining:

```bash
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

---

## Data

### Why Synthetic Data?

This project uses 10,000 synthetic developer profiles rather than real GitHub data. This is a deliberate architectural choice, not a shortcut:

1. **Privacy.** Real developer activity data raises GDPR/privacy concerns. Synthetic data lets us demonstrate the full ML pipeline without handling PII.

2. **Controlled archetypes.** We defined 8 developer archetypes (scrum_team_member, kanban_ops, platform_engineer, etc.) with distinct behavioral distributions. This lets us validate that the model learns archetype-specific patterns -- something harder to verify with noisy real data.

3. **Reproducibility.** Every run produces identical results. No dependency on API rate limits, network access, or data drift.

4. **Production-ready pipeline.** The feature extraction code (`features.py`) is designed to process real GitHub Archive data. Swapping synthetic data for real API calls requires changing the data source, not the model.

### Data Summary

| Dataset | Records | Description |
|---|---|---|
| Users | 10,000 | Synthetic developers across 8 archetypes |
| Workflows | 500 | Templates across 10 categories |
| Interactions | 156,645 | User-workflow pairs (20% positive, 80% negative) |
| Train/Val/Test | 80/10/10 | Split by user (no leakage) |

### User Features (11 dimensions)

| Feature | Description | Transform |
|---|---|---|
| issue_count | Total issues created | log-scale, min-max normalized |
| pr_count | Total pull requests | log-scale, min-max normalized |
| push_count | Total push events | log-scale, min-max normalized |
| repo_diversity | Unique repositories | log-scale, min-max normalized |
| label_diversity | Unique labels used | log-scale, min-max normalized |
| sprint_velocity | Issues closed per sprint | log-scale, min-max normalized |
| activity_score | Composite activity metric | log-scale, min-max normalized |
| collab_density | Fraction of collaborative events | min-max normalized |
| recency_weight | Recent vs. historical activity ratio | min-max normalized |
| pr_ratio | PRs / total actions | min-max normalized |
| issue_ratio | Issues / total actions | min-max normalized |

---

## Key Design Decisions

### 1. Temperature-Scaled BCE over Contrastive Loss

The model outputs cosine similarity in [-1, +1]. Raw BCE loss on `sigmoid(score)` compresses predictions to [0.27, 0.73] -- the model can never be confident. Dividing by temperature (T=0.1) stretches the logit range to [-10, +10], restoring full sigmoid dynamic range. This is simpler than margin-based contrastive losses and achieved 0.97 nDCG without tuning.

### 2. Log-Scale Count Features

Raw event counts follow a power law (most users have few events, some have thousands). Log-transform followed by min-max normalization maps these to [0, 1] with approximately Gaussian distribution. This prevents high-activity users from dominating the gradient signal and gives the model more uniform feature distributions to learn from.

### 3. Sentence-Transformers for Item Encoding

Workflow descriptions are short, domain-specific text. Using `all-MiniLM-L6-v2` (384-dim, 80MB) provides rich semantic embeddings without fine-tuning. The ItemTower then projects these to 128-dim, learning which semantic aspects are most relevant for recommendation. This also enables zero-shot encoding of new workflows without retraining.

### 4. FAISS IndexFlatIP over HNSW

With 500 items, exact brute-force search (IndexFlatIP) runs in 0.11ms. Approximate methods like HNSW add complexity (index build parameters, recall trade-offs) with zero latency benefit at this scale. The architecture supports a drop-in switch to IndexIVFFlat or IndexHNSW when scaling to millions of items.

### 5. User-Level Train/Val/Test Split

Interactions are split by user ID, not randomly. This prevents data leakage where the model sees a user's preferences during training and is evaluated on the same user's other preferences. All 8,000 training users are completely absent from the 1,000 validation and 1,000 test users.

---

## What I Would Do with Real Atlassian Data

- **Replace synthetic features with Jira/Confluence API signals.** Issue velocity from Jira boards, page edit frequency from Confluence, deployment frequency from Bitbucket Pipelines. The 11-feature UserTower input dim stays the same; only the feature extraction layer changes.

- **Add implicit feedback signals.** Workflow template view duration, configuration completion rate, and team adoption velocity provide richer training signal than binary positive/negative labels. The loss function would shift to weighted BCE or a learning-to-rank objective.

- **Scale to multi-tenant serving.** Pre-compute user embeddings nightly via batch inference, store in Redis, serve recommendations through a lightweight API. FAISS index would upgrade to IVF with product quantization for sub-millisecond search over 100K+ workflows.

---

## Interview Q&A

**Q: Why two towers instead of a single neural network that takes both user and item features?**

A single network (cross-network) computes interactions between every user-item pair at query time. That is O(N) per query where N is the number of items. Two-tower architectures decouple encoding: item embeddings are precomputed and indexed, reducing query-time cost to a single user-side forward pass plus a vector search. This is the standard architecture for candidate generation at YouTube, Google, and Spotify scale.

**Q: How do you handle cold-start users?**

The UserTower takes behavioral features as input, not interaction history. A new user with even a few days of GitHub activity produces a valid 11-dim feature vector. The model generalizes from behavioral similarity to existing users -- a new user who creates many issues and uses sprint labels will get recommendations similar to other scrum_team_members, even with zero prior interactions.

**Q: Why is your Recall@50 lower than nDCG@10?**

nDCG@10 measures whether relevant items appear near the top of the ranking. Recall@50 measures what fraction of all relevant items appear in the top 50. Each user has 100+ relevant workflows (category-matched), so retrieving all of them within 50 slots is structurally constrained. The 3.8x improvement over random demonstrates the model is learning meaningful patterns despite this ceiling effect.

**Q: How would you evaluate this system in production?**

Online A/B testing with workflow adoption as the primary metric: do teams that receive model-recommended workflows actually configure and use them at higher rates than teams receiving popularity-based recommendations? Secondary metrics: time-to-first-workflow (onboarding speed), workflow retention at 30 days, and recommendation diversity (preventing filter bubbles).

**Q: What would break if you deployed this model without retraining for 6 months?**

Feature drift. Developer behavior patterns evolve as new tools emerge and team structures change. The sentence-transformer embeddings for new workflows would remain valid (text semantics are stable), but the UserTower's learned feature-to-preference mapping would degrade. Monitoring the distribution of user feature vectors against the training distribution (via KL divergence or PSI) would trigger retraining alerts.

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | PyTorch 2.7, custom two-tower architecture |
| Item Encoding | sentence-transformers (all-MiniLM-L6-v2) |
| Retrieval | FAISS 1.13 (IndexFlatIP) |
| Evaluation | scikit-learn (AUC), custom nDCG/Recall |
| Demo | Streamlit 1.49 |
| Data | pandas, NumPy, SQLite |
| Language | Python 3.13 |

---

## License

MIT
