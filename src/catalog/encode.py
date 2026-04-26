"""
src/catalog/encode.py
─────────────────────
Encodes all 500 workflow descriptions into 384-dimensional dense vectors
using the sentence-transformer model 'all-MiniLM-L6-v2'.

What this script does:
    1. Reads data/catalog/workflows.json (500 workflow templates)
    2. For each workflow, combines name + category + description into one string
    3. Feeds all 500 strings through the sentence-transformer model
    4. Saves the resulting (500, 384) embedding matrix to disk

Output files (used by dataset.py):
    data/embeddings/workflow_embeddings.npy   ← (500, 384) float32 matrix
    data/embeddings/workflow_ids.json          ← ["w0000", "w0001", ..., "w0499"]

Why sentence-transformers?
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Neural networks operate on numbers, not text.                       │
    │ A sentence-transformer converts text → fixed-length numeric vector. │
    │                                                                      │
    │ Key property: SIMILAR text → SIMILAR vectors (close in 384-d space) │
    │                                                                      │
    │ "Sprint planning with standups"  → [0.12, -0.34, 0.56, ...]        │
    │ "Agile scrum daily check-ins"    → [0.11, -0.33, 0.55, ...]  close!│
    │ "Kubernetes deployment pipeline" → [0.78,  0.21, -0.44, ...] far!  │
    │                                                                      │
    │ Model: all-MiniLM-L6-v2                                             │
    │   - Output dimension: 384                                            │
    │   - Size: ~80 MB                                                     │
    │   - Speed: ~500 sentences/sec on CPU                                 │
    │   - Quality: very good for semantic similarity tasks                 │
    └──────────────────────────────────────────────────────────────────────┘

Run with:
    python src/catalog/encode.py
"""

import os       # for file path operations
import sys      # for reconfiguring stdout encoding
import json     # for reading/writing JSON files
import time     # for timing the encoding process
import numpy as np  # for saving the embedding matrix as .npy

# ── Windows console fix ───────────────────────────────────────────────────────
# Windows cmd/PowerShell defaults to cp1252 encoding, which can't print
# some Unicode characters. This forces stdout to use UTF-8 instead.
# The 'errors=replace' means any un-printable char becomes '?' rather than crashing.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# sentence_transformers is the library that provides pre-trained text encoders
# SentenceTransformer is the main class — it downloads the model on first use
# and caches it locally so subsequent runs are fast
from sentence_transformers import SentenceTransformer

# ── Path setup ─────────────────────────────────────────────────────────────────
# Navigate from this file (src/catalog/encode.py) up to project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Input: the workflow catalog we generated earlier
CATALOG_PATH = os.path.join(PROJECT_ROOT, "data", "catalog", "workflows.json")

# Output directory: we create this if it doesn't exist
EMBED_DIR = os.path.join(PROJECT_ROOT, "data", "embeddings")

# Output files — these names MUST match what dataset.py expects
EMBEDDINGS_PATH = os.path.join(EMBED_DIR, "workflow_embeddings.npy")
IDS_PATH = os.path.join(EMBED_DIR, "workflow_ids.json")


def build_text_for_encoding(workflow):
    """
    Combine a workflow's text fields into a single string for encoding.

    We concatenate name + category + description because:
      - name alone is too short (5-10 words) — not enough context
      - description alone misses the category signal
      - combining all three gives the sentence-transformer the full picture

    The ' | ' separator helps the model distinguish the parts.
    The tooling_tags are also appended since "Jira, Confluence" vs "GitHub, Linear"
    carries useful semantic signal about the type of team/workflow.

    Parameters:
        workflow : dict with keys 'name', 'category', 'description', 'tooling_tags'

    Returns:
        str : a single string ready for the sentence-transformer

    Example:
        "3-week Scrum with daily standups | Agile/Scrum | A 3-week sprint
         cycle with sprint planning... | Tools: Notion, Jira, Miro, Figma"
    """
    # Get each field, with safe defaults in case any are missing
    name = workflow.get("name", "")
    category = workflow.get("category", "")
    description = workflow.get("description", "")
    tools = workflow.get("tooling_tags", [])

    # Join tools into a comma-separated string: "Jira, Notion, Figma"
    tools_str = ", ".join(tools) if tools else ""

    # Build the combined string with ' | ' separators
    # Example: "Lightweight Scrum | Agile/Scrum | A 2-week sprint... | Tools: Jira, Notion"
    parts = [name, category, description]
    if tools_str:
        parts.append(f"Tools: {tools_str}")

    return " | ".join(parts)


def main():
    """
    Main encoding pipeline:
        1. Load workflows from JSON
        2. Build text strings for encoding
        3. Load the sentence-transformer model
        4. Encode all texts → (500, 384) matrix
        5. Save embeddings + ID mapping to disk
    """
    print("=" * 58)
    print("  Workflow Encoder -- all-MiniLM-L6-v2")
    print("=" * 58)

    # ── 1. Load the workflow catalog ──────────────────────────────────────
    print(f"\n[1] Loading workflows from {CATALOG_PATH}")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        workflows = json.load(f)

    print(f"    Loaded {len(workflows)} workflows")

    # ── 2. Build text strings for encoding ────────────────────────────────
    # We need a list of strings — one per workflow — in the SAME ORDER
    # as the workflow_ids list (so row i of the matrix = workflow_ids[i])
    print(f"\n[2] Building text strings for encoding...")

    workflow_ids = []    # ["w0000", "w0001", ...]
    texts = []           # ["3-week Scrum... | Agile/Scrum | ...", ...]

    for wf in workflows:
        workflow_ids.append(wf["workflow_id"])
        texts.append(build_text_for_encoding(wf))

    # Show a few examples so we can verify the text looks right
    print(f"    Built {len(texts)} text strings")
    print(f"\n    Example texts (first 3):")
    for i in range(min(3, len(texts))):
        # Truncate long strings for display
        display = texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
        print(f"      [{workflow_ids[i]}] {display}")

    # ── 3. Load the sentence-transformer model ───────────────────────────
    print(f"\n[3] Loading sentence-transformer model 'all-MiniLM-L6-v2'...")
    print(f"    (first run downloads ~80 MB model; subsequent runs use cache)")

    # SentenceTransformer() downloads the model from HuggingFace Hub
    # on the first call and caches it in ~/.cache/huggingface/
    # After that, it loads from cache (fast, no internet needed)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"    Model loaded. Output dimension: {model.get_embedding_dimension()}")

    # ── 4. Encode all workflow texts ──────────────────────────────────────
    print(f"\n[4] Encoding {len(texts)} workflows...")

    start_time = time.time()

    # model.encode() takes a list of strings and returns a numpy array
    # shape: (num_strings, embedding_dim) = (500, 384)
    #
    # Parameters:
    #   texts              : list of strings to encode
    #   batch_size=64      : encode 64 texts at a time (fits in CPU RAM easily)
    #   show_progress_bar  : displays a tqdm progress bar
    #   normalize_embeddings=True : L2-normalize each vector to unit length
    #       Why normalize? Same reason as our towers — so dot product = cosine sim.
    #       This makes the raw sentence-transformer embeddings directly comparable.
    #       Note: the ItemTower will ALSO normalize its output, but having pre-normalized
    #       inputs helps with training stability.
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # each row has L2 norm = 1.0
    )

    elapsed = time.time() - start_time
    print(f"    Encoded {len(texts)} workflows in {elapsed:.2f}s "
          f"({len(texts) / elapsed:.0f} workflows/sec)")

    # Verify the output shape and dtype
    print(f"    Embedding matrix shape: {embeddings.shape}")   # (500, 384)
    print(f"    Embedding matrix dtype: {embeddings.dtype}")   # float32

    # ── 5. Verify the embeddings look correct ─────────────────────────────
    print(f"\n[5] Quality checks:")

    # Check L2 norms — should all be ~= 1.0 since we used normalize_embeddings=True
    norms = np.linalg.norm(embeddings, axis=1)  # compute L2 norm of each row
    print(f"    L2 norms: min={norms.min():.6f}  max={norms.max():.6f}  "
          f"(should be ~= 1.0)")

    # Check that similar workflows get similar embeddings
    # Let's find the most similar pair among the first 50 workflows
    # Cosine similarity = dot product (since vectors are unit-length)
    subset = embeddings[:50]                      # first 50 embeddings
    sim_matrix = subset @ subset.T                # (50, 50) similarity matrix
    # Zero out the diagonal (self-similarity is always 1.0, not interesting)
    np.fill_diagonal(sim_matrix, -1)
    # Find the most similar pair
    most_similar_idx = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
    i, j = most_similar_idx
    print(f"    Most similar pair (first 50): "
          f"{workflow_ids[i]} <-> {workflow_ids[j]} "
          f"(cosine={sim_matrix[i, j]:.4f})")
    print(f"      -> {workflows[i]['name']}")
    print(f"      -> {workflows[j]['name']}")

    # Find the least similar pair
    np.fill_diagonal(sim_matrix, 2)  # reset diagonal to find true min
    least_similar_idx = np.unravel_index(sim_matrix.argmin(), sim_matrix.shape)
    i, j = least_similar_idx
    print(f"    Least similar pair (first 50): "
          f"{workflow_ids[i]} <-> {workflow_ids[j]} "
          f"(cosine={sim_matrix[i, j]:.4f})")
    print(f"      -> {workflows[i]['name']}")
    print(f"      -> {workflows[j]['name']}")

    # -- 6. Save to disk ---------------------------------------------------
    print(f"\n[6] Saving embeddings to disk...")

    # Create the output directory if it doesn't exist
    # os.makedirs with exist_ok=True is safe to call even if the dir exists
    os.makedirs(EMBED_DIR, exist_ok=True)

    # Save the embedding matrix as a .npy file
    # .npy is NumPy's native binary format - very fast to load/save
    # It preserves the exact shape, dtype, and values
    np.save(EMBEDDINGS_PATH, embeddings.astype(np.float32))
    print(f"    Saved: {EMBEDDINGS_PATH}")
    print(f"    Shape: {embeddings.shape}, dtype: float32")

    # Save the workflow ID list as JSON
    # This maps row index -> workflow_id:
    #   row 0 -> "w0000", row 1 -> "w0001", etc.
    with open(IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(workflow_ids, f, indent=2)
    print(f"    Saved: {IDS_PATH}")
    print(f"    {len(workflow_ids)} workflow IDs")

    # -- 7. Verify saved files load correctly ------------------------------
    print(f"\n[7] Verification - reloading from disk...")

    loaded_embeddings = np.load(EMBEDDINGS_PATH)
    with open(IDS_PATH, "r") as f:
        loaded_ids = json.load(f)

    print(f"    Reloaded embeddings shape: {loaded_embeddings.shape}")
    print(f"    Reloaded workflow IDs: {len(loaded_ids)} IDs")
    print(f"    First 5 IDs: {loaded_ids[:5]}")

    # Verify the data matches what we saved
    assert loaded_embeddings.shape == embeddings.shape, "Shape mismatch!"
    assert np.allclose(loaded_embeddings, embeddings), "Values mismatch!"
    assert loaded_ids == workflow_ids, "IDs mismatch!"
    print(f"    [OK] All data matches -- files are correct")

    # -- Summary -----------------------------------------------------------
    # Calculate file sizes
    emb_size = os.path.getsize(EMBEDDINGS_PATH)
    ids_size = os.path.getsize(IDS_PATH)

    print()
    print("=" * 58)
    print("  Encoding complete!")
    print(f"  Embeddings: {EMBEDDINGS_PATH}")
    print(f"    -> {embeddings.shape[0]} workflows x {embeddings.shape[1]} dims")
    print(f"    -> {emb_size / 1024:.1f} KB")
    print(f"  IDs: {IDS_PATH}")
    print(f"    -> {ids_size / 1024:.1f} KB")
    print()
    print("  dataset.py will now load REAL embeddings instead of")
    print("  random vectors. Re-run dataset.py to verify:")
    print("    python src/model/dataset.py")
    print("=" * 58)


if __name__ == "__main__":
    main()
