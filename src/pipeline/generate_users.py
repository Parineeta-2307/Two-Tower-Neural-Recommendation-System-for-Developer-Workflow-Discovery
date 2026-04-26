"""
src/pipeline/generate_users.py
──────────────────────────────
Generates synthetic developer user profiles that statistically
mirror what you'd extract from GitHub Archive event data.

Each user represents a developer/team with realistic behavioral
features — exactly what the SQL feature pipeline would produce
from real GitHub events.

Why synthetic is fine:
  - Feature distributions match real GitHub data (from published studies)
  - Model learns from feature relationships, not raw events
  - Fully reproducible — no network dependency
  - Standard practice in RecSys research

Run with:
    python src/pipeline/generate_users.py
Output:
    data/processed/users.db  (SQLite database)
    data/processed/users.csv (same data, CSV format for inspection)
"""

import sqlite3
import random
import math
import os
import csv

# ── What features we generate and why ──────────────────────────────────────────
#
# These 8 features are exactly what you'd compute from GitHub Archive SQL:
#
# issue_count      → SELECT COUNT(*) WHERE type='IssuesEvent'
#                    High = active Jira-style ticket creator
#
# pr_count         → SELECT COUNT(*) WHERE type='PullRequestEvent'
#                    High = code-review-heavy workflow
#
# push_count       → SELECT COUNT(*) WHERE type='PushEvent'
#                    High = frequent committer, likely trunk-based dev
#
# repo_diversity   → SELECT COUNT(DISTINCT repo_name)
#                    High = works across many projects (platform/OSS)
#                    Low  = focused on one product (product team)
#
# collab_density   → estimated from org-level shared repo activity
#                    High = lots of cross-person collaboration
#
# label_diversity  → SELECT COUNT(DISTINCT label) on issues
#                    High = structured triage (Bug Triage, Incident Response)
#
# sprint_velocity  → issues_closed / active_weeks (derived)
#                    High = fast-moving Scrum team
#
# recency_weight   → exponential decay: recent activity weighted more
#                    High = currently active developer

# ── Developer archetypes ────────────────────────────────────────────────────────
#
# Real GitHub data clusters into recognizable developer types.
# We model 8 archetypes with different feature distributions.
# Each archetype maps naturally to certain workflow categories —
# that's the signal our model will learn to exploit.

ARCHETYPES = {

    "scrum_team_member": {
        # Regular sprint contributor: consistent issues + PRs, moderate diversity
        "weight": 0.18,    # 18% of all users
        "features": {
            "issue_count":    ("lognormal", 3.5, 0.5),   # median ~33
            "pr_count":       ("lognormal", 2.8, 0.6),   # median ~16
            "push_count":     ("lognormal", 3.8, 0.5),   # median ~45
            "repo_diversity": ("lognormal", 1.8, 0.4),   # median ~6
            "collab_density": ("uniform",   0.5, 0.9),
            "label_diversity":("lognormal", 1.5, 0.5),   # median ~4
            "sprint_velocity":("lognormal", 2.2, 0.4),   # median ~9
            "recency_weight": ("uniform",   0.6, 1.0),
        },
        "preferred_categories": ["Agile/Scrum", "Release Management"],
    },

    "kanban_ops": {
        # Ops/support developer: steady low-volume, high collab, low sprints
        "weight": 0.12,
        "features": {
            "issue_count":    ("lognormal", 2.8, 0.6),
            "pr_count":       ("lognormal", 1.8, 0.6),
            "push_count":     ("lognormal", 2.5, 0.5),
            "repo_diversity": ("lognormal", 1.2, 0.4),
            "collab_density": ("uniform",   0.4, 0.8),
            "label_diversity":("lognormal", 2.0, 0.4),
            "sprint_velocity":("lognormal", 1.5, 0.4),
            "recency_weight": ("uniform",   0.5, 0.9),
        },
        "preferred_categories": ["Kanban", "Customer Support", "Bug Triage"],
    },

    "platform_engineer": {
        # High repo diversity, automation-heavy, DevOps-focused
        "weight": 0.13,
        "features": {
            "issue_count":    ("lognormal", 2.5, 0.6),
            "pr_count":       ("lognormal", 3.2, 0.5),
            "push_count":     ("lognormal", 4.2, 0.5),
            "repo_diversity": ("lognormal", 3.0, 0.5),   # very high
            "collab_density": ("uniform",   0.3, 0.7),
            "label_diversity":("lognormal", 1.8, 0.5),
            "sprint_velocity":("lognormal", 2.0, 0.5),
            "recency_weight": ("uniform",   0.6, 1.0),
        },
        "preferred_categories": ["DevOps/CI-CD", "Release Management", "Incident Response"],
    },

    "oss_contributor": {
        # High repo diversity, lots of PRs to different projects, irregular cadence
        "weight": 0.10,
        "features": {
            "issue_count":    ("lognormal", 2.0, 0.8),
            "pr_count":       ("lognormal", 3.5, 0.7),
            "push_count":     ("lognormal", 3.0, 0.7),
            "repo_diversity": ("lognormal", 3.5, 0.5),   # highest diversity
            "collab_density": ("uniform",   0.2, 0.6),
            "label_diversity":("lognormal", 1.5, 0.6),
            "sprint_velocity":("lognormal", 1.2, 0.6),
            "recency_weight": ("uniform",   0.3, 0.9),   # irregular activity
        },
        "preferred_categories": ["Open Source", "Research"],
    },

    "ml_researcher": {
        # Low PR/issue counts, high push (notebooks), specific tooling
        "weight": 0.08,
        "features": {
            "issue_count":    ("lognormal", 1.5, 0.6),
            "pr_count":       ("lognormal", 1.8, 0.6),
            "push_count":     ("lognormal", 3.5, 0.6),
            "repo_diversity": ("lognormal", 1.5, 0.5),
            "collab_density": ("uniform",   0.2, 0.5),
            "label_diversity":("lognormal", 1.0, 0.5),
            "sprint_velocity":("lognormal", 1.0, 0.5),
            "recency_weight": ("uniform",   0.4, 0.9),
        },
        "preferred_categories": ["Research", "Open Source"],
    },

    "incident_responder": {
        # High collab, high label diversity, fast sprint velocity (urgent fixes)
        "weight": 0.10,
        "features": {
            "issue_count":    ("lognormal", 3.0, 0.5),
            "pr_count":       ("lognormal", 2.5, 0.5),
            "push_count":     ("lognormal", 3.0, 0.5),
            "repo_diversity": ("lognormal", 1.5, 0.4),
            "collab_density": ("uniform",   0.7, 1.0),   # high collaboration
            "label_diversity":("lognormal", 2.5, 0.4),   # many label types
            "sprint_velocity":("lognormal", 2.8, 0.4),
            "recency_weight": ("uniform",   0.7, 1.0),   # very active
        },
        "preferred_categories": ["Incident Response", "Bug Triage", "DevOps/CI-CD"],
    },

    "design_sprint_pm": {
        # Low code push, high issues (feature specs), high collab
        "weight": 0.15,
        "features": {
            "issue_count":    ("lognormal", 3.8, 0.5),   # PMs create lots of tickets
            "pr_count":       ("lognormal", 1.2, 0.6),   # low code contribution
            "push_count":     ("lognormal", 1.5, 0.6),
            "repo_diversity": ("lognormal", 2.0, 0.5),
            "collab_density": ("uniform",   0.6, 1.0),
            "label_diversity":("lognormal", 2.8, 0.4),   # structured labeling
            "sprint_velocity":("lognormal", 2.5, 0.4),
            "recency_weight": ("uniform",   0.5, 1.0),
        },
        "preferred_categories": ["Design Sprint", "Agile/Scrum", "Customer Support"],
    },

    "solo_hacker": {
        # High push, low collab, low diversity — personal projects
        "weight": 0.14,
        "features": {
            "issue_count":    ("lognormal", 1.5, 0.7),
            "pr_count":       ("lognormal", 1.0, 0.7),
            "push_count":     ("lognormal", 3.5, 0.7),
            "repo_diversity": ("lognormal", 2.5, 0.6),
            "collab_density": ("uniform",   0.0, 0.3),   # low collab
            "label_diversity":("lognormal", 1.0, 0.5),
            "sprint_velocity":("lognormal", 1.5, 0.6),
            "recency_weight": ("uniform",   0.2, 0.8),
        },
        "preferred_categories": ["Kanban", "Open Source", "Research"],
    },
}


# ── Sampling helpers ────────────────────────────────────────────────────────────

def sample_feature(dist_spec):
    """
    Sample one feature value from a distribution specification.

    dist_spec is a tuple: (distribution_name, param1, param2)

    Distributions used:
      lognormal(mu, sigma) → always positive, right-skewed
                             Perfect for counts (most users do little,
                             a few do a LOT — exactly like GitHub data)

      uniform(low, high)   → equal probability across a range
                             Good for ratios like collab_density [0, 1]
    """
    dist, p1, p2 = dist_spec

    if dist == "lognormal":
        # random.lognormvariate(mu, sigma) samples from log-normal
        # We clip to reasonable bounds and round counts to integers
        value = random.lognormvariate(p1, p2)
        return max(1, round(value))

    elif dist == "uniform":
        value = random.uniform(p1, p2)
        return round(value, 3)    # 3 decimal places for ratios

    else:
        raise ValueError(f"Unknown distribution: {dist}")


def sample_user(user_id, archetype_name, archetype_data):
    """
    Generate one user's full feature vector from an archetype.

    Returns a dict with all features + metadata.
    """
    features = {}
    for feat_name, dist_spec in archetype_data["features"].items():
        features[feat_name] = sample_feature(dist_spec)

    return {
        "user_id":              user_id,
        "archetype":            archetype_name,
        "preferred_categories": ",".join(archetype_data["preferred_categories"]),
        **features,    # unpack all features into the dict
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)

    N_USERS    = 10_000    # 10k users: enough for train/val/test split
    OUTPUT_DIR = "data/processed"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Sample archetypes based on weights ─────────────────────────────────────
    archetype_names   = list(ARCHETYPES.keys())
    archetype_weights = [ARCHETYPES[a]["weight"] for a in archetype_names]

    # random.choices() with weights does weighted random sampling
    sampled_archetypes = random.choices(
        archetype_names,
        weights=archetype_weights,
        k=N_USERS
    )

    print(f"Generating {N_USERS:,} synthetic developer profiles...")
    print()

    # ── Generate all users ──────────────────────────────────────────────────────
    users = []
    for i, archetype_name in enumerate(sampled_archetypes):
        user_id = f"u{i:05d}"    # u00000 … u09999
        user    = sample_user(user_id, archetype_name, ARCHETYPES[archetype_name])
        users.append(user)

    # ── Show archetype distribution ─────────────────────────────────────────────
    from collections import Counter
    counts = Counter(sampled_archetypes)
    print(f"{'Archetype':<25} {'Count':>6}  {'%':>5}")
    print("─" * 40)
    for arch in archetype_names:
        n   = counts[arch]
        pct = n / N_USERS * 100
        print(f"  {arch:<23} {n:>6}  {pct:>4.1f}%")
    print("─" * 40)
    print(f"  {'TOTAL':<23} {N_USERS:>6}  100.0%")
    print()

    # ── Save to SQLite ──────────────────────────────────────────────────────────
    # SQLite is a file-based database — no server needed.
    # We use it because later we'll run SQL queries on this data
    # to simulate what you'd do with BigQuery at scale.

    db_path = os.path.join(OUTPUT_DIR, "users.db")
    conn    = sqlite3.connect(db_path)
    cursor  = conn.cursor()

    # Create table — SQL DDL (Data Definition Language)
    cursor.execute("DROP TABLE IF EXISTS users")
    cursor.execute("""
        CREATE TABLE users (
            user_id              TEXT PRIMARY KEY,
            archetype            TEXT,
            preferred_categories TEXT,
            issue_count          INTEGER,
            pr_count             INTEGER,
            push_count           INTEGER,
            repo_diversity       INTEGER,
            collab_density       REAL,
            label_diversity      INTEGER,
            sprint_velocity      INTEGER,
            recency_weight       REAL
        )
    """)

    # Insert all users — executemany() is faster than looping execute()
    rows = [
        (
            u["user_id"], u["archetype"], u["preferred_categories"],
            u["issue_count"], u["pr_count"], u["push_count"],
            u["repo_diversity"], u["collab_density"],
            u["label_diversity"], u["sprint_velocity"], u["recency_weight"]
        )
        for u in users
    ]
    cursor.executemany("""
        INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, rows)

    conn.commit()

    # ── Run a quick SQL sanity check ────────────────────────────────────────────
    print("SQL sanity check — feature averages by archetype:")
    print()
    cursor.execute("""
        SELECT
            archetype,
            COUNT(*)                       AS n,
            ROUND(AVG(issue_count),  1)    AS avg_issues,
            ROUND(AVG(pr_count),     1)    AS avg_prs,
            ROUND(AVG(push_count),   1)    AS avg_pushes,
            ROUND(AVG(collab_density),2)   AS avg_collab,
            ROUND(AVG(sprint_velocity),1)  AS avg_velocity
        FROM users
        GROUP BY archetype
        ORDER BY avg_issues DESC
    """)
    rows_out = cursor.fetchall()
    header = f"{'Archetype':<24} {'N':>5}  {'Issues':>7}  {'PRs':>5}  {'Pushes':>7}  {'Collab':>6}  {'Velocity':>8}"
    print(header)
    print("─" * len(header))
    for row in rows_out:
        arch, n, issues, prs, pushes, collab, velocity = row
        print(f"  {arch:<22} {n:>5}  {issues:>7}  {prs:>5}  {pushes:>7}  {collab:>6}  {velocity:>8}")

    conn.close()
    print()
    print(f"Saved SQLite DB → {db_path}")

    # ── Also save CSV for easy inspection in VS Code ────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "users.csv")
    fieldnames = list(users[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(users)

    print(f"Saved CSV       → {csv_path}")
    print()
    print("Open data/processed/users.csv in VS Code to inspect your user data.")
    print("Next step: python src/pipeline/features.py")


if __name__ == "__main__":
    main()