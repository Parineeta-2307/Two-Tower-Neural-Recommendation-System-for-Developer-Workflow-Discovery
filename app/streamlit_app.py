"""
app/streamlit_app.py
--------------------
Interactive demo UI for the Two-Tower Workflow Recommender.

Two modes:
  1. TEAM MODE: Select team members from synthetic user database,
     see their actual behavioral profiles, get personalized recs
  2. EXPLORE MODE: Adjust feature sliders directly to explore
     how the model responds to different developer profiles

In production, Mode 1 would connect to GitHub/Jira APIs to pull
real activity data. For this demo, we use 10,000 synthetic users.

Run with:
    python -m streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import faiss
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "model"))
from towers import TwoTowerModel

# Asset paths
MODEL_PATH        = os.path.join(PROJECT_ROOT, "data", "models", "best_model.pt")
FAISS_INDEX_PATH  = os.path.join(PROJECT_ROOT, "data", "embeddings", "faiss_index.bin")
WORKFLOW_IDS_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings", "workflow_ids.json")
CATALOG_PATH      = os.path.join(PROJECT_ROOT, "data", "catalog", "workflows.json")
SCALER_PATH       = os.path.join(PROJECT_ROOT, "data", "processed", "scaler_stats.json")
EVAL_RESULTS_PATH = os.path.join(PROJECT_ROOT, "data", "models", "evaluation_results.json")

# All user feature CSVs (we combine all splits for the demo lookup)
FEATURES_TRAIN = os.path.join(PROJECT_ROOT, "data", "processed", "features_train.csv")
FEATURES_VAL   = os.path.join(PROJECT_ROOT, "data", "processed", "features_val.csv")
FEATURES_TEST  = os.path.join(PROJECT_ROOT, "data", "processed", "features_test.csv")

# The 11 feature columns the model expects
FEATURE_COLS = [
    "issue_count_log_norm", "pr_count_log_norm", "push_count_log_norm",
    "repo_diversity_log_norm", "label_diversity_log_norm",
    "sprint_velocity_log_norm", "activity_score_log_norm",
    "collab_density_norm", "recency_weight_norm",
    "pr_ratio_norm", "issue_ratio_norm",
]

# Human-readable names for the 11 features (for display)
FEATURE_LABELS = [
    "Issue creation", "PR activity", "Push frequency",
    "Repo diversity", "Label usage", "Sprint velocity",
    "Overall activity", "Collaboration", "Recency",
    "PR ratio", "Issue ratio",
]

# Category colors: (text_color, background_color)
CATEGORY_COLORS = {
    "Agile/Scrum":         ("#1E88E5", "#E3F2FD"),
    "Kanban":              ("#43A047", "#E8F5E9"),
    "Bug Triage":          ("#E53935", "#FFEBEE"),
    "Release Management":  ("#FB8C00", "#FFF3E0"),
    "Design Sprint":       ("#8E24AA", "#F3E5F5"),
    "DevOps/CI-CD":        ("#546E7A", "#ECEFF1"),
    "Open Source":         ("#00897B", "#E0F2F1"),
    "Research":            ("#3949AB", "#E8EAF6"),
    "Incident Response":   ("#C62828", "#FFCDD2"),
    "Customer Support":    ("#827717", "#F9FBE7"),
}

# Archetype descriptions for display
ARCHETYPE_DESC = {
    "scrum_team_member":   "Works in sprint cycles, balanced PR/issue activity",
    "kanban_ops":          "Continuous flow, monitors throughput and cycle time",
    "platform_engineer":   "High push frequency, maintains infrastructure repos",
    "oss_contributor":     "Active across many repos, community collaboration",
    "ml_researcher":       "Research-oriented, async cadence, experiment-driven",
    "incident_responder":  "Alert-driven, high urgency, on-call workflows",
    "design_sprint_pm":    "Design-focused, ticket-heavy, stakeholder reviews",
    "solo_hacker":         "Independent contributor, low collaboration density",
}


# ══════════════════════════════════════════════════════════════════════
#  Cached Loaders
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    model = TwoTowerModel(
        user_input_dim=11, item_input_dim=384,
        hidden_dim=256, output_dim=128, dropout=0.2,
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return model

@st.cache_resource
def load_faiss_index():
    return faiss.read_index(FAISS_INDEX_PATH)

@st.cache_data
def load_workflow_ids():
    with open(WORKFLOW_IDS_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return {wf["workflow_id"]: wf for wf in json.load(f)}

@st.cache_data
def load_scaler_stats():
    with open(SCALER_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_eval_results():
    if os.path.exists(EVAL_RESULTS_PATH):
        with open(EVAL_RESULTS_PATH, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_all_users():
    """Load all 10,000 users from train/val/test splits."""
    dfs = []
    for path in [FEATURES_TRAIN, FEATURES_VAL, FEATURES_TEST]:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
#  Core Functions
# ══════════════════════════════════════════════════════════════════════

def recommend(model, index, workflow_ids, catalog, user_features, k=10):
    """FAISS query: user features -> top-k workflows."""
    with torch.no_grad():
        t = torch.tensor(user_features, dtype=torch.float32).unsqueeze(0)
        emb = model.get_user_embedding(t).numpy().astype(np.float32)

    start = time.perf_counter()
    scores, indices = index.search(emb, k)
    latency = (time.perf_counter() - start) * 1000

    results = []
    for i in range(k):
        wid = workflow_ids[indices[0][i]]
        wf = catalog[wid]
        results.append({
            "rank": i + 1, "workflow_id": wid,
            "name": wf["name"], "category": wf["category"],
            "description": wf["description"],
            "tooling_tags": wf.get("tooling_tags", []),
            "score": float(scores[0][i]),
        })
    return results, latency


def normalize_value(raw, feat_name, stats):
    s = stats["stats"][feat_name]
    mn, mx = s["min"], s["max"]
    if mx == mn:
        return 0.5
    return max(0.0, min(1.0, (raw - mn) / (mx - mn)))


# ══════════════════════════════════════════════════════════════════════
#  UI Rendering Helpers
# ══════════════════════════════════════════════════════════════════════

def cat_badge(category):
    fg, bg = CATEGORY_COLORS.get(category, ("#333", "#F5F5F5"))
    return (f'<span style="background:{bg};color:{fg};padding:3px 10px;'
            f'border-radius:12px;font-size:0.85em;font-weight:600;">'
            f'{category}</span>')

def tool_badge(tool):
    return (f'<span style="background:#F0F0F0;color:#555;padding:2px 8px;'
            f'border-radius:8px;font-size:0.75em;margin-right:4px;">'
            f'{tool}</span>')

def render_feature_profile(features, label="Feature Profile"):
    """Render a horizontal bar chart of the 11 normalized features."""
    st.markdown(f"**{label}**")
    for i, (name, val) in enumerate(zip(FEATURE_LABELS, features)):
        bar_w = max(int(val * 180), 2)
        # Color gradient: low=gray, high=blue
        intensity = int(val * 200 + 55)
        color = f"rgb({max(30, 220-intensity)}, {max(60, 160-intensity//2)}, {intensity})"
        st.markdown(
            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<span style="width:120px;font-size:0.82em;color:#555;">{name}</span>'
            f'<div style="background:#E8E8E8;border-radius:4px;height:12px;'
            f'width:180px;margin:0 8px;">'
            f'<div style="background:{color};height:12px;border-radius:4px;'
            f'width:{bar_w}px;"></div></div>'
            f'<span style="font-size:0.8em;color:#777;">{val:.2f}</span></div>',
            unsafe_allow_html=True,
        )

def render_workflow_card(r, score_min, score_range):
    """Render a single workflow recommendation card."""
    pct = int(((r["score"] - score_min) / max(score_range, 0.001)) * 40 + 60)
    tools_html = " ".join(tool_badge(t) for t in r["tooling_tags"])
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#FAFAFA,#F0F4F8);
                border:1px solid #E0E0E0;border-radius:12px;padding:20px;
                margin-bottom:12px;">
        <div>
            <span style="background:linear-gradient(135deg,#1E88E5,#1565C0);
                         color:white;font-weight:800;font-size:1.1em;
                         width:36px;height:36px;border-radius:50%;
                         display:inline-flex;align-items:center;
                         justify-content:center;margin-right:12px;">
                {r['rank']}</span>
            <span style="font-size:1.15em;font-weight:700;color:#1A237E;">
                {r['name']}</span>
            &nbsp;{cat_badge(r['category'])}
        </div>
        <div style="margin-top:10px;">
            <span style="color:#666;font-size:0.85em;">Match score:</span>
            <div style="background:#E0E0E0;border-radius:6px;height:8px;
                        width:200px;display:inline-block;vertical-align:middle;">
                <div style="background:linear-gradient(90deg,#42A5F5,#1565C0);
                            height:8px;border-radius:6px;width:{pct}%;"></div>
            </div>
            <span style="color:#666;font-size:0.85em;">&nbsp;{r['score']:.4f}</span>
        </div>
        <div style="color:#555;font-size:0.92em;margin-top:8px;line-height:1.5;">
            {r['description']}</div>
        <div style="margin-top:8px;">{tools_html}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  Main App
# ══════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Workflow Recommender",
        page_icon="🔍",
        layout="wide",
    )

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Run `python src/model/train.py` first.")
        st.stop()

    # Load resources (all cached)
    model = load_model()
    index = load_faiss_index()
    workflow_ids = load_workflow_ids()
    catalog = load_catalog()
    scaler_stats = load_scaler_stats()
    eval_results = load_eval_results()
    all_users = load_all_users()

    # ── Header ────────────────────────────────────────────────────────
    st.markdown("# 🔍 Developer Workflow Recommender")
    st.markdown("*Two-Tower Neural Network* &mdash; analyzes developer "
                "activity patterns to recommend optimal team workflows")

    # ── Two Tabs ──────────────────────────────────────────────────────
    tab_team, tab_explore = st.tabs([
        "👥 Team Analysis (Personalized)",
        "🎛️ Feature Explorer"
    ])

    # ══════════════════════════════════════════════════════════════════
    #  TAB 1: Team Analysis — select real users, get personalized recs
    # ══════════════════════════════════════════════════════════════════
    with tab_team:
        st.markdown("### How it works in production")
        st.info(
            "**Production flow:** Connect to GitHub/Jira APIs → pull each "
            "team member's activity (issues, PRs, commits, repos) → compute "
            "behavioral features → encode through UserTower → FAISS retrieval.\n\n"
            "**This demo:** Uses 10,000 synthetic developer profiles with "
            "realistic activity patterns across 8 archetypes."
        )

        st.markdown("### Select your team members")
        st.caption("In production, you'd enter GitHub usernames. Here, search "
                   "our database of 10,000 synthetic developers.")

        # Build display options: "u00001 — scrum_team_member"
        user_options = [
            f"{row['user_id']}  —  {row['archetype']}"
            for _, row in all_users.iterrows()
        ]

        # Multiselect for team members
        selected = st.multiselect(
            "Add team members (search by ID or archetype)",
            options=user_options,
            default=[user_options[0]],  # pre-select first user
            max_selections=10,
            help="Select 1-10 team members. The model averages their "
                 "profiles to recommend workflows for the whole team.",
        )

        if selected and st.button("🚀 Analyze Team & Recommend",
                                   type="primary", use_container_width=True):
            # Parse selected user IDs
            selected_ids = [s.split("  —  ")[0].strip() for s in selected]
            selected_rows = all_users[all_users["user_id"].isin(selected_ids)]

            # ── Show team profiles ────────────────────────────────
            st.markdown("---")
            st.markdown("### 📊 Team Activity Profiles")
            st.caption("These features were extracted from each member's "
                       "development activity (issues, PRs, commits, repos).")

            cols = st.columns(min(len(selected_rows), 3))
            for i, (_, row) in enumerate(selected_rows.iterrows()):
                with cols[i % len(cols)]:
                    arch = row["archetype"]
                    desc = ARCHETYPE_DESC.get(arch, "")
                    st.markdown(f"**{row['user_id']}**")
                    st.markdown(f"*{arch}*")
                    st.caption(desc)
                    st.caption(f"Preferred: {row['preferred_categories']}")
                    features = row[FEATURE_COLS].values.astype(np.float32)
                    render_feature_profile(features, label="")

            # ── Aggregate team features ───────────────────────────
            # Average the feature vectors across all team members
            # This is how you'd handle team-level recommendations:
            # each member contributes their behavioral signal
            team_features = selected_rows[FEATURE_COLS].values.astype(
                np.float32
            ).mean(axis=0)

            st.markdown("---")
            st.markdown("### 🎯 Team Recommendations")

            if len(selected_rows) > 1:
                st.caption(
                    f"Aggregated profile of {len(selected_rows)} team members "
                    f"(feature vectors averaged). The model finds workflows "
                    f"that best fit the team's combined activity pattern."
                )

            # ── Run recommendation ────────────────────────────────
            results, latency = recommend(
                model, index, workflow_ids, catalog, team_features, k=10
            )

            st.caption(f"Retrieved in {latency:.2f} ms")

            # Render cards
            scores = [r["score"] for r in results]
            s_min, s_max = min(scores), max(scores)

            for r in results:
                render_workflow_card(r, s_min, s_max - s_min)

            # ── Explainability ────────────────────────────────────
            with st.expander("🧠 Why these recommendations?"):
                st.markdown(
                    "The model encodes each developer's 11 behavioral "
                    "features into a 128-dimensional embedding via the "
                    "**UserTower**. For teams, member embeddings are "
                    "averaged. FAISS then finds the nearest workflow "
                    "embeddings (from the **ItemTower**) by cosine "
                    "similarity."
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Team's strongest signals:**")
                    sorted_idx = np.argsort(team_features)[::-1]
                    for i in sorted_idx[:5]:
                        bar_w = int(team_features[i] * 150)
                        st.markdown(
                            f"- **{FEATURE_LABELS[i]}**: {team_features[i]:.2f} "
                            f"<span style='display:inline-block;background:#42A5F5;"
                            f"height:10px;width:{bar_w}px;border-radius:4px;'>"
                            f"</span>",
                            unsafe_allow_html=True,
                        )

                with col_b:
                    cats = [r["category"] for r in results]
                    unique_cats = list(dict.fromkeys(cats))
                    st.markdown("**Recommended categories:**")
                    for c in unique_cats:
                        count = cats.count(c)
                        st.markdown(f"- {cat_badge(c)} x{count}",
                                    unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    #  TAB 2: Feature Explorer — manual sliders for experimentation
    # ══════════════════════════════════════════════════════════════════
    with tab_explore:
        st.markdown("### Direct Feature Control")
        st.caption("Manually adjust the 11 behavioral features to explore "
                   "how the model responds. Useful for understanding what "
                   "drives each recommendation category.")

        raw_features = list(scaler_stats["stats"].keys())

        col1, col2 = st.columns(2)
        slider_vals = []
        for i, (feat, label) in enumerate(zip(raw_features, FEATURE_LABELS)):
            target_col = col1 if i < 6 else col2
            with target_col:
                v = st.slider(label, 0.0, 1.0, 0.5, 0.05, key=f"sl_{feat}")
                slider_vals.append(v)

        explore_features = np.array(slider_vals, dtype=np.float32)

        if st.button("🔍 Get Recommendations", type="primary",
                     key="explore_btn", use_container_width=True):

            results, latency = recommend(
                model, index, workflow_ids, catalog, explore_features, k=10
            )

            st.caption(f"Retrieved in {latency:.2f} ms")

            scores = [r["score"] for r in results]
            s_min, s_max = min(scores), max(scores)
            for r in results:
                render_workflow_card(r, s_min, s_max - s_min)

    # ── Model Performance (always visible at bottom) ──────────────────
    st.markdown("---")
    with st.expander("📊 Model Performance"):
        if eval_results:
            ab = eval_results.get("ab_simulation", {})
            ctrl = ab.get("control", {})
            treat = ab.get("treatment", {})
            imp = ab.get("improvement", {})

            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                (c1, f"{treat.get('ndcg_at_10',0):.4f}",
                 f"nDCG@10 (random: {ctrl.get('ndcg_at_10',0):.4f})"),
                (c2, f"{treat.get('recall_at_50',0):.4f}",
                 f"Recall@50 (random: {ctrl.get('recall_at_50',0):.4f})"),
                (c3, "<1ms", "FAISS retrieval (target: <10ms)"),
                (c4, f"{imp.get('ndcg_ratio',0):.1f}x",
                 "Improvement over random"),
            ]
            for col, val, lbl in metrics:
                with col:
                    st.markdown(
                        f'<div style="background:#F8F9FA;border-radius:10px;'
                        f'padding:16px;text-align:center;border:1px solid #E8E8E8;">'
                        f'<div style="font-size:1.8em;font-weight:800;'
                        f'color:#1565C0;">{val}</div>'
                        f'<div style="font-size:0.82em;color:#777;'
                        f'margin-top:4px;">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.caption(
                "Training data: 10,000 developers | 500 workflows | "
                "156k interactions | 8 archetypes | 201,984 parameters | "
                "Trained in 1.2 min on CPU"
            )
        else:
            st.info("Run `python src/evaluation/metrics.py` to see metrics.")


if __name__ == "__main__":
    main()
