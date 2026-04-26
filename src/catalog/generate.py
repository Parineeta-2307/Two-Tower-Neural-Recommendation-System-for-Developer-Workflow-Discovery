"""
src/catalog/generate.py
───────────────────────
Generates 500 developer workflow templates — the "items"
your recommender will suggest to teams.

Each workflow has:
  - workflow_id    : unique identifier (e0001 … e0500)
  - name           : short human-readable title
  - category       : one of 10 high-level buckets
  - description    : 1-2 sentence explanation (this gets encoded by sentence-transformers later)
  - avg_team_size  : typical team size using this workflow
  - tooling_tags   : list of tools commonly used with this workflow

Run with:
    python src/catalog/generate.py
Output:
    data/catalog/workflows.json
"""

import json       # built-in: saves your data as a JSON file
import random     # built-in: for realistic variation in the generated data
import os         # built-in: for creating folders if they don't exist

# ── STEP 1: Define the building blocks ─────────────────────────────────────────
#
# Instead of writing 500 workflows by hand, we define components and
# combine them programmatically. This is called "synthetic data generation."
# Real companies do this all the time when they don't have labeled data yet.

# 10 categories — these are the high-level workflow families
CATEGORIES = [
    "Agile/Scrum",
    "Kanban",
    "Bug Triage",
    "Release Management",
    "Design Sprint",
    "DevOps/CI-CD",
    "Open Source",
    "Research",
    "Incident Response",
    "Customer Support",
]

# For each category, we define:
#   - name_templates : fill-in-the-blank names (we'll substitute {var} placeholders)
#   - desc_templates : description templates
#   - tools          : tools commonly associated with this category
#   - team_sizes     : realistic team size range [min, max]

CATEGORY_DATA = {

    "Agile/Scrum": {
        "name_templates": [
            "{sprint_len}-week Scrum with daily standups",
            "Scrum for {team_type} teams with {ceremony} ceremonies",
            "Scaled Scrum using {framework} framework",
            "Scrum with {estimation} estimation and {review} reviews",
            "Lightweight Scrum for {team_type} squads",
        ],
        "desc_templates": [
            "A {sprint_len}-week sprint cycle with sprint planning, daily standups, review, and retrospective. Designed for {team_type} teams that need predictable delivery cadence.",
            "Scrum implementation emphasizing {ceremony} ceremonies and {estimation} estimation. Teams maintain a groomed backlog and deliver working software every {sprint_len} weeks.",
            "Scaled agile approach using {framework} for coordinating multiple squads. Includes cross-team synchronization and program increment planning.",
        ],
        "vars": {
            "sprint_len": ["1", "2", "3", "4"],
            "team_type":  ["product", "platform", "infrastructure", "mobile", "data", "frontend", "backend", "full-stack"],
            "ceremony":   ["lightweight", "full", "async-friendly", "remote-first"],
            "estimation": ["story-point", "t-shirt sizing", "no-estimate", "fibonacci"],
            "review":     ["stakeholder", "internal", "demo-heavy", "written"],
            "framework":  ["SAFe", "LeSS", "Nexus", "custom"],
        },
        "tools": ["Jira", "Linear", "Confluence", "Notion", "Miro", "Figma"],
        "team_sizes": [4, 12],
    },

    "Kanban": {
        "name_templates": [
            "Kanban with {wip} WIP limits and {cadence} cadence",
            "Flow-based Kanban for {team_type} teams",
            "Kanban with {metric} as primary metric",
            "Continuous-flow Kanban with {review} reviews",
        ],
        "desc_templates": [
            "A continuous-flow Kanban system with explicit WIP limits of {wip} per stage. Optimised for {team_type} teams where work items arrive unpredictably.",
            "Lean Kanban board tracking {metric} as the primary flow metric. Includes weekly replenishment meetings and daily queue reviews.",
        ],
        "vars": {
            "wip":       ["3", "4", "5", "6", "unlimited with expedite lane"],
            "cadence":   ["weekly", "bi-weekly", "daily", "on-demand"],
            "team_type": ["support", "ops", "design", "research", "maintenance"],
            "metric":    ["cycle time", "throughput", "lead time", "queue age"],
            "review":    ["async", "synchronous", "written", "visual"],
        },
        "tools": ["Trello", "Jira", "Linear", "Notion", "Asana", "GitHub Projects"],
        "team_sizes": [2, 8],
    },

    "Bug Triage": {
        "name_templates": [
            "{severity}-severity bug triage with {sla} SLA",
            "Bug triage using {framework} priority framework",
            "Automated bug triage with {automation} routing",
            "{cadence} bug triage for {team_type} teams",
        ],
        "desc_templates": [
            "Structured bug triage process classifying issues by {severity} severity with defined {sla} SLA for each tier. Includes automated routing and weekly review.",
            "Priority-based bug management using {framework} framework. Bugs are triaged within 24 hours and assigned based on component ownership.",
        ],
        "vars": {
            "severity":   ["P0-P3", "critical/high/medium/low", "impact-based", "customer-facing-first"],
            "sla":        ["4-hour", "24-hour", "48-hour", "sprint-based"],
            "framework":  ["Eisenhower matrix", "MoSCoW", "RICE scoring", "custom impact-effort"],
            "automation": ["label-based", "ML-assisted", "rule-based", "owner-based"],
            "cadence":    ["daily", "twice-weekly", "weekly"],
            "team_type":  ["product", "platform", "infrastructure", "mobile"],
        },
        "tools": ["Jira", "GitHub Issues", "Linear", "PagerDuty", "Sentry"],
        "team_sizes": [3, 10],
    },

    "Release Management": {
        "name_templates": [
            "{release_type} release with {gating} quality gates",
            "Trunk-based {release_type} release process",
            "Release train with {cadence} cadence",
            "{rollout} rollout strategy for {team_type} services",
        ],
        "desc_templates": [
            "A {release_type} release process with automated {gating} quality gates and {rollout} deployment strategy. Enables confident and repeatable releases.",
            "Release train model with {cadence} fixed cadence. Includes feature flagging, automated smoke tests, and {rollout} rollout to production.",
        ],
        "vars": {
            "release_type": ["weekly", "bi-weekly", "monthly", "continuous", "hotfix"],
            "gating":       ["automated test", "manual QA", "canary", "staged"],
            "rollout":      ["blue-green", "canary", "feature-flag", "ring-based"],
            "cadence":      ["weekly", "bi-weekly", "monthly"],
            "team_type":    ["backend", "frontend", "mobile", "data", "platform"],
        },
        "tools": ["GitHub Actions", "Jenkins", "ArgoCD", "LaunchDarkly", "PagerDuty", "Confluence"],
        "team_sizes": [4, 15],
    },

    "Design Sprint": {
        "name_templates": [
            "{duration}-day design sprint for {problem_type} problems",
            "Lean design sprint with {team_type} team",
            "Remote-first design sprint using {tools}",
            "Design sprint focused on {outcome} outcomes",
        ],
        "desc_templates": [
            "A {duration}-day structured design sprint to answer critical {problem_type} questions through prototyping and user testing. Includes map, sketch, decide, prototype, and test phases.",
            "Compressed design sprint adapted for {team_type} teams. Focuses on rapid prototyping and validated learning within a {duration}-day window.",
        ],
        "vars": {
            "duration":    ["3", "4", "5"],
            "problem_type":["UX", "business", "technical", "product-market fit"],
            "team_type":   ["cross-functional", "product", "design", "startup"],
            "tools":       ["Miro + Figma", "FigJam + Notion", "Mural + Loom"],
            "outcome":     ["user-validated", "stakeholder-aligned", "prototype-ready"],
        },
        "tools": ["Figma", "Miro", "FigJam", "Notion", "Loom", "Maze"],
        "team_sizes": [4, 8],
    },

    "DevOps/CI-CD": {
        "name_templates": [
            "{pipeline} CI/CD pipeline with {testing} testing strategy",
            "GitOps workflow with {deployment} deployments",
            "DevOps with {frequency} deploy frequency",
            "Platform engineering workflow for {team_type} teams",
        ],
        "desc_templates": [
            "A {pipeline} CI/CD pipeline running {testing} test suites on every commit. Enables {frequency} deployments with full observability and automated rollback.",
            "GitOps-based {pipeline} workflow with {deployment} deployments. Infrastructure is version-controlled and changes are driven by pull requests.",
        ],
        "vars": {
            "pipeline":   ["GitHub Actions", "Jenkins", "GitLab CI", "CircleCI", "Buildkite"],
            "testing":    ["unit + integration", "full E2E", "contract", "snapshot"],
            "deployment": ["automated", "approval-gated", "scheduled", "on-demand"],
            "frequency":  ["multiple-times-daily", "daily", "weekly", "on-demand"],
            "team_type":  ["platform", "infrastructure", "SRE", "backend"],
        },
        "tools": ["GitHub Actions", "ArgoCD", "Terraform", "Datadog", "PagerDuty", "Grafana"],
        "team_sizes": [2, 10],
    },

    "Open Source": {
        "name_templates": [
            "Open source contribution workflow with {contribution_type} focus",
            "OSS governance using {model} model",
            "Community-driven workflow with {review_process} review",
            "Open source project with {release_cadence} releases",
        ],
        "desc_templates": [
            "Contributor workflow for open source projects emphasising {contribution_type}. Includes clear CONTRIBUTING.md, issue templates, and {review_process} code review process.",
            "Governance model based on {model} with transparent decision-making. Releases follow {release_cadence} cadence with changelog automation.",
        ],
        "vars": {
            "contribution_type": ["bug fixes", "feature development", "documentation", "security"],
            "model":     ["BDFL", "committee", "meritocracy", "foundation-backed"],
            "review_process":   ["async PR", "synchronous pair", "RFC-based", "automated + human"],
            "release_cadence":  ["semantic versioning", "time-based", "milestone-based"],
        },
        "tools": ["GitHub", "GitLab", "Discourse", "Slack", "Read the Docs", "Changelog.md"],
        "team_sizes": [1, 20],
    },

    "Research": {
        "name_templates": [
            "Research workflow with {output} output cadence",
            "Experiment tracking with {tool} and {review_type} reviews",
            "ML research workflow with {reproducibility} reproducibility",
            "Academic-industry hybrid workflow for {team_type} teams",
        ],
        "desc_templates": [
            "Structured research workflow with {output} deliverable cadence. Includes hypothesis tracking, experiment logging, and {review_type} peer review.",
            "ML experiment management using {tool} with enforced {reproducibility} reproducibility. Weekly research syncs and monthly result write-ups.",
        ],
        "vars": {
            "output":          ["weekly memo", "bi-weekly report", "monthly paper draft"],
            "tool":            ["MLflow", "Weights & Biases", "Neptune", "DVC", "ClearML"],
            "review_type":     ["peer", "manager", "cross-team", "external"],
            "reproducibility": ["full", "partial", "checkpoint-based"],
            "team_type":       ["ML", "data science", "academic", "applied AI"],
        },
        "tools": ["Weights & Biases", "MLflow", "Jupyter", "GitHub", "Overleaf", "Notion"],
        "team_sizes": [2, 8],
    },

    "Incident Response": {
        "name_templates": [
            "{severity}-severity incident response with {on_call} on-call",
            "SRE incident workflow with {postmortem} postmortems",
            "Incident management with {tool} and {communication} comms",
            "{mttr}-hour MTTR target incident response process",
        ],
        "desc_templates": [
            "Incident response process for {severity}-severity events with {on_call} on-call rotation. Includes automated alerting, war room procedures, and {postmortem} postmortem culture.",
            "SRE-style incident management targeting {mttr}-hour MTTR. Uses {tool} for incident coordination with structured {postmortem} retrospectives.",
        ],
        "vars": {
            "severity":   ["P0-P1", "all-severity", "customer-impacting", "production"],
            "on_call":    ["24/7 rotating", "business-hours", "follow-the-sun", "tiered"],
            "postmortem": ["blameless", "action-item-focused", "public", "internal"],
            "tool":       ["PagerDuty", "Opsgenie", "Incident.io", "Slack-native"],
            "communication": ["Slack-first", "email + Slack", "status-page-first"],
            "mttr":       ["1", "2", "4", "8"],
        },
        "tools": ["PagerDuty", "Opsgenie", "Datadog", "Slack", "Statuspage", "Confluence"],
        "team_sizes": [3, 12],
    },

    "Customer Support": {
        "name_templates": [
            "Customer support workflow with {sla} SLA and {tier} tiering",
            "{channel} support workflow with {escalation} escalation",
            "Support workflow with {automation} automation",
            "Technical support process for {customer_type} customers",
        ],
        "desc_templates": [
            "Customer support process with {tier}-tier structure and {sla} first-response SLA. Includes {channel} support channels and {escalation} escalation paths to engineering.",
            "Support workflow optimised for {customer_type} customers with {automation} automation for common queries. Maintains {sla} SLA across all {channel} channels.",
        ],
        "vars": {
            "sla":           ["1-hour", "4-hour", "24-hour", "business-hours"],
            "tier":          ["2", "3", "4"],
            "channel":       ["Intercom", "Zendesk", "email + chat", "multi-channel"],
            "escalation":    ["Jira-linked", "Slack-based", "PagerDuty", "linear"],
            "automation":    ["chatbot-first", "macro-based", "AI-assisted", "rule-based"],
            "customer_type": ["enterprise", "SMB", "developer", "consumer"],
        },
        "tools": ["Zendesk", "Intercom", "Jira", "Slack", "Salesforce", "Linear"],
        "team_sizes": [3, 20],
    },
}


# ── STEP 2: The generator function ────────────────────────────────────────────
#
# This function takes one category's data and generates `count` workflows from it.
# It uses random.choice() to pick different template combinations each time.

def generate_workflows_for_category(category_name, category_data, count, id_start):
    """
    Generate `count` workflow dictionaries for a given category.

    Parameters:
        category_name  : e.g. "Agile/Scrum"
        category_data  : the dict from CATEGORY_DATA for this category
        count          : how many to generate (50 per category × 10 categories = 500)
        id_start       : starting ID number so IDs don't overlap across categories

    Returns:
        list of dicts, each representing one workflow
    """
    workflows = []   # we'll append to this list

    for i in range(count):
        workflow_id = f"w{id_start + i:04d}"   # e.g. w0001, w0042, w0499
                                                # :04d means zero-pad to 4 digits

        # Pick a random name template and fill in variables
        name_template = random.choice(category_data["name_templates"])
        desc_template = random.choice(category_data["desc_templates"])

        # Fill in {var} placeholders with random choices from the vars dict
        # Example: "{sprint_len}-week Scrum" → "2-week Scrum"
        vars_dict = category_data.get("vars", {})
        filled_vars = {}   # stores the chosen value for each variable
        for var_name, options in vars_dict.items():
            filled_vars[var_name] = random.choice(options)

        # .format_map() replaces all {var} in the template string
        # We use a defaultdict-style approach so missing vars don't crash
        try:
            name = name_template.format(**filled_vars)
            description = desc_template.format(**filled_vars)
        except KeyError:
            # If a template references a var not in filled_vars, skip gracefully
            name = name_template.split("{")[0].strip()   # use just the prefix
            description = desc_template.split("{")[0].strip()

        # Pick 2–4 random tools from this category's tool list
        tools = random.sample(
            category_data["tools"],
            k=min(random.randint(2, 4), len(category_data["tools"]))
        )

        # Generate a realistic team size within this category's range
        min_size, max_size = category_data["team_sizes"]
        avg_team_size = random.randint(min_size, max_size)

        # Assemble the final workflow dict
        workflow = {
            "workflow_id":   workflow_id,
            "name":          name,
            "category":      category_name,
            "description":   description,
            "avg_team_size": avg_team_size,
            "tooling_tags":  tools,
        }
        workflows.append(workflow)

    return workflows


# ── STEP 3: Main execution ─────────────────────────────────────────────────────

def main():
    random.seed(42)   # seed = reproducibility: running this twice gives same output
                      # 42 is a convention (Hitchhiker's Guide to the Galaxy reference)

    all_workflows = []
    workflows_per_category = 50   # 50 × 10 categories = 500 total

    print("Generating workflow catalog...")
    print(f"{'Category':<25} {'Count':>6}")
    print("─" * 33)

    for idx, (category_name, category_data) in enumerate(CATEGORY_DATA.items()):
        id_start = idx * workflows_per_category
        workflows = generate_workflows_for_category(
            category_name,
            category_data,
            count=workflows_per_category,
            id_start=id_start,
        )
        all_workflows.extend(workflows)   # .extend() appends a whole list at once
        print(f"  {category_name:<23} {len(workflows):>6} workflows")

    print("─" * 33)
    print(f"  {'TOTAL':<23} {len(all_workflows):>6} workflows")

    # ── Save to JSON ───────────────────────────────────────────────────────────
    output_dir  = "data/catalog"
    output_path = os.path.join(output_dir, "workflows.json")

    # Create the folder if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)   # exist_ok=True: no error if already exists

    with open(output_path, "w", encoding="utf-8") as f:
        # indent=2 makes the JSON human-readable (pretty-printed)
        json.dump(all_workflows, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {output_path}")
    print("\nSample workflow (first item):")
    print(json.dumps(all_workflows[0], indent=2))
    print("\nSample workflow (last item):")
    print(json.dumps(all_workflows[-1], indent=2))


# ── Entry point ────────────────────────────────────────────────────────────────
# This block only runs when you execute this file directly.
# It does NOT run if another file imports this file.
# This is a Python best practice — always wrap your main() call like this.

if __name__ == "__main__":
    main()