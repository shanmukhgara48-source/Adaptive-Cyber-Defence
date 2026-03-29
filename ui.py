#!/usr/bin/env python3
"""
ui.py — Streamlit SOC Dashboard for Adaptive Cyber Defense Simulator

Run with:
    streamlit run adaptive_cyber_defense/ui.py

Or from the project directory:
    streamlit run ui.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# ── path bootstrap ───────────────────────────────────────────────────────────
# Allows running as `streamlit run ui.py` from inside the package dir
# or from the parent directory.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from adaptive_cyber_defense.models.action import Action, ActionInput
from adaptive_cyber_defense.models.state import AttackStage
from adaptive_cyber_defense.agents.baseline import BaselineAgent
from adaptive_cyber_defense.tasks import EasyTask, MediumTask, HardTask

# ── optional graph dep (streamlit-agraph) ────────────────────────────────────
try:
    from streamlit_agraph import agraph, Node, Edge, Config as AGraphConfig
    _AGRAPH = True
except ImportError:
    _AGRAPH = False

_TRAINING_DIR = _HERE / "training"
_QTABLE_PATH  = _HERE / "agents" / "ql_table.json"

# ---------------------------------------------------------------------------
# MITRE ATT&CK mapping  (attack_type → technique IDs + names)
# ---------------------------------------------------------------------------
_MITRE_MAP: dict[str, list[tuple[str, str]]] = {
    "phishing":      [("T1566", "Phishing"), ("T1204", "User Execution")],
    "apt":           [("T1059", "Command & Scripting"), ("T1071", "App Layer Protocol"),
                      ("T1027", "Obfuscated Files")],
    "ransomware":    [("T1486", "Data Encrypted for Impact"), ("T1490", "Inhibit Recovery")],
    "ddos":          [("T1498", "Network DoS"), ("T1499", "Endpoint DoS")],
    "insider":       [("T1078", "Valid Accounts"), ("T1565", "Data Manipulation")],
    "zero_day":      [("T1203", "Exploitation for Client Execution"),
                      ("T1190", "Exploit Public-Facing App")],
    "supply_chain":  [("T1195", "Supply Chain Compromise"), ("T1199", "Trusted Relationship")],
    "generic":       [("T1110", "Brute Force"), ("T1083", "File & Dir Discovery")],
}


# ---------------------------------------------------------------------------
# Incident Report Generator
# ---------------------------------------------------------------------------

def generate_incident_report(episode_data: dict) -> dict:
    """
    Build a structured incident report from episode_data.

    Parameters
    ----------
    episode_data : dict with keys:
        task_name, seed, steps, total_reward, step_rewards,
        actions_taken     : list[str]
        threats_seen      : list[Threat]
        compromised_nodes : list[str]  (max observed over episode)
        lateral_moves     : int
        final_state       : EnvironmentState
        episode_score     : float

    Returns
    -------
    dict — structured report (also rendered as Markdown in the UI)
    """
    import datetime

    threats         = episode_data.get("threats_seen", [])
    actions         = episode_data.get("actions_taken", [])
    compromised     = episode_data.get("compromised_nodes", [])
    steps           = episode_data.get("steps", 0)
    total_reward    = episode_data.get("total_reward", 0.0)
    episode_score   = episode_data.get("episode_score", 0.0)
    lateral_moves   = episode_data.get("lateral_moves", 0)
    final_state     = episode_data.get("final_state")
    task_name       = episode_data.get("task_name", "unknown")
    seed            = episode_data.get("seed", 0)
    step_rewards    = episode_data.get("step_rewards", [])

    # ── attack types ─────────────────────────────────────────────────────────
    attack_types = sorted({
        getattr(t, "attack_type", "generic") for t in threats
    }) or ["generic"]

    # ── MITRE techniques ─────────────────────────────────────────────────────
    mitre: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for at in attack_types:
        for tid, tname in _MITRE_MAP.get(at, _MITRE_MAP["generic"]):
            if tid not in seen_ids:
                mitre.append((tid, tname))
                seen_ids.add(tid)

    # ── peak severity ─────────────────────────────────────────────────────────
    peak_sev = max((getattr(t, "severity", 0.0) for t in threats), default=0.0)
    if peak_sev >= 0.75:   sev_label = "CRITICAL"
    elif peak_sev >= 0.50: sev_label = "HIGH"
    elif peak_sev >= 0.25: sev_label = "MEDIUM"
    else:                  sev_label = "LOW"

    # ── kill-chain stages reached ─────────────────────────────────────────────
    stages_reached = sorted({
        getattr(t, "stage", None) for t in threats
        if getattr(t, "stage", None) is not None
    }, key=lambda s: s.value)
    stage_names = [s.name for s in stages_reached]

    # ── action breakdown ──────────────────────────────────────────────────────
    from collections import Counter
    action_counts = Counter(actions)

    # ── outcome classification ────────────────────────────────────────────────
    final_threats = getattr(final_state, "active_threats", []) if final_state else []
    still_active  = [t for t in final_threats if not getattr(t, "is_contained", False)]
    n_compromised = len(compromised)

    if n_compromised == 0 and not still_active:
        outcome       = "✅ CONTAINED"
        outcome_label = "Contained"
    elif n_compromised <= 2 and episode_score >= 0.40:
        outcome       = "⚠️ PARTIALLY CONTAINED"
        outcome_label = "Partially Contained"
    elif lateral_moves > 3 or n_compromised >= 4:
        outcome       = "🚨 BREACH"
        outcome_label = "Breach"
    else:
        outcome       = "📈 ESCALATED"
        outcome_label = "Escalated"

    # ── avg reward trend (first half vs second half) ──────────────────────────
    trend = "N/A"
    if len(step_rewards) >= 10:
        mid      = len(step_rewards) // 2
        avg_fh   = sum(step_rewards[:mid])  / mid
        avg_sh   = sum(step_rewards[mid:])  / (len(step_rewards) - mid)
        trend    = f"{'↑ improving' if avg_sh > avg_fh else '↓ declining'} " \
                   f"({avg_fh:.3f} → {avg_sh:.3f})"

    report = {
        "generated_at":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task":          task_name,
        "seed":          seed,
        # Attack summary
        "attack_types":  attack_types,
        "mitre":         mitre,
        "stages_reached": stage_names,
        # Impact
        "compromised_nodes": list(compromised),
        "n_compromised":     n_compromised,
        "peak_severity":     round(peak_sev, 3),
        "severity_label":    sev_label,
        "lateral_moves":     lateral_moves,
        # Response
        "actions_taken":  dict(action_counts),
        "total_actions":  len(actions),
        # Outcome
        "outcome":        outcome,
        "outcome_label":  outcome_label,
        # Performance
        "total_reward":   round(total_reward, 4),
        "episode_score":  round(episode_score, 4),
        "steps":          steps,
        "reward_trend":   trend,
    }
    return report


def _report_markdown(r: dict) -> str:
    """Render report dict as a Markdown string for display and download."""
    atk  = ", ".join(r["attack_types"]) or "unknown"
    mit  = "  \n  ".join(f"`{tid}` — {tn}" for tid, tn in r["mitre"]) or "N/A"
    acts = "\n".join(
        f"  - `{a}` × {n}" for a, n in sorted(r["actions_taken"].items())
    ) or "  - None recorded"
    stages = " → ".join(r.get("stages_reached", [])) or "N/A"
    comp   = ", ".join(f"`{n}`" for n in r["compromised_nodes"]) or "None"

    return f"""# 🔒 Incident Report
**Generated:** {r['generated_at']}  |  **Task:** {r['task']}  |  **Seed:** {r['seed']}

---

## 🎯 Attack Summary
| Field | Value |
|---|---|
| Attack Types | {atk} |
| Kill-Chain Stages | {stages} |
| MITRE Techniques | {", ".join(tid for tid, _ in r["mitre"])} |

{chr(10).join(f"- `{tid}` {tn}" for tid, tn in r["mitre"])}

---

## 💥 Impact
| Metric | Value |
|---|---|
| Compromised Systems | **{r['n_compromised']}** ({comp}) |
| Peak Threat Severity | **{r['severity_label']}** ({r['peak_severity']:.0%}) |
| Lateral Movements | {r['lateral_moves']} |

---

## 🛡️ Response Actions
| Action | Count |
|---|---|
{"".join(f"| `{a}` | {n} |{chr(10)}" for a, n in sorted(r['actions_taken'].items()))}
**Total actions taken:** {r['total_actions']}

---

## 📊 Outcome
> ### {r['outcome']}

---

## 📈 Performance
| Metric | Value |
|---|---|
| Episode Score | **{r['episode_score']:.4f}** |
| Total Reward | {r['total_reward']:.4f} |
| Episode Duration | {r['steps']} steps |
| Reward Trend | {r['reward_trend']} |

---
*Generated by Adaptive Cyber Defense Simulator*
"""

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cyber Defense SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark terminal log */
.log-terminal {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    background: #0d1117;
    color: #39d353;
    padding: 12px 16px;
    border-radius: 6px;
    border: 1px solid #30363d;
    max-height: 260px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
}
/* Network legend pills */
.legend-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 6px;
}
/* Compact metric cards */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── constants ────────────────────────────────────────────────────────────────
TASK_MAP = {"Easy": EasyTask, "Medium": MediumTask, "Hard": HardTask}

ASSET_ICON = {
    "firewall":    "🔥",
    "router":      "🌐",
    "server":      "🖥️",
    "database":    "🗄️",
    "workstation": "💻",
}

STAGE_ICON = {
    "PHISHING":           "🎣",
    "CREDENTIAL_ACCESS":  "🔑",
    "MALWARE_INSTALL":    "💀",
    "LATERAL_SPREAD":     "🕸️",
    "EXFILTRATION":       "📤",
}

STAGE_COLOR = {
    "PHISHING":           "#f0c030",
    "CREDENTIAL_ACCESS":  "#e08030",
    "MALWARE_INSTALL":    "#d04040",
    "LATERAL_SPREAD":     "#c02020",
    "EXFILTRATION":       "#800000",
}


# ── Network graph static topology (matches NetworkGraph.build_default) ───────
# Pre-defined edges; derived from connected_to adjacency (undirected, deduped).
_TOPOLOGY_EDGES: list[tuple[str, str]] = [
    ("fw-01",    "router-01"),
    ("router-01","srv-web"),
    ("router-01","srv-db"),
    ("router-01","ws-01"),
    ("srv-web",  "srv-db"),
    ("srv-db",   "ws-02"),
    ("srv-db",   "db-01"),
    ("db-01",    "ws-02"),
    ("ws-01",    "ws-03"),
    ("ws-02",    "ws-03"),
]

# Node layout positions (x, y) in a roughly hub-and-spoke arrangement
_NODE_POS: dict[str, tuple[int, int]] = {
    "fw-01":    (400,  50),
    "router-01":(400, 180),
    "srv-web":  (200, 310),
    "srv-db":   (400, 310),
    "ws-01":    (600, 310),
    "db-01":    (300, 450),
    "ws-02":    (500, 450),
    "ws-03":    (650, 450),
}

_ASSET_ICON: dict[str, str] = {
    "firewall":    "🔥",
    "router":      "🌐",
    "server":      "🖥",
    "database":    "🗄",
    "workstation": "💻",
}


def _node_color(node_id: str, state) -> str:
    """Return hex fill colour based on current node status."""
    assets = getattr(state, "assets", {})
    asset  = assets.get(node_id)

    if asset is None:
        return "#58a6ff"                          # unknown → blue

    if asset.is_isolated:
        return "#8b949e"                          # grey  — isolated

    if asset.is_compromised:
        return "#f85149"                          # red   — compromised

    # Under active threat?
    threats = getattr(state, "active_threats", [])
    if any(t.current_node == node_id for t in threats):
        return "#e3b341"                          # yellow — under threat

    # Health-based green shade
    h = asset.health                              # 0.0 – 1.0
    if h > 0.70:
        return "#3fb950"                          # bright green
    elif h > 0.40:
        return "#d29922"                          # amber — degraded
    return "#f85149"                              # red   — critical health


def build_graph(state) -> tuple[list, list]:
    """
    Build agraph Node + Edge lists from current EnvironmentState.

    Returns (nodes, edges).  Falls back gracefully if state is incomplete.
    """
    assets = getattr(state, "assets", {})

    nodes: list[Node] = []
    for node_id, pos in _NODE_POS.items():
        asset      = assets.get(node_id)
        color      = _node_color(node_id, state)
        icon       = _ASSET_ICON.get(
            asset.asset_type.value if asset else "workstation", "📦"
        )
        # Status suffix on label
        if asset:
            if asset.is_isolated:    suffix = " 🔒"
            elif asset.is_compromised: suffix = " ⚠"
            else:                    suffix = ""
        else:
            suffix = ""

        label = f"{icon} {node_id}{suffix}"
        health_pct = int(asset.health * 100) if asset else 100

        nodes.append(Node(
            id    = node_id,
            label = label,
            size  = 28,
            color = color,
            title = (                            # tooltip on hover
                f"{node_id}\n"
                f"type: {asset.asset_type.value if asset else '?'}\n"
                f"health: {health_pct}%\n"
                f"compromised: {asset.is_compromised if asset else '?'}\n"
                f"isolated: {asset.is_isolated if asset else '?'}"
            ),
            x = pos[0],
            y = pos[1],
            # highlight most-threatened node with a border
            borderWidth      = 4 if any(
                getattr(t, "current_node", None) == node_id
                for t in getattr(state, "active_threats", [])
            ) else 1,
            borderWidthSelected = 4,
        ))

    edges: list[Edge] = []
    for src, dst in _TOPOLOGY_EDGES:
        # Red edge if both endpoints compromised — shows lateral path
        src_asset = assets.get(src)
        dst_asset = assets.get(dst)
        both_comp = (
            src_asset and dst_asset
            and src_asset.is_compromised
            and dst_asset.is_compromised
        )
        edges.append(Edge(
            source = src,
            target = dst,
            color  = "#f85149" if both_comp else "#30363d",
            width  = 3 if both_comp else 1,
        ))

    return nodes, edges


def render_graph(state, interactive: bool = True) -> None:
    """
    Render the network topology graph inside the current Streamlit container.

    Args:
        interactive: If True (default) render the agraph interactive graph.
                     Pass False to render a compact status table instead —
                     use this whenever a second graph would appear on the same
                     page to avoid StreamlitDuplicateElementId errors.
    """
    # ── legend ───────────────────────────────────────────────────────────────
    st.markdown(
        '<span class="legend-pill" style="background:#3fb950;color:#0d1117">● Healthy</span>'
        '<span class="legend-pill" style="background:#d29922;color:#0d1117">● Degraded</span>'
        '<span class="legend-pill" style="background:#e3b341;color:#0d1117">⚡ Under Threat</span>'
        '<span class="legend-pill" style="background:#f85149;color:#fff">✖ Compromised</span>'
        '<span class="legend-pill" style="background:#8b949e;color:#0d1117">🔒 Isolated</span>',
        unsafe_allow_html=True,
    )

    assets = getattr(state, "assets", {})

    # ── compact status table (always available; used for inline / fallback) ──
    if not interactive or not _AGRAPH:
        if not _AGRAPH and interactive:
            st.warning(
                "streamlit-agraph not installed.  "
                "Run: `pip3 install streamlit-agraph`"
            )
        rows = []
        threats_here = {
            t.current_node for t in getattr(state, "active_threats", [])
        }
        for nid in _NODE_POS:
            asset = assets.get(nid)
            if asset is None:
                continue
            if asset.is_isolated:        status = "🔒 Isolated"
            elif asset.is_compromised:   status = "⚠️ Compromised"
            elif nid in threats_here:    status = "⚡ Under Threat"
            else:                        status = "✅ Clean"
            rows.append({
                "Node":   nid,
                "Type":   asset.asset_type.value,
                "Health": f"{asset.health:.0%}",
                "Status": status,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        return

    # ── full interactive agraph ───────────────────────────────────────────────
    nodes, edges = build_graph(state)

    cfg = AGraphConfig(
        width           = "100%",
        height          = "420px",
        directed        = False,
        physics         = False,          # static layout — no jiggling
        hierarchical    = False,
        nodeHighlightBehavior = True,
        highlightColor  = "#ffffff",
        collapsible     = False,
        node            = {"labelProperty": "label", "fontSize": 11},
        link            = {"renderLabel": False},
        backgroundColor = "#0d1117",
    )

    agraph(nodes=nodes, edges=edges, config=cfg)


# ── session state helpers ────────────────────────────────────────────────────

def _init(task_name: str = "Easy", seed: int = 42) -> None:
    """Fully reset session state and build a fresh environment."""
    task = TASK_MAP[task_name]()
    env  = task.build_env()
    state = env.reset(seed=seed)

    st.session_state.update(
        env               = env,
        state             = state,
        task              = task,
        task_name         = task_name,
        seed              = seed,
        done              = False,
        step_count        = 0,
        total_reward      = 0.0,
        step_rewards      = [],
        logs              = [],
        last_info         = {},
        last_action       = None,
        agent             = BaselineAgent(),
        # ── episode tracking for incident report ─────────────────────────
        ep_actions        = [],          # list[str] — action names this episode
        ep_threats        = [],          # list[Threat] — all threats seen
        ep_compromised    = set(),       # set[str] — union of compromised nodes
        ep_lateral_moves  = 0,           # count of lateral movement events
        last_report       = None,        # dict — most recent incident report
    )


if "env" not in st.session_state:
    _init()


def _do_step(action: ActionInput) -> None:
    """Execute one env step and update session state."""
    env = st.session_state.env
    state, reward, done, info = env.step(action)

    st.session_state.state        = state
    st.session_state.done         = done
    st.session_state.last_info    = info
    st.session_state.last_action  = action
    st.session_state.step_count  += 1
    st.session_state.total_reward += reward
    st.session_state.step_rewards.append(reward)

    # ── episode data collection ───────────────────────────────────────────────
    st.session_state.ep_actions.append(action.action.name)
    for t in state.active_threats:
        st.session_state.ep_threats.append(t)
    for nid in state.compromised_nodes:
        st.session_state.ep_compromised.add(nid)
    lateral = info.get("lateral_movements", [])
    st.session_state.ep_lateral_moves += len(lateral)

    # Build a compact log line
    bd      = info.get("reward_breakdown", {})
    outcome = info.get("action_outcome", {})
    wasted  = "⚠ wasted" if outcome.get("wasted") else ""

    line = (
        f"[{info.get('step', '?'):>3}] "
        f"{action.action.name:<14}"
        f"{'@' + action.target_node if action.target_node else '':<12} "
        f"r={reward:+.3f}  "
        f"threats={info.get('threats_active', '?')}  "
        f"contain={bd.get('containment', 0):+.3f}  "
        f"survival={bd.get('survival', 0):.3f}"
    )
    if lateral:
        line += f"  🕸 lateral×{len(lateral)}"
    if wasted:
        line += f"  {wasted}"
    if done:
        line += "  🏁 DONE"
        # ── generate incident report at episode end ───────────────────────
        report = generate_incident_report({
            "task_name":       st.session_state.task_name,
            "seed":            st.session_state.seed,
            "steps":           st.session_state.step_count,
            "total_reward":    st.session_state.total_reward,
            "step_rewards":    st.session_state.step_rewards,
            "actions_taken":   st.session_state.ep_actions,
            "threats_seen":    st.session_state.ep_threats,
            "compromised_nodes": st.session_state.ep_compromised,
            "lateral_moves":   st.session_state.ep_lateral_moves,
            "final_state":     state,
            "episode_score":   state.episode_score,
        })
        st.session_state.last_report = report
        st.session_state.logs.append(
            f"[Episode End] 📋 Incident Report Generated — "
            f"outcome: {report['outcome_label']}  |  "
            f"score: {report['episode_score']:.4f}"
        )

    st.session_state.logs.append(line)


def _ai_step() -> None:
    recs   = st.session_state.env.recommend()
    action = st.session_state.agent.choose(
        st.session_state.state, recommendations=recs
    )
    _do_step(action)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🛡️ SOC Dashboard")
    st.markdown("---")

    # ── config
    st.markdown("### ⚙️ Configuration")
    task_name = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    seed_val  = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1)

    if st.button("🔄 Reset Episode", use_container_width=True, type="primary"):
        _init(task_name, int(seed_val))
        st.rerun()

    st.markdown("---")

    # ── agent mode
    st.markdown("### 🤖 Control Mode")
    mode = st.radio("Mode", ["Manual", "AI Baseline"], horizontal=True, label_visibility="collapsed")

    if mode == "Manual":
        action_name = st.selectbox(
            "Action",
            ["BLOCK_IP", "ISOLATE_NODE", "PATCH_SYSTEM", "RUN_DEEP_SCAN", "IGNORE"],
        )
        node_opts   = list(st.session_state.state.assets.keys())
        target_node = None
        if action_name != "IGNORE":
            target_node = st.selectbox("Target Node", node_opts)

        if st.button(
            "▶ Step",
            use_container_width=True,
            disabled=st.session_state.done,
        ):
            _do_step(ActionInput(action=Action[action_name], target_node=target_node))
            st.rerun()

    else:  # AI Baseline
        if st.button(
            "▶ AI Step",
            use_container_width=True,
            disabled=st.session_state.done,
        ):
            _ai_step()
            st.rerun()

    st.markdown("---")

    # ── auto-run
    st.markdown("### ⚡ Auto-Run (AI)")
    auto_run = st.toggle("Run continuously", value=False)
    speed    = st.slider("Delay (s)", 0.1, 3.0, 0.6, step=0.1)

    # trigger auto-run AFTER all widgets are rendered
    if auto_run and not st.session_state.done:
        _ai_step()
        time.sleep(speed)
        st.rerun()

    st.markdown("---")
    st.caption(
        f"Task: **{st.session_state.task_name}** | "
        f"Seed: **{st.session_state.seed}** | "
        f"Passing: **{st.session_state.task.config.passing_score}**"
    )


# ── grab current state once ──────────────────────────────────────────────────
state = st.session_state.state
info  = st.session_state.last_info

# ── tabs ─────────────────────────────────────────────────────────────────────
tab_soc, tab_topo, tab_train = st.tabs(["🛡️ SOC Dashboard", "🌐 Network Topology", "📈 Training"])

# ============================================================================
# TAB 2 — Network Topology
# ============================================================================
with tab_topo:
    st.markdown("## 🌐 Network Topology")
    st.caption(
        "Live view of all 8 network nodes. "
        "Colours update on every step — switch here after pressing **▶ Step** to see the change."
    )

    # ── per-step stats bar ───────────────────────────────────────────────────
    _topo_state = st.session_state.state
    _assets     = getattr(_topo_state, "assets", {})
    _threats    = getattr(_topo_state, "active_threats", [])

    _t1, _t2, _t3, _t4 = st.columns(4)
    _t1.metric("Total nodes",    len(_NODE_POS))
    _t2.metric("Compromised",    sum(1 for a in _assets.values() if a.is_compromised),
               delta_color="inverse")
    _t3.metric("Isolated",       sum(1 for a in _assets.values() if a.is_isolated))
    _t4.metric("Active threats", len(_threats))

    st.markdown("")

    # ── graph ────────────────────────────────────────────────────────────────
    render_graph(_topo_state, interactive=True)

    # ── node detail table ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Node Status")
    _rows = []
    for _nid in _NODE_POS:
        _asset = _assets.get(_nid)
        if _asset is None:
            continue
        _threat_here = any(t.current_node == _nid for t in _threats)
        if _asset.is_isolated:        _status = "🔒 Isolated"
        elif _asset.is_compromised:   _status = "⚠️ Compromised"
        elif _threat_here:            _status = "⚡ Under Threat"
        else:                         _status = "✅ Clean"
        _rows.append({
            "Node":         _nid,
            "Type":         _asset.asset_type.value,
            "Health":       f"{_asset.health:.0%}",
            "Patch":        f"{_asset.patch_level:.0%}",
            "Criticality":  f"{_asset.criticality:.0%}",
            "Status":       _status,
        })
    if _rows:
        st.dataframe(_rows, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 3 — Training (rendered before SOC so it doesn't interfere with SOC state)
# ============================================================================
with tab_train:
    st.markdown("## 📈 Q-Learning Training Results")

    p3_path  = _TRAINING_DIR / "phase3_scores.json"
    png_path = _TRAINING_DIR / "training_curves.png"

    # ── training curves image ────────────────────────────────────────────────
    if png_path.exists():
        st.image(str(png_path), use_container_width=True)
    else:
        st.info(
            "training_curves.png not found.  "
            "Run `python3 adaptive_cyber_defense/training/plot_results.py` to generate it."
        )

    # ── score comparison table ───────────────────────────────────────────────
    if p3_path.exists():
        import json as _json
        p3 = _json.loads(p3_path.read_text())
        eval_d = p3.get("eval", {})

        st.markdown("### 🏆 Final Evaluation — 20 seeds, Hard difficulty, ε=0")

        seeds    = eval_d.get("seeds", [])
        ql_sc    = eval_d.get("ql", [])
        base_sc  = eval_d.get("baseline", [])
        ign_sc   = eval_d.get("ignore", [])
        ql_avg   = eval_d.get("ql_avg",   0.0)
        base_avg = eval_d.get("base_avg", 0.0)
        ign_avg  = eval_d.get("ign_avg",  0.0)

        # KPI row
        k1, k2, k3 = st.columns(3)
        k1.metric("QL Agent avg",  f"{ql_avg:.4f}",   delta=f"{ql_avg - base_avg:+.4f} vs Baseline")
        k2.metric("Baseline avg",  f"{base_avg:.4f}")
        k3.metric("Ignore avg",    f"{ign_avg:.4f}",   delta=f"{ign_avg - base_avg:+.4f} vs Baseline")

        # Per-seed table
        if seeds and ql_sc:
            rows = []
            for s, ql, bl, ig in zip(seeds, ql_sc, base_sc, ign_sc):
                rows.append({
                    "Seed": s,
                    "QL Agent": round(ql, 4),
                    "Baseline": round(bl, 4),
                    "Ignore":   round(ig, 4),
                    "QL > Baseline": "✓" if ql >= bl else "",
                })
            st.dataframe(rows, use_container_width=True)

        # Phase-avg trend
        pavg = p3.get("phase_avgs", {})
        if pavg:
            st.markdown("### 📊 Phase-by-Phase Avg Score")
            st.bar_chart({"Phase": ["P1 (ep 1–50)", "P2 (ep 51–200)", "P3 (ep 201–500)"],
                          "Score": [pavg.get("p1",0), pavg.get("p2",0), pavg.get("p3",0)]})
    else:
        st.info("phase3_scores.json not found — run Phase 3 training to populate this tab.")

    st.markdown("---")

    # ── Live Training Performance ─────────────────────────────────────────────
    st.markdown("### 🏋️ Training Performance — Agent Learning Curve")

    # ── controls row ─────────────────────────────────────────────────────────
    _tc1, _tc2, _tc3 = st.columns([2, 2, 4])
    _tr_episodes = _tc1.number_input(
        "Episodes", min_value=10, max_value=500, value=50, step=10,
        key="tr_episodes",
    )
    _tr_task = _tc2.selectbox(
        "Task", ["Easy", "Medium", "Hard"], index=2, key="tr_task"
    )

    if _tc3.button("▶ Run Training", type="primary", use_container_width=True):
        _tr_task_cls = TASK_MAP[_tr_task]()
        _tr_env      = _tr_task_cls.build_env()

        try:
            from adaptive_cyber_defense.agents.ql_agent import (
                QLearningAgent, train as ql_train, run_baseline,
            )
        except ImportError as _e:
            st.error(f"Cannot import ql_agent: {_e}")
        else:
            _prog_bar = st.progress(0, text="Training…")

            # ── QL training ──────────────────────────────────────────────────
            _tr_agent  = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2)
            _tr_result = ql_train(
                _tr_agent, _tr_env,
                episodes   = int(_tr_episodes),
                verbose    = False,
            )

            # ── baseline run on same episode count ───────────────────────────
            _bl_env    = _tr_task_cls.build_env()
            _bl_result = run_baseline(_bl_env, episodes=int(_tr_episodes))

            _prog_bar.progress(1.0, text="Done!")

            st.session_state.training_rewards  = _tr_result["rewards"]
            st.session_state.baseline_rewards  = _bl_result["rewards"]
            st.session_state.training_episodes = int(_tr_episodes)
            st.session_state.training_task     = _tr_task

    # ── chart ─────────────────────────────────────────────────────────────────
    _tr_rewards = st.session_state.get("training_rewards")
    _bl_rewards = st.session_state.get("baseline_rewards")

    if not _tr_rewards:
        st.info("⬆ Set episodes & task, then click **▶ Run Training** to view the learning curve.")
    else:
        _n = len(_tr_rewards)
        _w = 5                                          # moving-average window

        def _moving_avg(vals: list, w: int) -> list:
            return [
                sum(vals[max(0, i - w + 1): i + 1]) / min(i + 1, w)
                for i in range(len(vals))
            ]

        # Build chart data dict  (Streamlit multi-line chart)
        _chart_data: dict = {
            "QL Agent":           _tr_rewards,
            "QL MA-5":            _moving_avg(_tr_rewards, _w),
        }
        if _bl_rewards and len(_bl_rewards) == _n:
            _chart_data["Random Baseline"] = _bl_rewards
            _chart_data["Baseline MA-5"]   = _moving_avg(_bl_rewards, _w)

        # KPI strip
        _ql_avg  = sum(_tr_rewards) / _n
        _bl_avg  = sum(_bl_rewards) / len(_bl_rewards) if _bl_rewards else None
        _ql_last = sum(_tr_rewards[max(0, _n - 10):]) / min(10, _n)

        _ka, _kb, _kc = st.columns(3)
        _ka.metric("QL avg reward",       f"{_ql_avg:.3f}")
        _kb.metric("QL last-10 avg",      f"{_ql_last:.3f}",
                   delta=f"{_ql_last - _ql_avg:+.3f}")
        if _bl_avg is not None:
            _kc.metric("Random baseline avg", f"{_bl_avg:.3f}",
                       delta=f"{_ql_avg - _bl_avg:+.3f} QL vs baseline")

        # Native Streamlit line chart  (updates on every rerun)
        st.line_chart(
            _chart_data,
            use_container_width = True,
            height              = 320,
            color               = ["#58a6ff", "#1f6feb", "#e74c3c", "#922b21"],
        )
        st.caption(
            f"Solid lines = raw reward per episode · "
            f"Faded lines = {_w}-episode moving average · "
            f"Task: **{st.session_state.get('training_task','?')}** · "
            f"{_n} episodes"
        )

        # Learning verdict
        _first10 = sum(_tr_rewards[:10]) / 10
        _last10  = sum(_tr_rewards[max(0, _n - 10):]) / min(10, _n)
        if _last10 > _first10 * 1.05:
            st.success(f"📈 **Learning detected** — last-10 avg ({_last10:.3f}) "
                       f"is {(_last10/_first10 - 1)*100:.0f}% above first-10 avg ({_first10:.3f})")
        elif _last10 >= _first10:
            st.info(f"➡ Reward is stable — try more episodes or reduce task difficulty.")
        else:
            st.warning(f"📉 Reward declined slightly — agent may need a longer warm-up.")

    st.markdown("---")

    # ── QL agent mode switch ─────────────────────────────────────────────────
    st.markdown("### 🤖 Switch Dashboard to QL Agent Mode")
    if _QTABLE_PATH.exists():
        if st.button("🔄 Load QL Agent into Dashboard", type="primary"):
            try:
                from adaptive_cyber_defense.agents.ql_agent import QLearningAgent
                ql_agent = QLearningAgent()
                ql_agent.load(str(_QTABLE_PATH))
                ql_agent.epsilon = 0.0
                st.session_state.agent = ql_agent
                st.success("✅ QL Agent loaded — switch to the SOC Dashboard tab and use AI Step.")
            except Exception as e:
                st.error(f"Failed to load QL agent: {e}")
    else:
        st.warning(
            "ql_table.json not found.  "
            "Run the Phase 1–3 training scripts to generate it, then reload this page."
        )

# ============================================================================
# TAB 1 — SOC Dashboard (the original content, wrapped in tab_soc)
# ============================================================================
with tab_soc:

# ── page header ──────────────────────────────────────────────────────────────
    st.markdown("# 🛡️ Adaptive Cyber Defense — Live SOC Dashboard")

    if st.session_state.done:
        outcome = info.get("action_outcome", {})
        st.success(
            f"🏁 **Episode complete** — "
            f"Final score: **{state.episode_score:.3f}** | "
            f"Passed: **{'✅' if state.episode_score >= st.session_state.task.config.passing_score else '❌'}** | "
            f"Steps: **{st.session_state.step_count}**"
        )

    # ── top KPI metrics ──────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    prev_rewards = st.session_state.step_rewards
    last_r = prev_rewards[-1] if prev_rewards else None
    prev_r = prev_rewards[-2] if len(prev_rewards) >= 2 else None

    m1.metric("🕐 Time Step",        state.time_step)
    m2.metric("⚡ Resources",        f"{state.resource_availability:.0%}")
    m3.metric("🔥 Threat Severity",  f"{state.threat_severity:.0%}")
    m4.metric("🌐 Network Load",     f"{state.network_load:.0%}")
    m5.metric(
        "📦 Step Reward",
        f"{last_r:.4f}" if last_r is not None else "—",
        delta=f"{last_r - prev_r:.4f}" if (last_r is not None and prev_r is not None) else None,
    )
    m6.metric("🏆 Cumulative Reward", f"{st.session_state.total_reward:.4f}")

    st.markdown("---")

    # ── inline network topology ──────────────────────────────────────────────
    with st.expander("🌐 Network Topology", expanded=True):
        render_graph(state, interactive=False)  # table view — avoids duplicate agraph element

    st.markdown("---")

    # ── main content: 3 columns ──────────────────────────────────────────────
    col_net, col_threats, col_reward = st.columns([3, 3, 2], gap="medium")


# ── COL 1 — Network Assets ───────────────────────────────────────────────────
with col_net:
    st.markdown("### 🖥️ Network Assets")

    # Summary bar
    total_assets      = len(state.assets)
    compromised_count = sum(1 for a in state.assets.values() if a.is_compromised)
    isolated_count    = sum(1 for a in state.assets.values() if a.is_isolated)
    avg_health        = sum(a.health for a in state.assets.values()) / total_assets

    na1, na2, na3 = st.columns(3)
    na1.metric("Compromised", f"{compromised_count}/{total_assets}", delta_color="inverse")
    na2.metric("Isolated",    f"{isolated_count}/{total_assets}")
    na3.metric("Avg Health",  f"{avg_health:.0%}")

    st.markdown("")

    for node_id, asset in state.assets.items():
        # Determine status badge
        badges = []
        if asset.is_compromised: badges.append("⚠️ COMPROMISED")
        if asset.is_isolated:    badges.append("🔒 ISOLATED")
        status_str = "  ".join(badges) if badges else "✅ Clean"

        # Health indicator
        if asset.health > 0.70:   h_dot = "🟢"
        elif asset.health > 0.40: h_dot = "🟡"
        else:                      h_dot = "🔴"

        icon = ASSET_ICON.get(asset.asset_type.value, "📦")
        label = f"{icon} {node_id}  {h_dot}  {status_str}"

        with st.expander(label, expanded=asset.is_compromised):
            c1, c2, c3 = st.columns(3)
            c1.metric("Health",      f"{asset.health:.0%}")
            c2.metric("Patch Level", f"{asset.patch_level:.0%}")
            c3.metric("Criticality", f"{asset.criticality:.0%}")
            st.progress(asset.health)


# ── COL 2 — Threats + AI Recommendations ─────────────────────────────────────
with col_threats:

    # ── Active Threats
    st.markdown("### 🚨 Active Threats")

    threats = state.active_threats
    if not threats:
        st.success("🎉 No active threats — network is clean!")
    else:
        # summary
        max_sev = max(t.effective_severity() for t in threats)
        st.caption(
            f"**{len(threats)}** active  |  "
            f"max severity **{max_sev:.0%}**  |  "
            f"compromised nodes: **{', '.join(state.compromised_nodes) or 'none'}**"
        )

        for threat in sorted(threats, key=lambda t: t.effective_severity(), reverse=True):
            sev = threat.effective_severity()

            if sev >= 0.70:   sev_dot, badge = "🔴", "CRITICAL"
            elif sev >= 0.45: sev_dot, badge = "🟠", "HIGH"
            elif sev >= 0.20: sev_dot, badge = "🟡", "MEDIUM"
            else:             sev_dot, badge = "🟢", "LOW"

            stage_icon = STAGE_ICON.get(threat.stage.name, "❓")
            label = (
                f"{sev_dot} [{badge}] {threat.id[:10]}…  "
                f"{stage_icon} {threat.stage.name}  @{threat.current_node}"
            )

            with st.expander(label, expanded=(sev >= 0.55)):
                t1, t2, t3 = st.columns(3)
                t1.metric("Eff. Severity",  f"{sev:.0%}")
                t2.metric("Confidence",     f"{threat.detection_confidence:.0%}")
                t3.metric("Steps Active",   threat.steps_active)

                st.progress(sev)
                st.caption(
                    f"Origin: `{threat.origin_node}`  →  Current: `{threat.current_node}`  |  "
                    f"Persistence: {threat.persistence:.2f}  |  "
                    f"Detected: {'✅' if threat.is_detected else '❌'}"
                )
                st.caption(
                    f"🔬 **MITRE:** `{threat.stage.technique_id}` — {threat.stage.technique_name}"
                )

    st.markdown("---")

    # ── AI Recommendations
    st.markdown("### 🤖 AI Recommendations")

    if st.session_state.done:
        st.info("Episode ended — reset to continue.")
    else:
        recs = st.session_state.env.recommend()
        if not recs:
            st.info("No recommendations available.")
        else:
            top_recs = [r for r in recs if r.action_input.action != Action.IGNORE][:4]
            if not top_recs:
                top_recs = recs[:1]

            for i, rec in enumerate(top_recs):
                afford_icon = "✅" if rec.is_affordable else "❌ (too costly)"
                target_str  = f"@{rec.action_input.target_node}" if rec.action_input.target_node else ""
                header = (
                    f"**#{i+1}** {afford_icon} "
                    f"`{rec.action_input.action.name}{target_str}`  "
                    f"— score **{rec.score:.3f}**"
                )
                with st.expander(header, expanded=(i == 0 and rec.is_affordable)):
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Score",      f"{rec.score:.3f}")
                    r2.metric("Confidence", f"{rec.confidence:.0%}")
                    r3.metric("Benefit",    f"{rec.expected_benefit:.3f}")
                    r4.metric("Cost",       f"{rec.resource_cost:.2f}")

                    if rec.threat_score:
                        ts = rec.threat_score
                        st.progress(
                            ts.composite_score,
                            text=f"Threat composite: {ts.composite_score:.2f}  |  driver: {ts.primary_driver}",
                        )
                        sub1, sub2, sub3, sub4 = st.columns(4)
                        sub1.caption(f"Impact\n{ts.impact_score:.2f}")
                        sub2.caption(f"Spread\n{ts.spread_score:.2f}")
                        sub3.caption(f"Likely\n{ts.likelihood_score:.2f}")
                        sub4.caption(f"Urgency\n{ts.urgency_score:.2f}")

                    st.info(f"💬 {rec.reasoning}")
                    if rec.expected_impact:
                        st.caption(f"📊 **Expected impact:** {rec.expected_impact}")


# ── AI Decision Insight (col_threats continued) ──────────────────────────────
with col_threats:
    st.markdown("---")
    st.markdown("### 🧠 AI Decision Insight")

    if st.session_state.done:
        st.info("Episode ended — reset to see live insights.")
    else:
        _ins_recs = st.session_state.env.recommend()
        _ins_top  = next(
            (r for r in _ins_recs
             if r.is_affordable and r.action_input.action != Action.IGNORE),
            _ins_recs[0] if _ins_recs else None,
        )
        if _ins_top:
            _ins_conf  = _ins_top.confidence
            _ins_color = (
                "#3fb950" if _ins_conf >= 0.65
                else "#d29922" if _ins_conf >= 0.35
                else "#f85149"
            )
            _ins_target = (
                f" @ <code>{_ins_top.action_input.target_node}</code>"
                if _ins_top.action_input.target_node else ""
            )
            st.markdown(
                f'<div style="background:#161b22;border:1px solid #30363d;'
                f'border-radius:8px;padding:10px 14px;margin-bottom:6px;">'
                f'<b>Top action:</b> '
                f'<code>{_ins_top.action_input.action.name}</code>{_ins_target}'
                f'&nbsp;&nbsp;'
                f'<span style="color:{_ins_color};font-weight:700;">'
                f'{_ins_conf:.0%} confidence</span>'
                f'&nbsp;·&nbsp;score <b>{_ins_top.score:.3f}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if _ins_top.expected_impact:
                st.caption(f"📊 {_ins_top.expected_impact}")
            st.caption(f"💬 {_ins_top.reasoning}")

            # MITRE context from highest-severity active threat
            _ins_threats = state.active_threats
            if _ins_threats:
                _ins_ht = max(_ins_threats, key=lambda t: t.effective_severity())
                _ins_stage_icon = STAGE_ICON.get(_ins_ht.stage.name, "❓")
                st.caption(
                    f"🔬 Active technique: "
                    f"**`{_ins_ht.stage.technique_id}`** "
                    f"{_ins_ht.stage.technique_name} "
                    f"({_ins_stage_icon} {_ins_ht.stage.name})"
                )
        else:
            st.caption("No actionable recommendations for current state.")


# ── COL 3 — Reward & Detection ───────────────────────────────────────────────
with col_reward:

    # ── Reward panel
    st.markdown("### 🏆 Reward")

    st.metric("Cumulative",  f"{st.session_state.total_reward:.4f}")
    st.metric("Steps Taken", st.session_state.step_count)

    # Reward history sparkline
    if st.session_state.step_rewards:
        st.line_chart(
            {"Reward": st.session_state.step_rewards},
            height=130,
            use_container_width=True,
        )

    # Breakdown of last step
    bd = info.get("reward_breakdown", {})
    if bd:
        st.markdown("**Last step breakdown:**")
        breakdown_rows = [
            ("Containment",  bd.get("containment", 0)),
            ("Severity ↓",   bd.get("severity",    0)),
            ("Efficiency",   bd.get("efficiency",   0)),
            ("Survival",     bd.get("survival",     0)),
            ("Spread ↓",     bd.get("spread_penalty",  0)),
            ("Waste ↓",      bd.get("waste_penalty",   0)),
            ("FP penalty",   bd.get("fp_penalty",      0)),
            ("Avail penalty",bd.get("avail_penalty",   0)),
        ]
        for label, val in breakdown_rows:
            if val == 0:
                continue
            dot = "🟢" if val > 0 else "🔴"
            st.caption(f"{dot} {label}: `{val:+.4f}`")

    st.markdown("---")

    # ── Detection events
    st.markdown("### 🔍 Detection (last step)")
    det = info.get("detection_events", [])
    if det:
        tp = sum(1 for e in det if e.get("type") == "true_positive")
        fp = sum(1 for e in det if e.get("type") == "false_positive")
        fn = sum(1 for e in det if e.get("type") == "false_negative")
        d1, d2, d3 = st.columns(3)
        d1.metric("TP", tp)
        d2.metric("FP", fp, delta=f"−{fp}" if fp else None, delta_color="inverse")
        d3.metric("FN", fn, delta=f"−{fn}" if fn else None, delta_color="inverse")
        for e in det[:4]:
            icon = {"true_positive": "✅", "false_positive": "⚠️", "false_negative": "❌"}.get(e.get("type",""), "•")
            st.caption(
                f"{icon} `{e.get('node','?')}` — {e.get('type','?')} "
                f"({e.get('method','?')}) conf={e.get('confidence',0):.2f}"
            )
    else:
        st.caption("No detection events this step.")

    st.markdown("---")

    # ── Last action outcome
    st.markdown("### ⚡ Last Action")
    outcome = info.get("action_outcome", {})
    if outcome:
        ok = outcome.get("success", False)
        wasted = outcome.get("wasted", False)
        st.markdown(
            f"{'✅' if ok else ('⚠️' if wasted else '❌')} "
            f"`{outcome.get('action','?')}` → `{outcome.get('target','—')}`"
        )
        contained = outcome.get("contained", [])
        if contained:
            st.success(f"Contained: {', '.join(contained)}")
        st.caption(outcome.get("message", ""))
        o1, o2 = st.columns(2)
        o1.metric("Cost",         f"{outcome.get('cost', 0):.2f}")
        o2.metric("Detect boost", f"{outcome.get('detection_boost', 0):.2f}")

        lat = info.get("lateral_movements", [])
        if lat:
            st.warning(f"🕸️ {len(lat)} lateral movement(s) this step!")
            for lm in lat:
                st.caption(f"  `{lm['from']}` → `{lm['to']}` (parent: {lm['parent'][:10]}…)")
    else:
        st.caption("No action taken yet — press Step.")


    # ── Live Log ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Live Step Log")

    log_lines = st.session_state.logs[-60:]         # keep last 60 lines
    log_text  = "\n".join(reversed(log_lines))      # newest first
    if not log_text:
        log_text = "[ No steps yet — press Step or enable Auto-Run ]"

    st.markdown(
        f'<div class="log-terminal">{log_text}</div>',
        unsafe_allow_html=True,
    )

    # ── Incident Report ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Incident Report")

    _rpt = st.session_state.get("last_report")

    if _rpt is None:
        st.info("🔎 No incident report yet — complete an episode to generate one.")
    else:
        # ── outcome badge ─────────────────────────────────────────────────────
        _oc_color = {"Contained": "#3fb950", "Partially Contained": "#d29922",
                     "Escalated": "#e3b341", "Breach": "#f85149"}.get(
                         _rpt["outcome_label"], "#58a6ff")
        st.markdown(
            f'<div style="background:{_oc_color};color:#0d1117;padding:8px 16px;'
            f'border-radius:6px;font-weight:700;font-size:16px;display:inline-block;">'
            f'{_rpt["outcome"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        # ── four-column KPI strip ─────────────────────────────────────────────
        _ri1, _ri2, _ri3, _ri4 = st.columns(4)
        _ri1.metric("Episode Score",   f"{_rpt['episode_score']:.4f}")
        _ri2.metric("Total Reward",    f"{_rpt['total_reward']:.4f}")
        _ri3.metric("Steps",           _rpt["steps"])
        _ri4.metric("Compromised",     _rpt["n_compromised"], delta_color="inverse")

        # ── two-column detail ─────────────────────────────────────────────────
        _rcol1, _rcol2 = st.columns(2)

        with _rcol1:
            st.markdown("**🎯 Attack Summary**")
            st.markdown(f"- **Types:** `{'`, `'.join(_rpt['attack_types'])}`")
            st.markdown(f"- **Kill-chain:** {' → '.join(_rpt['stages_reached']) or 'N/A'}")
            st.markdown(f"- **Severity:** **{_rpt['severity_label']}** ({_rpt['peak_severity']:.0%})")
            st.markdown(f"- **Lateral moves:** {_rpt['lateral_moves']}")

            st.markdown("**🔬 MITRE ATT&CK**")
            for _tid, _tname in _rpt["mitre"]:
                st.markdown(f"  - `{_tid}` {_tname}")

        with _rcol2:
            st.markdown("**🛡️ Response Actions**")
            for _act, _cnt in sorted(_rpt["actions_taken"].items()):
                _bar_w = int(_cnt / max(_rpt["actions_taken"].values()) * 100)
                st.markdown(
                    f'`{_act}` ×{_cnt} '
                    f'<div style="background:#1f6feb;height:6px;width:{_bar_w}%;'
                    f'border-radius:3px;margin-bottom:4px;"></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("**📊 Compromised Nodes**")
            if _rpt["compromised_nodes"]:
                st.markdown("  " + "  ".join(f"`{n}`" for n in _rpt["compromised_nodes"]))
            else:
                st.markdown("  ✅ None")

            st.markdown(f"**📈 Reward trend:** {_rpt['reward_trend']}")

        # ── full report expander ──────────────────────────────────────────────
        with st.expander("📄 Full Report (Markdown)", expanded=False):
            _md = _report_markdown(_rpt)
            st.markdown(_md)

        # ── download button ───────────────────────────────────────────────────
        _dl_text = _report_markdown(_rpt)
        _dl_name = (
            f"incident_report_{_rpt['task']}_seed{_rpt['seed']}_"
            f"{_rpt['generated_at'].replace(' ','_').replace(':','-')}.txt"
        )
        st.download_button(
            label     = "⬇ Download Report (.txt)",
            data      = _dl_text,
            file_name = _dl_name,
            mime      = "text/plain",
        )
