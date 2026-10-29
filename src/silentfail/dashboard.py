"""
SilentFail Hardener Web Dashboard & API Server.
Section 13 of the Project Spec.

Implements all 4 views:
1. Ledger Timeline (The Money Shot): dual-lane comparison of agent-observed events vs real world mutations with highlighted divergence.
2. Scoreboard: baseline vs patched metrics with Wilson 95% confidence intervals.
3. Patch Race: live grid of candidates with progress, fix rate, and guard rejections annotated (controls tagged).
4. Proof Panel: held-out confirmation, generated regression test, and Section 9.1 scale/cost table.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from pathlib import Path
from src.silentfail.orchestrator import Orchestrator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_OUTPUT_DIR = Path(os.getenv("SILENTFAIL_DEMO_OUTPUT_DIR", str(REPO_ROOT / "data" / "demo")))
DEMO_TESTS_DIR = DEMO_OUTPUT_DIR / "tests"

app = FastAPI(title="SilentFail Hardener Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator(data_dir=str(DEMO_OUTPUT_DIR), tests_dir=str(DEMO_TESTS_DIR))


def provider_status_label(gate_g0: dict, *, configured_live: bool = False) -> str:
    """Describe provider state without implying live success before a probe."""
    status = gate_g0.get("status")
    if status == "connected" and gate_g0.get("connected") is True:
        return "Live provider verified (model response received)"
    if status == "live_provider_error":
        return "Configured live provider failed; deterministic local simulation"
    if status == "live_provider_unavailable":
        return "Live provider configured but unavailable; deterministic local simulation"
    if status == "local_simulation":
        return "Deterministic local simulation"
    if configured_live:
        return "Live provider configured; not verified"
    return "Provider status: not probed"


def _dashboard_state() -> dict:
    state_file = orchestrator.data_dir / "latest_pipeline_run.json"
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = orchestrator.pipeline_state
    state = dict(state)
    state["provider_status"] = provider_status_label(
        state.get("gate_g0", {}),
        configured_live=bool(
            orchestrator.nebius_client.api_key
            and orchestrator.nebius_client.api_key != "your_nebius_or_nvidia_api_key_here"
        ),
    )
    return state

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SilentFail Hardener | Side-Effect Truth Engine</title>
    <style>
        :root {
            --bg: #0b0f19;
            --surface: #111827;
            --surface-hover: #1f2937;
            --border: #374151;
            --text: #f9fafb;
            --muted: #9ca3af;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.2);
            --danger: #ef4444;
            --success: #10b981;
            --warning: #f59e0b;
            --tag-ctrl: #8b5cf6;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 24px; }
        .badge { background: #1e1b4b; color: #a5b4fc; border: 1px solid #4338ca; padding: 4px 10px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }
        .tabs { display: flex; gap: 8px; margin-bottom: 24px; }
        .tab-btn { background: var(--surface); border: 1px solid var(--border); color: var(--muted); padding: 10px 18px; border-radius: 8px; cursor: pointer; font-size: 0.95rem; font-weight: 500; transition: all 0.2s; }
        .tab-btn.active { background: var(--primary); color: white; border-color: var(--primary); }
        .tab-btn:hover:not(.active) { background: var(--surface-hover); color: var(--text); }
        .panel { display: none; }
        .panel.active { display: block; }
        
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .card-title { font-size: 1.15rem; font-weight: 600; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
        
        /* Dual Lane Timeline */
        .timeline-container { display: flex; flex-direction: column; gap: 16px; }
        .lane { border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
        .lane-agent { border-left: 4px solid var(--warning); background: rgba(245, 158, 11, 0.05); }
        .lane-world { border-left: 4px solid var(--danger); background: rgba(239, 68, 68, 0.05); }
        .lane-header { font-size: 0.9rem; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; display: flex; justify-content: space-between; }
        .event-box { background: #1f2937; border: 1px solid #374151; border-radius: 6px; padding: 10px; margin-bottom: 8px; font-size: 0.88rem; }
        .event-divergence { border-left: 4px solid var(--danger); background: rgba(239, 68, 68, 0.15); }
        
        /* Scoreboard Table */
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid var(--border); font-size: 0.92rem; }
        th { color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 0.8rem; }
        .rate-val { font-weight: 700; font-size: 1rem; }
        .ci-tag { font-size: 0.78rem; color: var(--muted); margin-left: 4px; }
        
        /* Grid */
        .grid-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
        .race-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; position: relative; }
        .race-card.rejected { border-color: rgba(239, 68, 68, 0.5); background: rgba(239, 68, 68, 0.03); }
        .race-card.winner { border-color: var(--success); box-shadow: 0 0 15px rgba(16, 185, 129, 0.2); }
        .control-pill { background: var(--tag-ctrl); color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
        
        button.btn-run { background: #4f46e5; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600; transition: background 0.2s; }
        button.btn-run:hover { background: #4338ca; }
        pre { background: #000; padding: 14px; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; color: #34d399; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>SilentFail Hardener</h1>
            <p style="color: var(--muted); font-size: 0.95rem; margin-top: 4px;">Side-effect truth engine for tool-using AI agents & closed-loop patch verifier</p>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <span id="backend-status" class="badge">Provider status: not probed</span>
            <button class="btn-run" onclick="triggerPipeline()">Re-Run Hardening Pipeline</button>
        </div>
    </div>

    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('timeline')">1. Ledger Timeline (Money Shot)</button>
        <button class="tab-btn" onclick="switchTab('scoreboard')">2. Scoreboard</button>
        <button class="tab-btn" onclick="switchTab('race')">3. Patch Race</button>
        <button class="tab-btn" onclick="switchTab('proof')">4. Proof Panel</button>
    </div>

    <!-- 1. LEDGER TIMELINE (THE MONEY SHOT) -->
    <div id="tab-timeline" class="panel active">
        <div class="card">
            <div class="card-title">
                <span>Causal Side-Effect Ledger: Dual-Lane Timeline</span>
                <span style="font-size: 0.85rem; color: var(--danger); font-weight: bold;">● DIVERGENCE HIGHLIGHTED</span>
            </div>
            <p style="color: var(--muted); font-size: 0.9rem; margin-bottom: 16px;">
                Shows exactly what the agent saw and said (Top Lane) vs what actually mutated in the world (Bottom Lane). The gap reveals the silent failure.
            </p>
            <div class="timeline-container" id="timeline-lanes">
                Loading causal ledger timeline...
            </div>
        </div>
    </div>

    <!-- 2. SCOREBOARD -->
    <div id="tab-scoreboard" class="panel">
        <div class="card">
            <div class="card-title">Benchmark Scoreboard (Baseline vs Hardened)</div>
            <p style="color: var(--muted); font-size: 0.85rem; margin-bottom: 12px;">All metrics reported under published fault distribution with 95% Wilson confidence intervals.</p>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Baseline (Naive Agent)</th>
                        <th>Hardened (Winning Patch)</th>
                        <th>Impact / Delta</th>
                    </tr>
                </thead>
                <tbody id="scoreboard-body">
                    <tr><td colspan="4">Loading benchmark metrics...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- 3. PATCH RACE -->
    <div id="tab-race" class="panel">
        <div class="card">
            <div class="card-title">Candidate Patch Race with 4 Active Verifier Guards</div>
            <p style="color: var(--muted); font-size: 0.85rem; margin-bottom: 16px;">
                Guards enforce: 1. Regression Guard &nbsp;|&nbsp; 2. Legitimate-Completion Guard &nbsp;|&nbsp; 3. No New Failure Class &nbsp;|&nbsp; 4. Execution Cost. Controls prove guards work.
            </p>
            <div class="grid-cards" id="patch-race-cards">
                Loading candidate pool...
            </div>
        </div>
    </div>

    <!-- 4. PROOF PANEL -->
    <div id="tab-proof" class="panel">
        <div class="card">
            <div class="card-title">Held-Out Generalization & Nebius Scale Table (Section 9.1)</div>
            <div class="grid-cards" style="margin-bottom: 20px;">
                <div class="race-card winner">
                    <h4>Held-Out Scenarios (Unseen)</h4>
                    <p style="color: var(--muted); font-size: 0.85rem; margin: 8px 0;">Duplicate rate: <b style="color: var(--danger)" id="heldout-dup-before">--</b> ➔ <b style="color: var(--success)" id="heldout-dup-after">--</b></p>
                    <p style="color: var(--muted); font-size: 0.85rem;">Correct completion: <b style="color: var(--success)" id="heldout-correct-after">--</b></p>
                </div>
                <div class="race-card">
                    <h4>Nebius Scale & Serial vs Parallel</h4>
                    <p style="color: var(--muted); font-size: 0.85rem; margin: 6px 0;">Verification Runs: <b id="scale-verif-runs">--</b></p>
                    <p style="color: var(--muted); font-size: 0.85rem; margin: 6px 0;">Serial Wall-Clock: <b id="scale-serial-time">--</b></p>
                    <p style="color: var(--muted); font-size: 0.85rem; margin: 6px 0;">Actual Parallel Time: <b id="scale-parallel-time" style="color: var(--success);">--</b></p>
                    <p style="color: var(--muted); font-size: 0.85rem;">Speedup: <b id="scale-speedup">--</b></p>
                </div>
            </div>
            
            <div class="card-title">Exportable Runnable Pytest Regression Test</div>
            <pre><code id="code-snippet"># Regression test will appear here</code></pre>
        </div>
    </div>

    <script>
        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');
        }

        async function loadData() {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                renderUI(data);
            } catch (err) {
                console.error(err);
            }
        }

        async function triggerPipeline() {
            const btn = document.querySelector('.btn-run');
            btn.innerText = "Running Pipeline...";
            btn.disabled = true;
            try {
                const res = await fetch('/api/run', { method: 'POST' });
                const data = await res.json();
                renderUI(data);
            } finally {
                btn.innerText = "Re-Run Hardening Pipeline";
                btn.disabled = false;
            }
        }

        function renderUI(data) {
            // Gate G0
            if (data.provider_status) {
                document.getElementById('backend-status').innerText = data.provider_status;
            }

            // 1. Dual Lane Timeline
            if (data.sample_failing_record) {
                const rec = data.sample_failing_record;
                let agentHtml = '<div class="lane lane-agent"><div class="lane-header"><span>Agent Observed Lane</span><span>Customer Says: "Refund order"</span></div>';
                let worldHtml = '<div class="lane lane-world"><div class="lane-header"><span>Ground-Truth World State Lane</span><span>Database State</span></div>';
                
                rec.events.forEach(ev => {
                    if (ev.type === 'tool_request') {
                        agentHtml += `<div class="event-box"><b>[Seq ${ev.seq}] Requested:</b> ${ev.tool} (Args: ${JSON.stringify(ev.args)})</div>`;
                    } else if (ev.type === 'tool_response_to_agent') {
                        const isErr = ev.fault || (ev.observed && ev.observed.error);
                        agentHtml += `<div class="event-box ${isErr ? 'event-divergence' : ''}"><b>[Seq ${ev.seq}] Observed:</b> ${JSON.stringify(ev.observed)} ${ev.fault ? '<b>[FAULT: ' + ev.fault + ']</b>' : ''}</div>`;
                    } else if (ev.type === 'world_mutation') {
                        worldHtml += `<div class="event-box event-divergence"><b>[Seq ${ev.seq}] MUTATION:</b> ${ev.effect} (Amount: $${ev.amount})</div>`;
                    }
                });

                agentHtml += `<div class="event-box" style="margin-top:8px;"><b>Final Agent Message:</b> "${rec.final_message}"</div></div>`;
                worldHtml += `<div class="event-box" style="margin-top:8px;"><b>Final State:</b> ${JSON.stringify(rec.final_state)}</div></div>`;
                
                document.getElementById('timeline-lanes').innerHTML = agentHtml + worldHtml;
            }

            // 2. Scoreboard (paired held-out: same scenarios, baseline vs winning patch; all values from the artifact)
            if (data.heldout_metrics_before && data.heldout_metrics_after && data.heldout_metrics_after.correct_completion) {
                const b = data.heldout_metrics_before;
                const h = data.heldout_metrics_after;
                const formatCi = (ci) => ci ? `${(ci[0]*100).toFixed(1)}% <span class="ci-tag">[${(ci[1]*100).toFixed(1)}% - ${(ci[2]*100).toFixed(1)}%]</span>` : '--';
                const pp = (x, y) => { const d = (y[0] - x[0]) * 100; return (d > 0 ? '+' : '') + d.toFixed(1) + ' pp'; };

                const rows = [
                    { name: 'Duplicate-Action Rate', base: formatCi(b.duplicate_action_rate), hard: formatCi(h.duplicate_action_rate), delta: pp(b.duplicate_action_rate, h.duplicate_action_rate) },
                    { name: 'Silent Failure Rate (claim contradicts state)', base: formatCi(b.silent_failure_rate), hard: formatCi(h.silent_failure_rate), delta: pp(b.silent_failure_rate, h.silent_failure_rate) },
                    { name: `Claimed-Success Share of Incorrect Runs (${b.counts.cond_failed_total} vs ${h.counts.cond_failed_total} incorrect runs)`, base: formatCi(b.conditional_false_success_rate), hard: formatCi(h.conditional_false_success_rate), delta: pp(b.conditional_false_success_rate, h.conditional_false_success_rate) },
                    { name: 'Correct Completion Rate', base: formatCi(b.correct_completion), hard: formatCi(h.correct_completion), delta: pp(b.correct_completion, h.correct_completion) },
                    { name: 'Legitimate Completion Rate', base: formatCi(b.legitimate_completion_rate), hard: formatCi(h.legitimate_completion_rate), delta: pp(b.legitimate_completion_rate, h.legitimate_completion_rate) },
                    { name: 'Mean Execution Steps', base: `${b.mean_steps} steps`, hard: `${h.mean_steps} steps`, delta: (h.mean_steps - b.mean_steps).toFixed(2) }
                ];

                let bodyHtml = '';
                rows.forEach(r => {
                    bodyHtml += `<tr><td><b>${r.name}</b></td><td>${r.base}</td><td>${r.hard}</td><td>${r.delta}</td></tr>`;
                });
                document.getElementById('scoreboard-body').innerHTML = bodyHtml;
            }

            // 3. Patch Race
            if (data.race_summary) {
                let cardsHtml = '';
                Object.entries(data.race_summary).forEach(([pid, info]) => {
                    const isWinner = data.winning_patch && data.winning_patch.patch_id === pid;
                    const cardClass = isWinner ? 'race-card winner' : (!info.eligible ? 'race-card rejected' : 'race-card');
                    cardsHtml += `
                        <div class="${cardClass}">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <h4 style="font-size: 0.95rem;">${info.spec.name}</h4>
                                ${info.spec.is_control ? '<span class="control-pill">CONTROL</span>' : ''}
                            </div>
                            <p style="font-size: 0.8rem; color: var(--muted); margin-bottom: 10px;">${info.spec.description}</p>
                            <p style="font-size: 0.85rem;">Duplicate Reduction: <b>${(info.duplicate_reduction * 100).toFixed(1)}%</b></p>
                            <div style="margin-top: 10px; font-size: 0.8rem;">
                                ${info.eligible ? '<span style="color: var(--success); font-weight: bold;">✔ All 4 Guards Passed</span>' : 
                                  '<span style="color: var(--danger); font-weight: bold;">✖ Rejected by Verifier Guard:</span><ul style="padding-left:16px; margin-top:4px; color:#fca5a5;">' + 
                                  info.rejection_reasons.map(r => `<li>${r}</li>`).join('') + '</ul>'}
                            </div>
                            ${isWinner ? '<div style="margin-top:12px; background:var(--success); color:#000; font-weight:bold; text-align:center; padding:4px; border-radius:4px; font-size:0.75rem;">WINNING PATCH</div>' : ''}
                        </div>
                    `;
                });
                document.getElementById('patch-race-cards').innerHTML = cardsHtml;
            }

            // 4. Proof
            if (data.heldout_metrics_before && data.heldout_metrics_after) {
                document.getElementById('heldout-dup-before').innerText = (data.heldout_metrics_before.duplicate_action_rate[0]*100).toFixed(1) + '%';
                document.getElementById('heldout-dup-after').innerText = (data.heldout_metrics_after.duplicate_action_rate[0]*100).toFixed(1) + '%';
                document.getElementById('heldout-correct-after').innerText = (data.heldout_metrics_after.correct_completion[0]*100).toFixed(1) + '%';
            }
            if (data.scale_and_cost) {
                document.getElementById('scale-verif-runs').innerText = data.scale_and_cost.verification_runs + ' episodes';
                document.getElementById('scale-serial-time').innerText = data.scale_and_cost.serial_wall_clock_estimate_sec + 's';
                document.getElementById('scale-parallel-time').innerText = data.scale_and_cost.actual_parallel_wall_clock_sec + 's';
                document.getElementById('scale-speedup').innerText = data.scale_and_cost.speedup_factor + 'x';
            }
            if (data.generated_test_code) {
                document.getElementById('code-snippet').innerText = data.generated_test_code;
            }
        }

        window.onload = loadData;
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return DASHBOARD_HTML

@app.get("/api/state")
def get_pipeline_state():
    return _dashboard_state()

@app.post("/api/run")
def trigger_pipeline():
    orchestrator.run_full_pipeline(train_count=50, held_out_count=30)
    return _dashboard_state()
