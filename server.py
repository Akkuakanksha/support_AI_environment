"""
FastAPI server for the Support Env OpenEnv environment.
Serves the OpenEnv HTTP API + a browser UI on port 7860.
"""
import os
import sys
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── ensure project root is on path ───────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models import Action, Observation, Reward
from app.env import SupportEnv, TASKS

# ── App + single shared env instance ─────────────────────────────────────────
app = FastAPI(
    title="Support Env",
    description="AI Customer Support Ticket Resolution Environment",
    version="0.1.0",
)

_env = SupportEnv()

# ── Pydantic request schemas ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_index: Optional[int] = None

class StepRequest(BaseModel):
    action_type: str
    content: Optional[str] = None


# ── OpenEnv API endpoints ─────────────────────────────────────────────────────

@app.post("/reset")
def reset(req: ResetRequest = None):
    task_idx = req.task_index if req else None
    obs = _env.reset(task_index=task_idx)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    action = Action(action_type=req.action_type, content=req.content)
    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return _env.state()


@app.get("/tasks")
def tasks():
    return {"tasks": SupportEnv.task_list()}


@app.get("/health")
def health():
    return {"status": "ok", "env": "support-env", "version": "0.1.0"}


# ── Frontend ──────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Support Env — OpenEnv</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2d3149;
    --accent: #6c63ff; --accent2: #ff6584; --text: #e8e8f0;
    --muted: #8b8fa8; --success: #43d98c; --warn: #f5c842; --danger: #ff5f6d;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
  header span { color: var(--muted); font-size: 0.85rem; }
  .badge { background: var(--accent); color: #fff; font-size: 0.7rem; padding: 2px 8px; border-radius: 20px; font-weight: 600; }

  main { max-width: 1000px; margin: 0 auto; padding: 2rem; display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }
  .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 1rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.05em; }

  .ticket-box { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
  .ticket-id { font-size: 0.75rem; color: var(--muted); margin-bottom: 0.3rem; }
  .ticket-query { font-size: 0.95rem; line-height: 1.5; }
  .status-badge { display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 2px 10px; border-radius: 20px; margin-top: 0.5rem; }
  .status-open { background: #1a3a2a; color: var(--success); }
  .status-in_progress { background: #2a2a1a; color: var(--warn); }
  .status-closed { background: #1a1a3a; color: #8888ff; }

  .history { max-height: 160px; overflow-y: auto; font-size: 0.82rem; color: var(--muted); line-height: 1.8; }
  .history div::before { content: "▸ "; color: var(--accent); }

  .actions-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-bottom: 1rem; }
  button { cursor: pointer; border: none; border-radius: 8px; font-size: 0.9rem; font-weight: 600; padding: 0.6rem 1rem; transition: all 0.15s; }
  .btn-action { background: var(--border); color: var(--text); }
  .btn-action:hover { background: var(--accent); color: #fff; }
  .btn-action.active { background: var(--accent); color: #fff; }
  .btn-primary { background: var(--accent); color: #fff; width: 100%; padding: 0.75rem; margin-bottom: 0.5rem; }
  .btn-primary:hover { filter: brightness(1.15); }
  .btn-reset { background: var(--danger); color: #fff; width: 100%; padding: 0.75rem; }
  .btn-reset:hover { filter: brightness(1.15); }

  textarea { width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: 0.85rem; padding: 0.6rem; resize: vertical; min-height: 60px; margin-bottom: 0.8rem; outline: none; }
  textarea:focus { border-color: var(--accent); }

  .score-bar-wrap { background: var(--bg); border-radius: 20px; height: 10px; overflow: hidden; margin: 0.4rem 0 0.8rem; }
  .score-bar { height: 100%; border-radius: 20px; background: linear-gradient(90deg, var(--danger), var(--warn), var(--success)); transition: width 0.4s ease; }

  .reward-box { background: var(--bg); border-radius: 8px; padding: 1rem; margin-top: 0.5rem; font-size: 0.85rem; }
  .reward-score { font-size: 1.8rem; font-weight: 700; color: var(--success); }
  .reward-reason { color: var(--muted); margin-top: 0.3rem; }

  .task-selector { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
  .task-btn { flex: 1; padding: 0.5rem; background: var(--bg); border: 1px solid var(--border); color: var(--muted); border-radius: 8px; font-size: 0.8rem; }
  .task-btn.active { border-color: var(--accent); color: var(--text); background: rgba(108,99,255,0.1); }
  .task-btn:hover { border-color: var(--accent); }

  .done-banner { background: rgba(67,217,140,0.1); border: 1px solid var(--success); border-radius: 8px; padding: 0.8rem 1rem; text-align: center; color: var(--success); font-weight: 600; margin-bottom: 1rem; display: none; }

  @media(max-width: 700px){ main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>🎧 Support Env</h1>
  <span>Customer Support Ticket Resolution</span>
  <span class="badge">OpenEnv</span>
</header>
<main>
  <!-- LEFT: Ticket & history -->
  <div>
    <div class="card" style="margin-bottom:1.5rem">
      <h2>📋 Current Ticket</h2>
      <div class="task-selector">
        <button class="task-btn active" onclick="selectTask(0)">Easy</button>
        <button class="task-btn" onclick="selectTask(1)">Medium</button>
        <button class="task-btn" onclick="selectTask(2)">Hard</button>
      </div>
      <div class="ticket-box">
        <div class="ticket-id" id="ticketId">—</div>
        <div class="ticket-query" id="ticketQuery">Press Reset to start</div>
        <div><span class="status-badge status-open" id="statusBadge">open</span></div>
      </div>
    </div>
    <div class="card">
      <h2>📜 Action History</h2>
      <div class="history" id="historyBox"><div style="color:var(--muted);padding:0.5rem 0">No actions yet</div></div>
    </div>
  </div>

  <!-- RIGHT: Controls & reward -->
  <div>
    <div class="card" style="margin-bottom:1.5rem">
      <h2>🕹 Actions</h2>
      <div class="done-banner" id="doneBanner">✅ Episode Complete!</div>
      <div class="actions-grid">
        <button class="btn-action" onclick="setAction('classify')">🏷 Classify</button>
        <button class="btn-action" onclick="setAction('respond')">💬 Respond</button>
        <button class="btn-action" onclick="setAction('escalate')">🚨 Escalate</button>
        <button class="btn-action" onclick="setAction('close')">✅ Close</button>
      </div>
      <textarea id="responseContent" placeholder="Optional: type your response content here (for respond action)..."></textarea>
      <button class="btn-primary" onclick="doStep()">▶ Take Action</button>
      <button class="btn-reset" onclick="doReset()">↩ Reset Episode</button>
    </div>

    <div class="card">
      <h2>🏆 Reward</h2>
      <div class="reward-box" id="rewardBox">
        <div style="color:var(--muted)">No reward yet</div>
      </div>
    </div>
  </div>
</main>

<script>
let selectedAction = 'classify';
let currentTaskIndex = 0;
let episodeDone = false;

async function api(path, method='GET', body=null){
  const opts = { method, headers:{'Content-Type':'application/json'} };
  if(body) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  return r.json();
}

function setAction(a){
  selectedAction = a;
  document.querySelectorAll('.btn-action').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
}

function selectTask(idx){
  currentTaskIndex = idx;
  document.querySelectorAll('.task-btn').forEach((b,i) => b.classList.toggle('active', i===idx));
  doReset();
}

function updateObs(obs){
  document.getElementById('ticketId').textContent = obs.ticket_id;
  document.getElementById('ticketQuery').textContent = obs.customer_query;
  const sb = document.getElementById('statusBadge');
  sb.textContent = obs.status;
  sb.className = 'status-badge status-' + obs.status;
  const hb = document.getElementById('historyBox');
  if(obs.history && obs.history.length){
    hb.innerHTML = obs.history.map(h => `<div>${h}</div>`).join('');
    hb.scrollTop = hb.scrollHeight;
  } else {
    hb.innerHTML = '<div style="color:var(--muted);padding:0.5rem 0">No actions yet</div>';
  }
}

function updateReward(reward){
  const score = reward.score ?? 0;
  const pct = Math.round(score * 100);
  document.getElementById('rewardBox').innerHTML = `
    <div class="reward-score">${score.toFixed(2)}</div>
    <div class="score-bar-wrap"><div class="score-bar" style="width:${pct}%"></div></div>
    <div class="reward-reason">${reward.reason || ''}</div>
  `;
}

async function doReset(){
  episodeDone = false;
  document.getElementById('doneBanner').style.display = 'none';
  document.getElementById('rewardBox').innerHTML = '<div style="color:var(--muted)">No reward yet</div>';
  const obs = await api('/reset','POST',{task_index: currentTaskIndex});
  updateObs(obs);
}

async function doStep(){
  if(episodeDone){ alert('Episode is done! Press Reset to start a new one.'); return; }
  const content = document.getElementById('responseContent').value.trim();
  const body = { action_type: selectedAction };
  if(content) body.content = content;
  const data = await api('/step','POST', body);
  updateObs(data.observation);
  updateReward(data.reward);
  if(data.done){
    episodeDone = true;
    document.getElementById('doneBanner').style.display = 'block';
  }
}

// Auto-reset on load
doReset();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
