import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000";

const css = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #070a0e;
    --surface: #0e1318;
    --surface2: #141c24;
    --border: #1d2a38;
    --border2: #243345;
    --green: #00e87c;
    --green-dim: rgba(0,232,124,0.12);
    --orange: #ff6b35;
    --orange-dim: rgba(255,107,53,0.12);
    --blue: #38bdf8;
    --blue-dim: rgba(56,189,248,0.10);
    --purple: #a78bfa;
    --text: #d4dde8;
    --muted: #5a6a7a;
    --muted2: #3a4a5a;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Grid noise bg */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,232,124,0.02) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,232,124,0.02) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
  }

  .app { position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 0 24px 80px; }

  /* HEADER */
  .header {
    padding: 48px 0 32px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 40px;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 24px;
  }
  .header-left { flex: 1; }
  .eyebrow {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--green);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .eyebrow::before {
    content: '';
    width: 24px; height: 1px;
    background: var(--green);
  }
  .title {
    font-family: var(--font-head);
    font-size: clamp(28px, 4vw, 44px);
    font-weight: 800;
    line-height: 1.05;
    color: #fff;
    letter-spacing: -0.02em;
    margin-bottom: 12px;
  }
  .title span { color: var(--green); }
  .subtitle {
    font-size: 13px;
    color: var(--muted);
    line-height: 1.7;
    max-width: 520px;
  }
  .health-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 4px;
    border: 1px solid;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    white-space: nowrap;
    transition: all 0.3s;
    flex-shrink: 0;
    margin-top: 8px;
  }
  .health-badge.ok { border-color: var(--green); color: var(--green); background: var(--green-dim); }
  .health-badge.fail { border-color: var(--orange); color: var(--orange); background: var(--orange-dim); }
  .health-badge.checking { border-color: var(--muted2); color: var(--muted); background: transparent; }
  .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse-dot 1.8s ease-in-out infinite;
  }
  .health-badge.ok .dot { animation: pulse-dot 1.8s ease-in-out infinite; }
  .health-badge:not(.ok) .dot { animation: none; }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
  }

  /* COLUMNS */
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
  @media (max-width: 760px) { .columns { grid-template-columns: 1fr; } }

  /* CARD */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .card:hover { border-color: var(--border2); }
  .card-head {
    padding: 18px 20px 14px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .card-tag {
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .card-title {
    font-family: var(--font-head);
    font-size: 17px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 2px;
  }
  .phase-badge {
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 3px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
  }
  .phase-badge.p1 { background: var(--green-dim); color: var(--green); border: 1px solid rgba(0,232,124,0.25); }
  .phase-badge.p2 { background: var(--orange-dim); color: var(--orange); border: 1px solid rgba(255,107,53,0.25); }
  .card-body { padding: 20px; }

  /* FORM */
  .field { margin-bottom: 14px; }
  .field label {
    display: block;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .field input, .field select {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 4px;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 13px;
    padding: 9px 12px;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
    -webkit-appearance: none;
  }
  .field input:focus, .field select:focus {
    border-color: var(--green);
    box-shadow: 0 0 0 2px rgba(0,232,124,0.08);
  }
  .field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }

  /* BUTTON */
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: 100%;
    padding: 11px 20px;
    border-radius: 4px;
    border: none;
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.2s;
    margin-top: 6px;
    position: relative;
    overflow: hidden;
  }
  .btn::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, transparent 60%);
    pointer-events: none;
  }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-green { background: var(--green); color: #000; }
  .btn-green:not(:disabled):hover { background: #00ff8a; box-shadow: 0 0 20px rgba(0,232,124,0.35); transform: translateY(-1px); }
  .btn-orange { background: var(--orange); color: #000; }
  .btn-orange:not(:disabled):hover { background: #ff7f50; box-shadow: 0 0 20px rgba(255,107,53,0.35); transform: translateY(-1px); }
  .btn-ghost {
    background: transparent;
    border: 1px solid var(--border2);
    color: var(--muted);
    font-size: 11px;
    padding: 7px 14px;
    width: auto;
    margin-top: 0;
  }
  .btn-ghost:hover { border-color: var(--blue); color: var(--blue); }

  /* SPINNER */
  .spinner {
    width: 13px; height: 13px;
    border: 2px solid rgba(0,0,0,0.3);
    border-top-color: #000;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    flex-shrink: 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* METRICS GRID */
  .metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 16px; }
  .metric-cell {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 12px 14px;
  }
  .metric-label {
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 5px;
  }
  .metric-value {
    font-size: 20px;
    font-weight: 500;
    color: #fff;
    letter-spacing: -0.02em;
  }
  .metric-value.green { color: var(--green); }
  .metric-value.orange { color: var(--orange); }
  .metric-value.blue { color: var(--blue); }

  /* PROGRESS BAR */
  .progress-wrap { margin: 14px 0 0; }
  .progress-label { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-bottom: 5px; }
  .progress-bar {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s ease;
  }
  .progress-fill.green { background: var(--green); box-shadow: 0 0 8px var(--green); }
  .progress-fill.orange { background: var(--orange); box-shadow: 0 0 8px var(--orange); }

  /* RECEPTOR STATE */
  .receptor-card { margin-bottom: 24px; }
  .receptor-bar-grid { display: grid; gap: 6px; margin-top: 12px; }
  .rec-row { display: flex; align-items: center; gap: 10px; }
  .rec-label { font-size: 10px; color: var(--muted); width: 28px; text-align: right; flex-shrink: 0; }
  .rec-bar-bg { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .rec-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--blue), var(--green));
    transition: width 0.6s ease;
  }
  .rec-val { font-size: 10px; color: var(--muted); width: 40px; flex-shrink: 0; }

  /* FLOW LOG */
  .flow-card { margin-bottom: 24px; }
  .flow-head-actions { display: flex; align-items: center; gap: 8px; }
  .flow-log {
    background: #060a0d;
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 12px;
    height: 220px;
    overflow-y: auto;
    font-size: 11.5px;
    line-height: 1.75;
    scrollbar-width: thin;
    scrollbar-color: var(--border2) transparent;
  }
  .flow-log::-webkit-scrollbar { width: 4px; }
  .flow-log::-webkit-scrollbar-track { background: transparent; }
  .flow-log::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
  .flow-entry { display: flex; gap: 12px; padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.025); }
  .flow-entry:last-child { border-bottom: none; }
  .flow-ep { color: var(--muted); min-width: 40px; flex-shrink: 0; }
  .flow-event { color: var(--text); word-break: break-word; }
  .flow-event.llm { color: var(--blue); }
  .flow-event.escape { color: var(--orange); }
  .flow-event.reward { color: var(--green); }
  .flow-empty { color: var(--muted2); font-style: italic; }
  .tab-row { display: flex; gap: 0; border-bottom: 1px solid var(--border); }
  .tab {
    padding: 10px 18px;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: all 0.2s;
    background: none;
    border-top: none;
    border-left: none;
    border-right: none;
    font-family: var(--font-mono);
  }
  .tab:hover { color: var(--text); }
  .tab.active.green { color: var(--green); border-bottom-color: var(--green); }
  .tab.active.orange { color: var(--orange); border-bottom-color: var(--orange); }

  /* KEY METRICS BANNER */
  .banner {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 24px;
    overflow: hidden;
  }
  @media (max-width: 640px) { .banner { grid-template-columns: 1fr 1fr; } }
  .banner-cell {
    padding: 20px 22px;
    border-right: 1px solid var(--border);
    position: relative;
  }
  .banner-cell:last-child { border-right: none; }
  .banner-cell::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }
  .banner-cell.b-green::before { background: var(--green); }
  .banner-cell.b-orange::before { background: var(--orange); }
  .banner-cell.b-blue::before { background: var(--blue); }
  .banner-cell.b-purple::before { background: var(--purple); }
  .banner-num {
    font-family: var(--font-head);
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #fff;
    margin-bottom: 3px;
  }
  .banner-num.g { color: var(--green); }
  .banner-num.o { color: var(--orange); }
  .banner-num.b { color: var(--blue); }
  .banner-num.p { color: var(--purple); }
  .banner-label { font-size: 10px; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; }

  /* ALERT */
  .alert {
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 12px;
    margin-top: 12px;
    display: flex;
    align-items: flex-start;
    gap: 8px;
  }
  .alert.err { background: rgba(255,107,53,0.08); border: 1px solid rgba(255,107,53,0.25); color: var(--orange); }
  .alert.suc { background: var(--green-dim); border: 1px solid rgba(0,232,124,0.25); color: var(--green); }
  .alert-icon { flex-shrink: 0; font-size: 13px; }

  /* DIVIDER */
  .divider { border: none; border-top: 1px solid var(--border); margin: 4px 0 20px; }

  /* SECTION LABEL */
  .section-label {
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

  /* TOOLTIP-STYLE DESCRIPTION */
  .info-pill {
    display: inline-block;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 20px;
    background: var(--surface2);
    border: 1px solid var(--border2);
    color: var(--muted);
    margin-top: 4px;
  }

  /* FADE IN */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .fade-in { animation: fadeUp 0.35s ease forwards; }
`;

function fmt(val) {
  if (val == null) return "—";
  if (typeof val === "number") return val.toFixed(3);
  return String(val);
}

function MetricCell({ label, value, color = "" }) {
  return (
    <div className="metric-cell">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${color}`}>{fmt(value)}</div>
    </div>
  );
}

function Alert({ type, msg }) {
  if (!msg) return null;
  return (
    <div className={`alert ${type}`}>
      <span className="alert-icon">{type === "err" ? "⚠" : "✓"}</span>
      <span>{msg}</span>
    </div>
  );
}

function ReceptorBars({ receptor }) {
  if (!receptor) return <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 12 }}>Run Phase 1 to see receptor state.</div>;
  const binding = receptor.binding_sites || [];
  const allergy = receptor.allostery || [];
  const items = [...binding.slice(0, 4).map((v, i) => ({ label: `B${i}`, val: v })),
                 ...allergy.slice(0, 4).map((v, i) => ({ label: `A${i}`, val: v }))];
  return (
    <div className="receptor-bar-grid">
      {items.map(({ label, val }) => (
        <div className="rec-row" key={label}>
          <span className="rec-label">{label}</span>
          <div className="rec-bar-bg">
            <div className="rec-bar-fill" style={{ width: `${Math.min(Math.abs(val) * 100, 100)}%` }} />
          </div>
          <span className="rec-val">{val?.toFixed(2) ?? "—"}</span>
        </div>
      ))}
    </div>
  );
}

function FlowLog({ events }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [events]);

  if (!events || events.length === 0) {
    return <div className="flow-log"><span className="flow-empty">No events yet — run training to populate log.</span></div>;
  }

  return (
    <div className="flow-log" ref={ref}>
      {events.map((ev, i) => {
        const text = typeof ev === "string" ? ev : JSON.stringify(ev);
        const isLLM = text.toLowerCase().includes("llm");
        const isEscape = text.toLowerCase().includes("escape");
        const isReward = text.toLowerCase().includes("reward") || text.toLowerCase().includes("binding");
        const cls = isLLM ? "llm" : isEscape ? "escape" : isReward ? "reward" : "";
        return (
          <div className="flow-entry" key={i}>
            <span className="flow-ep">#{i + 1}</span>
            <span className={`flow-event ${cls}`}>{text}</span>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState("checking");
  const [state, setState] = useState({ phase1: null, phase2: null, receptor: null });
  const [flow, setFlow] = useState({ phase1: [], phase2: [] });
  const [flowTab, setFlowTab] = useState("phase1");

  const [p1, setP1] = useState({ episodes: 8, dimension: 8, max_steps: 6, llm_interval: 2, seed: 0 });
  const [p2, setP2] = useState({ episodes: 6, llm_interval: 2 });

  const [p1Loading, setP1Loading] = useState(false);
  const [p2Loading, setP2Loading] = useState(false);
  const [p1Err, setP1Err] = useState("");
  const [p2Err, setP2Err] = useState("");
  const [p1Ok, setP1Ok] = useState("");
  const [p2Ok, setP2Ok] = useState("");

  const checkHealth = useCallback(async () => {
    try {
      const r = await fetch(`${API}/health`);
      setHealth(r.ok ? "ok" : "fail");
    } catch { setHealth("fail"); }
  }, []);

  const fetchState = useCallback(async () => {
    try {
      const r = await fetch(`${API}/state`);
      if (r.ok) setState(await r.json());
    } catch {}
  }, []);

  const fetchFlow = useCallback(async (phase) => {
    try {
      const r = await fetch(`${API}/flow/${phase}`);
      if (r.ok) {
        const d = await r.json();
        setFlow(f => ({ ...f, [phase]: d.events || [] }));
      }
    } catch {}
  }, []);

  useEffect(() => {
    checkHealth();
    fetchState();
    const t = setInterval(checkHealth, 8000);
    return () => clearInterval(t);
  }, [checkHealth, fetchState]);

  async function runPhase1() {
    setP1Loading(true); setP1Err(""); setP1Ok("");
    try {
      const r = await fetch(`${API}/train/phase1`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...p1, episodes: +p1.episodes, dimension: +p1.dimension, max_steps: +p1.max_steps, llm_interval: +p1.llm_interval, seed: +p1.seed }),
      });
      const d = await r.json();
      if (!r.ok) { setP1Err(d.detail || "Training failed"); }
      else {
        setP1Ok(`Phase 1 complete — ${d.episodes} episodes`);
        await fetchState();
        await fetchFlow("phase1");
      }
    } catch (e) { setP1Err("Cannot reach API. Is the server running?"); }
    setP1Loading(false);
  }

  async function runPhase2() {
    setP2Loading(true); setP2Err(""); setP2Ok("");
    try {
      const body = { episodes: +p2.episodes };
      if (p2.llm_interval) body.llm_interval = +p2.llm_interval;
      const r = await fetch(`${API}/train/phase2`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!r.ok) { setP2Err(d.detail || "Training failed"); }
      else {
        setP2Ok(`Phase 2 complete — ${d.episodes} episodes`);
        await fetchState();
        await fetchFlow("phase2");
      }
    } catch (e) { setP2Err("Cannot reach API. Is the server running?"); }
    setP2Loading(false);
  }

  const m1 = state.phase1 || {};
  const m2 = state.phase2 || {};

  const peakBinding = m1.peak_binding ?? m1.best_binding ?? null;
  const peakSel = m1.peak_selectivity ?? m1.best_selectivity ?? null;
  const escapeMotifs = m2.escape_motifs_found ?? m2.escape_motifs ?? null;
  const escapeDisruption = m2.mean_escape_disruption ?? m2.disruption ?? null;

  function InputField({ label, value, onChange, min, max, help }) {
    return (
      <div className="field">
        <label>{label}</label>
        <input type="number" value={value} min={min} max={max}
          onChange={e => onChange(e.target.value)} />
        {help && <div className="info-pill">{help}</div>}
      </div>
    );
  }

  return (
    <>
      <style>{css}</style>
      <div className="app">

        {/* HEADER */}
        <header className="header">
          <div className="header-left">
            <div className="eyebrow">OpenEnv · Adversarial RL · Drug Discovery</div>
            <h1 className="title">Bidirectional<br /><span>Adversarial</span> RL</h1>
            <p className="subtitle">
              Two agents in an arms race — LigandDesigner binds the receptor, ReceptorMutator evolves to escape.
              Co-evolution drives both toward robust, resistance-aware strategies.
            </p>
          </div>
          <div>
            <div className={`health-badge ${health}`}>
              <span className="dot" />
              {health === "ok" ? "API Online" : health === "fail" ? "API Offline" : "Checking…"}
            </div>
          </div>
        </header>

        {/* BANNER */}
        <div className="banner fade-in">
          <div className="banner-cell b-green">
            <div className={`banner-num ${peakBinding != null ? "g" : ""}`}>
              {peakBinding != null ? peakBinding.toFixed(3) : "—"}
            </div>
            <div className="banner-label">Peak Binding Score</div>
          </div>
          <div className="banner-cell b-orange">
            <div className={`banner-num ${peakSel != null ? "o" : ""}`}>
              {peakSel != null ? peakSel.toFixed(3) : "—"}
            </div>
            <div className="banner-label">Best Selectivity</div>
          </div>
          <div className="banner-cell b-blue">
            <div className={`banner-num ${escapeMotifs != null ? "b" : ""}`}>
              {escapeMotifs != null ? escapeMotifs : "—"}
            </div>
            <div className="banner-label">Escape Motifs Found</div>
          </div>
          <div className="banner-cell b-purple">
            <div className={`banner-num ${escapeDisruption != null ? "p" : ""}`}>
              {escapeDisruption != null ? (escapeDisruption * 100).toFixed(1) + "%" : "—"}
            </div>
            <div className="banner-label">Mean Disruption</div>
          </div>
        </div>

        {/* PHASE 1 + PHASE 2 COLUMNS */}
        <div className="columns">

          {/* PHASE 1 */}
          <div className="card fade-in">
            <div className="card-head">
              <div>
                <div className="card-tag">Phase 1</div>
                <div className="card-title">Co-Training</div>
              </div>
              <span className="phase-badge p1">Ligand vs Receptor</span>
            </div>
            <div className="card-body">
              <div className="field-row">
                <InputField label="Episodes" value={p1.episodes} onChange={v => setP1(s => ({ ...s, episodes: v }))} min={1} max={100} />
                <InputField label="Dimension" value={p1.dimension} onChange={v => setP1(s => ({ ...s, dimension: v }))} min={2} max={64} />
              </div>
              <div className="field-row">
                <InputField label="Max Steps" value={p1.max_steps} onChange={v => setP1(s => ({ ...s, max_steps: v }))} min={1} max={50} />
                <InputField label="LLM Interval" value={p1.llm_interval} onChange={v => setP1(s => ({ ...s, llm_interval: v }))} min={1} max={50} help="episodes between LLM calls" />
              </div>
              <InputField label="Random Seed" value={p1.seed} onChange={v => setP1(s => ({ ...s, seed: v }))} min={0} />
              <button className="btn btn-green" onClick={runPhase1} disabled={p1Loading || health !== "ok"}>
                {p1Loading ? <><span className="spinner" /> Training…</> : "▶ Run Phase 1"}
              </button>
              <Alert type="err" msg={p1Err} />
              <Alert type="suc" msg={p1Ok} />

              {m1 && Object.keys(m1).length > 0 && (
                <>
                  <hr className="divider" />
                  <div className="section-label">Results</div>
                  <div className="metrics-grid">
                    <MetricCell label="Peak Binding" value={m1.peak_binding ?? m1.best_binding} color="green" />
                    <MetricCell label="Selectivity" value={m1.peak_selectivity ?? m1.best_selectivity} color="green" />
                    <MetricCell label="Ligand Reward" value={m1.mean_ligand_reward ?? m1.avg_ligand_reward} />
                    <MetricCell label="Receptor Reward" value={m1.mean_receptor_reward ?? m1.avg_receptor_reward} color="orange" />
                  </div>
                  {(m1.peak_binding ?? m1.best_binding) != null && (
                    <div className="progress-wrap">
                      <div className="progress-label">
                        <span>Binding Progress</span>
                        <span>{((m1.peak_binding ?? m1.best_binding ?? 0) * 100).toFixed(1)}%</span>
                      </div>
                      <div className="progress-bar">
                        <div className="progress-fill green" style={{ width: `${(m1.peak_binding ?? m1.best_binding ?? 0) * 100}%` }} />
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          {/* PHASE 2 */}
          <div className="card fade-in" style={{ animationDelay: "0.05s" }}>
            <div className="card-head">
              <div>
                <div className="card-tag">Phase 2</div>
                <div className="card-title">Escape Agent</div>
              </div>
              <span className="phase-badge p2">Adversarial Attack</span>
            </div>
            <div className="card-body">
              <div
                style={{
                  background: "rgba(255,107,53,0.06)",
                  border: "1px solid rgba(255,107,53,0.18)",
                  borderRadius: 4,
                  padding: "10px 12px",
                  fontSize: 12,
                  color: "var(--muted)",
                  marginBottom: 16,
                  lineHeight: 1.6,
                }}
              >
                Requires Phase 1 to be completed first. The EscapeAgent attacks the frozen best ligand
                to find receptor mutations that break binding — feeding hard negatives back into training.
              </div>
              <div className="field-row">
                <InputField label="Episodes" value={p2.episodes} onChange={v => setP2(s => ({ ...s, episodes: v }))} min={1} max={100} />
                <InputField label="LLM Interval" value={p2.llm_interval} onChange={v => setP2(s => ({ ...s, llm_interval: v }))} min={1} max={50} />
              </div>
              <button
                className="btn btn-orange"
                onClick={runPhase2}
                disabled={p2Loading || health !== "ok" || !state.phase1}
              >
                {p2Loading ? <><span className="spinner" style={{ borderTopColor: "#000" }} /> Escaping…</> : "▶ Run Phase 2"}
              </button>
              <Alert type="err" msg={p2Err} />
              <Alert type="suc" msg={p2Ok} />

              {m2 && Object.keys(m2).length > 0 && (
                <>
                  <hr className="divider" />
                  <div className="section-label">Results</div>
                  <div className="metrics-grid">
                    <MetricCell label="Escape Motifs" value={m2.escape_motifs_found ?? m2.escape_motifs} color="orange" />
                    <MetricCell label="Hard Negatives" value={m2.hard_negatives_injected ?? m2.hard_negatives} color="orange" />
                    <MetricCell label="Mean Disruption" value={m2.mean_escape_disruption ?? m2.disruption} />
                    <MetricCell label="Best Escape Rew" value={m2.best_escape_reward ?? m2.peak_escape} color="blue" />
                  </div>
                  {(m2.mean_escape_disruption ?? m2.disruption) != null && (
                    <div className="progress-wrap">
                      <div className="progress-label">
                        <span>Disruption Rate</span>
                        <span>{(((m2.mean_escape_disruption ?? m2.disruption) ?? 0) * 100).toFixed(1)}%</span>
                      </div>
                      <div className="progress-bar">
                        <div className="progress-fill orange" style={{ width: `${((m2.mean_escape_disruption ?? m2.disruption) ?? 0) * 100}%` }} />
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        {/* RECEPTOR STATE */}
        <div className="card receptor-card fade-in" style={{ animationDelay: "0.08s" }}>
          <div className="card-head">
            <div>
              <div className="card-tag">Live State</div>
              <div className="card-title">Receptor Configuration</div>
            </div>
            <button className="btn btn-ghost" onClick={() => { fetchState(); fetchFlow("phase1"); fetchFlow("phase2"); }}>
              ↻ Refresh
            </button>
          </div>
          <div className="card-body">
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
              <div>
                <div className="section-label">Binding Sites & Allostery</div>
                <ReceptorBars receptor={state.receptor} />
              </div>
              <div>
                <div className="section-label">System Metrics</div>
                <div className="metrics-grid">
                  <MetricCell label="Resistance" value={state.receptor?.resistance_factor} color="orange" />
                  <MetricCell label="Functionality" value={state.receptor?.functionality_score} color="green" />
                  <MetricCell label="Mutation Δ" value={state.receptor?.cumulative_mutation_delta} />
                  <MetricCell label="Generation" value={state.receptor?.generation} color="blue" />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* FLOW LOG */}
        <div className="card flow-card fade-in" style={{ animationDelay: "0.1s" }}>
          <div className="card-head">
            <div>
              <div className="card-tag">Training Log</div>
              <div className="card-title">Event Stream</div>
            </div>
            <div className="flow-head-actions">
              <button className="btn btn-ghost" onClick={() => fetchFlow(flowTab)}>↻</button>
            </div>
          </div>
          <div className="tab-row">
            <button className={`tab ${flowTab === "phase1" ? "active green" : ""}`} onClick={() => { setFlowTab("phase1"); fetchFlow("phase1"); }}>
              Phase 1
            </button>
            <button className={`tab ${flowTab === "phase2" ? "active orange" : ""}`} onClick={() => { setFlowTab("phase2"); fetchFlow("phase2"); }}>
              Phase 2
            </button>
          </div>
          <div style={{ padding: "14px 16px 16px" }}>
            <FlowLog events={flow[flowTab]} />
          </div>
        </div>

        {/* FOOTER */}
        <div style={{ textAlign: "center", color: "var(--muted2)", fontSize: 11, paddingTop: 16, letterSpacing: "0.1em", textTransform: "uppercase" }}>
          Bidirectional Adversarial RL · OpenEnv · {new Date().getFullYear()}
        </div>
      </div>
    </>
  );
}
