"""Session-safe FastAPI dashboard for running the existing trading cycle."""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse

from crew.crew import CONFIG_DIR, LOG_ROOT, ROOT, _read_config
from run_cycle import run_cycle

SESSION_COOKIE = "trading_session"
SESSION_TTL_SECONDS = 12 * 60 * 60
APPROVAL_TIMEOUT_SECONDS = 30 * 60
SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.-]*$")
SIGNALS = ("trend", "rotation", "breakout", "dip", "ml", "news")
PROVIDERS = ("ollama_local", "ollama_cloud", "groq")


INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trading Cycle Console</title>
<style>
:root{color-scheme:dark;--bg:#101715;--panel:#17211e;--panel2:#1b2924;--line:#31413a;--text:#e9f0eb;--muted:#9dafa5;--mint:#77d7ae;--green:#23785c;--amber:#e5b564;--red:#df7f73;--blue:#7cb8d8;font-family:ui-sans-serif,system-ui,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 12% 0%,#21352e 0,transparent 34rem),var(--bg);color:var(--text);font-size:14px}button,input,select,textarea{font:inherit;color:inherit}button{cursor:pointer;border:1px solid var(--line);background:#20312b;padding:9px 13px;border-radius:6px}button:hover{border-color:var(--mint)}button.primary{background:var(--green);border-color:#419979;color:#f4fff9;font-weight:650}button.danger{color:#ffd6d0;border-color:#754541;background:#382421}.shell{max-width:1440px;margin:0 auto;padding:24px}.top{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:22px}.brand{display:flex;gap:14px;align-items:center}.mark{width:38px;height:38px;border:1px solid #528b70;border-radius:8px;color:var(--mint);display:grid;place-items:center;font-weight:800;background:linear-gradient(145deg,#243b31,#15201b)}h1{font-size:20px;margin:0;font-weight:700}.subtitle,.muted{color:#a0b1a8}.subtitle{margin-top:4px;font-size:12px}.status{border:1px solid var(--line);border-radius:99px;padding:7px 11px;color:var(--muted);background:#131d19}.layout{display:grid;grid-template-columns:minmax(290px,370px) minmax(0,1fr);gap:20px}.column{display:grid;gap:16px;align-content:start}.panel{border:1px solid var(--line);border-radius:8px;background:linear-gradient(160deg,#1a2521,#151e1b);padding:16px}.panel h2{font-size:14px;margin:0 0 14px;letter-spacing:.02em}.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.field{display:grid;gap:6px;margin-bottom:12px}.field:last-child{margin-bottom:0}label,.label{font-size:12px;color:#bac9c1}.hint{font-size:11px;color:#8fa198;line-height:1.5}.input,select,textarea{width:100%;border:1px solid var(--line);background:#101916;border-radius:5px;padding:9px 10px;outline:none}.input:focus,select:focus,textarea:focus{border-color:var(--mint)}textarea{min-height:66px;resize:vertical;font-family:ui-monospace,Consolas,monospace;font-size:12px}.signal-list{display:grid;gap:8px}.signal{display:grid;grid-template-columns:22px 1fr 76px;gap:8px;align-items:center;border-bottom:1px solid #293731;padding:5px 0}.signal:last-child{border:0}.signal input[type=number]{width:76px}.key-actions{display:flex;gap:8px;margin-top:8px}.key-actions button{flex:1}.actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.actions button{min-width:92px}.results{display:grid;gap:16px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.metric{padding:12px;border-left:2px solid var(--mint);background:#131d19}.metric span{display:block;color:#9eafa6;font-size:11px}.metric strong{display:block;font-size:16px;margin-top:5px;overflow-wrap:anywhere}.chart-wrap{height:230px;position:relative}.chart-wrap canvas{width:100%;height:100%}.legend{display:flex;gap:14px;font-size:11px;color:#b3c2ba;margin-top:8px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.table-wrap{overflow:auto;max-height:320px}table{border-collapse:collapse;width:100%;font-size:12px}th{text-align:left;color:#9eafa6;font-weight:550;position:sticky;top:0;background:#17211e}th,td{padding:9px 8px;border-bottom:1px solid #2a3932;white-space:nowrap}td.right,th.right{text-align:right}.positive{color:var(--mint)}.negative{color:var(--red)}.banner{padding:10px 12px;border:1px solid var(--line);border-radius:6px;color:#c3d2ca;background:#121b17}.banner.waiting{border-color:#957643;color:#f4d69e;background:#302719}.banner.error{border-color:#784944;color:#ffd5d0;background:#311f1d}.logs{display:flex;gap:8px;flex-wrap:wrap}.logs a{color:var(--mint);font-size:12px}.approval{display:none}.approval.visible{display:block}.approval-actions{display:flex;gap:8px;margin-top:12px}.approval-actions button{min-width:100px}.qty-edit{width:90px;padding:5px 7px;border:1px solid var(--line);background:#101916;border-radius:4px}.hidden{display:none!important}.foot{font-size:11px;color:#82958b;border-top:1px solid var(--line);padding-top:12px;margin-top:4px}@media(max-width:900px){.layout{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.shell{padding:14px}.top{align-items:flex-start}}
</style>
</head>
<body><main class="shell">
<header class="top"><div class="brand"><div class="mark">TC</div><div><h1>Trading Cycle Console</h1><div class="subtitle">Alpaca paper execution · local CrewAI cycle</div></div></div><div id="topStatus" class="status">Ready</div></header>
<div class="layout">
<aside class="column">
<section class="panel"><h2>Cycle setup</h2>
<div class="row"><div class="field"><label for="student">Student</label><select id="student"><option>A</option><option>B</option><option>C</option></select></div><div class="field"><label for="mode">Mode</label><select id="mode"><option value="dry">Dry</option><option value="live">Live</option></select></div></div>
<div class="row"><div class="field"><label for="provider">LLM provider</label><select id="provider"></select></div><div class="field"><label for="approval">Approval</label><select id="approval"><option value="auto">Automatic</option><option value="manual">Require approval</option></select></div></div>
<label style="display:flex;gap:8px;align-items:center;margin:2px 0 14px"><input id="forceClock" type="checkbox"> Bypass market clock (dry only)</label>
<div class="field"><label for="highBeta">High-beta symbols</label><textarea id="highBeta" spellcheck="false"></textarea></div>
<div class="field"><label for="diversified">Diversified symbols</label><textarea id="diversified" spellcheck="false"></textarea></div>
<div class="field"><span class="label">Signals and weights</span><div class="signal-list" id="signals"></div></div>
<div class="actions"><button class="primary" id="runButton">Run cycle</button><button id="clearKeys">Clear session keys</button></div>
</section>
<section class="panel"><h2>Alpaca credentials</h2>
<div class="field"><label for="apiKey">API key</label><input class="input" id="apiKey" type="password" autocomplete="new-password" spellcheck="false"></div>
<div class="field"><label for="apiSecret">Secret</label><input class="input" id="apiSecret" type="password" autocomplete="new-password" spellcheck="false"></div>
<div class="key-actions"><button id="saveKeys">Keep for this browser session</button></div>
<p class="hint">Credentials remain in server memory for this session only. They are never written to files or returned by the API. Live mode is paper-only and still requires TRADING_ENABLED=true.</p>
</section>
<section class="panel"><h2>Run status</h2><div id="runStatus" class="banner">No cycle started.</div><p id="runMessage" class="hint"></p></section>
</aside>
<section class="results">
<div id="approvalPanel" class="panel approval"><h2>Order approval</h2><div class="hint">Edits may reduce quantities or remove proposed orders; symbols, sides, and quantities cannot be increased.</div><div class="table-wrap"><table><thead><tr><th>Symbol</th><th>Side</th><th class="right">Qty</th><th class="right">Notional</th></tr></thead><tbody id="approvalRows"></tbody></table></div><div class="approval-actions"><button class="primary" id="approveButton">Approve</button><button id="editButton">Approve edits</button><button class="danger" id="rejectButton">Reject</button></div></div>
<div class="panel"><h2>Cycle snapshot</h2><div class="metrics"><div class="metric"><span>Provider</span><strong id="metricProvider">—</strong></div><div class="metric"><span>Validation</span><strong id="metricValidation">—</strong></div><div class="metric"><span>Orders</span><strong id="metricOrders">—</strong></div><div class="metric"><span>Portfolio equity</span><strong id="metricEquity">—</strong></div></div></div>
<div class="panel"><h2>Proposed portfolio vs SPY</h2><div class="hint">Normalized 30-session curve from the cycle's OHLCV data and proposed target weights; not account-history equity.</div><div class="chart-wrap"><canvas id="equityChart" aria-label="Normalized portfolio and SPY comparison"></canvas></div><div class="legend"><span><i class="dot" style="background:#77d7ae"></i>Proposed portfolio</span><span><i class="dot" style="background:#7cb8d8"></i>SPY</span></div><div id="curveReason" class="hint"></div></div>
<div class="panel"><h2>Proposed orders</h2><div class="table-wrap"><table><thead><tr><th>Symbol</th><th>Side</th><th class="right">Qty</th><th>Client order ID</th></tr></thead><tbody id="proposedRows"></tbody></table></div></div>
<div class="panel"><h2 id="orderResultsHeading">Executed orders</h2><div class="table-wrap"><table><thead><tr><th>Symbol</th><th>Side</th><th class="right">Qty</th><th>Status</th><th>Client order ID</th></tr></thead><tbody id="executedRows"></tbody></table></div></div>
<div class="panel"><h2>Current positions</h2><div class="table-wrap"><table><thead><tr><th>Symbol</th><th class="right">Qty</th><th class="right">Market value</th><th class="right">Avg entry</th></tr></thead><tbody id="positionRows"></tbody></table></div></div>
<div class="panel"><h2>Run logs</h2><div class="logs" id="logLinks"><span class="hint">Reports appear after the cycle reaches the agent phases.</span></div></div>
</section></div>
</main>
<script>
let csrf='', currentJob='', currentConfig={}, lastOrders=[];
const $=id=>document.getElementById(id);
async function api(path,options={}){const headers={...(options.headers||{})};if(options.method&&options.method!=='GET'){headers['Content-Type']='application/json';headers['X-CSRF-Token']=csrf}const response=await fetch(path,{...options,headers});const data=await response.json().catch(()=>({}));if(!response.ok)throw new Error(data.detail||`Request failed (${response.status})`);return data}
function splitSymbols(value){return value.split(/[\s,;]+/).map(s=>s.trim().toUpperCase()).filter(Boolean)}
function showStatus(text,kind=''){ $('runStatus').textContent=text;$('runStatus').className='banner '+kind;$('topStatus').textContent=text }
async function loadConfig(){currentConfig=await api(`/api/config/${$('student').value}`);$('highBeta').value=currentConfig.high_beta.join(', ');$('diversified').value=currentConfig.diversified.join(', ');$('provider').innerHTML='';for(const item of currentConfig.providers){const option=document.createElement('option');option.value=item.name;option.textContent=`${item.name} · ${item.model}`;$('provider').append(option)}$('provider').value=currentConfig.provider_order[0];const container=$('signals');container.innerHTML='';for(const name of ['trend','rotation','breakout','dip','ml','news']){const row=document.createElement('div');row.className='signal';const check=document.createElement('input');check.type='checkbox';check.dataset.signal=name;check.checked=Number(currentConfig.signal_weights[name]||0)>0;const title=document.createElement('label');title.textContent=name[0].toUpperCase()+name.slice(1);const weight=document.createElement('input');weight.type='number';weight.className='input';weight.min='0';weight.max='1';weight.step='0.01';weight.value=Number(currentConfig.signal_weights[name]||0).toFixed(2);weight.dataset.weight=name;row.append(check,title,weight);container.append(row)}}
function selectedSignalWeights(){const weights={};for(const row of document.querySelectorAll('.signal')){const name=row.querySelector('input[type=checkbox]').dataset.signal;const weight=Number(row.querySelector('input[type=number]').value);weights[name]=row.querySelector('input[type=checkbox]').checked?weight:0}return weights}
async function saveCredentials(){const key=$('apiKey').value,secret=$('apiSecret').value;if(!key||!secret)throw new Error('Enter both credentials before saving them to this session.');await api('/api/session/credentials',{method:'POST',body:JSON.stringify({student:$('student').value,key,secret})});$('apiKey').value='';$('apiSecret').value='';$('runMessage').textContent='Credentials held in server memory for this browser session.'}
function drawChart(curve){const canvas=$('equityChart'),ctx=canvas.getContext('2d'),rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;canvas.width=rect.width*ratio;canvas.height=rect.height*ratio;ctx.scale(ratio,ratio);ctx.clearRect(0,0,rect.width,rect.height);if(!curve||curve.length<2){ctx.fillStyle='#9eafa6';ctx.font='12px system-ui';ctx.fillText('Curve data unavailable for this run.',12,24);return}const values=curve.flatMap(p=>[p.portfolio,p.spy]);const min=Math.min(...values),max=Math.max(...values),pad=18,w=rect.width-pad*2,h=rect.height-pad*2;ctx.strokeStyle='#33433c';ctx.lineWidth=1;for(let i=0;i<4;i++){const y=pad+h*i/3;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(pad+w,y);ctx.stroke()}function line(key,color){ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2;curve.forEach((p,i)=>{const x=pad+w*i/(curve.length-1),y=pad+h-(p[key]-min)/(max-min||1)*h;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}line('portfolio','#77d7ae');line('spy','#7cb8d8')}
function drawChartReason(curve,reason){window.lastCurve=curve||[];drawChart(window.lastCurve);$('curveReason').textContent=window.lastCurve.length<2?`Curve data unavailable for this run: ${reason||'no saved price data is available.'}`:''}
function addCell(row,text,cls=''){const cell=document.createElement('td');cell.textContent=text??'';if(cls)cell.className=cls;row.append(cell);return cell}
function renderRows(id,orders,executed=false){const body=$(id);body.innerHTML='';for(const order of orders||[]){const row=document.createElement('tr');addCell(row,order.symbol);addCell(row,order.side,(order.side==='buy'?'positive':'negative'));addCell(row,Number(order.qty||0).toFixed(4),'right');if(executed){addCell(row,order.status||'dry run');addCell(row,order.client_order_id||'')}else addCell(row,order.client_order_id||'');body.append(row)}}
function updateOrderResultsHeading(mode){$('orderResultsHeading').textContent=mode==='dry'?'Proposed orders (dry run)':'Executed orders'}
function renderApproval(orders){lastOrders=orders||[];const body=$('approvalRows');body.innerHTML='';for(const order of lastOrders){const row=document.createElement('tr');row.dataset.id=order.client_order_id;addCell(row,order.symbol);addCell(row,order.side);const q=addCell(row,'','right');const input=document.createElement('input');input.className='qty-edit';input.type='number';input.min='0';input.step='0.000001';input.max=order.qty;input.value=order.qty;q.append(input);addCell(row,(Number(order.qty)*Number(currentConfig.latest_prices?.[order.symbol]||0)).toFixed(2),'right');body.append(row)}$('approvalPanel').classList.toggle('visible',Boolean(currentJob)&&$('approval').value==='manual')}
async function sendApproval(action){if(!currentJob)return;let body={action};if(action==='edit'){body.orders=[];for(const row of $('approvalRows').querySelectorAll('tr')){const input=row.querySelector('input');if(Number(input.value)>0){const original=lastOrders.find(order=>order.client_order_id===row.dataset.id);body.orders.push({...original,qty:Number(input.value)})}}}await api(`/api/runs/${currentJob}/approval`,{method:'POST',body:JSON.stringify(body)});$('approvalPanel').classList.remove('visible');showStatus(action==='reject'?'Rejected':'Approval received','waiting')}
function renderJob(job){if(job.status==='awaiting_approval'){showStatus('Waiting for approval','waiting');renderApproval(job.proposed_orders)}else if(job.status==='running'){showStatus(job.stage||'Cycle running');$('approvalPanel').classList.remove('visible')}else if(job.status==='completed'){showStatus('Cycle complete','');$('approvalPanel').classList.remove('visible')}else if(job.status==='failed'){showStatus('Cycle failed','error');$('runMessage').textContent=job.error||'Run failed'}else showStatus(job.stage||job.status);$('runMessage').textContent=job.error||job.stage||'';renderRows('proposedRows',job.proposed_orders||[]);const execution=job.execution||{};const executed=[...(execution.sell_orders?.orders||[]),...(execution.buy_orders?.orders||[])];renderRows('executedRows',executed,true);const positions=execution.account?.positions||[];const pbody=$('positionRows');pbody.innerHTML='';for(const p of positions){const tr=document.createElement('tr');addCell(tr,p.symbol);addCell(tr,String(p.qty??''),'right');addCell(tr,Number(p.market_value||0).toFixed(2),'right');addCell(tr,Number(p.avg_entry_price||0).toFixed(2),'right');pbody.append(tr)}$('metricProvider').textContent=job.provider||'—';$('metricValidation').textContent=execution.validation?.valid?'Passed':(execution.validation?'Failed':'—');$('metricOrders').textContent=String(execution.order_count??'—');$('metricEquity').textContent=execution.account?.account?.equity?Number(execution.account.account.equity).toLocaleString(): '—';drawChart(job.curve);const links=$('logLinks');links.innerHTML='';for(const name of job.logs||[]){const a=document.createElement('a');a.href=`/api/runs/${job.id}/logs/${encodeURIComponent(name)}`;a.textContent=name;a.target='_blank';a.rel='noopener';links.append(a)}}
const renderJobBase=renderJob;renderJob=function(job){renderJobBase(job);updateOrderResultsHeading(job.execution?.mode||job.mode);if(job.status==='completed'){showStatus('Completed','positive')}else if(job.status==='skipped'){showStatus('Skipped','waiting');$('runMessage').textContent=`${job.reason||'Cycle skipped.'}${job.previous_run?' Showing the previous completed run.':''}`}else if(job.status==='failed'){showStatus('Failed','error');$('runMessage').textContent=job.reason||job.error||'Cycle failed.'}drawChartReason(job.curve,job.curve_reason)}
async function poll(){if(!currentJob)return;try{const job=await api(`/api/runs/${currentJob}`);currentConfig.latest_prices=job.latest_prices||{};renderJob(job);if(!['completed','failed'].includes(job.status))setTimeout(poll,1800)}catch(error){showStatus(error.message,'error')}}
$('student').addEventListener('change',()=>loadConfig().catch(e=>showStatus(e.message,'error')));$('saveKeys').addEventListener('click',()=>saveCredentials().catch(e=>showStatus(e.message,'error')));$('clearKeys').addEventListener('click',async()=>{try{await api('/api/session/credentials',{method:'DELETE'});$('apiKey').value='';$('apiSecret').value='';$('runMessage').textContent='Session credentials cleared.'}catch(e){showStatus(e.message,'error')}});$('approveButton').addEventListener('click',()=>sendApproval('approve').catch(e=>showStatus(e.message,'error')));$('rejectButton').addEventListener('click',()=>sendApproval('reject').catch(e=>showStatus(e.message,'error')));$('editButton').addEventListener('click',()=>sendApproval('edit').catch(e=>showStatus(e.message,'error')));
$('runButton').addEventListener('click',async()=>{try{const weights=selectedSignalWeights();if(Object.values(weights).every(value=>value<=0))throw new Error('Select at least one signal with a positive weight.');const config={high_beta:splitSymbols($('highBeta').value),diversified:splitSymbols($('diversified').value),signal_weights:weights,provider:$('provider').value};const response=await api('/api/runs',{method:'POST',body:JSON.stringify({student:$('student').value,mode:$('mode').value,approval_mode:$('approval').value,force_clock:$('forceClock').checked,config})});currentJob=response.id;currentConfig.latest_prices=response.latest_prices||{};$('approvalPanel').classList.remove('visible');showStatus('Cycle queued');poll()}catch(error){showStatus(error.message,'error')}});
(async()=>{try{const boot=await api('/api/bootstrap');csrf=boot.csrf_token;await loadConfig()}catch(error){showStatus(error.message,'error')}})();window.addEventListener('resize',()=>drawChart(window.lastCurve||[]));
</script></body></html>'''


@dataclass
class RunJob:
	job_id: str
	student: str
	mode: str
	approval_mode: str
	config: dict[str, Any] = field(repr=False)
	credentials: dict[str, str] = field(repr=False)
	force_clock: bool = False
	status: str = "queued"
	stage: str = "Queued"
	created_at: float = field(default_factory=time.time)
	updated_at: float = field(default_factory=time.time)
	proposed_orders: list[dict[str, Any]] = field(default_factory=list)
	latest_prices: dict[str, float] = field(default_factory=dict)
	execution: dict[str, Any] | None = None
	cycle_result: dict[str, Any] | None = field(default=None, repr=False)
	run_dir: str | None = None
	curve: list[dict[str, Any]] = field(default_factory=list)
	curve_reason: str | None = None
	provider: str | None = None
	logs: list[str] = field(default_factory=list)
	error: str | None = None
	reason: str | None = None
	previous_run: bool = False
	approval_decision: dict[str, Any] | None = field(default=None, repr=False)
	condition: threading.Condition = field(default_factory=threading.Condition, repr=False)


@dataclass
class BrowserSession:
	session_id: str
	csrf_token: str
	created_at: float = field(default_factory=time.time)
	last_seen: float = field(default_factory=time.time)
	credentials: dict[str, str] = field(default_factory=dict, repr=False)
	jobs: dict[str, RunJob] = field(default_factory=dict, repr=False)


_sessions: dict[str, BrowserSession] = {}
_sessions_lock = threading.RLock()
app = FastAPI(title="Trading Cycle Console", docs_url=None, redoc_url=None)


def _session(request: Request, response: Response) -> BrowserSession:
	now = time.time()
	with _sessions_lock:
		for key, value in list(_sessions.items()):
			if now - value.last_seen > SESSION_TTL_SECONDS:
				value.credentials.clear()
				del _sessions[key]
		token = request.cookies.get(SESSION_COOKIE, "")
		current = _sessions.get(token)
		if current is None:
			token = secrets.token_urlsafe(32)
			current = BrowserSession(token, secrets.token_urlsafe(32))
			_sessions[token] = current
			response.set_cookie(
				SESSION_COOKIE,
				token,
				httponly=True,
				secure=request.url.scheme == "https",
				samesite="strict",
				max_age=SESSION_TTL_SECONDS,
				path="/",
			)
		current.last_seen = now
		return current


def _require_csrf(request: Request, session: BrowserSession) -> None:
	if request.headers.get("x-csrf-token") != session.csrf_token:
		raise HTTPException(status_code=403, detail="Session token is missing or expired")


async def _json_body(request: Request) -> dict[str, Any]:
	try:
		body = await request.json()
	except Exception as error:
		raise HTTPException(status_code=400, detail="Request body must be valid JSON") from error
	if not isinstance(body, dict):
		raise HTTPException(status_code=400, detail="Request body must be an object")
	return body


def _symbols(value: Any, label: str) -> list[str]:
	if not isinstance(value, list) or not value:
		raise HTTPException(status_code=400, detail=f"{label} must contain at least one symbol")
	result: list[str] = []
	for raw in value:
		if not isinstance(raw, str):
			raise HTTPException(status_code=400, detail=f"{label} contains an invalid symbol")
		symbol = raw.strip().upper()
		if not SYMBOL_RE.fullmatch(symbol) or symbol in result:
			raise HTTPException(status_code=400, detail=f"{label} contains an invalid or duplicate symbol")
		result.append(symbol)
	return result


def _prepare_config(student: str, payload: Any) -> dict[str, Any]:
	if student not in {"A", "B", "C"}:
		raise HTTPException(status_code=400, detail="Unknown student")
	if not isinstance(payload, dict):
		raise HTTPException(status_code=400, detail="Config settings are required")
	config = _read_config(student)
	high_beta = _symbols(payload.get("high_beta"), "High-beta symbols")
	diversified = _symbols(payload.get("diversified"), "Diversified symbols")
	if set(high_beta) & set(diversified):
		raise HTTPException(status_code=400, detail="A symbol cannot be in both stock lists")
	weights = payload.get("signal_weights")
	if not isinstance(weights, dict) or set(weights) != set(SIGNALS):
		raise HTTPException(status_code=400, detail="All six signal weights are required")
	try:
		weights = {name: float(weights[name]) for name in SIGNALS}
	except (TypeError, ValueError) as error:
		raise HTTPException(status_code=400, detail="Signal weights must be numeric") from error
	if any(not math.isfinite(value) or value < 0 or value > 1 for value in weights.values()):
		raise HTTPException(status_code=400, detail="Signal weights must be between zero and one")
	if sum(weights.values()) <= 0:
		raise HTTPException(status_code=400, detail="At least one signal weight must be positive")
	provider = payload.get("provider")
	providers = config.get("llm", {}).get("providers", {})
	if provider not in providers or provider not in PROVIDERS:
		raise HTTPException(status_code=400, detail="Unknown configured LLM provider")
	provider_order = [provider, *[name for name in config["llm"].get("provider_order", []) if name != provider]]
	config["high_beta"] = high_beta
	config["diversified"] = diversified
	config["signal_weights"] = weights
	config["llm"] = {**config["llm"], "provider_order": provider_order}
	return config


def _portfolio_curve_result(
	rows: list[dict[str, Any]], weights: dict[str, float]
) -> tuple[list[dict[str, Any]], str | None]:
	if not rows:
		return [], "No OHLCV bars were returned for this cycle."
	prices: dict[str, dict[str, float]] = {}
	for row in rows:
		try:
			symbol = str(row["symbol"]).upper()
			close = float(row["close"])
			day = str(row["timestamp"])[:10]
		except (KeyError, TypeError, ValueError):
			continue
		if math.isfinite(close) and close > 0:
			prices.setdefault(symbol, {})[day] = close
	active_symbols = [symbol for symbol, weight in weights.items() if weight > 0]
	if not active_symbols:
		return [], "The proposed portfolio has no positive target weights."
	missing = [symbol for symbol in [*active_symbols, "SPY"] if symbol not in prices]
	if missing:
		return [], f"Missing closing-price history for: {', '.join(sorted(set(missing)))}."
	days = sorted(set.intersection(*(set(prices[symbol]) for symbol in [*active_symbols, "SPY"])))
	if len(days) < 2:
		return [], "Fewer than two common trading sessions are available for the proposed symbols and SPY."
	portfolio_value = 100.0
	spy_start = prices["SPY"][days[0]]
	points = [{"date": days[0], "portfolio": 100.0, "spy": 100.0}]
	for previous, current in zip(days, days[1:]):
		portfolio_return = sum(
			weights[symbol] * (prices[symbol][current] / prices[symbol][previous] - 1.0)
			for symbol in active_symbols
		)
		portfolio_value *= 1.0 + portfolio_return
		points.append({
			"date": current,
			"portfolio": round(portfolio_value, 4),
			"spy": round(100.0 * prices["SPY"][current] / spy_start, 4),
		})
	return points, None


def _portfolio_curve(rows: list[dict[str, Any]], weights: dict[str, float]) -> list[dict[str, Any]]:
	return _portfolio_curve_result(rows, weights)[0]


def _execution_from_log(run_dir: Path) -> dict[str, Any] | None:
	for log_path in sorted(run_dir.glob("Execution*.md")):
		try:
			content = log_path.read_text(encoding="utf-8")
			evidence = content.split("## Authoritative MCP evidence\n```json\n", 1)[1].split("\n```", 1)[0]
			parsed = json.loads(evidence)
			if isinstance(parsed, dict):
				return parsed
		except (OSError, IndexError, json.JSONDecodeError):
			continue
	return None


def _orders_from_execution(execution: Mapping[str, Any]) -> list[dict[str, Any]]:
	proposed = execution.get("proposed_orders")
	if isinstance(proposed, list) and proposed:
		return list(proposed)
	orders: list[dict[str, Any]] = []
	for side_key in ("sell_orders", "buy_orders"):
		result = execution.get(side_key, {})
		if isinstance(result, Mapping) and isinstance(result.get("orders"), list):
			orders.extend(result["orders"])
	return orders


def _run_log_names(run_dir: Path) -> list[str]:
	return sorted(path.name for path in run_dir.glob("*.md") if path.is_file())


def _latest_completed_snapshot(student: str) -> dict[str, Any] | None:
	student_dir = LOG_ROOT / student
	if not student_dir.is_dir():
		return None
	run_dirs = sorted(
		(path for path in student_dir.iterdir() if path.is_dir()),
		key=lambda path: path.stat().st_mtime,
		reverse=True,
	)
	for run_dir in run_dirs:
		summary_path = run_dir / "web_summary.json"
		try:
			if summary_path.is_file():
				summary = json.loads(summary_path.read_text(encoding="utf-8"))
				if isinstance(summary, dict) and summary.get("status") == "completed":
					summary["run_dir"] = str(run_dir)
					summary["logs"] = _run_log_names(run_dir)
					return summary
		except (OSError, json.JSONDecodeError):
			continue
		execution = _execution_from_log(run_dir)
		if execution is None:
			continue
		provider = ""
		provider_path = run_dir / "llm_provider.md"
		try:
			provider_text = provider_path.read_text(encoding="utf-8")
			provider = provider_text.partition(":")[2].strip().splitlines()[0]
		except (OSError, IndexError):
			pass
		return {
			"status": "completed",
			"execution": execution,
			"proposed_orders": _orders_from_execution(execution),
			"provider": provider,
			"run_dir": str(run_dir),
			"logs": _run_log_names(run_dir),
			"curve": [],
			"curve_reason": "This previous run did not save OHLCV rows for a curve.",
		}
	return None


def _store_completed_snapshot(job: RunJob, payload: dict[str, Any]) -> None:
	if not job.run_dir:
		return
	student_root = (LOG_ROOT / job.student).resolve()
	run_path = Path(job.run_dir).resolve()
	if not run_path.is_relative_to(student_root):
		job.run_dir = None
		return
	curve, curve_reason = _portfolio_curve_result(
		payload.get("ohlcv_rows", []), payload.get("target_weights", {})
	)
	job.curve = curve
	job.curve_reason = curve_reason
	job.execution = payload.get("execution") or job.execution
	if isinstance(job.execution, dict):
		job.proposed_orders = job.proposed_orders or list(job.execution.get("proposed_orders", []))
	job.run_dir = str(run_path)
	job.logs = _run_log_names(run_path)
	summary = {
		"status": "completed",
		"student": job.student,
		"mode": job.mode,
		"provider": job.provider,
		"execution": job.execution,
		"proposed_orders": job.proposed_orders,
		"target_weights": payload.get("target_weights", {}),
		"ohlcv_rows": payload.get("ohlcv_rows", []),
		"curve": job.curve,
		"curve_reason": job.curve_reason,
	}
	try:
		(Path(job.run_dir) / "web_summary.json").write_text(
			json.dumps(summary, indent=2, ensure_ascii=True, default=str) + "\n",
			encoding="utf-8",
		)
	except OSError:
		pass


def _newest_run_dir(student: str) -> Path | None:
	student_dir = LOG_ROOT / student
	if not student_dir.is_dir():
		return None
	run_dirs = [path for path in student_dir.iterdir() if path.is_dir()]
	return max(run_dirs, key=lambda path: path.stat().st_mtime) if run_dirs else None


def _job_json(job: RunJob) -> dict[str, Any]:
	with job.condition:
		execution = job.execution or {}
		account = execution.get("account", {}) if isinstance(execution, dict) else {}
		positions = account.get("positions", []) if isinstance(account, dict) else []
		return {
			"id": job.job_id,
			"student": job.student,
			"mode": job.mode,
			"approval_mode": job.approval_mode,
			"status": job.status,
			"stage": job.stage,
			"error": job.error,
			"reason": job.reason,
			"previous_run": job.previous_run,
			"proposed_orders": list(job.proposed_orders),
			"latest_prices": dict(getattr(job, "latest_prices", {})),
			"execution": execution,
			"positions": positions,
			"curve": list(job.curve),
			"curve_reason": job.curve_reason,
			"provider": job.provider,
			"logs": list(job.logs),
			"updated_at": job.updated_at,
		}


def _event_callback(job: RunJob, event: str, payload: dict[str, Any]) -> None:
	with job.condition:
		job.updated_at = time.time()
		if event == "stage":
			job.status = "running"
			job.stage = str(payload.get("name", "Cycle running"))
		elif event == "orders_proposed":
			job.proposed_orders = list(payload.get("orders", []))
			job.latest_prices = dict(payload.get("latest_prices", {}))
			job.stage = "Orders awaiting approval" if job.approval_mode == "manual" else "Orders prepared"
		elif event == "execution_complete":
			job.execution = dict(payload)
			job.mode = str(payload.get("mode", job.mode))
		elif event == "cycle_complete":
			job.cycle_result = dict(payload)
			job.status = "completed"
			job.reason = None
			job.execution = payload.get("execution") or job.execution
			job.run_dir = str(payload.get("run_dir", "")) or None
			job.provider = str(payload.get("provider", "")) or None
			job.mode = str(payload.get("mode", job.mode))
			job.proposed_orders = job.proposed_orders or list(
				_orders_from_execution(payload.get("execution") or {})
			)
			_store_completed_snapshot(job, payload)
			job.stage = "Cycle complete"
		elif event == "cycle_skipped":
			job.status = "skipped"
			job.reason = str(payload.get("reason", "Cycle skipped."))
			job.stage = "Previous completed run shown"
			account_snapshot = payload.get("account_snapshot")
			previous = _latest_completed_snapshot(job.student)
			if previous is not None:
				job.previous_run = True
				job.execution = dict(previous.get("execution") or {})
				if isinstance(account_snapshot, dict):
					job.execution["account"] = account_snapshot
				job.proposed_orders = list(previous.get("proposed_orders", []))
				job.provider = previous.get("provider")
				job.run_dir = previous.get("run_dir")
				job.logs = list(previous.get("logs", []))
				job.curve = list(previous.get("curve", []))
				job.curve_reason = previous.get("curve_reason") or "Previous run did not include a curve."
				job.mode = str(previous.get("mode", job.mode))
			else:
				if isinstance(account_snapshot, dict):
					job.execution = {"account": account_snapshot}
				job.reason += " No previous completed run is available yet."
		elif event == "cycle_failed":
			job.status = "failed"
			job.reason = str(payload.get("reason", "Cycle failed."))
			job.stage = "Cycle failed"
		job.condition.notify_all()


def _approval_handler(job: RunJob):
	def wait_for_decision(proposal: dict[str, Any]) -> dict[str, Any]:
		with job.condition:
			job.proposed_orders = list(proposal.get("orders", []))
			job.status = "awaiting_approval"
			job.stage = "Review proposed orders"
			job.updated_at = time.time()
			job.condition.notify_all()
			deadline = time.monotonic() + APPROVAL_TIMEOUT_SECONDS
			while job.approval_decision is None:
				remaining = deadline - time.monotonic()
				if remaining <= 0:
					job.approval_decision = {"action": "reject"}
					job.error = "Approval timed out; no orders were submitted."
					break
				job.condition.wait(remaining)
			decision = job.approval_decision
			job.approval_decision = None
			job.status = "running"
			job.stage = "Executing approved orders"
			return dict(decision)

	return wait_for_decision


def _run_job(job: RunJob) -> None:
	def event(event_name: str, payload: Any) -> None:
		if isinstance(payload, dict):
			_event_callback(job, event_name, payload)

	approval = _approval_handler(job) if job.approval_mode == "manual" else None
	try:
		code = run_cycle(
			job.student,
			job.mode,
			force_clock=job.force_clock,
			config_override=job.config,
			credential_override=job.credentials,
			approval_handler=approval,
			event_callback=event,
		)
		with job.condition:
			if job.status not in {"skipped", "failed"}:
				job.status = "completed" if code == 0 else "failed"
			if code != 0 and job.error is None:
				job.reason = job.reason or "Cycle failed. Review run logs for details."
			job.updated_at = time.time()
			job.condition.notify_all()
	except Exception as error:
		with job.condition:
			job.status = "failed"
			message = str(error)
			secret_values = set(job.credentials.values())
			for name in (
				f"ALPACA_KEY_{student}" for student in ("A", "B", "C")
			):
				if os.environ.get(name):
					secret_values.add(os.environ[name])
			for name in ("ALPACA_SECRET_A", "ALPACA_SECRET_B", "ALPACA_SECRET_C", "GROQ_API_KEY", "OLLAMA_API_KEY", "FINNHUB_API_KEY"):
				if os.environ.get(name):
					secret_values.add(os.environ[name])
			for secret in secret_values:
				if secret:
					message = message.replace(secret, "[REDACTED]")
			job.reason = f"{type(error).__name__}: {message or 'No additional details supplied.'}"
			job.updated_at = time.time()
			job.condition.notify_all()
	finally:
		job.credentials.clear()


def _get_job(session: BrowserSession, job_id: str) -> RunJob:
	job = session.jobs.get(job_id)
	if job is None:
		raise HTTPException(status_code=404, detail="Run not found")
	return job


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
	return HTMLResponse(INDEX_HTML)


@app.get("/api/bootstrap")
def bootstrap(request: Request, response: Response) -> dict[str, Any]:
	session = _session(request, response)
	return {"csrf_token": session.csrf_token, "students": ["A", "B", "C"]}


@app.get("/api/config/{student}")
def config_for_student(student: str, request: Request, response: Response) -> dict[str, Any]:
	_session(request, response)
	try:
		config = _read_config(student)
	except (ValueError, OSError) as error:
		raise HTTPException(status_code=404, detail="Student config not found") from error
	providers = config.get("llm", {}).get("providers", {})
	return {
		"student": student.upper(),
		"high_beta": config["high_beta"],
		"diversified": config["diversified"],
		"signal_weights": config["signal_weights"],
		"provider_order": config["llm"].get("provider_order", []),
		"providers": [
			{"name": name, "model": providers[name].get("model", "")}
			for name in config["llm"].get("provider_order", [])
			if name in providers
		],
	}


@app.post("/api/session/credentials")
async def set_credentials(request: Request, response: Response) -> dict[str, bool]:
	session = _session(request, response)
	_require_csrf(request, session)
	body = await _json_body(request)
	student = str(body.get("student", "")).upper()
	if student not in {"A", "B", "C"}:
		raise HTTPException(status_code=400, detail="Unknown student")
	key, secret = body.get("key"), body.get("secret")
	if not isinstance(key, str) or not isinstance(secret, str) or not key or not secret:
		raise HTTPException(status_code=400, detail="Both credentials are required")
	if len(key) > 512 or len(secret) > 512:
		raise HTTPException(status_code=400, detail="Credential value is invalid")
	session.credentials = {f"ALPACA_KEY_{student}": key, f"ALPACA_SECRET_{student}": secret}
	return {"stored": True}


@app.delete("/api/session/credentials")
def clear_credentials(request: Request, response: Response) -> dict[str, bool]:
	session = _session(request, response)
	_require_csrf(request, session)
	session.credentials.clear()
	return {"cleared": True}


@app.post("/api/runs")
async def start_run(request: Request, response: Response) -> dict[str, Any]:
	session = _session(request, response)
	_require_csrf(request, session)
	body = await _json_body(request)
	student = str(body.get("student", "")).upper()
	mode = body.get("mode")
	approval_mode = body.get("approval_mode", "auto")
	if mode not in {"dry", "live"} or approval_mode not in {"auto", "manual"}:
		raise HTTPException(status_code=400, detail="Invalid cycle mode or approval mode")
	config = _prepare_config(student, body.get("config"))
	force_clock = body.get("force_clock", False)
	if not isinstance(force_clock, bool) or (force_clock and mode != "dry"):
		raise HTTPException(status_code=400, detail="Clock bypass is available only in dry mode")
	if mode == "live" and __import__("os").environ.get("TRADING_ENABLED", "").lower() == "true":
		if not session.credentials.get(f"ALPACA_KEY_{student}") or not session.credentials.get(f"ALPACA_SECRET_{student}"):
			raise HTTPException(status_code=400, detail="Enter and save this student's Alpaca credentials for the session")
	credentials = {
		name: value
		for name, value in session.credentials.items()
		if name in {f"ALPACA_KEY_{student}", f"ALPACA_SECRET_{student}"}
	}
	job_id = secrets.token_urlsafe(18)
	job = RunJob(
		job_id=job_id,
		student=student,
		mode=mode,
		approval_mode=approval_mode,
		config=config,
		credentials=credentials,
		force_clock=force_clock,
	)
	session.jobs[job_id] = job
	threading.Thread(target=_run_job, args=(job,), name=f"cycle-{student}-{job_id[:6]}", daemon=True).start()
	return {"id": job_id, "status": job.status}


@app.get("/api/runs/{job_id}")
def get_run(job_id: str, request: Request, response: Response) -> dict[str, Any]:
	session = _session(request, response)
	job = _get_job(session, job_id)
	return _job_json(job)


@app.post("/api/runs/{job_id}/approval")
async def approve_run(job_id: str, request: Request, response: Response) -> dict[str, str]:
	session = _session(request, response)
	_require_csrf(request, session)
	job = _get_job(session, job_id)
	body = await _json_body(request)
	action = body.get("action")
	if action not in {"approve", "reject", "edit"}:
		raise HTTPException(status_code=400, detail="Approval must be approve, reject, or edit")
	decision: dict[str, Any] = {"action": action}
	if action == "edit":
		orders = body.get("orders")
		if not isinstance(orders, list):
			raise HTTPException(status_code=400, detail="Edited orders must be a list")
		decision["orders"] = orders
	with job.condition:
		if job.status != "awaiting_approval" or job.approval_decision is not None:
			raise HTTPException(status_code=409, detail="This run is not waiting for approval")
		job.approval_decision = decision
		job.updated_at = time.time()
		job.condition.notify_all()
	return {"status": "received"}


@app.get("/api/runs/{job_id}/logs/{filename}")
def get_log(job_id: str, filename: str, request: Request, response: Response) -> FileResponse:
	session = _session(request, response)
	job = _get_job(session, job_id)
	if not job.run_dir or Path(filename).name != filename or not filename.endswith(".md"):
		raise HTTPException(status_code=404, detail="Log not found")
	base = LOG_ROOT.resolve()
	path = (Path(job.run_dir) / filename).resolve()
	if not path.is_relative_to(base) or path.parent != Path(job.run_dir).resolve() or not path.is_file():
		raise HTTPException(status_code=404, detail="Log not found")
	return FileResponse(path, media_type="text/markdown", filename=filename)


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok"}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=False)
