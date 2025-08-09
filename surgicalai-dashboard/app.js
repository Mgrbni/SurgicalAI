/* SurgicalAI Dashboard – Wire hooks here.
   All “TODO” blocks mark integration points for your backend. */

const $ = (sel, ctx=document) => ctx.querySelector(sel);
const $$ = (sel, ctx=document) => Array.from(ctx.querySelectorAll(sel));

/* ---------- Navigation ---------- */
$$(".nav-item").forEach(btn=>{
  btn.addEventListener("click", ()=>{
    $$(".nav-item").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    const id = btn.getAttribute("data-section");
    $$(".section").forEach(s=>s.classList.remove("visible"));
    $("#"+id).classList.add("visible");
  });
});

/* ---------- Year & Signature ---------- */
$("#year").textContent = new Date().getFullYear();
const sigEl = $("#sigline");

/* ---------- About modal ---------- */
const about = $("#about");
$("#btn-about").addEventListener("click", ()=> about.showModal());
$("#btn-close-about").addEventListener("click", ()=> about.close());

/* ---------- Dummy data ---------- */
const recentCases = [
  {id:"CASE-2025-0809-1", date:"2025-08-09", dx:"Susp. Melanoma", p:51, flap:"Rotation (lat. cheek)", status:"Draft"},
  {id:"CASE-2025-0808-7", date:"2025-08-08", dx:"BCC", p:73, flap:"Bilobed (ala nasi)", status:"Signed"},
  {id:"CASE-2025-0807-3", date:"2025-08-07", dx:"SCC in situ", p:38, flap:"V‑Y advancement", status:"Review"}
];

function renderCases(){
  const ul = $("#recent-cases");
  ul.innerHTML = "";
  recentCases.forEach(c=>{
    const li = document.createElement("li");
    li.innerHTML = `<strong>${c.id}</strong><br><span class="muted tiny">${c.date} • ${c.dx} • P=${c.p}% • ${c.flap} • ${c.status}</span>`;
    ul.appendChild(li);
  });
}
renderCases();

/* ---------- KPIs (placeholder) ---------- */
$("#kpi-gpu").textContent = "Idle";
$("#kpi-queue").textContent = "0";

/* ---------- Upload (drag & drop) ---------- */
const dropzone = $("#dropzone");
const fileInput = $("#file-input");
$("#btn-browse").addEventListener("click", ()=>fileInput.click());
dropzone.addEventListener("click", ()=>fileInput.click());
["dragenter","dragover"].forEach(ev=>dropzone.addEventListener(ev, e=>{
  e.preventDefault(); dropzone.style.borderColor = "rgba(167,139,250,.55)";
}));
["dragleave","drop"].forEach(ev=>dropzone.addEventListener(ev, e=>{
  e.preventDefault(); dropzone.style.borderColor = "rgba(110,231,255,.35)";
}));
dropzone.addEventListener("drop", e=>{
  const file = e.dataTransfer.files?.[0];
  if(file) handleFile(file);
});
fileInput.addEventListener("change", e=>{
  const file = e.target.files?.[0];
  if(file) handleFile(file);
});

function handleFile(file){
  /* TODO: send file to backend for:
     1) Mesh parse + landmarking
     2) Lesion classification
     3) Grad‑CAM heatmap
     4) Flap planning */
  console.log("Selected:", file.name);
  fake3DPreview();
  fakeAnalysis();
}

/* ---------- 3D Viewport (placeholder) ---------- */
const viewport = $("#viewport");
function fake3DPreview(){
  const ctx = viewport.getContext("2d");
  ctx.clearRect(0,0,viewport.width=viewport.clientWidth, viewport.height=viewport.clientHeight);
  // Placeholder render
  ctx.fillStyle = "rgba(110,231,255,.15)";
  ctx.beginPath(); ctx.arc(viewport.width/2, viewport.height/2, 120, 0, Math.PI*2); ctx.fill();
  ctx.strokeStyle = "rgba(167,139,250,.6)"; ctx.lineWidth = 2;
  ctx.strokeRect(viewport.width/2-90, viewport.height/2-120, 180, 240);
}

/* ---------- Probabilities ---------- */
const classes = ["Melanoma","BCC","SCC","Benign nevus","Seborrheic keratosis"];
function renderProbs(probs){
  const root = $("#probs"); root.innerHTML = "";
  classes.forEach((c,i)=>{
    const p = probs[i] || 0;
    const row = document.createElement("div"); row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-label">${c}</div>
      <div class="prob-bar"><span style="width:${p}%"></span></div>
      <div class="prob-val" style="width:48px; text-align:right">${p}%</div>
    `;
    root.appendChild(row);
  });
}

/* ---------- Heatmap ---------- */
const heatCanvas = $("#heatmap-canvas");
function renderHeatmap(on=true){
  const ctx = heatCanvas.getContext("2d");
  ctx.clearRect(0,0,heatCanvas.width=heatCanvas.clientWidth, heatCanvas.height=240);
  if(!on) return;
  for(let i=0;i<70;i++){
    const x = Math.random()*heatCanvas.width;
    const y = Math.random()*heatCanvas.height;
    const r = 12+Math.random()*26;
    const grd = ctx.createRadialGradient(x,y,0,x,y,r);
    grd.addColorStop(0, "rgba(255,80,80,.35)");
    grd.addColorStop(1, "rgba(255,80,80,0)");
    ctx.fillStyle = grd; ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill();
  }
}
$("#toggle-heat").addEventListener("change", e=>renderHeatmap(e.target.checked));

/* ---------- Flap plan (placeholder SVG) ---------- */
function renderPlan(angle=35, arc=90, success=89){
  const w = $("#plan-view").clientWidth || 600;
  const h = 240;
  const cx = w/2, cy = h/2;
  const r = 80;

  const a1 = (angle - arc/2) * Math.PI/180;
  const a2 = (angle + arc/2) * Math.PI/180;

  const path = `
    M ${cx} ${cy}
    L ${cx + r*Math.cos(a1)} ${cy + r*Math.sin(a1)}
    A ${r} ${r} 0 0 1 ${cx + r*Math.cos(a2)} ${cy + r*Math.sin(a2)}
    Z
  `;
  const svg = `
    <svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
      <defs>
        <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="rgba(110,231,255,.5)"/>
          <stop offset="100%" stop-color="rgba(167,139,250,.5)"/>
        </linearGradient>
      </defs>
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="rgba(255,255,255,.15)" stroke-width="1.5"/>
      <path d="${path}" fill="url(#g1)" stroke="rgba(255,255,255,.25)" stroke-width="1.5"/>
      <text x="${cx}" y="${cy+r+20}" fill="#9ba7b6" font-size="12" text-anchor="middle">
        Rotation flap • Langer’s angle ${angle}° • Arc ${arc}° • Pred. Success ${success}%
      </text>
    </svg>
  `;
  $("#plan-view").innerHTML = svg;
}

$("#btn-plan").addEventListener("click", ()=>{
  const angle = +$("#langer-angle").value;
  const arc = +$("#arc").value;
  const success = +$("#success").value;
  renderPlan(angle, arc, success);
});

/* ---------- Biomechanics checks (dummy) ---------- */
function renderChecks(){
  const out = [
    {label:"Stretch force within elasticity", state:"ok"},
    {label:"Vascular territory intact", state:"ok"},
    {label:"Edge necrosis risk", state:"warn"},
    {label:"Tension distribution", state:"ok"},
  ];
  const ul = $("#checks"); ul.innerHTML = "";
  out.forEach(o=>{
    const li = document.createElement("li");
    const dot = o.state==="ok"?"🟢":o.state==="warn"?"🟡":"🔴";
    li.innerHTML = `<span>${dot}</span><span>${o.label}</span><span class="${o.state}" style="margin-left:auto">${o.state.toUpperCase()}</span>`;
    ul.appendChild(li);
  });
}

/* ---------- Reports table (dummy) ---------- */
function renderReports(){
  const rows = recentCases.map(c=>`
    <tr>
      <td>${c.id}</td>
      <td>${c.date}</td>
      <td>${c.dx}</td>
      <td>${c.p}</td>
      <td>${c.flap}</td>
      <td>${c.status}</td>
      <td><button class="btn btn-ghost" data-open="${c.id}">Open</button></td>
    </tr>
  `).join("");
  $("#reports-table tbody").innerHTML = rows;
}

/* ---------- Exports (client-side demo) ---------- */
$("#btn-export-json").addEventListener("click", ()=>{
  const blob = new Blob([JSON.stringify({cases:recentCases, signature:sigEl.textContent}, null, 2)], {type:"application/json"});
  triggerDownload(blob, "surgicalai_report.json");
});
$("#btn-export-pdf").addEventListener("click", ()=>{
  // Simple HTML -> PDF hint: in real app, call backend. Here we export HTML.
  const html = `
    <h1>SurgicalAI Report</h1>
    <p>Signed: ${sigEl.textContent}</p>
    <pre>${JSON.stringify(recentCases,null,2)}</pre>
  `;
  const blob = new Blob([html], {type:"text/html"});
  triggerDownload(blob, "surgicalai_report.html");
});
function triggerDownload(blob, name){
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
}

/* ---------- Settings actions ---------- */
$("#btn-save-settings").addEventListener("click", ()=>{
  const cfg = {
    key: $("#openai-key").value ? "••••••••" : "",
    model: $("#model-select").value,
    backbone: $("#backbone").value,
    local: $("#toggle-local").checked,
    anon: $("#toggle-anon").checked,
    audit: $("#toggle-audit").checked,
  };
  console.log("Saved settings:", cfg);
  toast("Settings saved.");
});
$("#btn-export-audit").addEventListener("click", ()=>{
  // TODO: Replace with real audit log from backend
  const log = [{t:new Date().toISOString(), event:"login"}, {t:new Date().toISOString(), event:"upload_scan"}];
  const blob = new Blob([JSON.stringify(log, null, 2)], {type:"application/json"});
  triggerDownload(blob, "audit_log.json");
});
$("#btn-apply-brand").addEventListener("click", ()=>{
  sigEl.textContent = $("#signature").value || sigEl.textContent;
  const mode = $("#wm-toggle").value;
  document.body.dataset.wm = (mode==="on") ? "on" : "off";
  toast("Branding applied.");
});

/* ---------- Build integration helper ---------- */
$("#btn-copy-version").addEventListener("click", async ()=>{
  const text = document.querySelector("pre.code").innerText.trim();
  await navigator.clipboard.writeText(text);
  toast("Version-file snippet copied.");
});

/* ---------- Contact form (demo) ---------- */
$("#btn-send").addEventListener("click", ()=>{
  const name = $("#c-name").value.trim();
  const email = $("#c-email").value.trim();
  const msg = $("#c-msg").value.trim();
  if(!name || !email || !msg){ toast("Fill all fields.", true); return; }
  console.log("Contact message:", {name,email,msg});
  toast("Message queued. We’ll reply soon.");
  $("#c-name").value = $("#c-email").value = $("#c-msg").value = "";
});

/* ---------- New case (demo) ---------- */
$("#btn-new-case").addEventListener("click", ()=>{
  toast("New case created (demo).");
});

/* ---------- Keyboard shortcuts ---------- */
document.addEventListener("keydown", (e)=>{
  if(e.key==="/"){ e.preventDefault(); $("#search").focus(); }
  if(e.key==="u"||e.key==="U"){ fileInput.click(); }
  if(e.key==="g"||e.key==="G"){ const t=$("#toggle-heat"); t.checked=!t.checked; t.dispatchEvent(new Event("change")); }
  if(e.key==="p"||e.key==="P"){ $("#btn-plan").click(); }
});

/* ---------- Toast ---------- */
let toastTimeout;
function toast(msg, error=false){
  let t = $("#toast");
  if(!t){
    t = document.createElement("div");
    t.id = "toast";
    t.style.position = "fixed";
    t.style.bottom = "20px";
    t.style.right = "20px";
    t.style.padding = "10px 14px";
    t.style.border = "1px solid var(--line)";
    t.style.borderRadius = "12px";
    t.style.background = error ? "#3a1212" : "rgba(18,22,29,.95)";
    t.style.color = "var(--text)";
    t.style.boxShadow = "var(--shadow)";
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.style.opacity = "1";
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(()=> t.style.opacity = "0", 2200);
}

/* ---------- Fake analysis sequence ---------- */
function fakeAnalysis(){
  // pipeline badges
  $$(".pipe-step").forEach((s,i)=> setTimeout(()=> s.classList.add("done"), i*300));
  // dummy probs
  const probs = [51, 23, 12, 9, 5];
  renderProbs(probs);
  renderHeatmap(true);
  renderPlan();
  renderChecks();
  renderReports();
}
renderHeatmap(true); renderPlan(); renderChecks(); renderReports();

/* ---------- Resize handlers ---------- */
window.addEventListener("resize", ()=>{ fake3DPreview(); renderPlan(); renderHeatmap($("#toggle-heat").checked); });

/* ---------- Buttons ---------- */
$("#btn-center").addEventListener("click", fake3DPreview);
$("#btn-reset").addEventListener("click", ()=>{ fake3DPreview(); renderHeatmap(true); });

/* ---------- Refresh cases (demo) ---------- */
$("#btn-refresh-cases").addEventListener("click", ()=>{
  toast("Cases refreshed.");
});

