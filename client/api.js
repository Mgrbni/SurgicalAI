const $ = (s)=>document.querySelector(s);
const out = $("#out");

$("#btnInfer").onclick = async ()=>{
  out.textContent = "Running…";
  const json_schema = $("#jsonMode").checked;
  const r = await fetch("/api/infer", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({
      prompt: $("#prompt").value,
      json_schema
    })
  });
  const data = await r.json();
  // Uniform output for demo
  if (json_schema) {
    out.textContent = JSON.stringify(data, null, 2);
  } else {
    out.textContent = data.text || JSON.stringify(data, null, 2);
  }
};

$("#btnStream").onclick = async ()=>{
  out.textContent = "";
  const r = await fetch("/api/stream", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ prompt: $("#prompt").value })
  });
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  while(true){
    const {done, value} = await reader.read();
    if (done) break;
    out.textContent += dec.decode(value);
  }
};
