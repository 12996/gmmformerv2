import json
import subprocess
from pathlib import Path

node = r"C:\Users\zp\AppData\Roaming\npm\node_modules\@jackwener\opencli\dist\src\main.js"
js = r"""(function(){
  const turn = document.querySelector('[data-testid="conversation-turn-2"]');
  const ps = turn ? [...turn.querySelectorAll('p, li, h1, h2, h3, td, th')].map(e => e.innerText.trim()).filter(Boolean) : [];
  return JSON.stringify({
    stop: !!document.querySelector('[data-testid="stop-button"]'),
    turnN: turn ? turn.innerText.length : 0,
    turn: turn ? turn.innerText : '',
    paras: ps
  });
})()"""
r = subprocess.run(["node", node, "browser", "eval", js], capture_output=True, text=True, encoding="utf-8")
raw = r.stdout or ""
start = raw.find("{")
end = raw.rfind("}")
data = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {"raw": raw}
out = Path(r"F:\论文\papers\wang-2024-gmmformer-v2\experiments\gpt6_answer.json")
out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
text = data.get("turn") or "\n".join(data.get("paras") or [])
Path(r"F:\论文\papers\wang-2024-gmmformer-v2\experiments\gpt6_answer.txt").write_text(text, encoding="utf-8")
print("stop", data.get("stop"), "turnN", data.get("turnN"), "paras", len(data.get("paras") or []))
print(text[:4000])
print("---TAIL---")
print(text[-1500:])
