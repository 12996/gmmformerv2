import json
import subprocess
from pathlib import Path

node = r"C:\Users\zp\AppData\Roaming\npm\node_modules\@jackwener\opencli\dist\src\main.js"
js = r"""(function(){
  const asst = [...document.querySelectorAll('[data-message-author-role="assistant"]')];
  const last = asst.length ? asst[asst.length-1] : null;
  const stop = !!document.querySelector('[data-testid="stop-button"]');
  return JSON.stringify({
    stop: stop,
    nAsst: asst.length,
    url: location.href,
    n: last ? last.innerText.length : 0,
    last: last ? last.innerText : ''
  });
})()"""
r = subprocess.run(["node", node, "browser", "eval", js], capture_output=True, text=True, encoding="utf-8")
raw = (r.stdout or "") + (r.stderr or "")
start, end = raw.find("{"), raw.rfind("}")
data = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {"raw": raw[-800:]}
out = Path(r"F:\论文\papers\wang-2024-gmmformer-v2\experiments")
out.joinpath("gpt6_e2_ans.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
out.joinpath("gpt6_e2_ans.txt").write_text(data.get("last") or "", encoding="utf-8")
print("stop", data.get("stop"), "nAsst", data.get("nAsst"), "n", data.get("n"), "url", data.get("url"))
print((data.get("last") or "")[:9000])
