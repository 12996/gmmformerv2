import json
import subprocess
import time
from pathlib import Path

node = r"C:\Users\zp\AppData\Roaming\npm\node_modules\@jackwener\opencli\dist\src\main.js"
js = r"""(function(){
  const t = document.body.innerText;
  const i = t.indexOf('我会只依据现有证据');
  return JSON.stringify({
    stop: !!document.querySelector('[data-testid="stop-button"]'),
    think: t.indexOf('Pro 思考中') >= 0,
    n: t.length,
    fromAns: i>=0 ? t.slice(i) : ''
  });
})()"""
outp = Path(r"F:\论文\papers\wang-2024-gmmformer-v2\experiments\gpt6_raw.json")
for i in range(30):
    r = subprocess.run(["node", node, "browser", "eval", js], capture_output=True, text=True, encoding="utf-8")
    raw = r.stdout or ""
    # last JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    data = {}
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            data = {"raw": raw[-300:]}
    print(i, "stop", data.get("stop"), "think", data.get("think"), "n", data.get("n"))
    outp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if data.get("stop") is False and data.get("think") is False and data.get("n", 0) > 2500:
        break
    time.sleep(8)
print("DONE")
