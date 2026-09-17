#!/usr/bin/env python3
"""Multi-connection resume download of MS-SL tvr.zip from Google Drive."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request

URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1RVG3qIy-nf9XdalMw8wpmWNSPcnOAaNK&export=download&confirm=t"
)
EXPECT = 17787514489
CHUNK = 8 * 1024 * 1024
NWORK = 16
PROXY = (
    os.environ.get("HTTPS_PROXY")
    or os.environ.get("https_proxy")
    or "http://127.0.0.1:17897"
)


def opener():
    if PROXY:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
        )
    return urllib.request.build_opener()


def main() -> int:
    default_out = "/data/zhaopu/wang-2024-gmmformer-v2/data/prvr/tvr.zip"
    if os.name == "nt":
        default_out = r"F:\论文\data\tvr\tvr.zip"
    out = sys.argv[1] if len(sys.argv) > 1 else default_out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    state_path = out + ".parts.json"
    lock = threading.Lock()
    n_chunks = (EXPECT + CHUNK - 1) // CHUNK
    done = set()
    if os.path.isfile(state_path):
        try:
            done = set(json.load(open(state_path, encoding="utf-8")).get("done", []))
        except Exception:
            done = set()
    if os.path.isfile(out) and os.path.getsize(out) == EXPECT and len(done) >= n_chunks:
        print("ALREADY_DONE", out, flush=True)
        return 0
    mode = "r+b" if os.path.isfile(out) else "w+b"
    f = open(out, mode)
    if os.path.getsize(out) != EXPECT:
        f.seek(EXPECT - 1)
        f.write(b"\0")
        f.flush()
    todo = [i for i in range(n_chunks) if i not in done]
    print(
        "START n_chunks=%s done=%s todo=%s proxy=%s out=%s"
        % (n_chunks, len(done), len(todo), PROXY, out),
        flush=True,
    )
    got = 0
    t0 = time.time()
    err = []
    stop = threading.Event()

    def save_state():
        tmp = state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as sf:
            json.dump({"done": sorted(done), "expect": EXPECT, "chunk": CHUNK}, sf)
        os.replace(tmp, state_path)

    def worker():
        nonlocal got
        op = opener()
        while not stop.is_set():
            with lock:
                if not todo:
                    return
                i = todo.pop(0)
            start = i * CHUNK
            end = min(EXPECT, start + CHUNK) - 1
            want = end - start + 1
            ok = False
            last_e = None
            for attempt in range(8):
                try:
                    req = urllib.request.Request(
                        URL,
                        headers={
                            "User-Agent": "Mozilla/5.0",
                            "Range": "bytes=%s-%s" % (start, end),
                        },
                    )
                    with op.open(req, timeout=90) as r:
                        ct = r.headers.get("Content-Type", "")
                        if "text/html" in ct:
                            raise IOError("html_not_zip")
                        buf = bytearray()
                        while len(buf) < want:
                            b = r.read(min(256 * 1024, want - len(buf)))
                            if not b:
                                break
                            buf.extend(b)
                    if len(buf) != want:
                        raise IOError("short %s != %s" % (len(buf), want))
                    if i == 0 and buf[:2] != b"PK":
                        raise IOError("not_zip_magic %s" % buf[:8])
                    with lock:
                        f.seek(start)
                        f.write(buf)
                        done.add(i)
                        got += want
                        if len(done) % 10 == 0 or len(done) == n_chunks:
                            save_state()
                            dt = max(time.time() - t0, 1e-6)
                            left = EXPECT - (len(done) * CHUNK)
                            if left < 0:
                                left = 0
                            bps = got / dt
                            eta = left / bps if bps > 0 else 0
                            print(
                                "PROGRESS %s/%s bytes=%s MBps=%.2f eta_min=%.1f"
                                % (
                                    len(done),
                                    n_chunks,
                                    min(EXPECT, len(done) * CHUNK),
                                    bps / 1024 / 1024,
                                    eta / 60,
                                ),
                                flush=True,
                            )
                    ok = True
                    break
                except Exception as e:
                    last_e = e
                    time.sleep(min(30, 1.5 ** attempt))
            if not ok:
                with lock:
                    todo.append(i)
                    err.append((i, repr(last_e)))
                    print("RETRY_FAIL chunk=%s nerr=%s %s" % (i, len(err), last_e), flush=True)

    th = [threading.Thread(target=worker, daemon=True) for _ in range(NWORK)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    f.flush()
    f.close()
    save_state()
    sz = os.path.getsize(out)
    print("END size=%s expect=%s done=%s/%s" % (sz, EXPECT, len(done), n_chunks), flush=True)
    if sz != EXPECT or len(done) < n_chunks:
        print("INCOMPLETE", flush=True)
        return 1
    print("DOWNLOAD_OK", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
