#!/usr/bin/env python3
"""Resume Google Drive tvr.zip. Never abort because SSL flickered.

Keeps .parts.json. Failed chunks go back on the queue. Workers are
respawned until every chunk is on disk. Health line every 30s.
"""
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
NWORK = int(os.environ.get("TVR_NWORK", "4"))
if os.name == "nt":
    PROXY = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or "http://127.0.0.1:7897"
    )
else:
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


def load_done(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    try:
        return set(json.load(open(path, encoding="utf-8")).get("done", []))
    except Exception:
        return set()


def main() -> int:
    default_out = "/data/zhaopu/wang-2024-gmmformer-v2/data/prvr/tvr.zip"
    if os.name == "nt":
        default_out = r"F:\论文\data\tvr\tvr.zip"
    out = sys.argv[1] if len(sys.argv) > 1 else default_out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    state_path = out + ".parts.json"
    n_chunks = (EXPECT + CHUNK - 1) // CHUNK
    done = load_done(state_path)
    if os.path.isfile(out) and os.path.getsize(out) == EXPECT and len(done) >= n_chunks:
        print("ALREADY_DONE", out, flush=True)
        return 0
    mode = "r+b" if os.path.isfile(out) else "w+b"
    f = open(out, mode)
    if os.path.getsize(out) != EXPECT:
        f.seek(EXPECT - 1)
        f.write(b"\0")
        f.flush()

    lock = threading.Lock()
    todo = [i for i in range(n_chunks) if i not in done]
    inflight = set()
    got_session = 0
    t0 = time.time()
    last_ok = time.time()
    ssl_fail = 0
    ok_n = 0
    stop = threading.Event()

    def save_state() -> None:
        tmp = state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as sf:
            json.dump({"done": sorted(done), "expect": EXPECT, "chunk": CHUNK}, sf)
        os.replace(tmp, state_path)

    def fetch_chunk(i: int) -> None:
        nonlocal got_session, last_ok, ssl_fail, ok_n
        start = i * CHUNK
        end = min(EXPECT, start + CHUNK) - 1
        want = end - start + 1
        delay = 1.0
        op = opener()
        while not stop.is_set():
            try:
                req = urllib.request.Request(
                    URL,
                    headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=%s-%s" % (start, end)},
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
                    inflight.discard(i)
                    got_session += want
                    ok_n += 1
                    last_ok = time.time()
                    save_state()
                    dt = max(time.time() - t0, 1e-6)
                    left = EXPECT - len(done) * CHUNK
                    if left < 0:
                        left = 0
                    bps = got_session / dt
                    eta = left / bps if bps > 0 else 0
                    print(
                        "PROGRESS %s/%s MBps=%.2f eta_min=%.1f ssl_fail=%s"
                        % (len(done), n_chunks, bps / 1024 / 1024, eta / 60, ssl_fail),
                        flush=True,
                    )
                return
            except Exception as e:
                msg = str(e)
                with lock:
                    if "SSL" in msg or "EOF" in msg:
                        ssl_fail += 1
                    print("RETRY chunk=%s wait=%.0fs %s" % (i, delay, e), flush=True)
                time.sleep(delay)
                delay = min(60.0, delay * 1.7)

    def worker() -> None:
        while not stop.is_set():
            with lock:
                if len(done) >= n_chunks:
                    return
                if not todo:
                    time.sleep(0.5)
                    continue
                i = todo.pop(0)
                inflight.add(i)
            fetch_chunk(i)

    def health() -> None:
        while not stop.is_set():
            time.sleep(30)
            with lock:
                stall = time.time() - last_ok
                print(
                    "HEALTH done=%s/%s inflight=%s todo=%s stall_s=%.0f ssl_fail=%s ok=%s workers=%s proxy=%s"
                    % (
                        len(done),
                        n_chunks,
                        len(inflight),
                        len(todo),
                        stall,
                        ssl_fail,
                        ok_n,
                        threading.active_count() - 2,
                        PROXY,
                    ),
                    flush=True,
                )
                if stall > 180:
                    print("UNHEALTHY stall>180s; still retrying, not exiting", flush=True)

    print(
        "START n_chunks=%s done=%s todo=%s nwork=%s proxy=%s out=%s"
        % (n_chunks, len(done), len(todo), NWORK, PROXY, out),
        flush=True,
    )
    th = [threading.Thread(target=worker, daemon=True) for _ in range(NWORK)]
    hh = threading.Thread(target=health, daemon=True)
    hh.start()
    for t in th:
        t.start()
    try:
        while len(done) < n_chunks and not stop.is_set():
            alive = sum(1 for t in th if t.is_alive())
            if alive == 0:
                print("WORKERS_DEAD respawn", flush=True)
                th = [threading.Thread(target=worker, daemon=True) for _ in range(NWORK)]
                for t in th:
                    t.start()
            time.sleep(2)
    except KeyboardInterrupt:
        stop.set()
        print("INTERRUPT save and exit; rerun to resume", flush=True)
        save_state()
        f.flush()
        f.close()
        return 1
    stop.set()
    f.flush()
    f.close()
    save_state()
    print("DOWNLOAD_OK done=%s/%s %s" % (len(done), n_chunks, out), flush=True)
    return 0 if len(done) >= n_chunks else 1


if __name__ == "__main__":
    raise SystemExit(main())
