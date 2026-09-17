#!/usr/bin/env python3
"""Multi-connection resume download of a Google Drive file via usercontent Range."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request

CHUNK = 8 * 1024 * 1024
PROXY_DEFAULT = "http://127.0.0.1:17897"


def opener(proxy: str):
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    return urllib.request.build_opener()


def file_url(file_id: str) -> str:
    return (
        "https://drive.usercontent.google.com/download?"
        "id=%s&export=download&confirm=t" % file_id
    )


def discover_size(op, url: str) -> int:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"},
    )
    with op.open(req, timeout=60) as r:
        cr = r.headers.get("Content-Range") or ""
        cd = r.headers.get("Content-Disposition") or ""
        ct = r.headers.get("Content-Type") or ""
        body = r.read(16)
        print("DISCOVER ct=%s cr=%s cd=%s head=%s" % (ct, cr, cd, body[:8]), flush=True)
        if "text/html" in ct:
            raise IOError("html_not_zip")
        if "/" in cr:
            return int(cr.split("/")[-1])
        cl = r.headers.get("Content-Length")
        if cl and int(cl) > 1:
            return int(cl)
    raise IOError("no_size")


def save_state(path: str, done, expect: int, chunk: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as sf:
        json.dump({"done": sorted(done), "expect": expect, "chunk": chunk}, sf)
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--expect", type=int, default=0)
    p.add_argument("--nwork", type=int, default=12)
    p.add_argument("--chunk", type=int, default=CHUNK)
    args = p.parse_args()
    proxy = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or PROXY_DEFAULT
    )
    url = file_url(args.id)
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    state_path = out + ".parts.json"
    op0 = opener(proxy)
    expect = args.expect
    if expect <= 0:
        expect = discover_size(op0, url)
        print("EXPECT_DISCOVERED", expect, flush=True)
    n_chunks = (expect + args.chunk - 1) // args.chunk
    done = set()
    if os.path.isfile(state_path):
        try:
            st = json.load(open(state_path, encoding="utf-8"))
            if int(st.get("expect", 0)) == expect and int(st.get("chunk", 0)) == args.chunk:
                done = set(st.get("done", []))
        except Exception:
            done = set()
    if os.path.isfile(out) and os.path.getsize(out) == expect and len(done) >= n_chunks:
        print("ALREADY_DONE", out, flush=True)
        return 0
    mode = "r+b" if os.path.isfile(out) else "w+b"
    f = open(out, mode)
    if os.path.getsize(out) != expect:
        f.seek(expect - 1)
        f.write(b"\0")
        f.flush()
    todo = [i for i in range(n_chunks) if i not in done]
    print(
        "START n_chunks=%s done=%s todo=%s nwork=%s proxy=%s out=%s expect=%s"
        % (n_chunks, len(done), len(todo), args.nwork, proxy, out, expect),
        flush=True,
    )
    lock = threading.Lock()
    got = 0
    t0 = time.time()
    err = []
    stop = threading.Event()

    def worker():
        nonlocal got
        op = opener(proxy)
        while not stop.is_set():
            with lock:
                if not todo:
                    return
                i = todo.pop(0)
            start = i * args.chunk
            end = min(expect, start + args.chunk) - 1
            want = end - start + 1
            ok = False
            last_e = None
            for attempt in range(8):
                try:
                    req = urllib.request.Request(
                        url,
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
                        save_state(state_path, done, expect, args.chunk)
                        dt = max(time.time() - t0, 1e-6)
                        left = expect - (len(done) * args.chunk)
                        if left < 0:
                            left = 0
                        bps = got / dt
                        eta = left / bps if bps > 0 else 0
                        print(
                            "PROGRESS %s/%s bytes=%s MBps=%.2f eta_min=%.1f"
                            % (
                                len(done),
                                n_chunks,
                                min(expect, len(done) * args.chunk),
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

    th = [threading.Thread(target=worker, daemon=True) for _ in range(args.nwork)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    f.flush()
    f.close()
    save_state(state_path, done, expect, args.chunk)
    sz = os.path.getsize(out)
    print(
        "END size=%s expect=%s done=%s/%s" % (sz, expect, len(done), n_chunks),
        flush=True,
    )
    if sz != expect or len(done) < n_chunks:
        print("INCOMPLETE", flush=True)
        return 1
    print("DOWNLOAD_OK", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
