# Fastest way to get MS-SL feature zips

Date: 2026-09-17  
Trigger: user asked to be given the fastest download method next time, after Drive quota / IDM / chunked gdown failed.

## Rule

For GMMFormer v2 **I3D/RoBERTa feature packs** (ActivityNet / TVR / Charades from MS-SL), do **not** start with Google Drive gdown or 16-way Range download.

Tell the user, in the same planning reply, this order:

1. **Baidu Netdisk** (official README): https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4 — usually fastest in this lab.
2. **Browser logged into Google**, save as a **new** filename (e.g. `tvr_browser.zip`). Drive virus-scan / “too many users” HTML is not the zip.
3. Only if 1–2 are impossible: gdown / resume script via laptop `127.0.0.1:7897`.

Do not ask the user to fight IDM against Drive HTML. After they say the file is local, verify size + `ZipFile` before upload/train.
