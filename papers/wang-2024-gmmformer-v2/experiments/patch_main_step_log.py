from pathlib import Path

p = Path("/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src/main.py")
t = p.read_text()
old_sig = "def train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer):"
new_sig = "def train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer, logger=None):"
old_upd = """        loss_meter.update(loss.cpu().item())

        train_bar.set_description('exp: {} epoch:{:2d} iter:{:3d} loss:{:.4f}'.format(cfg['model_name'], epoch, idx, loss))
"""
new_upd = """        loss_meter.update(loss.cpu().item())
        if logger is not None and (idx == 0 or idx % 10 == 0):
            logger.info('epoch:{:2d} iter:{:3d} loss:{:.4f}'.format(epoch, idx, float(loss.detach().cpu())))

        train_bar.set_description('exp: {} epoch:{:2d} iter:{:3d} loss:{:.4f}'.format(cfg['model_name'], epoch, idx, loss))
"""
old_call = "        loss_meter = train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer)"
new_call = "        loss_meter = train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer, logger)"
if "idx % 10 == 0" in t and "logger.info('epoch:" in t:
    print("PATCH_ALREADY")
else:
    if old_sig not in t or old_upd not in t or old_call not in t:
        raise SystemExit("PATCH_FAIL")
    t = t.replace(old_sig, new_sig, 1).replace(old_upd, new_upd, 1).replace(old_call, new_call, 1)
    p.write_text(t)
    print("PATCH_OK")
