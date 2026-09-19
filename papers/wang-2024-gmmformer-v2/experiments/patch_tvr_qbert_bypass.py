#!/usr/bin/env python3
"""TVR query-Bert bypass: skip query_encoder at encode_query only.

Keeps query_encoder construction. Does not edit BertAttention or encode_input.
Routes qbert_bypass PRVR_ROOT to CLIP token-sequence proj h5, not EOT.
"""
from pathlib import Path

SRC = Path("/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src")


def bak(p: Path, tag: str) -> None:
    b = Path(str(p) + tag)
    if not b.exists():
        b.write_bytes(p.read_bytes())
        print("BAK", b)
    else:
        print("BAK_KEEP", b)


def patch_model() -> None:
    p = SRC / "Models" / "gmmformerV2" / "model.py"
    t = p.read_text()
    if "self.query_encoder = BertAttention" not in t:
        raise SystemExit("MODEL_FAIL query_encoder init missing")
    if "def encode_input(feat, mask, input_proj_layer, encoder_layer" not in t:
        raise SystemExit("MODEL_FAIL encode_input missing")
    old = (
        "    def encode_query(self, query_feat, query_mask):\n"
        "        encoded_query = self.encode_input_for_query(query_feat, query_mask, self.query_input_proj, self.query_encoder,\n"
        "                                          self.query_pos_embed)  # (N, Lq, D)\n"
    )
    new = (
        "    def encode_query(self, query_feat, query_mask):\n"
        "        if self.config.get(\"bypass_query_encoder\", False):\n"
        "            encoded_query = self.query_pos_embed(self.query_input_proj(query_feat))\n"
        "        else:\n"
        "            encoded_query = self.encode_input_for_query(query_feat, query_mask, self.query_input_proj, self.query_encoder,\n"
        "                                              self.query_pos_embed)  # (N, Lq, D)\n"
    )
    if "bypass_query_encoder" in t and "self.query_pos_embed(self.query_input_proj(query_feat))" in t:
        print("MODEL_ALREADY")
        return
    if old not in t:
        raise SystemExit("MODEL_FAIL no encode_query official block")
    bak(p, ".bak_before_qbert_bypass")
    t = t.replace(old, new, 1)
    if t.count("self.query_encoder = BertAttention") != 1:
        raise SystemExit("MODEL_FAIL query_encoder init count")
    if "def encode_input(feat, mask, input_proj_layer, encoder_layer" not in t:
        raise SystemExit("MODEL_FAIL encode_input lost")
    p.write_text(t)
    print("MODEL_OK")


def patch_models_builder() -> None:
    p = SRC / "Models" / "builder.py"
    t = p.read_text()
    old = "        sft_factor=cfg['sft_factor'])\n"
    new = (
        "        sft_factor=cfg['sft_factor'],\n"
        "        bypass_query_encoder=bool(cfg.get('bypass_query_encoder', False)))\n"
    )
    if "bypass_query_encoder=bool(cfg.get('bypass_query_encoder', False))" in t:
        print("MODELS_BUILDER_ALREADY")
        return
    if old not in t:
        raise SystemExit("MODELS_BUILDER_FAIL no sft_factor tail")
    bak(p, ".bak_before_qbert_bypass")
    t = t.replace(old, new, 1)
    p.write_text(t)
    print("MODELS_BUILDER_OK")


def patch_data_builder() -> None:
    p = SRC / "Datasets" / "builder.py"
    t = p.read_text()
    old = (
        "        if 'i3d_cliptext_e4_eot' in os.environ.get('PRVR_ROOT', ''):\n"
        "            text_feat_path = os.path.join(rootpath, collection, 'TextData', "
        "'%s_clip_B32_eot.h5' % collection)\n"
        "        elif 'i3d_cliptext_e4' in os.environ.get('PRVR_ROOT', ''):\n"
        "            text_feat_path = os.path.join(rootpath, collection, 'TextData', "
        "'%s_clip_B32_proj.h5' % collection)\n"
        "        else:\n"
        "            text_feat_path = os.path.join(rootpath, collection, 'TextData', "
        "'roberta_%s_query_feat.hdf5' % collection)\n"
    )
    new = (
        "        if 'qbert_bypass' in os.environ.get('PRVR_ROOT', ''):\n"
        "            text_feat_path = os.path.join(rootpath, collection, 'TextData', "
        "'%s_clip_B32_proj.h5' % collection)\n"
        "        elif 'i3d_cliptext_e4_eot' in os.environ.get('PRVR_ROOT', ''):\n"
        "            text_feat_path = os.path.join(rootpath, collection, 'TextData', "
        "'%s_clip_B32_eot.h5' % collection)\n"
        "        elif 'i3d_cliptext_e4' in os.environ.get('PRVR_ROOT', ''):\n"
        "            text_feat_path = os.path.join(rootpath, collection, 'TextData', "
        "'%s_clip_B32_proj.h5' % collection)\n"
        "        else:\n"
        "            text_feat_path = os.path.join(rootpath, collection, 'TextData', "
        "'roberta_%s_query_feat.hdf5' % collection)\n"
    )
    if "if 'qbert_bypass' in os.environ.get('PRVR_ROOT', '')" in t and "%s_clip_B32_proj.h5" in t:
        print("DATA_BUILDER_ALREADY")
    elif old not in t:
        raise SystemExit("DATA_BUILDER_FAIL no eot/e4 text block")
    else:
        bak(p, ".bak_before_qbert_bypass")
        n = t.count(old)
        t = t.replace(old, new)
        p.write_text(t)
        print("DATA_BUILDER_OK", n)
    t = p.read_text()
    if "clip_ViT_B_32_tvr_query_feat.hdf5" in t:
        raise SystemExit("ZIP_EOT_WIRED")
    if "tvr_train_release.jsonl" in t:
        raise SystemExit("JSONL_STILL_PRESENT")
    if "PRVR_EVAL_SPLIT" not in t:
        raise SystemExit("SPLIT_MISSING")
    if "qbert_bypass" not in t:
        raise SystemExit("QBERT_ROOT_MISSING")
    print("BUILDER_JSONL_GONE", "tvr_train_release.jsonl" not in t)
    print("BUILDER_EVAL_SPLIT", "PRVR_EVAL_SPLIT" in t)
    print("BUILDER_PROJ_H5", "%s_clip_B32_proj.h5" in t)
    print("BUILDER_EOT_KEPT", "%s_clip_B32_eot.h5" in t)
    print("ZIP_EOT_ABSENT", "clip_ViT_B_32_tvr_query_feat.hdf5" not in t)


def main() -> None:
    patch_model()
    patch_models_builder()
    patch_data_builder()
    mp = (SRC / "Models" / "gmmformerV2" / "model.py").read_text()
    if "self.query_encoder = BertAttention" not in mp:
        raise SystemExit("FINAL_FAIL query_encoder init")
    if "bypass_query_encoder" not in mp:
        raise SystemExit("FINAL_FAIL bypass missing")
    if "class BertAttention" in mp:
        raise SystemExit("FINAL_FAIL BertAttention class moved into model.py")
    print("PATCH_QBERT_BYPASS_OK")


if __name__ == "__main__":
    main()
