import os
import json

def build_manifest(playlist_path, manifest_path, base_dir):
    with open(playlist_path, "r", encoding="utf-8") as f, open(manifest_path, "w", encoding="utf-8") as outf:
        for line in f:
            fn = line.strip()
            if not fn or fn.startswith("#"):
                continue
            filepath = os.path.join(base_dir, fn)
            if not os.path.isfile(filepath):
                print(f"[Warning] 没找到音频文件: {filepath}")
                continue
            num_spk = 1 if "T10" in fn else 2
            obj = {
                "audio_filepath": filepath,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": num_spk
            }
            outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Manifest 文件已生成：{manifest_path}")

