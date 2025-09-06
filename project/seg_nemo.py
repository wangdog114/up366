import torch, torchaudio, json
import whisper
import os, shutil
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import soundfile as sf
from tqdm import tqdm  # 添加进度条

def copy_rttm(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)

def parse_rttm(rttm_path):
    segments = {}
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            speaker = parts[7]
            start = float(parts[3])
            dur = float(parts[4])
            segments.setdefault(speaker, []).append((start, start + dur))
    return segments

def extract_per_speaker(input_wav, rttm_path, output_dir):
    wav, sr = torchaudio.load(input_wav)
    segments = parse_rttm(rttm_path)
    for spk, segs in segments.items():
        out = torch.zeros_like(wav)
        for (s, e) in segs:
            start_i = int(s * sr)
            end_i = int(e * sr)
            out[:, start_i:end_i] = wav[:, start_i:end_i]
        torchaudio.save(f"{output_dir}/{spk}.wav", out, sr)

def parse_rttm_2(rttm_path):
    diar = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            s, dur, sp = float(parts[3]), float(parts[4]), parts[7]
            diar.append((s, s + dur, sp))
    return diar

def assign_speaker(diar, start, end):
    for s, e, spk in diar:
        overlap = max(0, min(e, end) - max(s, start))
        if overlap >= (end - start) * 0.5:
            return spk
    return None

def process_audio_set(playlist_path, manifest_path, output_text_path, temp_dir, yaml_path, gender_model_name, whisper_model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用的设备: {device}")

    config = OmegaConf.load(yaml_path)
    config.device = device
    config.diarizer.manifest_filepath = manifest_path
    config.diarizer.out_dir = os.path.join(temp_dir, 'nemo_output')
    config.diarizer.speaker_embeddings.model_path = 'titanet_large'

    print("配置完成，开始实例化模型...")
    diarizer = ClusteringDiarizer(cfg=config).to(device)

    print("模型实例化成功，开始进行声纹分离...")
    diarizer.diarize()

    print("声纹分离完成！结果已保存在 nemo_output 文件夹中。")
    copy_rttm(os.path.join(config.diarizer.out_dir, 'pred_rttms'), temp_dir)

    if not os.path.exists(os.path.join(temp_dir, 'out_split')):
        os.mkdir(os.path.join(temp_dir, 'out_split'))

    with open(playlist_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Extracting speakers"):
            fn = line.strip()
            if not fn or fn.startswith("#"):
                continue
            wav_path = os.path.join(temp_dir, fn)
            split_dir = os.path.join(temp_dir, 'out_split', os.path.splitext(fn)[0])
            os.makedirs(split_dir, exist_ok=True)
            rttm_path = os.path.join(temp_dir, os.path.splitext(fn)[0] + ".rttm")
            extract_per_speaker(wav_path, rttm_path, split_dir)

    print("Gender infering...")
    speaker_gender = {}
    out_text = open(output_text_path, "w")
    
    print(f"加载 Whisper 模型: {whisper_model_name}")
    whisper_model = whisper.load_model(whisper_model_name, device=device)
    
    print(f"加载性别识别模型: {gender_model_name}")
    processor = AutoFeatureExtractor.from_pretrained(gender_model_name)
    model = AutoModelForAudioClassification.from_pretrained(
        gender_model_name, num_labels=2,
        label2id={"female": 0, "male": 1},
        id2label={0: "female", 1: "male"},
    ).to(device)
    model.eval()

    with open(playlist_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Gender and transcription"):
            fn = line.strip()
            if not fn or fn.startswith("#"):
                continue
            print(fn)
            speaker_gender[fn] = {}
            split_dir = os.path.join(temp_dir, 'out_split', os.path.splitext(fn)[0])
            for speaker_num in os.listdir(split_dir):
                speaker_path = os.path.join(split_dir, speaker_num)
                audio, orig_sr = sf.read(speaker_path, dtype="float32")
                target_sr = processor.sampling_rate
                if orig_sr != target_sr:
                    # print(f"Resampling from {orig_sr} -> {target_sr}")
                    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr, resampling_method="sinc_interp_hann")
                    audio = resampler(torch.from_numpy(audio).unsqueeze(0))[0].numpy()

                inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True).to(device)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    # print(f"female: {probs[0]:.4f}, male: {probs[1]:.4f}, preds: {model.config.id2label[int(probs.argmax())]}")
                    speaker_gender[fn][os.path.splitext(speaker_num)[0]] = model.config.id2label[int(probs.argmax())]
            
            rttm = os.path.join(temp_dir, fn.replace(".wav", ".rttm"))
            diar = parse_rttm_2(rttm)
            result = whisper_model.transcribe(os.path.join(temp_dir, fn), language="en")
            
            print(f"\n=== Transcription of {fn} ===")
            out_text.write(os.path.splitext(fn)[0] + "\n")
            for seg in result["segments"]:
                spk = assign_speaker(diar, seg["start"], seg["end"]) or "speaker_?"
                gender = speaker_gender[fn].get(spk, "?")
                print(f"[{spk}({gender}) {seg['start']:.2f}-{seg['end']:.2f}] {seg['text'].strip()}")
                out_text.write(("M:" if gender == "male" else "W:") + seg['text'].strip() + " ")
            out_text.write("\n")
    out_text.close()
