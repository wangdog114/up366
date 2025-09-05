# 使用说明:
# IMAGE_DIR 中的图片名称与 AUDIO_DIR 下音频集的名称一致，图片使用 png 保存
import os
import subprocess
import shutil
from tqdm import tqdm
from build_manifest import build_manifest
from seg_nemo import process_audio_set
from config_loader import config

# 从配置文件读取路径和设置
paths_config = config['paths']
processing_config = config['processing']
audio_config = config['audio_processing']

AUDIO_DIR = paths_config['audio_dir']
OUTPUT_DIR = paths_config['output_dir']
TEMP_DIR = paths_config['temp_dir']
YAML_PATH = paths_config['nemo_yaml']
CLEAN_TEMP = processing_config['clean_temp']

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def convert_mp3_to_wav(input_dir, temp_set_dir):
    playlist_path = os.path.join(temp_set_dir, "playlist.m3u")
    with open(playlist_path, "w", encoding="utf-8") as playlist:
        for file in os.listdir(input_dir):
            if file.endswith(".mp3"):
                mp3_path = os.path.join(input_dir, file)
                wav_file = f"{os.path.splitext(file)[0]}.wav"
                wav_path = os.path.join(temp_set_dir, wav_file)
                try:
                    # 使用 -loglevel error 来减少不必要的 ffmpeg 输出
                    subprocess.run(["ffmpeg", "-i", mp3_path, "-ac", "1", wav_path, "-loglevel", "error"], check=True)
                    playlist.write(wav_file + "\n")
                except subprocess.CalledProcessError as e:
                    print(f"转换失败: {mp3_path} - {e}")
    return playlist_path

def process_one_set(set_name, input_set_dir):
    temp_set_dir = os.path.join(TEMP_DIR, set_name)
    os.makedirs(temp_set_dir, exist_ok=True)
    
    print(f"处理音频集: {set_name}")
    
    # 步骤1: MP3 转 WAV 并生成 playlist
    playlist_path = convert_mp3_to_wav(input_set_dir, temp_set_dir)
    
    # 步骤2: 生成 manifest
    manifest_path = os.path.join(temp_set_dir, "manifest.json")
    build_manifest(playlist_path, manifest_path, temp_set_dir)
    
    # 步骤3: 运行 seg_nemo
    output_text_path = os.path.join(temp_set_dir, "text.txt")
    process_audio_set(
        playlist_path=playlist_path,
        manifest_path=manifest_path,
        output_text_path=output_text_path,
        temp_dir=temp_set_dir,
        yaml_path=YAML_PATH,
        gender_model_name=audio_config['gender_model'],
        whisper_model_name=audio_config['whisper_model']
    )
    
    # 步骤4: 移动输出到 output_dir
    output_set_dir = os.path.join(OUTPUT_DIR, set_name)
    os.makedirs(output_set_dir, exist_ok=True)
    shutil.move(output_text_path, os.path.join(output_set_dir, "text.txt"))
    
    # 步骤5: 清理临时文件
    if CLEAN_TEMP:
        shutil.rmtree(temp_set_dir)
        print(f"临时文件已清理: {temp_set_dir}")


if __name__ == "__main__":
    if not os.path.isdir(AUDIO_DIR):
        print(f"错误：音频目录不存在: {AUDIO_DIR}")
        print("请检查 config.yaml 文件中的 'audio_dir' 配置。" )
    else:
        audio_sets = [d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))]
        for set_name in tqdm(audio_sets, desc="处理音频集"):
            input_set_dir = os.path.join(AUDIO_DIR, set_name)
            try:
                process_one_set(set_name, input_set_dir)
            except Exception as e:
                print(f"处理 {set_name} 失败: {e}")
        print(f"所有音频集处理完成！输出在 {OUTPUT_DIR} 下。")