import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm  # 进度条库（需安装：pip install tqdm）

# 参数设置
root_dir = r"processed_bird_data"  # 根目录（包含多个分类文件夹）
chunk_duration = 7.0          # 每段切片长度（秒）
threshold_db = 30             # 静音判断的分贝阈值
min_bird_ratio = 1/3          # 鸟叫最小时间占比阈值

def estimate_active_duration(y, sr, top_db=30):
    """估算音频中有效声音的总时长"""
    intervals = librosa.effects.split(y, top_db=top_db)
    active = sum((end - start) for start, end in intervals)
    return active / sr

def process_file(file_path):
    """处理单个音频文件"""
    # 加载音频
    y, sr = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)
    num_chunks = int(np.ceil(total_duration / chunk_duration))

    # 生成输出路径（与原文件同目录）
    base_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # 存储有效切片
    valid_chunks = []

    for i in range(num_chunks):
        start = int(i * chunk_duration * sr)
        end = int(min((i + 1) * chunk_duration * sr, len(y)))
        y_chunk = y[start:end]

        # 鸟叫活跃时长检测
        active_duration = estimate_active_duration(y_chunk, sr, top_db=threshold_db)

        if active_duration >= chunk_duration * min_bird_ratio:
            output_path = os.path.join(base_dir, f"{base_name}_chunk{i}.wav")
            sf.write(output_path, y_chunk, sr)
            valid_chunks.append(output_path)
            print(f" 保留: {output_path} (鸟叫占比: {active_duration / chunk_duration:.2f})")
        else:
            print(f"✘ 丢弃: {base_name}_chunk{i} (鸟叫占比: {active_duration / chunk_duration:.2f})")

    # 删除原始文件（仅当存在有效切片时）
    if valid_chunks:
        os.remove(file_path)
        print(f"\n原始文件已删除: {file_path}")
    else:
        print(f"\n未生成有效切片，保留原始文件: {file_path}")

# 批量处理所有子目录
for foldername, _, filenames in tqdm(os.walk(root_dir), desc="处理进度"):
    for filename in filenames:
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(foldername, filename)
            print(f"\n{'-'*50}\n处理文件: {file_path}")
            process_file(file_path)

print("\n所有音频处理完毕！")