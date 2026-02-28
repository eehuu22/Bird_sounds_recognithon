import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# 加载音频（自动重采样到32kHz）
audio_path = r"E:/文档/TheFinals/Dominica_dataset/Signal_parts/SW_2_filtered.wav"
y, sr = librosa.load(audio_path, sr=32000)

# 参数设置
n_fft = 3200
hop_length = 80
n_mels = 128
fmin = 0
fmax = 16000

# 计算梅尔频谱
S = librosa.feature.melspectrogram(y=y, sr=sr,
                                   n_fft=n_fft,
                                   hop_length=hop_length,
                                   n_mels=n_mels,
                                   fmin=fmin,
                                   fmax=fmax)

S_db = librosa.power_to_db(S, ref=np.max)

# 创建输出目录
output_dir = r"纯净频谱图"
os.makedirs(output_dir, exist_ok=True)

# 生成唯一文件名
base_name = os.path.splitext(os.path.basename(audio_path))[0]
output_path = os.path.join(output_dir, f"{base_name}_pure_mel.png")

# 正确创建无边框画布和坐标系
plt.figure(figsize=(12, 6), frameon=False)
ax = plt.subplot(111)  # 创建单个子图
# ax.axis('off')          # 关闭所有坐标轴元素
ax.set_position([0, 0, 1, 1])  # 扩展坐标系到整个画布

# 绘制梅尔频谱
librosa.display.specshow(S_db,
                         sr=sr,
                         hop_length=hop_length,
                         x_axis='time',
                         y_axis='mel',
                         fmin=fmin,
                         fmax=fmax,
                         ax=ax)

# 保存配置（关键参数）
plt.savefig(output_path,
           dpi=300,
           bbox_inches='tight',
           pad_inches=0,
           facecolor='none')

plt.close()
print(f"纯净梅尔频谱图已保存至：{output_path}")