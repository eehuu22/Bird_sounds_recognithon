import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from scipy.signal import butter, lfilter
from tqdm import tqdm

# 参数配置
root_dir = r"processed_bird_data"
output_suffix = "_mel_spectrogram"
n_fft = 3200
hop_length = 80
n_mels = 128
fmin = 0
fmax = 16000
sr = 32000


def add_white_noise(y, snr_db=20):
    """添加指定信噪比的白噪声"""
    signal_rms = np.sqrt(np.mean(y ** 2))
    snr = 10 ** (snr_db / 20.0)
    noise = np.random.normal(0, 1, len(y))
    noise_rms = np.sqrt(np.mean(noise ** 2))
    scale = signal_rms / (snr * noise_rms)
    return y + noise * scale


def add_pink_noise(y, snr_db=20):
    """添加指定信噪比的粉红噪声"""
    white_noise = np.random.normal(0, 1, len(y))
    # 设计一阶巴特沃斯带通滤波器
    b, a = butter(1, [0.01, 0.5], btype='band', analog=False)
    pink_noise = lfilter(b, a, white_noise)

    signal_rms = np.sqrt(np.mean(y ** 2))
    snr = 10 ** (snr_db / 20.0)
    noise_rms = np.sqrt(np.mean(pink_noise ** 2))
    scale = signal_rms / (snr * noise_rms)
    return y + pink_noise * scale


def add_bandpass_noise(y, lowcut=500, highcut=5000, snr_db=18):
    """添加指定频段的带通噪声"""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    noise = np.random.normal(0, 1, len(y))
    filtered_noise = lfilter(b, a, noise)

    signal_rms = np.sqrt(np.mean(y ** 2))
    snr = 10 ** (snr_db / 20.0)
    noise_rms = np.sqrt(np.mean(filtered_noise ** 2))
    scale = signal_rms / (snr * noise_rms)
    return y + filtered_noise * scale


def lowpass_filter(y, cutoff=4000):
    """低通滤波（6阶巴特沃斯）"""
    nyq = 0.5 * sr
    b, a = butter(6, cutoff / nyq, btype='low')
    return lfilter(b, a, y)


def process_audio_to_spectrogram(file_path):
    """处理单个音频文件并保存增强后的梅尔频谱图"""
    try:
        # 加载音频
        y, original_sr = librosa.load(file_path, sr=sr)
        y_processed = y.copy()

        # 增强流程（带独立概率控制）
        # 白噪声（20dB SNR）
        if 0.4 <= random.random() <= 0.7:
            y_processed = add_white_noise(y_processed, snr_db=20)


        # 粉红噪声（15dB SNR）
        if 0.4 <= random.random() <= 0.7:
            y_processed = add_pink_noise(y_processed, snr_db=15)


        # 带通噪声（500-5000Hz，18dB SNR）
        if 0.4 <= random.random() <= 0.7:
            y_processed = add_bandpass_noise(y_processed, lowcut=500, highcut=5000, snr_db=18)


        # 低通滤波（4kHz截止）
        if 0.4 <= random.random() <= 0.7:
            y_processed = lowpass_filter(y_processed, cutoff=4000)


        # 计算梅尔频谱
        S = librosa.feature.melspectrogram(y=y_processed, sr=sr,
                                           n_fft=n_fft,
                                           hop_length=hop_length,
                                           n_mels=n_mels,
                                           fmin=fmin,
                                           fmax=fmax)

        S_db = librosa.power_to_db(S, ref=np.max)

        # 构建输出路径
        base_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}{output_suffix}.png")

        # 绘制并保存频谱图
        plt.figure(figsize=(12, 6), frameon=False)
        ax = plt.subplot(111)
        ax.axis('off')
        ax.set_position([0, 0, 1, 1])

        librosa.display.specshow(S_db,
                                 sr=sr,
                                 hop_length=hop_length,
                                 x_axis='time',
                                 y_axis='mel',
                                 fmin=fmin,
                                 fmax=fmax,
                                 ax=ax)

        plt.savefig(output_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0,
                    facecolor='none')
        plt.close()

        # 删除原始音频文件
        os.remove(file_path)
        print(f"成功处理并增强: {file_path} → {output_path}")
        return True

    except Exception as e:
        print(f"处理失败 {file_path}: {str(e)}")
        return False


# 批量处理所有子目录
for foldername, _, filenames in tqdm(os.walk(root_dir), desc="处理进度"):
    for filename in filenames:
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(foldername, filename)
            process_audio_to_spectrogram(file_path)

print("\n所有音频处理并增强完毕！")