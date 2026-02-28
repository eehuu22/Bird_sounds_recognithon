import sys
import numpy as np
import sounddevice as sd
import librosa
import librosa.display
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QMessageBox, QSizePolicy, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import json
from inference import inference

# 参数配置
samplerate = 32000
duration = 7
n_fft = 3200
hop_length = 80
n_mels = 128
fmin = 0
fmax = 16000

class BirdPredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("鸟鸣识别器")
        self.audio_data = np.array([])
        self.is_recording = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_recording)

        self.details_dict = self.load_details()
        self.init_ui()

    def init_ui(self):
        # 图片展示
        self.image_label = QLabel("频谱图将显示在这里")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet(
            "font-size: 16px; border: 1px solid #ccc; background-color: #222; color: #aaa;"
        )
        self.image_label.setFixedHeight(280)  # 适当减小高度

        # 识别结果
        self.result_label = QLabel("识别结果：")
        self.result_label.setAlignment(Qt.AlignLeft)
        self.result_label.setStyleSheet("font-size: 16px;")

        # 状态信息
        self.time_label = QLabel("录音时长：0秒")
        self.status_label = QLabel("状态：就绪")
        self.time_label.setStyleSheet("font-size: 14px;")
        self.status_label.setStyleSheet("font-size: 14px;")

        # 录制进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(duration * 10)  # 以0.1秒为单位
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m")

        # 按钮
        self.start_btn = QPushButton("开始录制")
        self.stop_btn = QPushButton("结束录制")
        self.plot_btn = QPushButton("识别并显示频谱图")
        for btn in [self.start_btn, self.stop_btn, self.plot_btn]:
            btn.setMinimumHeight(40)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setStyleSheet("font-size: 14px;")

        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn.clicked.connect(self.stop_recording)
        self.plot_btn.clicked.connect(self.plot_and_predict)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label, stretch=4)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.time_label)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.plot_btn)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.resize(600, 480)  # 缩小窗口尺寸

    def load_details(self):
        try:
            with open("birds_detail.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "未找到 birds_detail.json 文件！")
            return {}
        except json.JSONDecodeError:
            QMessageBox.critical(self, "错误", "JSON 文件格式错误！")
            return {}

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"音频设备状态: {status}", file=sys.stderr)
        if self.is_recording:
            self.audio_data = np.append(self.audio_data, indata[:, 0])

    def update_recording(self):
        if self.is_recording:
            elapsed = len(self.audio_data) / samplerate
            self.time_label.setText(f"录音时长：{elapsed:.1f}/{duration}秒")
            progress_value = min(int(elapsed * 10), self.progress_bar.maximum())
            self.progress_bar.setValue(progress_value)
            if elapsed >= duration - 0.1:
                self.stop_recording()
                self.status_label.setText("状态：录音自动停止")

    def start_recording(self):
        try:
            self.audio_data = np.array([])
            self.is_recording = True
            self.stream = sd.InputStream(
                callback=self.audio_callback, channels=1, samplerate=samplerate
            )
            self.stream.start()
            self.status_label.setText("状态：录音中...")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            self.timer.start(100)
        except sd.PortAudioError as e:
            QMessageBox.critical(self, "错误", f"音频设备访问失败: {str(e)}")
            self.stop_recording()

    def stop_recording(self):
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()
        self.is_recording = False
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.time_label.setText("录音时长：0秒")
        self.progress_bar.setValue(0)
        self.status_label.setText("状态：就绪")

    def plot_and_predict(self):
        if len(self.audio_data) == 0:
            QMessageBox.warning(self, "警告", "没有可用的音频数据！")
            return
        try:
            target_len = samplerate * duration
            if len(self.audio_data) < target_len:
                padded = np.zeros(target_len, dtype=np.float32)
                padded[: len(self.audio_data)] = self.audio_data
                audio = padded
            else:
                audio = self.audio_data[:target_len]

            # 计算梅尔频谱
            S = librosa.feature.melspectrogram(
                y=audio,
                sr=samplerate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
            )
            S_db = librosa.power_to_db(S, ref=np.max)

            # 动态色阶范围，防止黑图
            vmin = S_db.min()
            vmax = 0

            # 绘图，缩小图像尺寸，dpi也调低
            fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
            ax.axis("off")
            librosa.display.specshow(
                S_db,
                sr=samplerate,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
                x_axis=None,
                y_axis=None,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
            )
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="none")
            plt.close(fig)
            buf.seek(0)

            # 载入QImage并缩放显示
            image_data = buf.read()
            qimage = QImage.fromData(image_data, "PNG")
            pixmap = QPixmap.fromImage(qimage)

            # 缩放图像到控件宽度，保持比例
            w = self.image_label.width()
            scaled_pixmap = pixmap.scaledToWidth(w, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

            # 用PIL图像推理
            pil_img = Image.open(BytesIO(image_data)).convert("RGB")
            label = inference(pil_img)

            if label in self.details_dict:
                d = self.details_dict[label]
                result_text = (
                    f"识别结果：{label}\n"
                    f"学名：{d.get('sp', '无')}\n"
                    f"亚种：{d.get('ssp', '无')}\n"
                    f"俗名：{d.get('en', '无')}"
                )
            else:
                result_text = f"识别结果：{label}（无详细信息）"

            self.result_label.setText(result_text)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中发生错误: {str(e)}")

    def closeEvent(self, event):
        self.stop_recording()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BirdPredictorApp()
    window.show()
    sys.exit(app.exec_())
