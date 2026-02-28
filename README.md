# Bird_sounds_recognithon
可以实时听一段鸟叫声识别鸟的名字
## 先下载数据集
地址：https://hf-mirror.com/datasets/sakei/Bird_audio_in_China/tree/main  ，下载解压至当前文件夹，目录形式如
目录形式如\
当前文件夹->A/B/C 


## 数据预处理阶段：
按顺序运行   移动音频文件.py  ->  音频切割.py  ->  音频转增强频谱.py
移动音频.py运行完后目录形如

D:bird_data\
├─Abroscopus\
├─Acridotheres\
├─Acrocephalus\
├─Actinodura\
…（鸟名文件夹，每个文件夹中为鸟叫音频，如Abroscopus.3568353_1_chunk0.wav、Abroscopus.3568353_1_chunk1.wav）

**birds_names.json**记录了鸟类的名字，这作为分类的列表，因为我们按照鸟类的名称，即鸟类数据文件夹的文件名来作为标签，数据集给出的详细名字分类顺序是乱的会导致映射错误。**birds_detail.json**记录了鸟类的更详细信息，靠分类得到的学名再这个文件中查找。
数据集下载的网站有给一个**birds_list.json**的文件，我们**没有使用**。

预处理完毕，即运行完 音频转增强频谱.py 后会有一个processed_bird_data的目录，形如

D:processed_bird_data\
├─Abroscopus\
├─Acridotheres\
├─Acrocephalus\
├─Actinodura\
…（鸟名文件夹，每个文件夹中为鸟叫音频，如Abroscopus.3568353_1_chunk0_mel_spectrogram.png、Abroscopus.3568353_1_chunk1_mel_spectrogram.png）

此时运行模型训练.py，待跑完模型可以得到一个best_model.pth文件。epoch不用很大，低于30轮就可以，我们用的是26轮后得到的模型（26_best_model.pth）。

## 运行阶段：
确保gui_main.py、inference.py、birds_detail.json、26_best_model.pth、class_names.json在同一个文件夹。
运行gui_main.py


**另外**，若想做其他研究，音频转频谱.py 文件，可以将音频转为普通纯净的频谱图。但是这个文件和我们现在的项目没有关联。

