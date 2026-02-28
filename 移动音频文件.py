import os
import shutil
from tqdm import tqdm  # 进度条库（需安装：pip install tqdm）

# 配置路径
source_root = r"F:\智能系统设计"    #原始数据集位置（原数据集的目录为形如"C->Cacomantis->xxx.wav")
target_root = os.path.join(source_root, "processed_bird_data")     #汇总后数据集位置(形如"processed_bird_data->Cacomantis->xxx.wav")

# 创建目标根目录
os.makedirs(target_root, exist_ok=True)

# 遍历所有子目录
for foldername, subfolders, filenames in os.walk(source_root):
    # 跳过目标目录自身（避免重复处理）
    if foldername == target_root:
        continue

    # 处理每个文件
    for filename in tqdm(filenames, desc=f"处理 {os.path.basename(foldername)}"):
        # 跳过非wav文件
        if not filename.lower().endswith(".wav"):
            continue

        # 提取文件名前缀（以第一个点分隔）
        prefix = filename.split('.')[0]

        # 构建目标路径
        target_folder = os.path.join(target_root, prefix)
        target_path = os.path.join(target_folder, filename)

        # 创建目标文件夹（如果不存在）
        os.makedirs(target_folder, exist_ok=True)

        # 构建源文件完整路径
        src_path = os.path.join(foldername, filename)

        # 移动文件（如果目标不存在）
        if not os.path.exists(target_path):
            shutil.move(src_path, target_path)
        else:
            # 处理文件名冲突（可选：添加计数后缀）
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(target_path):
                new_name = f"{base}_{counter}{ext}"
                target_path = os.path.join(target_folder, new_name)
                counter += 1
            shutil.move(src_path, target_path)