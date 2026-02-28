import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json

# ===== 配置 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "26_best_model.pth"
CLASS_JSON = "class_names.json"

# ===== 加载类别 =====
with open(CLASS_JSON, "r", encoding="utf-8") as f:
    class_names = json.load(f)
class_names = sorted(class_names)
num_classes = len(class_names)

# ===== 构建模型并加载权重 =====
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ===== 图像预处理 =====
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # 如训练中使用 Normalize，请取消以下注释：
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
])

def inference(image: Image.Image) -> str:
    """推理预测输入图像的类别标签"""
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
    return label
