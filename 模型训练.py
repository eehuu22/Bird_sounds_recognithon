import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 配置参数
BATCH_SIZE = 64
TARGET_SIZE = (224, 224)
DATA_PATH = "processed_bird_data"
EPOCHS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "best_model.pth"

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(TARGET_SIZE),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(TARGET_SIZE),
    transforms.ToTensor(),
])

def main():
    # 加载数据
    full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=train_transform)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"检测到 {num_classes} 个鸟类类别")

    # 构建模型
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(DEVICE)

    # 损失函数、优化器、混合精度工具
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()

    best_val_acc = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # 训练函数
    def train_one_epoch(epoch):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return running_loss / total, correct / total

    # 验证函数
    def evaluate():
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with autocast(device_type=DEVICE.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return running_loss / total, correct / total

    # 训练主循环
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(epoch)

        if (epoch + 1) % 2 == 0:
            val_loss, val_acc = evaluate()
            print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), CHECKPOINT_PATH)

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
        else:
            history["val_loss"].append(None)
            history["val_acc"].append(None)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

    # 保存训练历史
    with open("training_history.json", "w") as f:
        json.dump(history, f)

    # 加载最佳模型并评估
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    final_loss, final_acc = evaluate()
    print(f"验证准确率: {final_acc:.4f}")

    # 保存最终模型
    torch.save(model, "birdcall_model.pth")

if __name__ == '__main__':
    main()
