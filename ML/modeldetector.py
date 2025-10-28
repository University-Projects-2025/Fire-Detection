import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import io
from sklearn.metrics import precision_score, recall_score, f1_score
import json
class FireSmokeDataset(Dataset):

    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = f"{self.img_dir}/{row['filename']}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row['detect'])
        
        # Безопасное создание bbox с проверкой на класс 2
        if label == 2:
            bbox = torch.zeros(4, dtype=torch.float32)
        else:
            bbox = torch.tensor([
                float(row['coord1']), 
                float(row['coord2']), 
                float(row['coord3']), 
                float(row['coord4'])
            ], dtype=torch.float32)
            
        return image, label, bbox
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones((dim,)) * 1e-6)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + x

class FireSmokeDetectorDualHead(nn.Module):
    def __init__(self, num_classes=3, detection_threshold_fire=0.5, detection_threshold_smoke=0.5):
        super().__init__()
        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]
        
        # Backbone (без изменений)
        self.stem = nn.Conv2d(3, dims[0], kernel_size=4, stride=4)
        self.stem_norm = LayerNorm(dims[0], eps=1e-6)
        
        self.stage1 = nn.Sequential(*[Block(dim=dims[0]) for _ in range(depths[0])])
        self.downsample1 = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
        )
        
        self.stage2 = nn.Sequential(*[Block(dim=dims[1]) for _ in range(depths[1])])
        self.downsample2 = nn.Sequential(
            LayerNorm(dims[1], eps=1e-6),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2),
        )
        
        self.stage3 = nn.Sequential(*[Block(dim=dims[2]) for _ in range(depths[2])])
        self.downsample3 = nn.Sequential(
            LayerNorm(dims[2], eps=1e-6),
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2),
        )
        
        self.stage4 = nn.Sequential(*[Block(dim=dims[3]) for _ in range(depths[3])])
        
        # Global features для классификации
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        # Голова 1: Классификатор (без изменений)
        self.classifier = nn.Linear(dims[-1], num_classes)
        
        # Голова 2: Детектор bounding boxes
        # Используем features из stage4 для детекции
        self.bbox_head = nn.Sequential(
            nn.Conv2d(dims[-1], 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.GELU(),
            nn.Linear(512, 4)  # [x, y, width, height]
        )
        
        # Пороги срабатывания
        self.detection_threshold_fire = detection_threshold_fire
        self.detection_threshold_smoke = detection_threshold_smoke
        
        # Активация для bounding boxes
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward через backbone
        features = self.stem(x)
        features = self.stem_norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        features = self.stage1(features)
        features = self.downsample1[0](features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        features = self.downsample1[1](features)
        
        features = self.stage2(features)
        features = self.downsample2[0](features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        features = self.downsample2[1](features)
        
        features = self.stage3(features)
        features = self.downsample3[0](features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        features = self.downsample3[1](features)
        
        features = self.stage4(features)
        
        # Голова 1: Классификация (ВСЕГДА вычисляется первой)
        global_features = features.mean([-2, -1])  # Global Average Pooling
        global_features = self.norm(global_features)
        classification_logits = self.classifier(global_features)
        classification_probs = F.softmax(classification_logits, dim=1)
        
        # Инициализация outputs для детекции
        batch_size = x.size(0)
        device = x.device
        bounding_boxes = torch.zeros(batch_size, 4, device=device)
        fire_smoke_detected_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # КРИТИЧЕСКИ ВАЖНО: проверка классификатора ПРЕЖДЕ вычисления детектора
        fire_probs = classification_probs[:, 1]  # класс 1 = огонь
        smoke_probs = classification_probs[:, 0]
        
        # Маска для срабатывания детектора (любая пожарная угроза)
        detection_mask = (fire_probs > self.detection_threshold_fire) | (smoke_probs > self.detection_threshold_smoke)
        
        # ВЫЧИСЛЯЕМ ДЕТЕКТОР ТОЛЬКО ДЛЯ УГРОЖАЮЩИХ ПРИМЕРОВ
        if detection_mask.any():
            # Берем только features для примеров с пожарной угрозой
            threat_features = features[detection_mask]
            
            # Вычисляем bounding boxes только для угрожающих примеров
            detected_boxes = self.bbox_head(threat_features)
            detected_boxes = self.sigmoid(detected_boxes)  # Нормализуем в [0, 1]
            
            # Записываем результаты обратно
            bounding_boxes[detection_mask] = detected_boxes
            fire_smoke_detected_mask[detection_mask] = True
        
        return classification_logits, bounding_boxes, fire_smoke_detected_mask

    def set_detection_thresholds(self, fire_threshold=None, smoke_threshold=None):
        """Динамическое изменение порогов срабатывания"""
        if fire_threshold is not None:
            self.detection_threshold_fire = fire_threshold
        if smoke_threshold is not None:
            self.detection_threshold_smoke = smoke_threshold

def dual_head_loss(classification_logits, bbox_predictions, targets, bbox_targets, detection_mask, 
                  cls_weight=1.0, bbox_weight=1.0):
    """
    Вычисляет общий loss для классификации и детекции
    
    Args:
        classification_logits: выход классификатора [batch, 3]
        bbox_predictions: предсказанные bbox [batch, 4]
        targets: истинные классы [batch]
        bbox_targets: истинные bbox [batch, 4]
        detection_mask: маска срабатывания детектора [batch]
        cls_weight: вес классификационного loss
        bbox_weight: вес детекционного loss
    """
    labels, bbox_gt = targets, bbox_targets
    
    # Classification loss (всегда вычисляется)
    cls_loss = F.cross_entropy(classification_logits, labels)
    
    # Detection loss (только для примеров с fire/smoke классами И валидными bbox)
    bbox_loss = torch.tensor(0.0, device=classification_logits.device)
    
    # Маска для обучения детектора: примеры с fire/smoke (классы 1,2) И bbox не нулевой
    train_detection_mask = (labels == 0) | (labels == 1)  # огонь ИЛИ дым
    
    if train_detection_mask.any():
        bbox_loss = F.smooth_l1_loss(
            bbox_predictions[train_detection_mask], 
            bbox_gt[train_detection_mask]
        )
    
    total_loss = cls_weight * cls_loss + bbox_weight * bbox_loss
    
    return total_loss, cls_loss, bbox_loss


# Модифицированная функция обучения
def train_dual_head(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    device="cpu",
    epochs=5,
    ckpt_path="best_dual_head.pt",
    experiment_name="dual_head_experiment"
):
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()
    
    best_accuracy = 0.0
    model.to(device)
    
    with mlflow.start_run():
        # Логируем параметры
        mlflow.log_params({
            "epochs": epochs,
            "device": device,
            "optimizer": optimizer.__class__.__name__,
            "detection_threshold_fire": model.detection_threshold_fire,
            "detection_threshold_smoke": model.detection_threshold_smoke,
            "batch_size": train_loader.batch_size
        })
        
        for epoch in range(epochs):
            # Обучение
            model.train()
            train_loop = tqdm(
                enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"
            )
            total_train_loss = 0.0
            total_cls_loss = 0.0
            total_bbox_loss = 0.0
            
            for i, data in train_loop:
                inputs, labels, bbox_targets = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                bbox_targets = bbox_targets.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                classification_logits, bbox_predictions, detection_mask = model(inputs)
                
                # Вычисление потерь
                total_loss, cls_loss, bbox_loss = loss_fn(
                    classification_logits, 
                    bbox_predictions, 
                    labels, 
                    bbox_targets, 
                    detection_mask,
                    cls_weight=1.0, 
                    bbox_weight=1.0
                )
                
                total_loss.backward()
                optimizer.step()
                
                total_train_loss += total_loss.item()
                total_cls_loss += cls_loss.item()
                total_bbox_loss += bbox_loss.item() if bbox_loss > 0 else 0.0
                
                train_loop.set_postfix({
                    "total_loss": total_loss.item(),
                    "cls_loss": cls_loss.item(),
                    "bbox_loss": bbox_loss.item() if bbox_loss > 0 else 0.0
                })
            
            # Валидация
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                val_loop = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
                for i, data in val_loop:
                    inputs, labels, bbox_targets = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    bbox_targets = bbox_targets.to(device)
                    
                    classification_logits, bbox_predictions, detection_mask = model(inputs)
                    
                    # Classification accuracy
                    _, predicted = torch.max(classification_logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Loss
                    total_loss, _, _ = loss_fn(
                        classification_logits, bbox_predictions, labels, bbox_targets, detection_mask
                    )
                    val_loss += total_loss.item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    val_loop.set_postfix({"val_acc": val_correct / val_total})
            
            val_accuracy = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            avg_train_loss = total_train_loss / len(train_loader)
            avg_cls_loss = total_cls_loss / len(train_loader)
            avg_bbox_loss = total_bbox_loss / len(train_loader)
            
            # Логирование в MLflow
            metrics = {
                "train_loss": avg_train_loss,
                "train_cls_loss": avg_cls_loss,
                "train_bbox_loss": avg_bbox_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy
            }
            mlflow.log_metrics(metrics, step=epoch)
            
            # Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Epoch {epoch}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            pil_image = Image.open(buf)
            mlflow.log_image(pil_image, f"confusion_matrix/epoch_{epoch}.png")
            buf.close()
            
            # Сохранение лучшей модели
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), ckpt_path)
                mlflow.log_artifact(ckpt_path)
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
            
            print(f"Epoch {epoch}: "
                  f"Train Loss: {avg_train_loss:.4f} (cls: {avg_cls_loss:.4f}, bbox: {avg_bbox_loss:.4f}), "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Логируем лучшую точность
        mlflow.log_metric("best_val_accuracy", best_accuracy)
        mlflow.pytorch.log_model(model, "model")
    
    print(f"Training completed. Best validation accuracy: {best_accuracy:.4f}")

import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split

def save_dataset_splits(train_df, val_df, test_df, img_dir, save_path="dataset_splits.pkl"):
    """Сохраняет разделения датасета в файл"""
    split_data = {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'img_dir': img_dir,
        'split_params': {
            'test_size': 0.1,
            'val_size': 0.1,
            'random_state': 42
        }
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(split_data, f)
    print(f"Dataset splits saved to {save_path}")

def load_dataset_splits(save_path="dataset_splits.pkl"):
    """Загружает разделения датасета из файла"""
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            split_data = pickle.load(f)
        print(f"Dataset splits loaded from {save_path}")
        return split_data
    else:
        print(f"No saved splits found at {save_path}")
        return None

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_dir = "C:\\Users\\LevPe\\OneDrive\\Рабочий стол\\Innopolis\\3 course\\DML\\Project\\test\\images"
    splits_file = "dataset_splits.pkl"
    
    # Пытаемся загрузить сохраненные разделения
    split_data = load_dataset_splits(splits_file)
    
    if split_data is not None:
        # Используем сохраненные разделения
        train_df = split_data['train_df']
        val_df = split_data['val_df']
        test_df = split_data['test_df']
        img_dir = split_data['img_dir']  # На случай если путь изменился
        print("Using saved dataset splits")
    else:
        # Создаем новые разделения
        df = pd.read_csv("labels-for-images.csv")
        train_val_df, test_df = train_test_split(
            df, test_size=0.1, random_state=42, stratify=df['detect']
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.1, random_state=42, stratify=train_val_df['detect']
        )
        
        # Сохраняем разделения для будущего использования
        save_dataset_splits(train_df, val_df, test_df, img_dir, splits_file)
        print("Created new dataset splits and saved them")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Создаем датасеты и даталоадеры
    train_dataset = FireSmokeDataset(train_df, img_dir, transform=transform)
    val_dataset = FireSmokeDataset(val_df, img_dir, transform=transform)
    test_dataset = FireSmokeDataset(test_df, img_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Модель и оптимизатор
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FireSmokeDetectorDualHead(
        num_classes=3, 
        detection_threshold_fire=0.5, 
        detection_threshold_smoke=0.5
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Обучение
    train_dual_head(
        model=model,
        optimizer=optimizer,
        loss_fn=dual_head_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=5,
        ckpt_path="best_dual_head.pt",
        experiment_name="fire_smoke_dual_head"
    )