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
        """
        dataframe: pd.DataFrame с колонками filename и detect
        img_dir: директория с изображениями
        """
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
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

df = pd.read_csv("ML/labels-for-images.csv")
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['detect'])
train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42, stratify=train_val_df['detect'])

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")



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

class FireSmokeDetector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]
        
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
        
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stem_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        x = self.stage1(x)
        
        x = self.downsample1[0](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.downsample1[1](x)
        
        x = self.stage2(x)
        
        x = self.downsample2[0](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.downsample2[1](x)
        
        x = self.stage3(x)
        
        x = self.downsample3[0](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.downsample3[1](x)
        
        x = self.stage4(x)
        
        x = x.mean([-2, -1])
        
        x = self.norm(x)
        x = self.head(x)
        
        return x
def load_training_img(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def predict(model, image_tensor, device="cpu"):
    model.eval()
    model.to(device)
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()
    return pred_class, confidence

def train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    writer=None,
    epochs=1,
    device="cpu",
    ckpt_path="best.pt",
    experiment_name="pytorch_experiment",
    run_name=None
):
    mlflow.set_experiment(experiment_name)
    
    mlflow.pytorch.autolog()
    
    best = 0.0
    model.to(device)
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": epochs,
            "device": device,
            "optimizer": optimizer.__class__.__name__,
            "loss_function": loss_fn.__class__.__name__,
            "train_dataset_size": len(train_loader.dataset),
            "val_dataset_size": len(val_loader.dataset),
            "batch_size": train_loader.batch_size
        })
        
        for key, value in optimizer.param_groups[0].items():
            if key != 'params':
                mlflow.log_param(f"optimizer_{key}", value)
        
        for epoch in range(epochs):
            train_loop = tqdm(
                enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch}"
            )
            model.train()
            train_loss = 0.0
            
            for i, data in train_loop:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_loop.set_postfix({"loss": loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            if writer:
                writer.add_scalar("Loss/train", avg_train_loss, epoch)
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            
           
            correct = 0
            total = 0
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                model.eval()
                val_loop = tqdm(enumerate(val_loader, 0), total=len(val_loader), desc="Val")
                for i, data in val_loop:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    val_loop.set_postfix({"acc": correct / total})

                val_accuracy = correct / total
                avg_val_loss = val_loss / len(val_loader)
                
                
                cm = confusion_matrix(all_labels, all_preds)
                print("Confusion Matrix:")
                print(cm)
                
                

                
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
                
                if writer:
                    writer.add_scalar("Loss/val", avg_val_loss, epoch)
                    writer.add_scalar("Accuracy/val", val_accuracy, epoch)
                
                mlflow.log_metrics({
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy
                }, step=epoch)
                
                
                try:
                    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
                    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
                    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                    
                    mlflow.log_metrics({
                        "val_precision": precision,
                        "val_recall": recall,
                        "val_f1": f1
                    }, step=epoch)
                except Exception as e:
                    print(f"Could not calculate additional metrics: {e}")

                if val_accuracy > best:
                    torch.save(model.state_dict(), ckpt_path)
                    
                    best = val_accuracy
                    print(f"New best model saved with accuracy: {best:.4f}")

            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        mlflow.log_metrics({
            "best_val_accuracy": best
        })
        
        mlflow.pytorch.log_model(model, "model")
        
        model_info = {
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "best_accuracy": best
        }
        with open("model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        mlflow.log_artifact("model_info.json")
        
        if os.path.exists("model_info.json"):
            os.remove("model_info.json")
    
    print(f"Training completed. Best validation accuracy: {best:.4f}")

if __name__ == "__main__":
    img_dir = "C:/Lev/Project_PMLDL/Fire-Detection/ML/test/images"

    train_dataset = FireSmokeDataset(train_df, img_dir, transform=transform)
    val_dataset   = FireSmokeDataset(val_df, img_dir, transform=transform)
    test_dataset  = FireSmokeDataset(test_df, img_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FireSmokeDetector(num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train(model, optimizer, loss_fn, train_loader, val_loader, device=device, epochs=5)