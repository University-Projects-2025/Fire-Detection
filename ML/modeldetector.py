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
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle

class FireSmokeDataset(Dataset):
    """Custom PyTorch Dataset for fire/smoke detection with bounding boxes"""
    def __init__(self, dataframe, img_dir, transform=None):
        """
        Args:
            dataframe: DataFrame containing image metadata and labels
            img_dir: Directory containing the images
            transform: Image transformations to apply
        """
        self.data = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets a single sample from the dataset
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            tuple: (image_tensor, class_label, bounding_box_tensor)
        """
        row = self.data.iloc[idx]
        # Construct image path from filename
        img_path = f"{self.img_dir}/{row['filename']}.jpg"
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            
        label = int(row['detect'])
        
        # Handle bounding boxes: zero bbox for class 2 (no fire/smoke)
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
    """Custom Layer Normalization implementation"""
    def __init__(self, normalized_shape, eps=1e-6):
        """
        Args:
            normalized_shape: Shape of the normalized dimension
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # Learnable scale parameter
        self.bias = nn.Parameter(torch.zeros(normalized_shape))   # Learnable shift parameter
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """Apply layer normalization to input tensor"""
        u = x.mean(-1, keepdim=True)  # Mean along last dimension
        s = (x - u).pow(2).mean(-1, keepdim=True)  # Variance
        x = (x - u) / torch.sqrt(s + self.eps)  # Normalize
        x = self.weight * x + self.bias  # Scale and shift
        return x

class Block(nn.Module):
    """Basic building block of the neural network architecture"""
    def __init__(self, dim):
        """
        Args:
            dim: Dimension of input features
        """
        super().__init__()
        # Depthwise convolution - processes each channel separately
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        # Pointwise convolutions for channel mixing
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Expand channels by 4x
        self.act = nn.GELU()  # Activation function
        self.pwconv2 = nn.Linear(4 * dim, dim)  # Compress channels back to original
        self.gamma = nn.Parameter(torch.ones((dim,)) * 1e-6)  # Residual scaling parameter

    def forward(self, x):
        """Forward pass with residual connection"""
        input = x  # Save input for residual connection
        x = self.dwconv(x)
        # Change dimension order for layer norm (B,C,H,W) -> (B,H,W,C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Channel mixing with expansion and compression
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x  # Scale residual connection
        # Restore dimension order (B,H,W,C) -> (B,C,H,W)
        x = x.permute(0, 3, 1, 2)
        return input + x  # Residual connection

class FireSmokeDetectorDualHead(nn.Module):
    """Dual-head neural network for fire/smoke classification and bounding box detection"""
    def __init__(self, num_classes=3, dropout_rate=0.3):
        """
        Args:
            num_classes: Number of output classes (smoke, fire, no_fire)
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        # Feature dimensions at each stage of the network
        dims = [96, 192, 384, 768]
        # Number of blocks at each stage
        depths = [3, 3, 9, 3]
        
        # Initial feature extraction - patch embedding
        self.stem = nn.Conv2d(3, dims[0], kernel_size=4, stride=4)
        self.stem_norm = LayerNorm(dims[0], eps=1e-6)
        
        # Four stages of processing with downsampling between stages
        self.stage1 = nn.Sequential(*[Block(dim=dims[0]) for _ in range(depths[0])])
        self.downsample1 = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),  # 2x downsampling
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
        
        # Global feature processing for classification
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        # Classification head for fire/smoke/no_fire prediction
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  # Regularization
            nn.Linear(dims[-1], 512),  # Feature projection
            nn.BatchNorm1d(512),       # Batch normalization
            nn.GELU(),                 # Activation
            nn.Dropout(dropout_rate * 0.7),  # Reduced dropout in deeper layers
            nn.Linear(512, 256),       # Further compression
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, num_classes)  # Final classification layer
        )
        
        # Spatial attention for focusing on relevant regions for bbox detection
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dims[-1], 256, kernel_size=1),  # Channel reduction
            nn.GELU(),
            nn.Conv2d(256, 1, kernel_size=1),         # Single channel attention map
            nn.Sigmoid()  # Attention weights between 0-1
        )
        
        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(dims[-1], 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((7, 7)),  # Fixed size features for FC layers
            nn.Flatten(),                   # Convert to 1D vector
            nn.Linear(128 * 7 * 7, 512),   # Fully connected layer
            nn.GELU(),
            nn.Linear(512, 4)  # Output: [x_center, y_center, width, height]
        )
        
        self.sigmoid = nn.Sigmoid()  # For bounding box coordinate normalization to [0,1]

    def _compute_bbox(self, features):
        """
        Compute bounding boxes only for the provided features
        
        Args:
            features: Input feature tensor
            
        Returns:
            torch.Tensor: Predicted bounding boxes in normalized coordinates
        """
        attention_weights = self.spatial_attention(features)  # Compute attention map
        attended_features = features * attention_weights      # Apply attention
        bbox_predictions = self.bbox_head(attended_features)  # Predict bbox coordinates
        return self.sigmoid(bbox_predictions)                 # Normalize to [0,1]

    def forward(self, x, targets=None):
        """
        Forward pass through the entire network
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            targets: Ground truth labels (required during training)
            
        Returns:
            tuple: (classification_logits, bbox_predictions, detection_mask)
        """
        # Feature extraction backbone (always executed)
        features = self.stem(x)  # Initial patch embedding
        # Apply layer norm with dimension permutation
        features = self.stem_norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Process through four stages with downsampling
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
        
        # Classification head (always executed)
        global_features = features.mean([-2, -1])  # Global average pooling
        global_features = self.norm(global_features)
        classification_logits = self.classifier(global_features)
        classification_probs = F.softmax(classification_logits, dim=1)
        
        # Bounding box - CONDITIONAL execution for efficiency
        batch_size = x.size(0)
        device = x.device
        # Initialize with zeros - will be filled conditionally
        bbox_predictions = torch.zeros(batch_size, 4, device=device)
        detection_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        if self.training:
            # During training - ALWAYS compute bbox for fire/smoke classes
            if targets is None:
                raise ValueError("Targets must be provided during training")
            
            # Create mask for fire (class 1) and smoke (class 0) samples
            train_detection_mask = (targets == 0) | (targets == 1)
            detection_mask = train_detection_mask
            
            if train_detection_mask.any():
                # Compute bbox ONLY for fire/smoke samples
                features_for_bbox = features[train_detection_mask]
                bbox_for_detections = self._compute_bbox(features_for_bbox)
                bbox_predictions[train_detection_mask] = bbox_for_detections
        else:
            # During inference - compute ONLY for predicted fire/smoke
            predicted_classes = torch.argmax(classification_probs, dim=1)
            detection_mask = (predicted_classes == 0) | (predicted_classes == 1)
            
            if detection_mask.any():
                # Compute bbox ONLY for detected fire/smoke samples
                features_for_bbox = features[detection_mask]
                bbox_for_detections = self._compute_bbox(features_for_bbox)
                bbox_predictions[detection_mask] = bbox_for_detections
        
        return classification_logits, bbox_predictions, detection_mask

def dual_head_loss(classification_logits, bbox_predictions, targets, bbox_targets, detection_mask, 
                  cls_weight=1.0, bbox_weight=1.0):
    """
    Combined loss function for classification and bounding box regression
    
    Args:
        classification_logits: Raw class scores from model
        bbox_predictions: Predicted bounding boxes
        targets: Ground truth class labels
        bbox_targets: Ground truth bounding boxes
        detection_mask: Which samples should have bounding boxes
        cls_weight: Weight for classification loss
        bbox_weight: Weight for bounding box loss
        
    Returns:
        tuple: (total_loss, classification_loss, bbox_loss)
    """
    # Classification loss (always computed for all samples)
    cls_loss = F.cross_entropy(classification_logits, targets)
    
    # Bounding box loss (only for fire/smoke samples with valid bboxes)
    bbox_loss = torch.tensor(0.0, device=classification_logits.device)
    
    # Mask for training bbox regression: only fire/smoke classes
    train_detection_mask = (targets == 0) | (targets == 1)
    
    if train_detection_mask.any():
        # Smooth L1 loss for bounding box regression (less sensitive to outliers than L2)
        bbox_loss = F.smooth_l1_loss(
            bbox_predictions[train_detection_mask], 
            bbox_targets[train_detection_mask]
        )
    
    # Combined weighted loss
    total_loss = cls_weight * cls_loss + bbox_weight * bbox_loss
    
    return total_loss, cls_loss, bbox_loss

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
    """
    Training loop for the dual-head fire/smoke detector
    
    Args:
        model: The neural network model to train
        optimizer: Optimization algorithm
        loss_fn: Loss function
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Computation device (cpu/cuda)
        epochs: Number of training epochs
        ckpt_path: Path to save best model
        experiment_name: MLflow experiment name
    """
    # Setup MLflow experiment tracking
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()
    
    # Training state variables
    best_accuracy = 0.0
    early_stop_patience = 3
    early_stop_counter = 0
    model.to(device)
    
    with mlflow.start_run():
        # Log training parameters
        mlflow.log_params({
            "epochs": epochs,
            "device": device,
            "optimizer": optimizer.__class__.__name__,
            "batch_size": train_loader.batch_size,
            "early_stop_patience": early_stop_patience
        })
        
        # Training loop over epochs
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loop = tqdm(
                enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"
            )
            total_train_loss = 0.0
            total_cls_loss = 0.0
            total_bbox_loss = 0.0
            
            for i, data in train_loop:
                inputs, labels, bbox_targets = data
                # Move data to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                bbox_targets = bbox_targets.to(device)
                
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                
                # Forward pass with targets for training
                classification_logits, bbox_predictions, detection_mask = model(inputs, targets=labels)
                
                # Compute loss
                total_loss, cls_loss, bbox_loss = loss_fn(
                    classification_logits, 
                    bbox_predictions, 
                    labels, 
                    bbox_targets, 
                    detection_mask,
                    cls_weight=1.0, 
                    bbox_weight=1.0
                )
                
                # Backward pass and optimization
                total_loss.backward()
                optimizer.step()
                
                # Accumulate losses for logging
                total_train_loss += total_loss.item()
                total_cls_loss += cls_loss.item()
                total_bbox_loss += bbox_loss.item() if bbox_loss > 0 else 0.0
                
                # Update progress bar with current losses
                train_loop.set_postfix({
                    "total_loss": total_loss.item(),
                    "cls_loss": cls_loss.item(),
                    "bbox_loss": bbox_loss.item() if bbox_loss > 0 else 0.0
                })
            
            # Validation phase
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
                    
                    # Model predictions (without targets for inference)
                    classification_logits, bbox_predictions, detection_mask = model(inputs)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(classification_logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Validation loss
                    total_loss, _, _ = loss_fn(
                        classification_logits, bbox_predictions, labels, bbox_targets, detection_mask
                    )
                    val_loss += total_loss.item()
                    
                    # Store predictions for metrics calculation
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    val_loop.set_postfix({"val_acc": val_correct / val_total})
            
            # Calculate epoch metrics
            val_accuracy = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            avg_train_loss = total_train_loss / len(train_loader)
            avg_cls_loss = total_cls_loss / len(train_loader)
            avg_bbox_loss = total_bbox_loss / len(train_loader)
            
            # Additional classification metrics
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # Log metrics to MLflow
            metrics = {
                "train_loss": avg_train_loss,
                "train_cls_loss": avg_cls_loss,
                "train_bbox_loss": avg_bbox_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1
            }
            mlflow.log_metrics(metrics, step=epoch)
            
            # Early stopping and model checkpointing
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                early_stop_counter = 0
                torch.save(model.state_dict(), ckpt_path)
                mlflow.log_artifact(ckpt_path)
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Print epoch summary
            print(f"Epoch {epoch}: "
                  f"Train Loss: {avg_train_loss:.4f} (cls: {avg_cls_loss:.4f}, bbox: {avg_bbox_loss:.4f}), "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                  f"F1: {f1:.4f}")
        
        # Final logging
        mlflow.log_metric("best_val_accuracy", best_accuracy)
        mlflow.pytorch.log_model(model, "model")
    
    print(f"Training completed. Best validation accuracy: {best_accuracy:.4f}")

def save_dataset_splits(train_df, val_df, test_df, img_dir, save_path="dataset_splits.pkl"):
    """
    Save dataset splits to file for reproducible experiments
    
    Args:
        train_df: Training dataset DataFrame
        val_df: Validation dataset DataFrame  
        test_df: Test dataset DataFrame
        img_dir: Image directory path
        save_path: Path to save the splits
    """
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
    """
    Load previously saved dataset splits
    
    Args:
        save_path: Path to saved splits file
        
    Returns:
        dict: Loaded dataset splits or None if file doesn't exist
    """
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            split_data = pickle.load(f)
        print(f"Dataset splits loaded from {save_path}")
        return split_data
    else:
        print(f"No saved splits found at {save_path}")
        return None

if __name__ == "__main__":
    # Image preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_dir = "C:\\Users\\LevPe\\OneDrive\\Рабочий стол\\Innopolis\\3 course\\DML\\Project\\test\\images"
    splits_file = "dataset_splits.pkl"
    
    # Load or create dataset splits
    split_data = load_dataset_splits(splits_file)
    
    if split_data is not None:
        # Use existing splits
        train_df = split_data['train_df']
        val_df = split_data['val_df']
        test_df = split_data['test_df']
        img_dir = split_data['img_dir']
        print("Using saved dataset splits")
    else:
        # Create new splits
        df = pd.read_csv("labels-for-images.csv")
        # Split into train/val/test with stratification
        train_val_df, test_df = train_test_split(
            df, test_size=0.1, random_state=42, stratify=df['detect']
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.1, random_state=42, stratify=train_val_df['detect']
        )
        
        # Save splits for future use
        save_dataset_splits(train_df, val_df, test_df, img_dir, splits_file)
        print("Created new dataset splits and saved them")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets and data loaders
    train_dataset = FireSmokeDataset(train_df, img_dir, transform=transform)
    val_dataset = FireSmokeDataset(val_df, img_dir, transform=transform)
    test_dataset = FireSmokeDataset(test_df, img_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Model setup with new architecture
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FireSmokeDetectorDualHead(num_classes=3, dropout_rate=0.3)
    
    # Optimizer with L2 regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    # Start training with new architecture
    train_dual_head(
        model=model,
        optimizer=optimizer,
        loss_fn=dual_head_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=15,
        ckpt_path="best_dual_head_v2.pt",  # New filename for version 2
        experiment_name="fire_smoke_dual_head_v2"  # New experiment name
    )


def create_final_confusion_matrix(model_path="best_dual_head_v2.pt", 
                                splits_file="dataset_splits.pkl",
                                experiment_name="confusion_matrix_final"):
    """
    Creates and logs confusion matrix for saved model
    
    Args:
        model_path: Path to saved model weights
        splits_file: Path to dataset splits file
        experiment_name: MLflow experiment name
        
    Returns:
        tuple: (confusion_matrix, true_labels, predictions)
    """
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FireSmokeDetectorDualHead(num_classes=3, dropout_rate=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load data
    split_data = load_dataset_splits(splits_file)
    if not split_data:
        print("Split file not found!")
        return
    
    test_df = split_data['test_df']
    img_dir = split_data['img_dir']
    
    # Transformations (same as during training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Test dataset and loader
    test_dataset = FireSmokeDataset(test_df, img_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Collect predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, bbox_targets in tqdm(test_loader, desc="Creating Confusion Matrix"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            classification_logits, bbox_predictions, detection_mask = model(inputs)
            _, predicted = torch.max(classification_logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ['Fire', 'Smoke', 'No Fire/Smoke']
    
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title('Confusion Matrix - Final Model on Test Set', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Logging to MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="final_confusion_matrix"):
        # Log the confusion matrix figure
        mlflow.log_figure(plt.gcf(), "confusion_matrix_final.png")
        
        # Log additional metrics
        from sklearn.metrics import classification_report, accuracy_score
        
        accuracy = accuracy_score(all_labels, all_preds)
        mlflow.log_metric("test_accuracy", accuracy)
        
        # Log the matrix itself as artifact
        np.save("confusion_matrix.npy", cm)
        mlflow.log_artifact("confusion_matrix.npy")
        
        # Classification report
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        import json
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("classification_report.json")
    
    plt.show()
    
    print(f"Confusion Matrix created and saved to MLflow!")
    print(f"Test set accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return cm, all_labels, all_preds

# Run after training completion
if __name__ == "__main__":
    # Your existing training code...
    
    # After training add:
    print("\n" + "="*50)
    print("Creating Confusion Matrix for trained model...")
    print("="*50)
    
    cm, true_labels, predictions = create_final_confusion_matrix()