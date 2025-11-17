import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

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
def load_trained_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Loads trained model for inference"""
    model = FireSmokeDetectorDualHead(num_classes=3)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model

def preprocess_image(image, transform=None):
    """Preprocesses PIL image or numpy array for model inference"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    if isinstance(image, Image.Image):
        processed_image = transform(image).unsqueeze(0)
    else:
        # Assume it's a numpy array (OpenCV image)
        image_rgb = Image.fromarray(image)
        processed_image = transform(image_rgb).unsqueeze(0)
    
    return processed_image

def predict_image(model, image, device="cpu"):
    """Run prediction on a single image"""
    processed_tensor = preprocess_image(image)
    processed_tensor = processed_tensor.to(device)
    
    with torch.no_grad():
        classification_logits, bbox_predictions, detection_mask = model(processed_tensor)
    
    # Get results
    probabilities = torch.softmax(classification_logits, dim=1)[0]
    predicted_class = torch.argmax(classification_logits, dim=1)[0].item()
    
    # Map class indices to names
    class_names = {0: "smoke", 1: "fire", 2: "clear"}
    prediction = class_names.get(predicted_class, "clear")
    
    bbox = bbox_predictions[0].cpu().numpy() if detection_mask[0].item() else None
    
    return {
        'class': predicted_class,
        'class_name': prediction,
        'probabilities': {
            'smoke': float(probabilities[0]),
            'fire': float(probabilities[1]),
            'clear': float(probabilities[2])
        },
        'bbox': bbox,
        'detected': bool(detection_mask[0].item())
    }