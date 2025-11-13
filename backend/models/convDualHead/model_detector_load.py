import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

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
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super().__init__()
        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]
        
        # Backbone
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
        
        # Global features for classification
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(dims[-1], 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )
        
        # Attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dims[-1], 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Bounding box detector
        self.bbox_head = nn.Sequential(
            nn.Conv2d(dims[-1], 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.GELU(),
            nn.Linear(512, 4)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward through backbone
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
        
        # Classification head
        global_features = features.mean([-2, -1])
        global_features = self.norm(global_features)
        classification_logits = self.classifier(global_features)
        classification_probs = F.softmax(classification_logits, dim=1)
        
        # Detection logic
        predicted_classes = torch.argmax(classification_probs, dim=1)
        detection_mask = (predicted_classes == 0) | (predicted_classes == 1)  # smoke or fire
        
        if self.training or detection_mask.any():
            attention_weights = self.spatial_attention(features)
            attended_features = features * attention_weights
            
            bbox_predictions = self.bbox_head(attended_features)
            bbox_predictions = self.sigmoid(bbox_predictions)
            
            if not self.training:
                bbox_predictions = bbox_predictions * detection_mask.unsqueeze(1).float()
        else:
            batch_size = x.size(0)
            bbox_predictions = torch.zeros(batch_size, 4, device=x.device)
        
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