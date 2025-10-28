import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

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
        
        # Backbone (unchanged)
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
        
        # Head 1: Classifier (unchanged)
        self.classifier = nn.Linear(dims[-1], num_classes)
        
        # Head 2: Bounding box detector
        # Using features from stage4 for detection
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
        
        # Detection thresholds
        self.detection_threshold_fire = detection_threshold_fire
        self.detection_threshold_smoke = detection_threshold_smoke
        
        # Activation for bounding boxes
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
        
        # Head 1: Classification (ALWAYS computed first)
        global_features = features.mean([-2, -1])  # Global Average Pooling
        global_features = self.norm(global_features)
        classification_logits = self.classifier(global_features)
        classification_probs = F.softmax(classification_logits, dim=1)
        
        # Initialize outputs for detection
        batch_size = x.size(0)
        device = x.device
        bounding_boxes = torch.zeros(batch_size, 4, device=device)
        fire_smoke_detected_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # CRITICALLY IMPORTANT: classifier check BEFORE detector computation
        fire_probs = classification_probs[:, 1]  # class 1 = fire
        smoke_probs = classification_probs[:, 0]
        
        # Mask for detector activation (any fire threat)
        detection_mask = (fire_probs > self.detection_threshold_fire) | (smoke_probs > self.detection_threshold_smoke)
        
        # COMPUTE DETECTOR ONLY FOR THREATENING EXAMPLES
        if detection_mask.any():
            # Take only features for examples with fire threat
            threat_features = features[detection_mask]
            
            # Compute bounding boxes only for threatening examples
            detected_boxes = self.bbox_head(threat_features)
            detected_boxes = self.sigmoid(detected_boxes)  # Normalize to [0, 1]
            
            # Write results back
            bounding_boxes[detection_mask] = detected_boxes
            fire_smoke_detected_mask[detection_mask] = True
        
        return classification_logits, bounding_boxes, fire_smoke_detected_mask

    def set_detection_thresholds(self, fire_threshold=None, smoke_threshold=None):
        """Dynamically change detection thresholds"""
        if fire_threshold is not None:
            self.detection_threshold_fire = fire_threshold
        if smoke_threshold is not None:
            self.detection_threshold_smoke = smoke_threshold


def load_trained_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Loads trained model"""
    model = FireSmokeDetectorDualHead(
        num_classes=3, 
        detection_threshold_fire=0.5, 
        detection_threshold_smoke=0.5
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model

def preprocess_image(image_path, transform=None):
    """Preprocesses image"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()
    processed_image = transform(image).unsqueeze(0)  # Add batch dimension
    
    return processed_image, original_image

def draw_detection_results(image, bbox, class_name, confidence, ax=None):
    """Draws bounding box and class information on image"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Show original image
    ax.imshow(image)
    
    # If there is bounding box, draw it
    if bbox is not None and any(coord > 0 for coord in bbox):
        # Convert coordinates from [0,1] to pixels
        img_width, img_height = image.size
        x_center, y_center, width, height = bbox
        
        # Convert from center format to corner format
        x1 = (x_center - width/2) * img_width
        y1 = (y_center - height/2) * img_height
        x2 = (x_center + width/2) * img_width
        y2 = (y_center + height/2) * img_height
        
        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=3, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add text with class and confidence
        text = f"{class_name}: {confidence:.2f}"
        ax.text(x1, y1-10, text, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                fontsize=12, color='white', weight='bold')
    
    ax.axis('off')
    return ax

def detect_fire_smoke_on_image(model, image_path, device="cpu", show_result=True):
    """Detects fire/smoke on image and displays results"""
    
    # Load and preprocess image
    processed_tensor, original_image = preprocess_image(image_path)
    processed_tensor = processed_tensor.to(device)
    
    # Prediction
    with torch.no_grad():
        classification_logits, bbox_predictions, detection_mask = model(processed_tensor)
    
    # Get class probabilities
    probabilities = torch.softmax(classification_logits, dim=1)[0]
    predicted_class = torch.argmax(classification_logits, dim=1)[0].item()
    
    # Class names
    class_names = {0: "smoke", 1: "fire", 2: "no_fire"}
    
    # Get bounding box if something detected
    bbox = None
    if detection_mask[0].item():  # If detector activated
        bbox = bbox_predictions[0].cpu().numpy()
        print(f"Detected bounding box: {bbox}")
    else:
        print("Fire/smoke not detected")
    
    # Prediction information
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Probabilities: smoke={probabilities[0]:.3f}, fire={probabilities[1]:.3f}, no_fire={probabilities[2]:.3f}")
    
    # Display results
    if show_result:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Determine frame color based on class
        if predicted_class == 1:  # fire
            color = 'red'
            class_name = "FIRE"
        elif predicted_class == 0:  # smoke
            color = 'orange'
            class_name = "SMOKE"
        else:  # no fire
            color = 'green'
            class_name = "NO FIRE"
        
        # Draw image with bounding box
        ax.imshow(original_image)
        
        if bbox is not None and any(coord > 0 for coord in bbox):
            img_width, img_height = original_image.size
            x_center, y_center, width, height = bbox
            
            # Convert coordinates
            x1 = (x_center - width/2) * img_width
            y1 = (y_center - height/2) * img_height
            x2 = (x_center + width/2) * img_width
            y2 = (y_center + height/2) * img_height
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=4, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add information panel
            text = f"{class_name}\nConfidence: {probabilities[predicted_class]:.3f}"
            ax.text(x1, y1-15, text, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                    fontsize=14, color='white', weight='bold')
        
        ax.set_title(f"Fire/Smoke Detection Result: {class_name}", fontsize=16, weight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    return {
        'class': predicted_class,
        'class_name': class_names[predicted_class],
        'probabilities': probabilities.cpu().numpy(),
        'bbox': bbox,
        'detected': detection_mask[0].item()
    }

def batch_detect_images(model, image_paths, device="cpu"):
    """Processes multiple images and shows results in grid"""
    n_images = len(image_paths)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    results = []
    
    for i, image_path in enumerate(image_paths):
        if i >= len(axes):
            break
            
        try:
            # Process image
            processed_tensor, original_image = preprocess_image(image_path)
            processed_tensor = processed_tensor.to(device)
            
            with torch.no_grad():
                classification_logits, bbox_predictions, detection_mask = model(processed_tensor)
            
            probabilities = torch.softmax(classification_logits, dim=1)[0]
            predicted_class = torch.argmax(classification_logits, dim=1)[0].item()
            
            class_names = {0: "smoke", 1: "fire", 2: "no_fire"}
            bbox = bbox_predictions[0].cpu().numpy() if detection_mask[0].item() else None
            
            # Draw result
            ax = axes[i]
            ax.imshow(original_image)
            
            # Determine color
            if predicted_class == 1:  # fire
                color = 'red'
                title_color = 'red'
            elif predicted_class == 0:  # smoke
                color = 'orange'
                title_color = 'orange'
            else:  # no fire
                color = 'green'
                title_color = 'black'
            
            # Draw bounding box if exists
            if bbox is not None and any(coord > 0 for coord in bbox):
                img_width, img_height = original_image.size
                x_center, y_center, width, height = bbox
                
                x1 = (x_center - width/2) * img_width
                y1 = (y_center - height/2) * img_height
                x2 = (x_center + width/2) * img_width
                y2 = (y_center + height/2) * img_height
                
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=3, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
            
            ax.set_title(f"{class_names[predicted_class].upper()}\n"
                        f"fire: {probabilities[1]:.3f}, smoke: {probabilities[0]:.3f}", 
                        color=title_color, weight='bold')
            ax.axis('off')
            
            results.append({
                'image_path': image_path,
                'class': predicted_class,
                'class_name': class_names[predicted_class],
                'probabilities': probabilities.cpu().numpy(),
                'bbox': bbox,
                'detected': detection_mask[0].item()
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            axes[i].axis('off')
            axes[i].set_title("Error", color='red')
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Usage example
if __name__ == "__main__":
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_path = "best_dual_head.pt"  # path to your saved model
    model = load_trained_model(model_path, device)
    
    # Example for single image
    image_path = "C:\\Users\\LevPe\\OneDrive\\Рабочий стол\\Innopolis\\3 course\\DML\\Project\\test\\images\\fire453.jpg"  # replace with path to your image
    result = detect_fire_smoke_on_image(model, image_path, device)
    
    # Example for multiple images
    # image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    # results = batch_detect_images(model, image_paths, device)
    
    # Output result information
    print("\n" + "="*50)
    print("DETECTION DETAILS:")
    print(f"Class: {result['class_name']}")
    print(f"Detected: {'Yes' if result['detected'] else 'No'}")
    if result['bbox'] is not None:
        print(f"Bounding Box: {result['bbox']}")
    print(f"Probabilities: smoke={result['probabilities'][0]:.3f}, "
          f"fire={result['probabilities'][1]:.3f}, "
          f"no_fire={result['probabilities'][2]:.3f}")