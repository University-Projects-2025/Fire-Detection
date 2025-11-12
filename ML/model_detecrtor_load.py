import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class LayerNorm(nn.Module):
    """Layer normalization for training stabilization"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # Scaling parameter
        self.bias = nn.Parameter(torch.zeros(normalized_shape))   # Shifting parameter
        self.eps = eps  # Small number for numerical stability
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # Calculate mean and standard deviation
        u = x.mean(-1, keepdim=True)  # Mean over last dimension
        s = (x - u).pow(2).mean(-1, keepdim=True)  # Variance
        # Normalization
        x = (x - u) / torch.sqrt(s + self.eps)
        # Scale and shift
        x = self.weight * x + self.bias
        return x

class Block(nn.Module):
    """Basic neural network architecture block"""
    def __init__(self, dim):
        super().__init__()
        # Depthwise convolution - convolution per channel separately
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)  # Normalization
        # Pointwise convolution 1 - feature expansion
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()  # Activation function
        # Pointwise convolution 2 - feature compression
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # Parameter for scaling residual connection
        self.gamma = nn.Parameter(torch.ones((dim,)) * 1e-6)

    def forward(self, x):
        input = x  # Save input for residual connection
        # Depthwise convolution
        x = self.dwconv(x)
        # Change dimension order for LayerNorm application
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Two sequential linear transformations with activation
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # Scaling and return to original dimension order
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        # Residual connection - add input to output
        return input + x

class FireSmokeDetectorDualHead(nn.Module):
    """Main model for fire and smoke detection with two heads"""
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super().__init__()
        # Feature dimensions at different stages
        dims = [96, 192, 384, 768]
        # Number of blocks at each stage
        depths = [3, 3, 9, 3]
        
        # Initial layer - input image transformation
        self.stem = nn.Conv2d(3, dims[0], kernel_size=4, stride=4)
        self.stem_norm = LayerNorm(dims[0], eps=1e-6)
        
        # Sequential processing stages with resolution reduction
        self.stage1 = nn.Sequential(*[Block(dim=dims[0]) for _ in range(depths[0])])
        self.downsample1 = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),  # 2x resolution reduction
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
        
        # Global features normalization
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        # Classification head - detects presence of fire/smoke
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  # Regularization to prevent overfitting
            nn.Linear(dims[-1], 512),
            nn.BatchNorm1d(512),  # Batch normalization
            nn.GELU(),  # Activation
            nn.Dropout(dropout_rate * 0.7),  # Smaller dropout
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, num_classes)  # Output: smoke, fire, no_fire
        )
        
        # Attention mechanism for improving bounding box detection
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dims[-1], 256, kernel_size=1),  # 1x1 convolution
            nn.GELU(),
            nn.Conv2d(256, 1, kernel_size=1),  # Attention map
            nn.Sigmoid()  # Normalization to [0, 1]
        )
        
        # Bounding box detection head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(dims[-1], 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((7, 7)),  # Fixed size pooling
            nn.Flatten(),  # Convert to vector
            nn.Linear(128 * 7 * 7, 512),
            nn.GELU(),
            nn.Linear(512, 4)  # Output: [x_center, y_center, width, height]
        )
        
        # Activation for bounding box coordinates in range [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through backbone network
        # Initial transformation
        features = self.stem(x)
        features = self.stem_norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Sequential processing stages
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
        # Global features via average pooling
        global_features = features.mean([-2, -1])
        global_features = self.norm(global_features)
        classification_logits = self.classifier(global_features)
        classification_probs = F.softmax(classification_logits, dim=1)  # Class probabilities
        
        # Detection mask determination (fire or smoke)
        predicted_classes = torch.argmax(classification_probs, dim=1)
        detection_mask = (predicted_classes == 0) | (predicted_classes == 1)  # smoke or fire
        
        # Bounding box detection head
        if self.training or detection_mask.any():
            # Apply attention mechanism
            attention_weights = self.spatial_attention(features)
            attended_features = features * attention_weights  # Weighted features
            
            # Bounding box prediction
            bbox_predictions = self.bbox_head(attended_features)
            bbox_predictions = self.sigmoid(bbox_predictions)  # Coordinate normalization
            
            # During inference, zero out bbox for non-detection cases
            if not self.training:
                bbox_predictions = bbox_predictions * detection_mask.unsqueeze(1).float()
        else:
            # If no detection - zero bounding boxes
            batch_size = x.size(0)
            bbox_predictions = torch.zeros(batch_size, 4, device=x.device)
        
        return classification_logits, bbox_predictions, detection_mask

def load_trained_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads a pre-trained model
    
    Args:
        model_path (str): Path to model weights file
        device (str): Device to load model on (cuda/cpu)
    
    Returns:
        FireSmokeDetectorDualHead: Loaded and prepared model
    """
    # Create model instance
    model = FireSmokeDetectorDualHead(num_classes=3)
    
    # Load weights and move to device
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Switch to inference mode
    
    print(f"Model loaded from {model_path}")
    return model

def preprocess_image(image_path, transform=None):
    """
    Preprocesses image for model input
    
    Args:
        image_path (str): Path to source image
        transform: Transformations to apply (if None - standard ones are used)
    
    Returns:
        tuple: (processed tensor, original image)
    """
    if transform is None:
        # Standard image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to fixed size
            transforms.ToTensor(),  # Convert to tensor [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                               std=[0.229, 0.224, 0.225])
        ])
    
    # Load and convert image
    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()  # Save original for visualization
    processed_image = transform(image).unsqueeze(0)  # Add batch dimension
    
    return processed_image, original_image

def draw_detection_results(image, bbox, class_name, confidence, ax=None):
    """
    Draws detection results on image
    
    Args:
        image (PIL.Image): Source image
        bbox (list): Bounding box coordinates [x_center, y_center, width, height]
        class_name (str): Class name
        confidence (float): Model confidence
        ax: Axes for drawing (if None - new one is created)
    
    Returns:
        matplotlib.axes.Axes: Axes with drawn results
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display source image
    ax.imshow(image)
    
    # If valid bounding box exists - draw it
    if bbox is not None and any(coord > 0 for coord in bbox):
        img_width, img_height = image.size
        x_center, y_center, width, height = bbox
        
        # Convert from center format to corner format
        x1 = (x_center - width/2) * img_width
        y1 = (y_center - height/2) * img_height
        x2 = (x_center + width/2) * img_width
        y2 = (y_center + height/2) * img_height
        
        # Create bounding box rectangle
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
    """
    Detects fire/smoke on single image and displays results
    
    Args:
        model: Trained detection model
        image_path (str): Path to input image
        device (str): Computation device
        show_result (bool): Whether to display visualization
    
    Returns:
        dict: Detection results with class, probabilities, and bounding box
    """
    # Load and preprocess image
    processed_tensor, original_image = preprocess_image(image_path)
    processed_tensor = processed_tensor.to(device)
    
    # Model prediction
    with torch.no_grad():
        classification_logits, bbox_predictions, detection_mask = model(processed_tensor)
    
    # Get class probabilities
    probabilities = torch.softmax(classification_logits, dim=1)[0]
    predicted_class = torch.argmax(classification_logits, dim=1)[0].item()
    
    # Class names mapping
    class_names = {0: "smoke", 1: "fire", 2: "no_fire"}
    
    # Get bounding box if detection occurred
    bbox = None
    if detection_mask[0].item():
        bbox = bbox_predictions[0].cpu().numpy()
        print(f"Detected bounding box: {bbox}")
    else:
        print("Fire/smoke not detected")
    
    # Print prediction information
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Probabilities: smoke={probabilities[0]:.3f}, fire={probabilities[1]:.3f}, no_fire={probabilities[2]:.3f}")
    
    # Display results if requested
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
        
        # Draw image
        ax.imshow(original_image)
        
        # Draw bounding box if exists
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
    """
    Processes multiple images and shows results in grid
    
    Args:
        model: Trained detection model
        image_paths (list): List of image paths
        device (str): Computation device
    
    Returns:
        list: Detection results for each image
    """
    n_images = len(image_paths)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    # Create subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    results = []
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        if i >= len(axes):
            break
            
        try:
            # Process image
            processed_tensor, original_image = preprocess_image(image_path)
            processed_tensor = processed_tensor.to(device)
            
            # Model prediction
            with torch.no_grad():
                classification_logits, bbox_predictions, detection_mask = model(processed_tensor)
            
            probabilities = torch.softmax(classification_logits, dim=1)[0]
            predicted_class = torch.argmax(classification_logits, dim=1)[0].item()
            
            class_names = {0: "smoke", 1: "fire", 2: "no_fire"}
            bbox = bbox_predictions[0].cpu().numpy() if detection_mask[0].item() else None
            
            # Draw result
            ax = axes[i]
            ax.imshow(original_image)
            
            # Determine color based on class
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
            
            # Set subplot title with class and probabilities
            ax.set_title(f"{class_names[predicted_class].upper()}\n"
                        f"fire: {probabilities[1]:.3f}, smoke: {probabilities[0]:.3f}", 
                        color=title_color, weight='bold')
            ax.axis('off')
            
            # Store results
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

if __name__ == "__main__":
    # Determine computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load trained model
    model_path = "best_dual_head_continued.pt"
    model = load_trained_model(model_path, device)
    
    # Detect fire/smoke on single image
    image_path = "C:\\Users\\LevPe\\OneDrive\\Рабочий стол\\Innopolis\\3 course\\DML\\Project\\test\\images\\fire453.jpg"
    result = detect_fire_smoke_on_image(model, image_path, device)
    
    # Print detailed results
    print("\n" + "="*50)
    print("DETECTION DETAILS:")
    print(f"Class: {result['class_name']}")
    print(f"Detected: {'Yes' if result['detected'] else 'No'}")
    if result['bbox'] is not None:
        print(f"Bounding Box: {result['bbox']}")
    print(f"Probabilities: smoke={result['probabilities'][0]:.3f}, "
          f"fire={result['probabilities'][1]:.3f}, "
          f"no_fire={result['probabilities'][2]:.3f}")