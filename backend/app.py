from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import sys
import torch
import traceback
import logging
import os
import time

# Env variables for Yolo
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['ULTRALYTICS_UPDATE'] = '0'

# Add model_rf_structure to path
sys.path.append('models/model_rf_structure')
sys.path.append('models/convDualHead')

from ultralytics import YOLO
from model_cv import SmokeDetectionPipeline, SmokeTextureFeatureExtractor
from segmentation_fire import FireSegmnetation, analyze_single_image
from segmentation_smoke import SmokeSegmentation, analyze_single_image_smoke
from model_detector_load import load_trained_model, predict_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize models
pipeline = SmokeDetectionPipeline()
MODEL_PATH = "models/model_rf_structure/smoke_fire_3class_model_full_final.pkl"

YOLO_MODEL_PATH = "models/yolo/best.pt"
yolo_model = None

CONVNEXT_MODEL_PATH = "models/convDualHead/best_dual_head_continued.pt"
convnext_model = None

# Patch to use weights_only=False for Yolo
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, weights_only=None, **kwargs):
    """Patched torch.load that forces weights_only=False for .pt files"""
    if isinstance(f, str) and f.endswith('.pt'):
        # Force weights_only=False for Yolo model files
        return original_torch_load(f, map_location=map_location, weights_only=False, **kwargs)
    else:
        # Use default behavior for other files
        if weights_only is None:
            weights_only = True
        return original_torch_load(f, map_location=map_location, weights_only=weights_only, **kwargs)

# Apply the patch
torch.load = patched_torch_load
logger.info("Applied torch.load patch for YOLO model loading")

def load_yolo_model():
    """Load YOLO model with patched torch.load"""
    global yolo_model
    logger.info(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    if yolo_model:
        return True
    return False

def load_convnext_model():
    """Load ConvNeXt Dual-Head model"""
    global convnext_model
    
    try:        
        convnext_model = load_trained_model(CONVNEXT_MODEL_PATH)
        logger.info("ConvNeXt model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading ConvNeXt model: {e}")
        logger.error(f"ConvNeXt traceback: {traceback.format_exc()}")
        return False

def initialize_models():
    """Initialize all models"""
    global pipeline, yolo_model, convnext_model
    
    # Initialize RF model

    pipeline.load_pipeline(MODEL_PATH)
    logger.info("Random Forest model loaded successfully")

    # Initialize YOLO model
    yolo_success = load_yolo_model()
    
    # Initialize ConvNeXt model
    convnext_success = load_convnext_model()
    
    if yolo_success and convnext_success:
        logger.info("All models initialized successfully")
        return True
    else:
        logger.warning("Some models failed to load, but continuing with available models")
        return True

# Initialize models on startup
initialize_models()

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    except Exception as e:
        raise ValueError(f"Error converting base64 to image: {e}")

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    try:
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and then to base64
        pil_image = Image.fromarray(image)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        raise ValueError(f"Error converting image to base64: {e}")

@app.route('/api/detect', methods=['POST'])
def detect_smoke_fire():
    start_time = time.time()
    try:
        data = request.get_json()
        logger.info(f"Received detection request with model type: {data.get('model', 'rf')}")
        
        if not data or 'image' not in data:
            logger.error("No image data provided in request")
            return jsonify({"error": "No image data provided"}), 400
        
        model_type = data.get('model', 'rf')  # Default to RF model
        
        # Convert base64 to image
        logger.info("Converting base64 to image...")
        image_bgr = base64_to_image(data['image'])
        logger.info(f"Image converted successfully. Shape: {image_bgr.shape}")
        
        # Route to appropriate model
        if model_type == 'yolo':
            logger.info("Using YOLO model for detection")
            result = detect_with_yolo(image_bgr)
        elif model_type == 'convnext':
            logger.info("Using ConvNeXt model for detection")
            result = detect_with_convnext(image_bgr)
        else:
            logger.info("Using Random Forest model for detection")
            result = detect_with_rf(image_bgr)
        
        # Add timing information to the response
        if isinstance(result, tuple):
            response_data = result[0].get_json() if hasattr(result[0], 'get_json') else result[0]
            response_data['processing_time'] = round((time.time() - start_time) * 1000, 2)
            return jsonify(response_data), result[1] if len(result) > 1 else 200
        else:
            result['processing_time'] = round((time.time() - start_time) * 1000, 2)
            return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in detect_smoke_fire: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

def detect_with_rf(image_bgr):
    """Detection using Random Forest model with intensity analysis"""
    rf_start_time = time.time()
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (128, 128))
    
    # Make prediction
    class_name, _, _ = pipeline.predict_image_3class(image_resized)
    
    result = {
        "prediction": class_name,
        "model": "rf",
        "detailed_analysis": {},
        "model_inference_time": round((time.time() - rf_start_time) * 1000, 2)
    }
    
    # Perform detailed analysis based on prediction
    if class_name == "fire":
        fire_segmenter = FireSegmnetation("")
        fire_segmenter.scene = image_bgr
        combined_fire_mask = fire_segmenter.combined_threshhold_fire()
        feat, _ = analyze_single_image(image_bgr, combined_fire_mask)
        
        result["detailed_analysis"] = {
            "intensity_score": float(feat['intensity_score']),
        }
        
        # Create overlay image
        overlay_image = create_intensity_analysis_image(image_bgr, combined_fire_mask)
        result["overlay_image"] = image_to_base64(overlay_image)
        
    elif class_name == "smoke":
        smoke_segmenter = SmokeSegmentation("")
        smoke_segmenter.scene = image_bgr
        smoke_mask = smoke_segmenter.combined_threshhold_smoke()
        feat, _ = analyze_single_image_smoke(image_bgr, smoke_mask)
        
        result["detailed_analysis"] = {
            "intensity_score": float(feat['intensity_score']),
        }
        
        # Create overlay image
        overlay_image = create_smoke_analysis_image(image_bgr, smoke_mask)
        result["overlay_image"] = image_to_base64(overlay_image)
    
    result["original_image"] = image_to_base64(image_bgr)
    
    return result

def detect_with_yolo(image_bgr):
    """Detection using YOLO model"""
    logger.info("Starting YOLO detection...")
    
    if yolo_model is None:
        error_msg = "YOLO model not available"
        logger.error(error_msg)
        return {"error": error_msg}, 500
    
    try:
        # Ensure image is in the right format
        if image_bgr.dtype != np.uint8:
            image_bgr = image_bgr.astype(np.uint8)
        
        # Run YOLO prediction with timing
        inference_start = time.time()
        results = yolo_model.predict(image_bgr, conf=0.25, verbose=False)
        inference_time = round((time.time() - inference_start) * 1000, 2)
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                class_names = []
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    class_names.append(class_name)
                
                # Determine prediction
                if 'fire' in class_names:
                    prediction = 'fire'
                elif 'smoke' in class_names:
                    prediction = 'smoke'
                else:
                    prediction = 'clear'
            else:
                prediction = 'clear'
            
            # Create result image with proper RGB conversion
            try:
                result_image = result.plot()                
                overlay_b64 = image_to_base64(result_image)
            except Exception as e:
                logger.warning(f"Could not create result image: {e}")
                # Fallback to original image converted to RGB
                original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                overlay_b64 = image_to_base64(original_rgb)
            
            result_data = {
                "prediction": prediction,
                "model": "yolo",
                "detailed_analysis": {},
                "overlay_image": overlay_b64,
                "original_image": image_to_base64(image_bgr),
                "detection_count": len(boxes) if boxes else 0,
                "model_inference_time": inference_time
            }
            
            return result_data
        else:
            return {
                "prediction": "clear",
                "model": "yolo",
                "detailed_analysis": {},
                "original_image": image_to_base64(image_bgr),
                "detection_count": 0,
                "model_inference_time": inference_time
            }
            
    except Exception as e:
        error_msg = f"YOLO detection error: {str(e)}"
        logger.error(error_msg)
        logger.error(f"YOLO traceback: {traceback.format_exc()}")
        return {"error": error_msg}, 500

def detect_with_convnext(image_bgr):
    """Detection using ConvNeXt Dual-Head model"""
    logger.info("Starting ConvNeXt detection...")
    
    if convnext_model is None:
        error_msg = "ConvNeXt model not available"
        logger.error(error_msg)
        return {"error": error_msg}, 500
    
    try:
        inference_start = time.time()
        
        # Convert OpenCV image to PIL
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Get prediction
        result = predict_image(convnext_model, pil_image)
        inference_time = round((time.time() - inference_start) * 1000, 2)
        
        # Convert numpy arrays to Python native types for JSON serialization
        bbox = None
        if result['detected'] and result['bbox'] is not None:
            # Convert numpy array to Python list
            bbox = result['bbox'].tolist() if hasattr(result['bbox'], 'tolist') else result['bbox']
        
        # Create visualization
        overlay_image = create_convnext_visualization(
            image_bgr, 
            result['bbox'],
            result['detected'], 
            result['class_name'], 
            result['probabilities']
        )
        
        result_data = {
            "prediction": result['class_name'],
            "model": "convnext",
            "detailed_analysis": {
                "probabilities": result['probabilities'],
                "detected": result['detected'],
                "bounding_box": bbox
            },
            "overlay_image": image_to_base64(overlay_image),
            "original_image": image_to_base64(image_bgr),
            "model_inference_time": inference_time
        }
        
        return result_data
        
    except Exception as e:
        error_msg = f"ConvNeXt detection error: {str(e)}"
        logger.error(error_msg)
        logger.error(f"ConvNeXt traceback: {traceback.format_exc()}")
        return {"error": error_msg}, 500

def create_convnext_visualization(frame, bbox, detected, prediction, probabilities):
    """Create visualization for ConvNeXt model results"""
    display_frame = frame.copy()
    
    if detected and bbox is not None and any(coord > 0 for coord in bbox):
        # Convert normalized coordinates to pixel coordinates
        img_height, img_width = display_frame.shape[:2]
        x_center, y_center, width, height = bbox
        
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        x2 = int((x_center + width/2) * img_width)
        y2 = int((y_center + height/2) * img_height)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Choose color based on prediction
        if prediction == "fire":
            color = (0, 0, 255)
        elif prediction == "smoke":
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
        
        # Add label
        confidence = probabilities.get(prediction, 0.5)
        label = f"{prediction.upper()} ({confidence:.2f})"
        cv2.putText(display_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return display_frame

def create_intensity_analysis_image(frame, fire_mask):
    """Create fire analysis overlay image"""
    display_frame = frame.copy()
    fire_overlay = display_frame.copy()
    fire_overlay[fire_mask == 255] = [255, 0, 0]
    cv2.addWeighted(fire_overlay, 0.6, display_frame, 0.7, 0, display_frame)
    return display_frame

def create_smoke_analysis_image(frame, smoke_mask):
    """Create smoke analysis overlay image"""
    display_frame = frame.copy()
    smoke_overlay = display_frame.copy()
    smoke_overlay[smoke_mask == 255] = [0, 0, 255]
    cv2.addWeighted(smoke_overlay, 0.4, display_frame, 0.7, 0, display_frame)
    return display_frame

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=False)