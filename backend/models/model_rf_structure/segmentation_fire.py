import cv2
import numpy as np

class FireSegmnetation:
    def __init__(self, path):
        self.path = path
        self.scene = cv2.imread(path)

    def _threshold_rgb_fire(self, scene_rgb):
        """
        Pixel value thresholding for fire in RGB space
        Based on fire's characteristic red-orange-yellow colors
        """
        #extracting channels
        R = scene_rgb[:, :, 0] #red (0-255)
        G = scene_rgb[:, :, 1] #green (0-255)
        B = scene_rgb[:, :, 2] #blue (0-255)
        
        #creating empty mask
        fire_mask_rgb = np.zeros(scene_rgb.shape[:2], dtype=np.uint8)
        
        #rule1: red channel should be dominant (R > G > B)
        rule1 = (R > G) & (G > B)
        
        #rule2: red channel should be strong (R > threshold)
        rule2 = R > 140
        
        #rule3: color ratio constraints (fire pixels have R much larger than B)
        rule3= R > B * 1.5
        
        #rule4: avoid grayscale (color_saturation has to be high)
        color_saturation = (R - np.minimum(G, B)) / (R + 1e-6)
        rule4 = color_saturation > 0.3
        
        #rule combinations
        fire_mask_rgb[(rule1 & rule2 & rule3 & rule4)] = 255
        
        return fire_mask_rgb

    def _threshold_hsv_fire(self, scene_hsv):
        """
        Pixel value thresholding for fire in HSV space
        Based on hue, saturation, and value characteristics
        """
        #extracting HSV channels
        H = scene_hsv[:, :, 0]  #hue (0-179)
        S = scene_hsv[:, :, 1]  #saturation (0-255)
        V = scene_hsv[:, :, 2]  #value/brightness (0-255)
        
        #creating empty mask
        fire_mask_hsv = np.zeros(scene_hsv.shape[:2], dtype=np.uint8)
        
        #rule1: hue range for fire colors (red 0-10, orange 11-25, yellow 26-35)
        hue_rule = ((H >= 0) & (H <= 35))
        
        #rule2: high saturation (fire is highly saturated)
        saturation_rule = S > 100
        
        #rule3: high brightness/value (fire is bright)
        value_rule = V > 150
        
        #rule combinations
        fire_mask_hsv[(hue_rule & saturation_rule & value_rule)] = 255
        
        return fire_mask_hsv

    def combined_threshhold_fire(self):
        scene_hsv = cv2.cvtColor(self.scene, cv2.COLOR_BGR2HSV)
        scene_rgb = cv2.cvtColor(self.scene, cv2.COLOR_BGR2RGB)
        rgb_mask = self._threshold_rgb_fire(scene_rgb)
        hsv_mask = self._threshold_hsv_fire(scene_hsv)
        combined = cv2.bitwise_and(rgb_mask, hsv_mask)
        return combined

class FireIntensityAnalyzer:
    def __init__(self):
        
        #weights for intensity score
        self.weights = {
            'area': 0.4,
            'brightness': 0.3,
            'color_temperature': 0.2,
            'flicker_frequency': 0.1
        }
    
    def extract_intensity_features(self, image_rgb, fire_mask):
        """
        Extract all intensity features from fire regions
        
        Args:
            image_rgb: Original RGB image
            fire_mask: Binary mask where 255 indicates fire pixels
            
        Returns:
            Dictionary containing all intensity features
        """
        features = {}
        
        #converting to other color spaces
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        image_ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        
        #getting fire region coordinates
        fire_pixels = fire_mask == 255
        num_fire_pixels = np.sum(fire_pixels)
        
        if num_fire_pixels == 0:
            #no fire detected
            return {
                'area': 0,
                'brightness': 0,
                'color_temperature_rb': 0,
                'color_temperature_rg': 0,
                'color_temperature_br': 0,
                'intensity_score': 0,
                'has_fire': False
            }
        
        #feature1: area/size
        features['area'] = self._calculate_fire_area(fire_mask)
        
        #feature2: average brightness
        features['brightness'] = self._calculate_average_brightness(image_hsv, image_ycrcb, fire_pixels)
        
        #feature3: color temperature
        color_temp_features = self._calculate_color_temperature(image_rgb, fire_pixels)
        features.update(color_temp_features)
        
        #calculating overall intensity score
        features['intensity_score'] = self._calculate_intensity_score(features)
        features['has_fire'] = True
        
        return features
    
    def _calculate_fire_area(self, fire_mask):
        """
        Calculate fire area in pixels
        """
        return cv2.countNonZero(fire_mask)
    
    def _calculate_average_brightness(self, image_hsv, image_ycrcb, fire_pixels):
        """
        Calculate average brightness in fire regions using multiple color spaces
        """
        #using HSV value channel
        v_channel = image_hsv[:, :, 2]
        brightness_hsv = np.mean(v_channel[fire_pixels])
        
        #using YCbCr Y channel (luminance)
        y_channel = image_ycrcb[:, :, 0]
        brightness_y = np.mean(y_channel[fire_pixels])
        
        #combining both methods for robustness
        average_brightness = (brightness_hsv + brightness_y) / 2
        
        return average_brightness
    
    def _calculate_color_temperature(self, image_rgb, fire_pixels):
        """
        Calculate color temperature ratios in fire regions
        """
        R = image_rgb[:, :, 0][fire_pixels]
        G = image_rgb[:, :, 1][fire_pixels]
        B = image_rgb[:, :, 2][fire_pixels]
        
        #avoiding division by zero
        R = np.maximum(R, 1)
        G = np.maximum(G, 1)
        B = np.maximum(B, 1)
        
        #red-to-blue ratio (higher = hotter flames)
        red_blue_ratio = np.mean(R / B)
        
        #red-to-green ratio
        red_green_ratio = np.mean(R / G)
        
        #blue-to-red ratio (inverse temperature)
        blue_red_ratio = np.mean(B / R)
        
        return {
            'color_temperature_rb': red_blue_ratio,
            'color_temperature_rg': red_green_ratio,
            'color_temperature_br': blue_red_ratio
        }
    
    def _normalize_features(self, features):
        """
        Normalize features to 0-1 range for combination
        """
        normalized = {}
        
        #area normalization (assuming max 10000 pixels for fire region)
        normalized['area'] = min(features['area'] / 10000.0, 1.0)
        
        #brightness normalization (0-255 range)
        normalized['brightness'] = features['brightness'] / 255.0
        
        #color temperature normalization (red-blue ratio)
        normalized['color_temperature'] = min(features['color_temperature_rb'] / 5.0, 1.0)
        
        return normalized

    def _calculate_intensity_score(self, features):
        """
        Calculate overall fire intensity score using weighted combination
        """
        #normalizing features to similar scales
        normalized_features = self._normalize_features(features)
        
        #weighted combination
        score = (
            self.weights['area'] * normalized_features['area'] +
            self.weights['brightness'] * normalized_features['brightness'] +
            self.weights['color_temperature'] * normalized_features['color_temperature'])
        
        return min(score, 1.0)  #max score = 1

def classify_intensity_level(intensity_score):
    """
    Classify fire intensity into categories based on score
    """
    if intensity_score == 0:
        return "No Fire"
    elif intensity_score < 0.3:
        return "Low Intensity"
    elif intensity_score < 0.6:
        return "Medium Intensity"
    elif intensity_score < 0.8:
        return "High Intensity"
    else:
        return "Very High Intensity"

def create_intensity_analysis_image(frame, fire_mask):
    """
    Create an image with fire overlayed fire mask
    """
    display_frame = frame.copy()
    
    #overlay fire mask
    fire_overlay = display_frame.copy()
    fire_overlay[fire_mask == 255] = [255, 0, 0]
    cv2.addWeighted(fire_overlay, 0.6, display_frame, 0.7, 0, display_frame)
    
    return display_frame

def analyze_single_image(image_bgr, fire_mask):
    """
    Analyze fire intensity in a single image
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    analyzer = FireIntensityAnalyzer()
    features = analyzer.extract_intensity_features(image_rgb, fire_mask)
    intensity_level = classify_intensity_level(features['intensity_score'])
    
    print("FIRE INTENSITY ANALYSIS RESULTS:")
    print(f"Intensity Level: {intensity_level}")
    print(f"Overall Score: {features['intensity_score']:.3f}")
    print(f"Fire Area: {features['area']} pixels")
    print(f"Average Brightness: {features['brightness']:.1f}")
    print(f"Red/Blue Ratio: {features['color_temperature_rb']:.2f}")
    
    return features, intensity_level