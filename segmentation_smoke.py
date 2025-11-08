import cv2
import numpy as np
import matplotlib.pyplot as plt

class SmokeSegmentation:
    def __init__(self, path):
        self.path = path
        self.scene = cv2.imread(path)
    
    def _threshold_rgb_smoke(self, scene_rgb):
        """
        Pixel value thresholding for smoke in RGB space
        Based on smoke's characteristic grayish-white colors
        """
        #extracting channels
        R = scene_rgb[:, :, 0]
        G = scene_rgb[:, :, 1]
        B = scene_rgb[:, :, 2]
        
        #creating empty mask
        smoke_mask_rgb = np.zeros(scene_rgb.shape[:2], dtype=np.uint8)
        
        #rule1: smoke colors are typically desaturated (have similar RGB values)
        #so all channels should be relatively close to each other
        max_channel = np.maximum(np.maximum(R, G), B)
        min_channel = np.minimum(np.minimum(R, G), B)
        rule1 = (max_channel - min_channel) < 200
        
        #rule2: smoke is typically bright but not pure white
        rule2 = (max_channel > 100) & (max_channel < 240)
        
        #rule3: smoke pixels should have moderate to high values in all channels
        rule3 = (R > 60) & (G > 60) & (B > 60)
        
        #rule combinations
        smoke_mask_rgb[(rule1 & rule2 & rule3)] = 255
        
        return smoke_mask_rgb

    def _threshold_ycrcb_smoke(self, scene_ycrcb):
        """
        Smoke detection in YCrCb color space
        """
        #extract channels
        Y = scene_ycrcb[:, :, 0]   #luminance (0-255)
        Cr = scene_ycrcb[:, :, 1]  #chrominance red (0-255)
        Cb = scene_ycrcb[:, :, 2]  #chrominance blue (0-255)
        
        #creating empty mask
        smoke_mask = np.zeros(scene_ycrcb.shape[:2], dtype=np.uint8)
        
        #smoke has specific chrominance characteristics
        #rule1: values 110-175 exclude green objects (Cr < 110) and bright red/fire (Cr > 175)
        rule1 = (Cr > 110) & (Cr < 175)

        #rule2: values 90-135 exclude: yellowish objects (Cb < 90) and blue sky/water (Cb > 135)
        rule2 = (Cb > 90) & (Cb < 135)

        #rule3: Y (Luminance/brightness) range for smoke
        # values 80-220 exclude dark shadows/objects (Y < 80) and bright reflections/pure white (Y > 220)
        rule3 = (Y > 80) & (Y < 220)

        #rule4: chrominance similarity (smoke is desaturated - red and blue components are similar)
        rule4 = np.abs(Cr - Cb) < 250

        #rule combinations
        smoke_mask[(rule1 & rule2 & rule3 & rule4)] = 255
        
        return smoke_mask

    def combined_threshhold_smoke(self):
        scene_rgb = cv2.cvtColor(self.scene, cv2.COLOR_BGR2RGB)
        scene_ycrcb = cv2.cvtColor(self.scene, cv2.COLOR_BGR2YCrCb)
        rgb_mask = self._threshold_rgb_smoke(scene_rgb)
        ycrcb_mask = self._threshold_ycrcb_smoke(scene_ycrcb)
        combined = cv2.bitwise_and(rgb_mask, ycrcb_mask)
        return combined

class SmokeIntensityAnalyzer:
    def __init__(self):
        
        #weights for intensity score
        self.weights = {
            'area': 0.3,
            'density': 0.4,      #more important for smoke than brightness
            'color_purity': 0.2  #how "gray" the smoke is
        }
    
    def extract_intensity_features(self, image_rgb, smoke_mask):
        """
        Extract all intensity features from smoke regions
        
        Args:
            image_rgb: Original RGB image
            smoke_mask: Binary mask where 255 indicates smoke pixels
            
        Returns:
            Dictionary containing all smoke intensity features
        """
        features = {}
        
        #converting to other color spaces
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        image_ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        
        # getting smoke region coordinates
        smoke_pixels = smoke_mask == 255
        num_smoke_pixels = np.sum(smoke_pixels)
        
        if num_smoke_pixels == 0:
            #if no smoke detected
            return {
                'area': 0,
                'density': 0,
                'color_purity': 0,
                'opacity': 0,
                'intensity_score': 0,
                'has_smoke': False
            }
        
        #feature 1: area/size
        features['area'] = self._calculate_smoke_area(smoke_mask)
        
        #feature 2: smoke density
        features['density'] = self._calculate_smoke_density(image_rgb, image_hsv, image_ycrcb, smoke_pixels)
        
        #feature 3: color purity (how gray/desaturated the smoke is)
        features['color_purity'] = self._calculate_color_purity(image_rgb, smoke_pixels)
        
        #feature 4: opacity (how much smoke obscures background)
        features['opacity'] = self._calculate_opacity(image_rgb, smoke_pixels)
        
        #calculating intensity score
        features['intensity_score'] = self._calculate_intensity_score(features)
        features['has_smoke'] = True
        
        return features
    
    def _calculate_smoke_area(self, smoke_mask):
        """
        Calculate smoke area in pixels
        """
        return cv2.countNonZero(smoke_mask)
    
    def _calculate_smoke_density(self, image_rgb, image_hsv, image_ycrcb, smoke_pixels):
        """
        Calculate smoke density using multiple metrics
        """
        #method 1: texture complexity (dense smoke has more texture)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        smoke_region_gray = gray[smoke_pixels]
        
        if len(smoke_region_gray) > 1:
            texture_complexity = np.std(smoke_region_gray)
        else:
            texture_complexity = 0
        
        #method 2: edge density (dense smoke has fewer sharp edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density_in_smoke = np.mean(edges[smoke_pixels]) / 255.0
        
        #method 3: color consistency (dense smoke is more uniform)
        r_channel = image_rgb[:, :, 0][smoke_pixels]
        g_channel = image_rgb[:, :, 1][smoke_pixels]
        b_channel = image_rgb[:, :, 2][smoke_pixels]
        
        color_variation = (np.std(r_channel) + np.std(g_channel) + np.std(b_channel)) / 3
        
        #combine metrics - dense smoke has medium texture, low edge density, low color variation
        density_score = (
            min(texture_complexity / 30.0, 1.0) * 0.4 +
            (1.0 - edge_density_in_smoke) * 0.4 +
            max(0, 1.0 - color_variation / 40.0) * 0.2
        )
        
        return min(density_score, 1.0)
    
    def _calculate_color_purity(self, image_rgb, smoke_pixels):
        """
        Calculate how "pure" the smoke color is (how close to ideal gray)
        """
        R = image_rgb[:, :, 0][smoke_pixels]
        G = image_rgb[:, :, 1][smoke_pixels]
        B = image_rgb[:, :, 2][smoke_pixels]
        
        #calculating color balance (ideal smoke has R ≈ G ≈ B)
        mean_r, mean_g, mean_b = np.mean(R), np.mean(G), np.mean(B)
        
        #calculating deviation from perfect gray
        max_channel = max(mean_r, mean_g, mean_b)
        min_channel = min(mean_r, mean_g, mean_b)
        
        if max_channel > 0:
            color_deviation = (max_channel - min_channel) / max_channel
            # 1.0 = perfect gray, 0.0 = colorful
            color_purity = 1.0 - color_deviation
        else:
            color_purity = 0
        
        return color_purity
    
    def _calculate_opacity(self, image_rgb, smoke_pixels):
        """
        Calculate smoke opacity (how much it obscures background)
        """
        #converting to LAB color space for better perception
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel = image_lab[:, :, 0]  # Lightness channel
        
        #smoke typically reduces contrast and lightness
        smoke_lightness = l_channel[smoke_pixels]
        
        if len(smoke_lightness) > 0:
            #thick smoke has lower and more uniform lightness
            lightness_std = np.std(smoke_lightness)
            lightness_mean = np.mean(smoke_lightness)
            
            #opacity is higher when lightness is low and uniform
            opacity = (
                (1.0 - lightness_mean / 255.0) * 0.7 +
                max(0, 1.0 - lightness_std / 50.0) * 0.3
            )
        else:
            opacity = 0
        
        return min(opacity, 1.0)
    
    def _calculate_intensity_score(self, features):
        """
        Calculate overall smoke intensity score using weighted combination
        """
        #normalizing features to similar scales
        normalized_features = self._normalize_features(features)
        
        #weighted combination
        score = (
            self.weights['area'] * normalized_features['area'] +
            self.weights['density'] * normalized_features['density'] +
            self.weights['color_purity'] * normalized_features['color_purity']
        )
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _normalize_features(self, features):
        """
        Normalize features to 0-1 range for combination
        """
        normalized = {}
        
        #area normalization
        normalized['area'] = min(features['area'] / 50000.0, 1.0)
        normalized['density'] = features['density']
        normalized['color_purity'] = features['color_purity']
        
        return normalized
    
    def update_weights(self, new_weights):
        """
        Update the weights for intensity score calculation
        """
        self.weights.update(new_weights)

        total = sum(self.weights.values())
        if total != 1.0:
            # normalizing weights
            for key in self.weights:
                self.weights[key] /= total

def classify_smoke_intensity_level(intensity_score):
    """
    Classify smoke intensity into categories based on score
    """
    if intensity_score == 0:
        return "No Smoke"
    elif intensity_score < 0.2:
        return "Light Smoke"
    elif intensity_score < 0.4:
        return "Moderate Smoke"
    elif intensity_score < 0.6:
        return "Dense Smoke"
    elif intensity_score < 0.8:
        return "Very Dense Smoke"
    else:
        return "Extremely Dense Smoke"

def create_smoke_analysis_image(frame, smoke_mask):
    """
    Create an image with smoke overlay and analysis information
    """
    display_frame = frame.copy()
    
    #overlaying smoke mask
    smoke_overlay = display_frame.copy()
    smoke_overlay[smoke_mask == 255] = [255, 0, 0]
    cv2.addWeighted(smoke_overlay, 0.4, display_frame, 0.7, 0, display_frame)
    
    return display_frame

def analyze_single_image_smoke(image_bgr, smoke_mask):
    """
    Analyze smoke intensity in a single image
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    analyzer = SmokeIntensityAnalyzer()
    features = analyzer.extract_intensity_features(image_rgb, smoke_mask)
    intensity_level = classify_smoke_intensity_level(features['intensity_score'])
    
    print("SMOKE INTENSITY ANALYSIS RESULTS:")
    print(f"Intensity Level: {intensity_level}")
    print(f"Overall Score: {features['intensity_score']:.3f}")
    print(f"Smoke Area: {features['area']} pixels")
    print(f"Smoke Density: {features['density']:.3f}")
    print(f"Color Purity: {features['color_purity']:.3f}")
    print(f"Opacity: {features['opacity']:.3f}")
    
    return features, intensity_level