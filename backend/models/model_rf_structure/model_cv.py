import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from skimage import feature
from sklearn.preprocessing import StandardScaler
import pickle

class SmokeTextureFeatureExtractor:
    """
    Extracts comprehensive texture features for smoke detection
    """
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        self.pca = None
        
    def extract_features(self, image):
        """
        Extract multiple texture features from an image
        Returns: feature vector and feature names
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        features = []
        self.feature_names = []
        
        # 1. Statistical Features
        stats_features = self._extract_statistical_features(gray)
        features.extend(stats_features)
        
        # 2. GLCM Features
        glcm_features = self._extract_glcm_features(gray)
        features.extend(glcm_features)
        
        # 3. LBP Features
        lbp_features = self._extract_lbp_features(gray)
        features.extend(lbp_features)
        
        # 4. Gabor Filter Features
        gabor_features = self._extract_gabor_features(gray)
        features.extend(gabor_features)
        
        # 5. Wavelet Features
        wavelet_features = self._extract_wavelet_features(gray)
        features.extend(wavelet_features)
        
        # 6. Edge-based Features
        edge_features = self._extract_edge_features(gray)
        features.extend(edge_features)
        
        return np.array(features), self.feature_names
    
    def _extract_statistical_features(self, gray):
        """Extract statistical texture features"""
        features = []
        gray_flat = gray.flatten()
        
        # Basic statistics
        features.extend([np.mean(gray), np.std(gray), np.var(gray)])
        self.feature_names.extend(['mean_intensity', 'std_intensity', 'variance_intensity'])
        
        # Higher order statistics
        try:
            gray_skew = skew(gray_flat)
            gray_kurt = kurtosis(gray_flat)
            features.extend([gray_skew, gray_kurt])
            self.feature_names.extend(['skewness', 'kurtosis'])
        except:
            features.extend([0.0, 0.0])
            self.feature_names.extend(['skewness', 'kurtosis'])
        
        # Histogram-based features
        hist, _ = np.histogram(gray, bins=32, range=(0, 256))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        features.append(entropy)
        self.feature_names.append('histogram_entropy')
        
        # Local entropy
        local_entropy = self._calculate_local_entropy(gray)
        features.extend([np.mean(local_entropy), np.std(local_entropy)])
        self.feature_names.extend(['local_entropy_mean', 'local_entropy_std'])
        
        return features
    
    def _extract_glcm_features(self, gray, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """Extract Gray Level Co-occurrence Matrix features"""
        try:
            from skimage.feature import graycomatrix, graycoprops
            
            # Quantizing image to reduce computation
            gray_quantized = (gray / 16).astype(np.uint8)
            
            features = []
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            
            for distance in distances:
                for angle in angles:
                    glcm = graycomatrix(gray_quantized, [distance], [angle], 
                                      levels=16, symmetric=True, normed=True)
                    
                    for prop in properties:
                        feature_val = graycoprops(glcm, prop)[0, 0]
                        features.append(feature_val)
                        self.feature_names.append(f'glcm_{prop}_d{distance}_a{int(angle*180/np.pi)}')
            
            return features
        except ImportError:
            # Return empty list if GLCM not available
            return []
    
    def _extract_lbp_features(self, gray, radius=2, n_points=16):
        """Extract Local Binary Patterns features"""
        try:
            # Uniform LBP
            lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # LBP histogram
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points+2, range=(0, n_points+2))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            
            features = list(lbp_hist)
            self.feature_names.extend([f'lbp_bin_{i}' for i in range(len(lbp_hist))])
            
            # LBP statistics
            features.extend([np.mean(lbp), np.std(lbp), np.var(lbp)])
            self.feature_names.extend(['lbp_mean', 'lbp_std', 'lbp_var'])
            
            return features
        except:
            # Return zeros for all LBP features if failed
            return [0.0] * (n_points + 2 + 3)
    
    def _extract_gabor_features(self, gray):
        """Extract Gabor filter responses"""
        features = []
        kernels = []
        
        # Different frequencies and orientations
        frequencies = [0.1, 0.3, 0.5]
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for freq in frequencies:
            for theta in thetas:
                kernel = cv2.getGaborKernel((15, 15), 4.0, theta, freq, 0.5, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
        
        for i, kernel in enumerate(kernels):
            filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            features.extend([np.mean(filtered), np.std(filtered), np.var(filtered)])
            self.feature_names.extend([
                f'gabor_{i}_mean', f'gabor_{i}_std', f'gabor_{i}_var'
            ])
        
        return features
    
    def _extract_wavelet_features(self, gray):
        """Extract wavelet-based texture features"""
        features = []
        
        try:
            # Simple wavelet-like decomposition using Gaussian pyramid
            level1 = cv2.pyrDown(gray)
            level2 = cv2.pyrDown(level1)
            
            # Calculating energy at different levels
            for i, level in enumerate([gray, level1, level2]):
                energy = np.sum(level**2) / level.size
                features.append(energy)
                self.feature_names.append(f'wavelet_energy_level_{i}')
        except:
            # Fallback if pyramid fails
            features.extend([0.0, 0.0, 0.0])
            self.feature_names.extend(['wavelet_energy_level_0', 'wavelet_energy_level_1', 'wavelet_energy_level_2'])
        
        return features
    
    def _extract_edge_features(self, gray):
        """Extract edge-based texture features"""
        features = []
        
        # Sobel edges
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Edge statistics
        features.extend([np.mean(gradient_mag), np.std(gradient_mag), np.var(gradient_mag)])
        self.feature_names.extend(['edge_mean', 'edge_std', 'edge_var'])
        
        # Edge density
        edge_threshold = np.percentile(gradient_mag, 90)
        edge_density = np.sum(gradient_mag > edge_threshold) / gradient_mag.size
        features.append(edge_density)
        self.feature_names.append('edge_density')
        
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        features.append(edge_ratio)
        self.feature_names.append('canny_edge_ratio')
        
        return features
    
    def _calculate_local_entropy(self, gray, kernel_size=7):
        """Calculate local entropy map"""
        gray_normalized = gray.astype(np.float32) / 255.0
        pad_size = kernel_size // 2
        
        padded = cv2.copyMakeBorder(gray_normalized, pad_size, pad_size, pad_size, pad_size, 
                                  cv2.BORDER_REFLECT)
        
        entropy_map = np.zeros_like(gray_normalized)
        
        for i in range(gray_normalized.shape[0]):
            for j in range(gray_normalized.shape[1]):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                hist, _ = np.histogram(window, bins=16, range=(0, 1))
                hist = hist / (hist.sum() + 1e-7)
                entropy = -np.sum(hist * np.log2(hist + 1e-7))
                entropy_map[i, j] = entropy
        
        return entropy_map
    
    def transform_features(self, feature_matrix, use_pca=False, n_components=50):
        """Transform features using fitted scaler and optionally PCA"""
        features_scaled = self.scaler.transform(feature_matrix)
        
        if use_pca and self.pca is not None:
            return self.pca.transform(features_scaled)
        
        return features_scaled


class SmokeTextureClassifier:
    """
    Classification for 3-class smoke/fire/clear detection - Prediction Only
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_extractor = SmokeTextureFeatureExtractor()
        self.is_trained = False
        self.use_pca = False
        self.n_components = 50
        self.class_names = ['smoke', 'fire', 'clear']
    
    def predict_multiclass(self, image):
        """Predict single image with multiclass probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features, _ = self.feature_extractor.extract_features(image)
        features = features.reshape(1, -1)
        
        # The same transformation used during training
        features_transformed = self.feature_extractor.transform_features(
            features, use_pca=self.use_pca, n_components=self.n_components
        )
        
        prediction = self.model.predict(features_transformed)[0]
        probabilities = self.model.predict_proba(features_transformed)[0]
        
        return prediction, probabilities
    
    def load_model(self, filepath):
        """Load trained model and feature extractor"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_extractor = model_data['feature_extractor']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        self.use_pca = model_data.get('use_pca', False)
        self.n_components = model_data.get('n_components', 50)
        self.class_names = model_data.get('class_names', ['smoke', 'fire', 'clear'])


class SmokeDetectionPipeline:
    """
    Complete pipeline for 3-class smoke/fire/clear detection - Prediction Only
    """
    
    def __init__(self, model_type='random_forest'):
        self.classifier = SmokeTextureClassifier(model_type)
        self.is_trained = False
        self.class_names = ['smoke', 'fire', 'clear']
    
    def predict_image_3class(self, image):
        """Predict single image with 3-class output"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        prediction, probabilities = self.classifier.predict_multiclass(image)
        
        # Mapping numeric predictions to class names
        class_name = self.class_names[prediction]
        
        # Getting probabilities for all classes
        prob_dict = {
            "smoke": probabilities[0],
            "fire": probabilities[1], 
            "clear": probabilities[2]
        }
        
        return class_name, prob_dict, prediction
    
    def load_pipeline(self, filepath):
        """Load entire pipeline"""
        self.classifier.load_model(filepath)
        self.is_trained = True


def test_single_image_3class(pipeline, image_path):
    """Test the trained 3-class model on a single image"""
    import os
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Loading and preprocess image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (128, 128))
    
    # Predicting
    class_name, probabilities, prediction = pipeline.predict_image_3class(image_resized)
    
    print(f"Prediction: {class_name.upper()}")
    print(f"Probabilities:")
    for class_name, prob in probabilities.items():
        print(f"  {class_name}: {prob:.3f}")
    
    return prediction, probabilities[class_name]