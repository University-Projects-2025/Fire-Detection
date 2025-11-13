class FireWatch {
    constructor() {
        this.apiBaseUrl = window.location.port === '8080' 
            ? 'http://localhost:5001/api' 
            : '/api';
        this.selectedModel = 'rf'; // Default model
        this.selectedFileName = ''; // Store the file name
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const uploadBtn = document.getElementById('uploadBtn');
        const imageInput = document.getElementById('imageInput');
        const detectBtn = document.getElementById('detectBtn');

        // Upload button click
        uploadBtn.addEventListener('click', () => {
            imageInput.click();
        });

        // File input change
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageSelect(e.target.files[0]);
            }
        });

        // Detect button
        detectBtn.addEventListener('click', () => {
            this.analyzeImage();
        });

        // Model selection
        const modelRadios = document.querySelectorAll('input[name="model"]');
        modelRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.selectedModel = e.target.value;
                this.updateUIForModel();
            });
        });
    }

    updateUIForModel() {
        const intensityItem = document.getElementById('intensityItem');
        
        if (this.selectedModel === 'yolo') {
            // Hide intensity display for YOLO model
            intensityItem.style.display = 'none';
        } else {
            // Show intensity display for RF model (when results are available)
            // It will be shown/hidden dynamically in displayResults
        }
    }

    handleImageSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG, JPEG)');
            return;
        }

        const detectBtn = document.getElementById('detectBtn');
        const originalImage = document.getElementById('originalImage');
        const originalImageLabel = document.getElementById('originalImageLabel');

        // Enable analyze button
        detectBtn.disabled = false;
        this.selectedFile = file;
        this.selectedFileName = file.name;

        // Update the original image label with file name
        const fileNameWithoutExt = file.name.replace(/\.[^/.]+$/, ""); // Remove file extension
        originalImageLabel.textContent = `Original Image: ${fileNameWithoutExt}`;

        // Show uploaded image in the original image frame
        const reader = new FileReader();
        reader.onload = (e) => {
            this.selectedImageData = e.target.result;
            originalImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    async analyzeImage() {
        if (!this.selectedImageData) return;

        this.showLoading(true);
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/detect`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: this.selectedImageData,
                    model: this.selectedModel
                })
            });

            if (!response.ok) {
                throw new Error('Analysis failed. Please try again.');
            }

            const result = await response.json();
            this.displayResults(result);
            
        } catch (error) {
            console.error('Error:', error);
            alert('Analysis failed. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(result) {
        const resultImage = document.getElementById('resultImage');
        const predictionBadge = document.getElementById('predictionBadge');
        const intensityItem = document.getElementById('intensityItem');
        const intensityFill = document.getElementById('intensityFill');
        const intensityValue = document.getElementById('intensityValue');

        // Display result image
        resultImage.src = result.overlay_image || result.original_image;

        // Update prediction badge
        const statusText = predictionBadge.querySelector('.status-text');
        
        if (statusText) {
            predictionBadge.className = `prediction-badge ${result.prediction}`;
            
            // Set appropriate text
            if (result.prediction === 'fire') {
                statusText.textContent = 'Fire Detected';
            } else if (result.prediction === 'smoke') {
                statusText.textContent = 'Smoke Detected';
            } else {
                statusText.textContent = 'Clear - No Detection';
            }
        }

        // Update intensity display (only for RF model)
        if (intensityItem && intensityFill && intensityValue) {
            if (this.selectedModel === 'rf' && (result.prediction === 'smoke' || result.prediction === 'fire')) {
                const score = result.detailed_analysis?.intensity_score || 0;
                const percentage = Math.round(score * 100);
                
                intensityFill.style.width = `${percentage}%`;
                intensityFill.className = `intensity-fill ${result.prediction}`;
                intensityValue.textContent = `${percentage}%`;
                intensityValue.className = `intensity-value ${result.prediction}`;
                intensityItem.style.display = 'block';
            } else {
                // Hide intensity display for YOLO, ConvNeXt or clear results
                intensityItem.style.display = 'none';
            }
        }

        // Display timing information
        this.displayTimingInfo(result);
    }

    displayTimingInfo(result) {
        // Create or update timing display
        let timingElement = document.getElementById('timingInfo');
        if (!timingElement) {
            timingElement = document.createElement('div');
            timingElement.id = 'timingInfo';
            timingElement.className = 'timing-info';
            
            // Add it to the results section after the results grid
            const resultsGrid = document.getElementById('resultsGrid');
            resultsGrid.parentNode.insertBefore(timingElement, resultsGrid.nextSibling);
        }

        let timingHTML = `
            <div class="timing-stats">
                <div class="timing-label">Performance Metrics</div>
                <div class="timing-item">
                    <span class="timing-label">Total Processing:</span>
                    <span class="timing-value">${result.processing_time || 'N/A'} ms</span>
                </div>
        `;

        if (result.model_inference_time) {
            timingHTML += `
                <div class="timing-item">
                    <span class="timing-label">Model Inference:</span>
                    <span class="timing-value">${result.model_inference_time} ms</span>
                </div>
            `;
        }

        timingHTML += `</div>`;
        timingElement.innerHTML = timingHTML;
    }

    showLoading(show) {
        const loading = document.getElementById('loading');
        const resultsGrid = document.getElementById('resultsGrid');
        const timingInfo = document.getElementById('timingInfo');
        const convnextInfo = document.getElementById('convnextInfo');
        
        if (loading && resultsGrid) {
            if (show) {
                // Show loading, hide results
                loading.style.display = 'block';
                resultsGrid.style.display = 'none';
                
                // Hide additional info sections during loading
                if (timingInfo) timingInfo.style.display = 'none';
                if (convnextInfo) convnextInfo.style.display = 'none';
            } else {
                // Hide loading, show results
                loading.style.display = 'none';
                resultsGrid.style.display = 'grid';
                
                // Show additional info sections after loading
                if (timingInfo) timingInfo.style.display = 'block';
                if (convnextInfo) convnextInfo.style.display = 'block';
            }
        }
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new FireWatch();
});