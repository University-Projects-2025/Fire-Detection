# Automatic Fire and Smoke Detection System
"Practical Machine Learning and Deep Learning" and "Introduction to Computer Vision" Course Project (Fall 2025)

Contributors: Lev Permiakov, Arina Petuhova, Aleliya Turushkina

## Repository Structure
```
|
└── data # Some data for fire/smoke segmentation testing
|
└── segmentation # Experiments on fire/smoke segmentation and intensity measurement
|
└── training_model # Model related experiments and training 
|
└── model_cv.py # Code with model related classes
|
└── segmentation_fire.py # Code with fire segmentation and intensity calculation related classes
|
└── segmentation_smoke.py # Code with smoke segmentation and intensity calculation related classes
|
└── prediction.py # Code for making a prediction on an image
|
└── README.md # Description of cv_testing branch
```

## About Testing
- Fire and smoke related segmentation and intensity calculations can be found in segmentation/fire_smoke_segmentation.ipynb
- Model related experiments and training process can be found in training_model/feature_extr_model.ipynb