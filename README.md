# Automatic Fire and Smoke Detection System - YOLO
"Practical Machine Learning and Deep Learning" and "Introduction to Computer Vision" Course Project (Fall 2025)

Contributors: Lev Permiakov, Arina Petuhova, Aleliya Turushkina

## Repository Structure
```
├── .dvc/                                               # Directory for local DVC
│  
├── data/                                               # Dataset-related files and notebooks  
│   └── data_clean.ipynb                                # Jupyter notebook for cleaning and labeling dataset  
│   └── eda.ipynb                                       # Notebook for exploratory data analysis and visualization 
│   └── dataset.dvc                                     # pointer file used by DVC to track dataset
│
├── model/                                              # Folder with all models and training  
│   ├── metrics/                                        # Folder with yolo11 model metrics and comparative metrics
│   │   ├── model_comparison.csv                        # Comparison of metrics of models of different sizes
│   │   ├── YOLO11m_class_metrics.csv                   # Statistics for the YOLOv11m model by its epochs
│   │   ├── YOLO11n_class_metrics.csv                   # Statistics for the YOLOv11n model by its epochs
│   │   └── YOLO11s_class_metrics.csv                   # Statistics for the YOLOv11s model by its epochs
│   ├── runs/detect                                     # Folder with training runs and detection results
│   │   ├── train_yolo11m_50/                           # Yolo11m 50 epoch 
│   │   ├── train_yolo11n_50/                           # Yolo11n 50 epoch
│   │   ├── train_yolo11n_100/                          # Yolo11n 100 epoch
│   │   ├── train_yolo11s_50/                           # Yolo11s 50 epoch
│   │   └── train_yolo8n/                               # Yolo8n 10 epoch
│   ├── data.yaml                                       # Configuration file with data paths 
│   ├── yolo.ipynb                                      # Main file of YOLOv11 Model Training, comparative analysis and results
│   └── yolo8.ipynb                                     # YOLOv8n Model Training and its result
│ 
├── .dvcignore                                          # Specifies which files/folders to exclude from DVC tracking  
├── .gitignore                                          # Files/folders excluded from Git
├── LICENSE                                             # MIT License file  
└── README.md                                           # Documentation for yolo_test branch
```

## Testing

- The yolo11 training process is located in the section `model/yolo.ipynb`

- Detection results for each model and different epochs is located in the section `model/runs/detect`

- The yolo8 training process is located in the section `model/yolo8.ipynb`

- Statistics on yolo11 models and comparative metrics can be found in the section `model/metrics`