# Automatic Fire and Smoke Detection System
"Practical Machine Learning and Deep Learning" and "Introduction to Computer Vision" Course Project (Fall 2025)

Contributors: Lev Permiakov, Arina Petuhova, Aleliya Turushkina

## Project Overview
This project aims to develop an automatic detection system for identifying fire and smoke in images using a Deep Learning approach. The core of the system is a Convolutional Neural Network (CNN) model trained to classify images into three distinct categories: Fire, Smoke, and No Threat.

## Repository Structure
```
├── 📁 ML/                                         # 🚀 Model Training Runs & Outputs
│   ├── 🏆 best_dual_head.pt                          # Best performing dual-head model weights (initial version)
│   ├── 🔄 best_dual_head_continued.pt                # Dual-head model weights from continued training
│   ├── 🆕 best_dual_head_v2.pt                       # Improved dual-head model v2 with architectural modifications
│   ├── 📊 labels-for-images.csv                      # CSV with image paths and corresponding labels for training
│   ├── ⚙️ main.py                                    # Main training script - model initialization & training loop
│   ├️ 🔧 model_detector_load.py                      # Utility functions for loading trained models
│   └── 🧠 modeldetector.py                           # Core detector class with dual-head architecture
│
├── 📁 data/                                           # 📊 Dataset Management
│   └── 🧹 data_clean.ipynb                           # Data cleaning, filtering, and preprocessing notebook
│   └── 📈 eda.ipynb                                  # Exploratory Data Analysis with visualizations
│   └── 🔗 dataset.dvc                                # DVC pointer for dataset version tracking
│
├── 📁 miruns/                                         # ⚡ Additional Model Runs & Configs
│   └── 📋 .dvcignore                                 # DVC ignore rules for model runs
│   └── ⚙️ .gitattributes                             # Git attributes configuration
│   └── 📄 LICENSE                                    # MIT License file
│   └── 📖 README.md                                  # Main project documentation
│   └── 🗂️ dataset_splits.pkl                        # Serialized train/val/test splits for reproducibility
│
├── 🔒 .dvcignore                                      # Global DVC ignore rules
├── 🔒 .gitignore                                      # Global Git ignore rules  
├── 📄 LICENSE                                         # MIT License file
└── 📖 README.md                                       # Project documentation
```

## Testing

- The ConvNeXt-based model training process is located in the section `ML/main.py`

- The downoland model and make predictian in `ML/model_detecrtor_load.py`