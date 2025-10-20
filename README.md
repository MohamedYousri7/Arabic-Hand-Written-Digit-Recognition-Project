# 🔢 Arabic Handwritten Digit Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![Keras](https://img.shields.io/badge/Keras-2.12%2B-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive **machine learning project** for recognizing **Arabic handwritten digits (0–9)** using deep learning and ensemble methods.  
Implements **CNN**, **ANN**, and **Random Forest classifiers**, combined with an **interactive GUI application** for real-time digit recognition.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Models](#-models)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [GUI Application](#-gui-application)
- [Project Structure](#-project-structure)
- [Technologies](#-key-technologies)
- [Data Preprocessing](#-data-preprocessing)
- [Learning Outcomes](#-learning-outcomes)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project explores **handwritten digit recognition** using the **Arabic Handwritten Digits Dataset**.  
It implements and compares three machine learning models:

- 🧠 **Convolutional Neural Network (CNN)**
- ⚙️ **Artificial Neural Network (ANN)**
- 🌲 **Random Forest Classifier**

The project leverages **data augmentation** to improve model generalization and includes a **Tkinter-based GUI** for real-time digit recognition.

---

## ✨ Features
- 🔢 **Multiple Model Architectures** — Compare CNN, ANN, and Random Forest  
- 🧩 **Data Augmentation** — Improve model robustness via rotation, shifting, shearing, and zooming  
- 🧮 **Interactive GUI** — Draw and predict digits live with Tkinter  
- 📊 **Comprehensive Evaluation** — Confusion matrices and accuracy metrics  
- 🔍 **Feature Importance Visualization** — Understand model decision-making  
- 💾 **Model Persistence** — Save and load trained models easily  

---

## 📊 Dataset
- **Source:** Arabic Handwritten Digits Dataset (CSV format)  
- **Training Samples:** 60,000 images  
- **Test Samples:** 10,000 images  
- **Image Format:** 28×28 grayscale pixels  
- **Labels:** Integers 0–9 representing handwritten Arabic digits  

---

## 🧠 Models

### 🧩 1. Convolutional Neural Network (CNN)
**Architecture:**
- Input Layer: 28×28×1 images  
- Conv2D (32 filters, 3×3) + ReLU  
- MaxPooling2D (2×2)  
- Conv2D (64 filters, 3×3) + ReLU  
- MaxPooling2D (2×2)  
- Flatten → Dense (128 neurons, ReLU)  
- Dropout (0.5) → Dense (10 neurons, Softmax)

**Optimizer:** Adam  
**Loss Function:** Sparse Categorical Crossentropy  
**Best Performance:** Achieved the highest accuracy among all models (~98.5%)

---

### 🧠 2. Artificial Neural Network (ANN)
**Architecture:**
- Flatten Layer (28×28 → 784)
- Dense (128 neurons) + ReLU  
- Dense (64 neurons) + ReLU  
- Output Layer (10 neurons) + Softmax  

**Optimizer:** Adam  
**Loss Function:** Sparse Categorical Crossentropy  

---

### 🌲 3. Random Forest Classifier
- Ensemble learning using multiple decision trees  
- Trained on flattened 784-feature vectors  
- Includes feature importance analysis for interpretability  

---

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Required Libraries
```bash
pip install numpy pandas matplotlib tensorflow plotly cufflinks scikit-learn seaborn pillow
