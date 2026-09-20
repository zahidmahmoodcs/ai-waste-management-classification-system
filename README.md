# AI-Based Smart Waste Management and Classification System

An AI-powered waste classification and management system developed as a CS619 final-year project. The system uses deep learning and computer vision to classify waste images and provides a web-based interface for submitting images and viewing prediction results.

## 📌 Project Overview

The system is designed to assist in automated waste classification using image-based deep learning.

Users can submit a waste image through the web interface, and the trained deep-learning model analyzes the image and predicts its waste category. The application also provides a database-backed system for managing users, categories, predictions, feedback, and classification records.

## ✨ Key Features

- 🖼️ Waste image upload and classification
- 🤖 Deep-learning-based image classification
- 🧠 MobileNetV2 transfer learning
- 🌐 Flask-based web application
- 🗄️ SQLite database integration
- 👤 User and administrator functionality
- 📊 Classification result management
- 📝 Feedback and classification logging
- 📱 Web-based user interface

## 🛠️ Technologies Used

### Programming & Web Technologies

- Python
- HTML
- CSS
- JavaScript

### Machine Learning

- TensorFlow
- Keras
- MobileNetV2
- Computer Vision
- Transfer Learning

### Backend & Database

- Flask
- SQLite

### Development Tools

- Git
- GitHub

## 🧠 Machine Learning Model

The system uses **MobileNetV2** with transfer learning for waste image classification.

### Input

- Image size: `224 × 224 × 3`

### Main preprocessing steps

- Image resizing
- Image normalization
- Data augmentation
- Dataset preparation

### Model approach

The project uses a pre-trained MobileNetV2 architecture as the foundation and applies transfer learning for waste classification.

## 📊 Model Performance

The evaluated model achieved the following results:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 83.44% |
| Precision | 81.43% |
| Recall    | 79.14% |
| F1 Score  | 80.00% |

These results are based on the project's reported evaluation.

## 🔄 System Workflow

```text
User
  │
  ▼
Web Interface
  │
  ▼
Upload Waste Image
  │
  ▼
Image Preprocessing
  │
  ▼
MobileNetV2 Model
  │
  ▼
Waste Classification
  │
  ▼
Prediction Result
  │
  ▼
SQLite Database
```
