# Diabetic Retinopathy Detection

This project focuses on detecting Diabetic Retinopathy (DR) stages from retinal fundus images using a deep learning model. The system allows users to upload retinal images through a Flask web app and receive instant predictions of DR severity.

---

## Overview

Diabetic Retinopathy is a progressive eye disease caused by diabetes and is one of the leading causes of preventable blindness. Early detection is essential, and this deep learning-based system assists by classifying fundus images into DR levels.

The model was trained on retinal image datasets such as the APTOS/organized dataset and deployed through a simple web interface for real-time prediction.

---

## Features

* Deep learning model for DR stage classification
* Flask web interface for uploading and predicting images
* Image preprocessing pipeline
* Organized dataset structure for clearer training workflow
* User-friendly prediction results through the browser

---

## Project Structure

```
├── app.py                       # Flask application for prediction
├── diabetic_retinopathy_model.h5  # Trained DL model
├── static/                      # Static assets (CSS, images)
├── templates/                   # HTML frontend templates
├── APTOS_2019_dataset/          # Original dataset (not uploaded fully)
├── organized_data/              # Cleaned/organized dataset
├── balanced_data/               # Augmented/balanced dataset
└── README.md
```

---

## Technologies Used

* Python
* TensorFlow / Keras
* Flask
* OpenCV
* NumPy, Pandas
* Matplotlib / Seaborn

---

## Image Preprocessing

* Resizing and normalization
* Histogram and contrast enhancement
* Data augmentation (flip, rotate, zoom)
* Class balancing to improve accuracy

---

## Model Training

* Custom CNN
* Categorical cross-entropy loss
* Optimizer tuning and callback usage
* Evaluation with accuracy/loss curves and confusion matrix

---

## Running the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run the Flask app

```
python app.py
```

### 3. Open the link shown in the terminal

Upload a retinal fundus image → get instant model prediction.

---

## Dataset

The project uses publicly available retinal image datasets such as:

* APTOS 2019 
* Other Kaggle datasets

Dataset files are not included due to size restrictions.

---

