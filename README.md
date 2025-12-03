Pneumonia Detection Using Deep Learning (Chest X-Ray Classification)

This project is part of the AAI-521course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

Project Status: Completed

Installation

Follow the steps below to run this project on any machine:

1. Clone the repository
git clone https://github.com/yourusername/Pneumonia-Detection.git
cd Pneumonia-Detection

2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Download the dataset

This project uses the Kaggle Chest X-Ray Pneumonia Dataset.
Download it from Kaggle and place it in the following folder structure:

dataset/
   train/
      NORMAL/
      PNEUMONIA/
   val/
      NORMAL/
      PNEUMONIA/
   test/
      NORMAL/
      PNEUMONIA/

5. Run the training script
python train.py

6. Launch the Gradio-based Pneumonia Detector App
python app.py

Project Intro / Objective

The objective of this project is to develop a computer vision framework for automated pneumonia detection using chest X-ray (CXR) images. The system applies deep learning techniques to classify images as Normal or Pneumonia, aiming to support clinical decision-making by offering fast, consistent, and accessible diagnostic assistance.

The project demonstrates end-to-end AI workflow: data preprocessing, EDA, transfer learning using CNN models, performance evaluation, interpretability (Grad-CAM), dimensionality reduction (PCA/t-SNE), and deployment via a web app.

Partner(s) / Contributor(s)

Sharon Karabel
(If you have teammates or contacts, add them here. Otherwise, leave as is.)

Methods Used

Deep Learning

Computer Vision

Transfer Learning

Data Preprocessing & Augmentation

Exploratory Data Analysis

Model Evaluation (Accuracy, F1-score, Confusion Matrix, ROC-AUC)

Explainable AI (Grad-CAM)

Dimensionality Reduction (PCA, t-SNE)

Cloud Deployment (Hugging Face Spaces)

Technologies

Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

Gradio

Hugging Face Spaces

Jupyter Notebook

Project Description

This project presents a complete deep learning pipeline for pneumonia detection using the Kaggle Chest X-Ray Pneumonia Dataset, which contains ~5,800 pediatric radiographs labeled as Normal or Pneumonia. The dataset is split into training, validation, and testing partitions.

Dataset Details

Classes: NORMAL, PNEUMONIA

Total Images: ~5,800

Image Source: Kaggle (NICHD pediatric chest X-rays)

Data Challenges:

Class imbalance (Pneumonia > Normal)

Variation in image brightness, dimension, and quality

Noise and artifacts common in medical images

A data dictionary and image-level exploratory analysis were created, including pixel intensity, aspect ratio, variance distribution, and correlation heatmaps.

Research Questions

Can deep learning models accurately differentiate between Pneumonia and Normal chest X-rays?

Which pretrained CNN architecture—MobileNetV2, VGG16, or ResNet50—performs best for this task?

How does fine-tuning affect model accuracy and generalization?

Can explainability tools (Grad-CAM, PCA, t-SNE) confirm clinically meaningful predictions?

Approach

Image preprocessing: resizing (224×224), normalization, RGB formatting

Data augmentation: rotation, shifting, zooming, flipping

Transfer learning using:

MobileNetV2

VGG16

ResNet50

Metrics used: Accuracy, Precision, Recall, F1-score, ROC-AUC

Interpretability: Grad-CAM visualizations

Deployment: Gradio app hosted on Hugging Face Spaces

Key Results

MobileNetV2 achieved the best performance:

Accuracy: 88.46%

Precision: 0.90

Recall: 0.92

F1-score: 0.91

Fine-tuning further improved validation accuracy to 96.66%.

Grad-CAM confirmed model focus on clinically relevant lung regions.

t-SNE and PCA demonstrated clear feature separability between classes.

Challenges

Class imbalance requiring class-weight adjustments

Overfitting in ResNet50

Variability in image quality and dimensions

Threshold tuning to reduce false negatives (critical in medical diagnosis)

License

This project is licensed under the MIT License.
(Include a LICENSE file in your repository.)

Acknowledgments

Instructor: Dr. Azka Azka

University of San Diego, Applied Artificial Intelligence Program

Kaggle for providing open-source datasets

TensorFlow/Keras open-source community

Support from teammates, peers, and faculty who contributed feedback
