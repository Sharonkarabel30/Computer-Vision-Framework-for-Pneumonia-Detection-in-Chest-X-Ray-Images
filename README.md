**Computer Vision Framework for Pneumonia 
Detection in Chest X-Ray Images **

This project is part of the AAI-521 course within the Applied Artificial Intelligence Program at the University of San Diego (USD).

**Overview**
This project presents a complete deep learning pipeline for detecting Pneumonia from chest X-ray images using computer vision, transfer learning, and explainable AI.
A lightweight, fine-tuned MobileNetV2 model was deployed as an interactive web application on Hugging Face Spaces to classify images as Normal or Pneumonia.

🚀** Project Status**
Completed

🛠️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/Sharonkarabel30/Computer-Vision-Framework-for-Pneumonia-Detection-in-Chest-X-Ray-Images.git
cd Pneumonia-Detection

2️⃣ Create & activate a virtual environment
python -m venv venv
source venv/bin/activate     # Mac / Linux
venv\Scripts\activate        # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Download the dataset

Download the Chest X-Ray Pneumonia Dataset from Kaggle and place it like this:

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

5️⃣ Train the model
python train.py

6️⃣ Run the web application
python app.py

🎯 **Project Objective**

The goal of this project is to develop a robust AI-powered pneumonia detection system that can:

✔ Classify chest X-rays as NORMAL or PNEUMONIA
✔ Assist clinicians by providing fast, consistent screening
✔ Demonstrate model interpretability using Grad-CAM
✔ Deploy an end-to-end working medical imaging application

👥** Contributors**
Sharon Karabel

📚 **Methods Used**
1. Deep Learning
2. Transfer Learning
3. Computer Vision
4. Data Preprocessing & Augmentation
5. Exploratory Data Analysis (EDA)
6. Explainable AI (Grad-CAM)
7. PCA & t-SNE Embedding Visualizatio
8. Deployment (Hugging Face Spaces)

**Technologies**
1. Python
2. TensorFlow / Keras
3. Pandas, NumPy
4. Scikit-learn
5. Matplotlib, Seaborn
6. Gradio
7. Hugging Face Spaces
8. Jupyter Notebook

📊 **Dataset Description**
The project uses the Kaggle Chest X-Ray Pneumonia Dataset, containing ~5,800 pediatric chest radiographs.

**Dataset Summary**
Class	Count	Description
NORMAL	~1,583	Healthy lungs
PNEUMONIA	~4,273	Bacterial or viral pneumonia

**Characteristics:**

1. Image variability in resolution, brightness, contrast
2. Class imbalance (Pneumonia cases outnumber normals)
3. Real-world imaging artifacts

🔍 **Exploratory Data Analysis (EDA)**

Performed analyses included:

📌 Pixel intensity distribution
📌 Image shape & aspect ratio variation
📌 Class imbalance visualization
📌 Correlation heatmaps
📌 Random image grids

These insights guided preprocessing, augmentation, and model tuning.

⚙️ **Modeling Approach**

Three pretrained CNNs were evaluated:
1. Model	Notes
2. MobileNetV2	Lightweight, fastest, best performing
3. VGG16	Strong feature extractor, heavier model
4. ResNet50	Deep architecture, prone to overfitting
5. Training Steps
   
Data augmentation: rotation, shift, zoom, horizontal flip
Resize: 224 × 224, normalization
Transfer learning with frozen base layers
Custom head: GAP + BatchNorm + Dropout + Sigmoid
Optimizer: Adam
Loss: Binary Cross-Entropy
Handling imbalance using class weights

🏆 **Results**
Best Model: MobileNetV2
Metric	Score
Accuracy	88.46%
Precision	0.90
Recall	0.92
F1-score	0.91
Validation accuracy improved during fine-tuning:

📈 96.19% → 96.66%

**Explainability**
Grad-CAM highlighted pneumonia-affected lung regions, confirming clinical relevance.

**Feature Visualization**
PCA and t-SNE showed strong cluster separation between NORMAL vs PNEUMONIA samples.

🌐 **Deployment**
A full working app was deployed on Hugging Face Spaces using Gradio.

**Features**
1. Upload chest X-ray images
2. View prediction (Normal/Pneumonia)
3. Confidence score
4. Instant inference (sub-second)
5. Privacy-safe (no image stored)

📄 **License**
This project is licensed under the APACHE License.

🙏 **Acknowledgments**

Special thanks to:

Dr. Azka Azka – Instructor
University of San Diego, Applied Artificial Intelligence Program
Kaggle for open-access medical imaging datasets
TensorFlow & Gradio open-source communities
