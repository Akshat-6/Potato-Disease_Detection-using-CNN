# Potato-Disease_Detection-using-CNN
Potato Disease Detection is a web application that identifies common potato leaf diseases using deep learning. The app utilizes Convolutional Neural Networks (CNN) to classify potato leaves as Early Blight, Late Blight, or Healthy. The model has been trained using the PlantVillage dataset, achieving an overall accuracy of 88%.

This project is aimed at helping farmers and agronomists detect potato diseases early, enabling quick and effective disease management.

Features

Classifies potato leaf images into three categories: Early Blight, Late Blight, or Healthy.
Provides confidence percentages for each prediction.
Simple and interactive user interface built using HTML, CSS, and JavaScript.
Uses FastAPI for the backend and TensorFlow Serving for model deployment.
Data augmentation and preprocessing using ImageDataGenerator.
Deployed using GCP.

Technologies Used

Deep Learning: TensorFlow, Keras, CNN
Backend: FastAPI, TensorFlow Serving
Frontend: HTML, CSS, JavaScript
Deployment: Google Cloud Platform (GCP)
Data Augmentation: ImageDataGenerator
Visualization: Matplotlib, Seaborn
Web Server: Uvicorn
Model Format: Keras (.keras)
Project Structure

Potato-Disease-Detection/
├── model/                   # Saved model files
├── frontend/                
│   ├── index1.html           # Frontend HTML file
│   ├── style.css             # CSS styles for frontend
│   └── script.js             # JavaScript logic for frontend
├── backend/
│   └── main.py               # FastAPI backend script
├── imgGen/
│   ├── imgGen.py             # Image generation and processing script
│   └── imgGen.ipynb          # Notebook for training and evaluation
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies


Model Training and Testing

The model is trained on the PlantVillage dataset from Kaggle. Data preprocessing includes:
Normalization and augmentation using ImageDataGenerator.
CNN model with multiple convolutional and pooling layers.
Early stopping and model checkpointing for optimal training.
Accuracy: 88%

Usage

Upload a potato leaf image using the web interface.
Click on the "Classify Image" button.
View the classification result and confidence score.
Get suggestions on disease management if applicable.

Dataset

Dataset used: PlantVillage Dataset on Kaggle

Contributing

Contributions are welcome! Please feel free to submit a Pull Request or report an issue.

Contact

For any issues or improvements, contact Akshat Sharma.(akshatsharma6604@gmail.com).
