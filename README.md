🥔 Potato Leaf Disease Classification

A deep learning–based image classification system to detect potato leaf diseases from images.
The model classifies images into Early Blight, Late Blight, and Healthy categories and is deployed as a live web application using Streamlit on Hugging Face Spaces.

🔗 Live Demo

👉 Hugging Face Space:
https://huggingface.co/spaces/P-r-e-e-t-a-m/potato-streamlit

Upload a potato leaf image and get real-time predictions with confidence scores.

📌 Problem Statement

Potato crops are highly susceptible to leaf diseases such as Early Blight and Late Blight, which can significantly reduce yield if not detected early. Manual inspection is time-consuming and error-prone.

This project aims to automate potato leaf disease detection using deep learning and make it accessible through a simple web interface.

🧠 Approach

Used a Convolutional Neural Network (CNN) with transfer learning

Trained on RGB potato leaf images resized to 256 × 256

Applied preprocessing and normalization for stable inference

Focused on generalization and deployability, not just accuracy
📊 Model Performance

Validation Accuracy: ~98%

Classes:

Early Blight

Late Blight

Healthy

Note: Real-world confidence may vary depending on image quality, lighting, and leaf orientation.

🚀 Deployment

Built a Streamlit web application for inference

Deployed on Hugging Face Spaces

Users can upload images and get predictions instantly

Handled TensorFlow–Keras version compatibility issues during deployment to ensure stable model loading

🛠️ Tech Stack

Programming Language: Python

Deep Learning: TensorFlow, Keras

Image Processing: NumPy, PIL

Web App: Streamlit

Deployment: Hugging Face Spaces

Version Control: Git, GitHub

📁 Project Structure
.
├── streamlit_app.py        # Streamlit application
├── requirements.txt        # Dependencies
├── model/
│   └── potato_model_hf.keras
└── README.md

⚙️ How to Run Locally

Clone the repository:

git clone https://github.com/Preetam-b/<repo-name>.git
cd <repo-name>


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run streamlit_app.py

🧪 Example Usage

Open the app

Upload a potato leaf image (.jpg, .jpeg, .png)

View predicted disease class and confidence score

🧩 Key Learnings

Handling model serialization and compatibility issues is critical during deployment

Matching TensorFlow versions between training and production environments avoids runtime failures

Deployment is as important as model accuracy in real-world ML systems

📌 Future Improvements

Add Grad-CAM for visual explainability

Improve robustness on low-quality images

Extend to multi-crop disease classification

Add REST API using FastAPI

