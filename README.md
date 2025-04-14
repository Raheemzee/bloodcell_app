# 🧬 Blood Cell Classification & Counting App

A Flask web application that allows users to:

- Upload **blood smear images**
- **Count white blood cells** by class (e.g., neutrophils, lymphocytes, etc.)
- **Classify individual cells** in cropped images

Built using **Fastai**, **OpenCV**, and **Flask**, this project showcases the use of computer vision in the biomedical domain.

---

## 🔍 Features

- 🧠 Custom trained Fastai CNN model (MobileNetV2)
- 🧪 Image preprocessing and segmentation using OpenCV
- 🧬 Cell classification into multiple classes (e.g. neutrophil, eosinophil, monocyte...)
- 🌐 Flask-powered web interface
- 🖼️ Upload multiple images for counting or classification
- 📊 Output predictions and cell counts

---

## 🧰 Tech Stack

- Python 3.10+
- Flask
- Fastai
- Torch
- OpenCV

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/blood-cell-app.git
cd blood-cell-app
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask app
```bash
python app.py
```

Then open your browser and go to:
```
http://localhost:5000
```

---

## 📦 Model Details
- Trained using Fastai with `mobilenet_v2` architecture
- Supports classification of white blood cell types
- Exported as `blood_cell_classifier.pkl` (kept under 25MB for easy GitHub hosting)

---

## 📸 Screenshots
*(Insert screenshots here of the UI and example outputs)*<img width="942" alt="app2" src="https://github.com/user-attachments/assets/66aeae66-0616-4cf1-baa1-1ab3eaf5959a" />
<img width="939" alt="app1" src="https://github.com/user-attachments/assets/a5ce2b2b-b4fa-4537-95c3-71bf055bb9a1" />


---

## 🧠 Acknowledgements
- Fastai Team (for an amazing deep learning framework)
- OpenCV (for image processing)
- Blood cell image datasets used for training

---

## 📬 Contact
Made with ❤️ by Abdulraheem Akinola  
Connect with me on [LinkedIn](www.linkedin.com/in/abdulraheem-akinola-a03936235)



