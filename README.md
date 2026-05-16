# YOLOv8 Helmet Detection

Real-time helmet detection system using YOLOv8 for industrial and traffic safety applications. This project provides a complete pipeline including dataset preparation, model training, inference, REST API integration, and Docker deployment.

---

## Project Overview

### Problem Statement
Ensuring helmet compliance in industrial and road environments is critical for safety. Manual monitoring is inefficient and error-prone.

### Goal
Build a real-time helmet detection system using YOLOv8 capable of detecting helmets accurately in images, videos, and live streams.

### Features
- Real-time helmet detection
- YOLOv8 object detection pipeline
- Image and video inference
- REST API integration using FastAPI
- Docker support for deployment
- Modular project structure
- Jupyter Notebook experimentation

---

## Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- FastAPI
- NumPy
- Docker
- Jupyter Notebook

---

## Project Structure

```bash
helmet-detection/
│
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── notebooks/
│   └── helmet_detection_yolov8.ipynb
│
├── model/
│   └── yolov8/
│
├── src/
│   ├── train.py
│   ├── detect.py
│   └── utils.py
│
├── app/
│   └── app.py
│
├── results/
│   ├── sample_predictions/
│   ├── confusion_matrix.png
│   └── training_curves.png
│
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
└── .env
```

---

## Dataset Preparation

### Dataset Sources
- Roboflow
- Kaggle datasets
- CCTV footage
- Custom collected images

### Annotation Tools
- LabelImg
- Roboflow

Annotations are stored in YOLO `.txt` format.

### Dataset Organization

```bash
data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## Installation

### 1. Clone Repository

```bash
git clone <repository_url>
cd helmet-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install YOLOv8

```bash
pip install ultralytics
```

---

## Model Training

### Using Jupyter Notebook

Open:

```bash
notebooks/helmet_detection_yolov8.ipynb
```

Configure:
- dataset paths
- `data.yaml`
- training parameters

Run notebook cells to start training.

---

### Using Python Script

```bash
python src/train.py --data data/data.yaml --epochs 50 --img-size 640 --batch-size 16
```

---

## Inference

```python
from ultralytics import YOLO

model = YOLO("model/yolov8/best.pt")

results = model.predict(
    source="data/images/val/sample1.jpg",
    conf=0.5,
    save=True,
    save_txt=True
)

results.show()
```

### Output
Prediction results are saved in:

```bash
runs/detect/exp/
```

---

## REST API Integration

### Start FastAPI Server

```bash
uvicorn app.app:app --reload
```

### API Endpoint

```bash
POST /predict
```

### Features
- Upload image
- Get prediction results
- JSON response with:
  - labels
  - confidence scores
  - bounding boxes

### Test Using cURL

```bash
curl -X POST -F "file=@sample.jpg" http://localhost:8000/predict
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t helmet-detector .
```

### Run Container

```bash
docker run -p 8000:8000 helmet-detector
```

### Access API

```bash
http://localhost:8000/predict
```

---

## Results

### Model Capabilities
- Helmet detection in traffic scenarios
- Industrial safety monitoring
- Real-time object detection

### Improvements Tested
- Webcam feed detection
- Batch image inference
- Accuracy evaluation
- False positive analysis

---

## Future Enhancements

- Helmet type classification
- Safety violation alerts
- Multi-camera integration
- Dashboard visualization
- Real-time analytics

---

## Author

**Het Vyas**

BTech Student | Aspiring Data Scientist | AI & ML Enthusiast
