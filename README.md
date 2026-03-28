# Age Recognition System for Content Recommendation 📸🩺

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Taipy](https://img.shields.io/badge/Taipy-3.1-red.svg)](https://www.taipy.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent system that recognizes a user's age via webcam and provides personalized health screening recommendations ("Profilaktyka zdrowotna"). The project uses a **ResNet50** deep learning model for age estimation and a **Taipy**-based web interface.

## 🌟 Key Features

- **Real-time Face Capture**: Integrated webcam support using OpenCV.
- **Automated Face Detection**: Uses Haar Cascades to crop and focus on the user's face.
- **Age Prediction**: Deep Learning model (ResNet50) trained for age regression.
- **Smart Recommendations**: Dynamic health checkup suggestions based on the predicted age group.
- **Modern UI**: Clean and interactive web interface built with Taipy.

## 🛠 Tech Stack

- **Core**: Python 3.10+
- **Deep Learning**: TensorFlow, Keras (ResNet50 Architecture)
- **Computer Vision**: OpenCV
- **GUI Framework**: Taipy
- **Data Handling**: NumPy, Pandas

## 📁 Project Structure

```text
├── app/                    # Production application
│   ├── temp/               # Temporary image storage
│   ├── camera_service.py   # Webcam handling logic
│   ├── config.py           # Application configuration & paths
│   ├── image_utils.py      # Face detection & preprocessing
│   ├── model_service.py    # Model loading & inference
│   └── model_on_prod.py    # Main GUI application (Taipy)
├── src/                    # Training & Research
│   ├── data_generator.py   # Custom TF Data Generators
│   ├── model.py            # Model architecture definition
│   ├── train.py            # Training pipeline
│   └── prepare_data.py     # Dataset preprocessing
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- A working webcam

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/age-recognition-system.git
   cd age-recognition-system
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the model**:
   Ensure your trained model (`best_model.keras`) is placed in the `models/` directory or update the path in `app/config.py`.

### Running the App

```bash
python app/model_on_prod.py
```

## 🧠 Model Information

The system utilizes the **ResNet50** architecture, fine-tuned for age estimation. 
- **Input Size**: 224x224x3 (RGB)
- **Preprocessing**: ResNet50 specific normalization.
- **Output**: Numerical age prediction (Regression).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed as part of a Bachelor's project focused on Healthcare Prevention and AI.*
