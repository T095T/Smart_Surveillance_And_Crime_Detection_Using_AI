# Smart Surveillance anc Crime Detection using AI 🕵️‍♂

**SSCDAI** (Smart Surveillance anc Crime Detection using AI) is a smart surveillance and crime detection system that leverages AI and computer vision to enhance public safety and automate threat detection in real time.

---

## 📌 Project Overview

The CRIME System uses surveillance camera feeds to:
- Detect suspicious activity or criminal behavior using machine learning.
- Alert authorities in real-time with incident snapshots and metadata.
- Maintain searchable logs of incidents for later review.
- Operate efficiently on edge devices or central servers.

---

## 🎯 Key Features

- 🎥 Live video stream monitoring
- 🧠 AI-based crime detection (e.g., violence, theft)
- 🚨 Real-time alerts via email/SMS/notification
- 📦 Event logging and image storage
- 📈 Criminal face detection and weapon detection
  

---

## 🛠️ Tech Stack

| Component       | Technology                           |
|----------------|-------------------------------------- |
| **Backend**     | Python, Flask                        |
| **AI/ML**       | OpenCV, TensorFlow / PyTorch, YOLOv8 |
| **Frontend**    |  HTML + Bootstrap                    |


---

## 🧪 Installation & Setup

### 🔧 Requirements

- Python **3.10**
- CUDA Toolkit **11.8**
- pip ≥ 22.0
- Git

---

### ⚙️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/rcime-system.git
cd rcime-system

# Set up Python 3.10 virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required libraries without pulling extra dependencies
pip install --no-deps -r flask-req.txt

# Run the Flask application
python app.py
