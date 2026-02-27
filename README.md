
# Quantum-Inspired Edge Detection for Real-Time Surveillance

## 📌 Overview

This project implements a real-time edge detection system using frequency-domain processing inspired by quantum-style transformations. The system enhances edge information using Fast Fourier Transform (FFT)-based high-pass filtering followed by Laplacian refinement. Performance is evaluated using FPS (Frame Per Second) and PSNR (Peak Signal-to-Noise Ratio).

---

## 🚀 Features

* Real-time video processing
* FFT-based frequency-domain edge detection
* Comparison with Sobel and Canny methods
* FPS and PSNR performance evaluation
* Performance graph generation

---

## 📊 Performance Results

| Method  | Average FPS | Average PSNR (dB) |
| ------- | ----------- | ----------------- |
| Quantum | ~50         | ~28               |
| Sobel   | ~160        | ~28               |
| Canny   | ~260        | ~28               |

---

## 🛠 Technology Stack

* Python
* OpenCV
* NumPy
* Matplotlib

---

## ▶ How to Run

1. Install required dependencies:

```
pip install -r requirements.txt
```

2. Run the real-time system:

```
python src/app.py
```

3. Generate performance graphs:

```
python src/plot_graph.py
```

---

## 📂 Project Structure

```
Quantum-Edge-Detection/
│
├── src/
├── docs/
├── fps.json
├── psnr.json
├── architecture.png
├── requirements.txt
├── README.md
└── setup_instructions.md
```

---

## 🎯 Application

Suitable for surveillance monitoring, security systems, and real-time video analytics requiring stable edge detection.

