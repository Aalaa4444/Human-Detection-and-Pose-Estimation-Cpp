# AI Inference Conversion – YOLOv8 & Pose Estimation (Python ➡ C++)

This project demonstrates how to convert minimal AI inference scripts written in Python to equivalent C++ implementation using ONNX, OpenCV, and other inference engines.

---

## 🔍 Task Overview

### Task 1: YOLOv8 Object Detection
- **Original Language**: Python
- **Converted To**: C++
- **Model Used**: [YOLOv8n](https://github.com/ultralytics/ultralytics)
- **Inference Framework**: ONNX Runtime / OpenCV DNN
- **Input**: `person.jpg`
- **Output**: Detected bounding boxes overlaid on image

### Task 2: Human Pose Estimation
- **Original Language**: Python (Hugging Face model: `microsoft/posex`)
- **Converted To**: C++
- **Note**: Due to unavailability of a working ONNX version of `microsoft/posex`, a COCO-based OpenPose alternative was used.
- **Inference Framework**: OpenCV / ONNX
- **Input**: `person.jpg`
- **Output**: Human keypoints (poses) visualized

### Requirements
- C++ Compiler (e.g., g++)
- OpenCV with DNN module
- ONNX Runtime (depending on code path)
