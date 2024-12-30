# YOLO (You Only Look Once) Project

## Overview
YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. Unlike traditional object detection methods that apply a classifier to different regions of an image, YOLO processes the entire image in a single pass, making it both fast and accurate. This repository provides an implementation of YOLO using OpenCV's Deep Neural Network (DNN) module, enabling real-time object detection for various applications.

---

## Features
- **Real-Time Object Detection**: Detect objects in images or video streams with high speed and accuracy.
- **Single Neural Network Architecture**: YOLO uses a single convolutional neural network to predict bounding boxes and class probabilities simultaneously.
- **Customizable Thresholds**: Adjust confidence thresholds and non-maximum suppression (NMS) parameters for optimal detection performance.
- **Pre-Trained Models**: Leverage pre-trained weights for common object detection tasks using the COCO dataset.

---

## Architecture
The YOLO architecture divides the input image into a grid and predicts bounding boxes, objectness scores, and class probabilities for each grid cell. Key components include:
1. **Input Image**: Resized to a fixed dimension (e.g., 448x448 or 416x416).
2. **DarkNet Backbone**: A convolutional neural network that extracts spatial and semantic features from the input image.
3. **Output Tensor**: Encodes bounding box coordinates, objectness scores, and class probabilities in a compact representation.

### Output Representation
- **Grid Division**: The image is divided into an `S x S` grid (e.g., 7x7).
- **Bounding Boxes**: Each grid cell predicts `B` bounding boxes, represented by:
  - `(x, y)`: Center coordinates relative to the grid cell.
  - `(w, h)`: Width and height relative to the image dimensions.
  - `Objectness Score`: Confidence that the box contains an object.
- **Class Probabilities**: Probabilities for `C` object classes.

The total output length for each grid cell is `5B + C`.

---

## Implementation Details
This repository includes a Python implementation of YOLO using OpenCV. The key steps are as follows:

### 1. Loading the Model
The YOLO configuration file (`.cfg`), pre-trained weights (`.weights`), and class labels (`.names`) are loaded using OpenCV's DNN module.

### 2. Preprocessing
The input image is resized and converted into a blob for compatibility with the YOLO model.

### 3. Forward Pass
The blob is passed through the network to obtain predictions, including bounding boxes, confidence scores, and class probabilities.

### 4. Post-Processing
- **Non-Maximum Suppression (NMS)**: Filters overlapping bounding boxes to retain the most confident predictions.
- **Thresholding**: Discards predictions with low confidence scores.

### 5. Visualization
Bounding boxes and class labels are drawn on the image or video frame using OpenCV.

---

## Code Structure
- **`load_labels`**: Loads class labels from a `.names` file.
- **`load_yolo_model`**: Loads the YOLO configuration and weights, and retrieves the output layer names.
- **`perform_detection`**: Processes the input image, performs object detection, and returns bounding boxes, confidences, and class IDs.
- **`draw_predictions`**: Draws bounding boxes and class labels on the image.
- **`main`**: Captures video frames, performs detection, and displays the results in real-time.

---

## Usage
### Prerequisites
- Python 3.x
- OpenCV
- NumPy

### Steps to Run
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Place the YOLO configuration, weights, and class labels in the `yolo_files` directory.
3. Run the script:
   ```bash
   python yolo_detection.py
   ```
4. Press `q` to exit the real-time detection window.

---

## Example
### Input
An image or video stream containing objects to detect.

### Output
The same image or video stream with bounding boxes and class labels drawn around detected objects.

---

## Applications
- Autonomous Vehicles
- Surveillance Systems
- Robotics
- Retail Analytics
- Medical Imaging

---

## References
- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [DarkNet Framework](https://pjreddie.com/darknet/)

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
This implementation is based on the YOLO architecture developed by Joseph Redmon and Ali Farhadi. Special thanks to the OpenCV community for their contributions to computer vision.
