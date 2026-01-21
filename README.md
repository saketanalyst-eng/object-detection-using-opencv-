# object-detection-using-opencv-
🧠 Real-Time Object Detection using YOLOv3 & OpenCV

This project implements a real-time object detection system using YOLOv3 and OpenCV’s DNN module in Python. The model detects and labels multiple objects from a live webcam feed with high accuracy while running efficiently on CPU.
📌 Features

Real-time object detection using webcam

YOLOv3 pre-trained on the COCO dataset

CPU-based inference (no GPU required)

Non-Maximum Suppression (NMS) for accurate bounding boxes

Modular and scalable code structure

Displays object name and confidence score
🛠️ Tech Stack

Programming Language: Python

Libraries: OpenCV, NumPy

Model: YOLOv3

Dataset: COCO (Common Objects in Context)

Framework: OpenCV DNN Module
├── tripti.py                # Main Python script
├── yolov3-320.cfg           # YOLOv3 configuration file
├── yolov3.weights           # Pre-trained YOLOv3 weights
├── coco_names.txt           # Class labels (COCO dataset)
└── README.md                # Project documentation
git clone https://github.com/saketanalyst-eng/Object-Detection-YOLOv3.git
cd Object-Detection-YOLOv3
3️⃣ Download YOLOv3 Files

Download the following files and place them in the project directory:

yolov3.weights

yolov3-320.cfg

coco_names.txt
python tripti.py
▶️ How to Run
python tripti.py


The webcam will open automatically

Detected objects will be highlighted with bounding boxes

Class name and confidence percentage will be displayed
displayed

📊 Detection Details

Confidence Threshold: 0.5

NMS Threshold: 0.1

Input Image Size: 320 × 320

Accuracy: ~95–99% (depending on lighting and object visibility)
📈 Output Example

Bounding boxes drawn around detected objects

Labels displayed as:

PERSON 98%
CAR 95%
🚀 Future Enhancements

GPU (CUDA) support for faster inference

Integration with video file input

Export detection results to CSV / database

Upgrade to YOLOv5 / YOLOv8

Deployment as a web or desktop application

👨‍💻 Author

Saket Pandey

🔗 LinkedIn: https://www.linkedin.com/in/saket-pandey-43859528a/

💻 GitHub: https://github.com/saketanalyst-eng

📜 License

This project is for educational and portfolio purposes.
YOLOv3 weights are subject to their original license.
