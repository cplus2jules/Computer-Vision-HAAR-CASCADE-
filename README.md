# Computer Vision Detection Web Application

This web application demonstrates various computer vision techniques using OpenCV and Python. It allows users to upload images or videos, or use their webcam to detect faces, eyes, pedestrians, and vehicles.

## Project Structure

```
computer-vision-detection/
│
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css      # CSS styles
│   ├── js/
│   │   └── main.js        # Frontend JavaScript
│   ├── uploads/           # Temporary storage for uploaded files
│   └── processed/         # Storage for processed images and videos
└── templates/
    └── index.html         # HTML template
```

## Features

- Face and Eye Detection: Detect faces and eyes in images, videos, or webcam feed
- Pedestrian Detection: Detect pedestrians in images, videos, or webcam feed
- Vehicle Detection: Detect vehicles in images, videos, or webcam feed
- Live Webcam Detection: Real-time detection using your computer's webcam

## Setup and Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python app.py
```

3. Open your web browser and go to `http://127.0.0.1:5000/`

## How It Works

### Backend (Python/Flask/OpenCV)

- The backend uses OpenCV's Haar cascade classifiers for object detection
- Flask provides the web server and API endpoints for processing images and videos
- Detection results are returned to the frontend as JSON responses

### Frontend (HTML/CSS/JavaScript)

- The frontend provides a user-friendly interface for interacting with the application
- JavaScript handles file uploads, webcam access, and communication with the backend
- Detection results are displayed in real-time

## API Endpoints

- `/detect`: Process uploaded images
- `/detect_video`: Process uploaded videos
- `/detect_webcam`: Process webcam frames

## Notes

- This application uses Haar cascade classifiers which are pre-trained models for object detection
- For production use, consider implementing more advanced detection models like YOLO or SSD
