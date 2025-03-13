import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import uuid
import base64
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/processed')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def load_cascade(cascade_name):
    cascade_path = cv2.data.haarcascades + cascade_name
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print(f"Warning: Could not load {cascade_path}")
    return cascade

face_cascade = load_cascade('haarcascade_frontalface_default.xml')
eye_cascade = load_cascade('haarcascade_eye.xml')
pedestrian_cascade = load_cascade('haarcascade_fullbody.xml')

try:
    car_cascade = load_cascade('haarcascade_car.xml')
except:
    print("Warning: Car cascade file not found, using frontalface as fallback")
    car_cascade = face_cascade 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    feature = request.form.get('feature', 'face')
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)

    try:
        processed_image, detections = process_image(file_path, feature)

        processed_filename = f"processed_{unique_filename}"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        cv2.imwrite(processed_path, processed_image)

        return jsonify({
            'image_url': f"/static/processed/{processed_filename}",
            'detections': detections
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400
    
    file = request.files['video']
    feature = request.form.get('feature', 'face')
    
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)
    
    try:
        processed_filename = f"processed_{unique_filename}"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

        processed_video, detections = process_video(file_path, processed_path, feature)

        return jsonify({
            'video_url': f"/static/processed/{processed_filename}",
            'original_filename': filename,
            'detections': detections
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect_webcam', methods=['POST'])
def detect_webcam():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    feature = request.form.get('feature', 'face')
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    filename = "webcam_frame.jpg"
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)

    try:

        img = cv2.imread(file_path)
        
        boxes = []
        colors = []
        detections = []
        
        if feature == 'face':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            face_count = 0
            eye_count = 0
            
            for (x, y, w, h) in faces:
                boxes.append({'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)})
                colors.append('rgba(255, 0, 255, 0.8)')
                face_count += 1

                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes:
                    boxes.append({'x': int(x+ex), 'y': int(y+ey), 'width': int(ew), 'height': int(eh)})
                    colors.append('rgba(255, 255, 0, 0.8)')
                    eye_count += 1
            
            if face_count > 0 or eye_count > 0:
                detections.append(f"Detected {face_count} face(s) and {eye_count} eye(s).")
            else:
                detections.append("No faces or eyes detected.")
                
        elif feature == 'pedestrian':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            pedestrians = pedestrian_cascade.detectMultiScale(gray, 1.1, 3)
            
            for (x, y, w, h) in pedestrians:
                boxes.append({'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)})
                colors.append('rgba(0, 255, 255, 0.8)')
            
            if len(pedestrians) > 0:
                detections.append(f"Detected {len(pedestrians)} pedestrian(s).")
            else:
                detections.append("No pedestrians detected.")
                
        elif feature == 'vehicle':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            vehicles = car_cascade.detectMultiScale(gray, 1.1, 3)
            
            for (x, y, w, h) in vehicles:
                boxes.append({'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)})
                colors.append('rgba(0, 255, 0, 0.8)')
            
            if len(vehicles) > 0:
                detections.append(f"Detected {len(vehicles)} vehicle(s).")
            else:
                detections.append("No vehicles detected.")
        
        os.remove(file_path)
        
        return jsonify({
            'boxes': boxes,
            'colors': colors,
            'detections': detections
        })
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

def process_image(image_path, feature):
    img = cv2.imread(image_path)
    
    detections = []
    
    if feature == 'face':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_count = 0
        eye_count = 0
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
            face_count += 1
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 2)
                eye_count += 1
        
        if face_count > 0 or eye_count > 0:
            detections.append(f"Detected {face_count} face(s) and {eye_count} eye(s).")
        else:
            detections.append("No faces or eyes detected.")
            
    elif feature == 'pedestrian':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        pedestrians = pedestrian_cascade.detectMultiScale(gray, 1.1, 3)
        
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        if len(pedestrians) > 0:
            detections.append(f"Detected {len(pedestrians)} pedestrian(s).")
        else:
            detections.append("No pedestrians detected.")
            
    elif feature == 'vehicle':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        vehicles = car_cascade.detectMultiScale(gray, 1.1, 3)
        
        for (x, y, w, h) in vehicles:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if len(vehicles) > 0:
            detections.append(f"Detected {len(vehicles)} vehicle(s).")
        else:
            detections.append("No vehicles detected.")
    
    return img, detections

def process_video(video_path, output_path, feature):
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    detection_counts = {'face': 0, 'eye': 0, 'pedestrian': 0, 'vehicle': 0}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if feature == 'face':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                detection_counts['face'] += 1
                
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 2)
                    detection_counts['eye'] += 1
                
        elif feature == 'pedestrian':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            pedestrians = pedestrian_cascade.detectMultiScale(gray, 1.1, 3)
            
            for (x, y, w, h) in pedestrians:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                detection_counts['pedestrian'] += 1
                
        elif feature == 'vehicle':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            vehicles = car_cascade.detectMultiScale(gray, 1.1, 3)
            
            for (x, y, w, h) in vehicles:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                detection_counts['vehicle'] += 1

        out.write(frame)

    cap.release()
    out.release()

    detections = []
    if feature == 'face':
        detections.append(f"Processed {frame_count} frames.")
        detections.append(f"Detected {detection_counts['face']} face instances.")
        detections.append(f"Detected {detection_counts['eye']} eye instances.")
    elif feature == 'pedestrian':
        detections.append(f"Processed {frame_count} frames.")
        detections.append(f"Detected {detection_counts['pedestrian']} pedestrian instances.")
    elif feature == 'vehicle':
        detections.append(f"Processed {frame_count} frames.")
        detections.append(f"Detected {detection_counts['vehicle']} vehicle instances.")
    
    return output_path, detections

if __name__ == '__main__':
    app.run(debug=True, port=5001)
