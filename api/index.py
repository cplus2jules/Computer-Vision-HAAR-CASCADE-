import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import uuid
import base64
from werkzeug.utils import secure_filename
import time

app = Flask(__name__, 
    template_folder='../templates',
    static_folder='../static'
)

# Use /tmp directory for Vercel's serverless environment
UPLOAD_FOLDER = '/tmp/uploads'
PROCESSED_FOLDER = '/tmp/processed'
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

        # Convert processed image to base64 for serverless environment
        _, buffer = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Clean up temporary files
        os.remove(file_path)
        os.remove(processed_path)

        return jsonify({
            'image_data': f"data:image/jpeg;base64,{img_base64}",
            'detections': detections
        })
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
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

        # Read the processed video file and convert to base64
        with open(processed_path, 'rb') as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')

        # Clean up temporary files
        os.remove(file_path)
        os.remove(processed_path)

        return jsonify({
            'video_data': f"data:video/mp4;base64,{video_base64}",
            'original_filename': filename,
            'detections': detections
        })
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(processed_path):
            os.remove(processed_path)
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
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 255, 0), 2)
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
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    total_detections = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, frame_detections = process_image(frame, feature)
        out.write(processed_frame)
        
        if frame_detections:
            total_detections.extend([f"Frame {frame_count}: {detection}" for detection in frame_detections])
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    return output_path, total_detections

if __name__ == '__main__':
    app.run()
