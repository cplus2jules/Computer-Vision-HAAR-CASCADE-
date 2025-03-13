let currentFeature = null;
let webcamStream = null;
let detectionActive = false;
let detectionInterval = null;

function selectFeature(feature) {
    currentFeature = feature;
    document.getElementById('featureContainer').classList.remove('hidden');
    document.getElementById('resultContainer').classList.add('hidden');
    
    const titleElement = document.getElementById('featureTitle');
    switch (feature) {
        case 'face':
            titleElement.textContent = 'Face & Eye Detection';
            break;
        case 'pedestrian':
            titleElement.textContent = 'Pedestrian Detection';
            break;
        case 'vehicle':
            titleElement.textContent = 'Vehicle Detection';
            break;
    }
    
    document.getElementById('imageInput').value = '';
    document.getElementById('videoInput').value = '';
    
    document.getElementById('imageInput').onchange = handleImageUpload;
    document.getElementById('videoInput').onchange = handleVideoUpload;
}

function cancelOperation() {
    document.getElementById('featureContainer').classList.add('hidden');
    document.getElementById('resultContainer').classList.add('hidden');
    currentFeature = null;
}

function addLogEntry(message) {
    const logContainer = document.getElementById('logContainer');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.textContent = message;
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

function addWebcamLogEntry(message) {
    const logContainer = document.getElementById('webcamLog');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.textContent = message;
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

function handleImageUpload(event) {
    if (!event.target.files.length) return;
    
    const file = event.target.files[0];
    const reader = new FileReader();
    
    document.getElementById('resultContainer').classList.remove('hidden');
    document.getElementById('loadingIndicator').classList.remove('hidden');
    document.getElementById('resultView').innerHTML = '';
    document.getElementById('logContainer').innerHTML = '';
    
    addLogEntry(`Processing image: ${file.name}`);
    
    reader.onload = function(e) {
        const img = new Image();
        img.onload = function() {
            const formData = new FormData();
            formData.append('image', file);
            formData.append('feature', currentFeature);
            
            fetch('/detect', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                document.getElementById('loadingIndicator').classList.add('hidden');
                
                const resultView = document.getElementById('resultView');
                resultView.innerHTML = `<img src="${data.image_data || data.image_url}" alt="Processed Image">`;
                
                data.detections.forEach(detection => {
                    addLogEntry(detection);
                });
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('loadingIndicator').classList.add('hidden');
                addLogEntry(`Error: ${error.message}`);
            });
        };
        img.src = e.target.result;
    };
    
    reader.readAsDataURL(file);
}

function handleVideoUpload(event) {
    if (!event.target.files.length) return;
    
    const file = event.target.files[0];
    
    document.getElementById('resultContainer').classList.remove('hidden');
    document.getElementById('loadingIndicator').classList.remove('hidden');
    document.getElementById('resultView').innerHTML = '';
    document.getElementById('logContainer').innerHTML = '';
    
    addLogEntry(`Processing video: ${file.name}`);
    
    const formData = new FormData();
    formData.append('video', file);
    formData.append('feature', currentFeature);
    
    fetch('/detect_video', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('loadingIndicator').classList.add('hidden');
        
        const resultView = document.getElementById('resultView');
        resultView.innerHTML = `<video src="${data.video_data || data.video_url}" controls></video>`;
        
        data.detections.forEach(detection => {
            addLogEntry(detection);
        });
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('loadingIndicator').classList.add('hidden');
        addLogEntry(`Error: ${error.message}`);
    });
}

function startWebcam() {
    document.getElementById('webcamContainer').classList.remove('hidden');
    
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            const video = document.getElementById('webcam');
            webcamStream = stream;
            video.srcObject = stream;
            
            addWebcamLogEntry('Webcam started successfully.');
        })
        .catch(function(error) {
            console.error('Error accessing webcam:', error);
            addWebcamLogEntry(`Error accessing webcam: ${error.message}`);
        });
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    document.getElementById('webcamContainer').classList.add('hidden');

    if (detectionActive) {
        toggleDetection();
    }
}

function toggleDetection() {
    const button = document.getElementById('startStop');
    
    if (!detectionActive) {
        detectionActive = true;
        button.textContent = 'Stop Detection';
        
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('webcam-canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        
        addWebcamLogEntry('Starting detection...');
        
        function captureFrame() {
            if (!detectionActive) return;
            
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            canvas.toBlob(function(blob) {
                const formData = new FormData();
                formData.append('image', blob, 'webcam-frame.jpg');
                formData.append('feature', currentFeature || 'face');
                
                fetch('/detect_webcam', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    ctx.lineWidth = 3;
                    
                    if (data.detections && data.detections.length > 0) {
                        data.boxes.forEach((box, index) => {
                            ctx.strokeStyle = data.colors[index] || 'rgba(255, 0, 255, 0.8)';
                            ctx.strokeRect(box.x, box.y, box.width, box.height);
                        });
                        
                        if (data.detections.length > 0 && data.detections[0] !== '') {
                            addWebcamLogEntry(data.detections[0]);
                        }
                        if (data.detections.length > 0 && data.detections[0] !== '') {
                            addWebcamLogEntry(data.detections[0]);
                        }
                    }
                    
                    if (detectionActive) {
                        setTimeout(captureFrame, 100); 
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    addWebcamLogEntry(`Error: ${error.message}`);
                    
                    if (detectionActive) {
                        setTimeout(captureFrame, 500); 
                    }
                });
            }, 'image/jpeg');
        }
        
        captureFrame();
    } else {
        detectionActive = false;
        button.textContent = 'Start Detection';
        addWebcamLogEntry('Detection stopped.');
        
        const canvas = document.getElementById('webcam-canvas');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}
