import os
import uuid
import json
from flask import Flask, request, jsonify, url_for, send_from_directory, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import time

# Import our detector functions
from simple_deepfake_detector import predict_deepfake
from audio_deepfake_detector import predict_audio_deepfake, check_audio_file

app = Flask(__name__, static_folder='static')

# Basic CORS configuration
CORS(app)

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response
    
# Handle preflight requests
@app.route('/api/upload/video', methods=['OPTIONS'])
@app.route('/api/upload/audio', methods=['OPTIONS'])
@app.route('/api/task/<task_id>', methods=['OPTIONS'])
def handle_options():
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['ALLOWED_AUDIO_EXTENSIONS'] = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create static directories if they don't exist
os.makedirs('static/spectrograms', exist_ok=True)

# Track processing tasks
tasks = {}

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_VIDEO_EXTENSIONS']

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_AUDIO_EXTENSIONS']

def process_video_task(file_path, task_id, frames):
    """Background task to process the video"""
    try:
        # Update task status to processing
        tasks[task_id]['status'] = 'processing'
        
        # Run the prediction
        result = predict_deepfake(file_path, frames)
        
        # Update task with result
        if 'error' in result:
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['error'] = result['error']
        else:
            tasks[task_id]['status'] = 'completed'
            tasks[task_id]['result'] = {
                'prediction': result['prediction'],
                'confidence': result['confidence']
            }
            
    except Exception as e:
        # Update task with error
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
    finally:
        # Clean up the uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete file {file_path}: {str(e)}")

def process_audio_task(file_path, task_id):
    """Background task to process the audio"""
    try:
        # Update task status to processing
        tasks[task_id]['status'] = 'processing'
        tasks[task_id]['message'] = 'Analyzing audio patterns...'
        
        # Check if audio file is valid
        if not check_audio_file(file_path):
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['error'] = "Invalid audio file or format not supported."
            return
            
        # Run the prediction
        result = predict_audio_deepfake(file_path)
        
        # Update task with result
        if 'error' in result:
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['error'] = result['error']
        else:
            base_url = request.host_url.rstrip('/')
            spectrogram_url = f"{base_url}/static/{result.get('spectrogram_path', '')}"
            
            tasks[task_id]['status'] = 'completed'
            tasks[task_id]['result'] = {
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'message': result.get('message', ''),
                'spectrogram_url': spectrogram_url,
                'features': result.get('features', {})
            }
            
    except Exception as e:
        # Update task with error
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
    finally:
        # Clean up the uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete file {file_path}: {str(e)}")

@app.route('/')
def index():
    """Return API status"""
    return jsonify({
        "status": "online",
        "message": "Deepfake Detection API is running",
        "endpoints": {
            "upload_video": "/api/upload/video",
            "upload_audio": "/api/upload/audio",
            "task_status": "/api/task/<task_id>"
        }
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/api/upload/video', methods=['POST'])
def upload_video():
    """Handle video file upload and start processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_video_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: mp4, avi, mov, mkv, webm'}), 400
    
    # Get sequence_length param or use default
    try:
        frames = int(request.form.get('frames', 20))
        if frames < 10 or frames > 50:
            frames = 20
    except ValueError:
        frames = 20
    
    # Generate a task ID and save the file
    task_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
    
    try:
        file.save(file_path)
        
        # Initialize task
        tasks[task_id] = {
            'id': task_id,
            'filename': filename,
            'type': 'video',
            'status': 'queued',
            'frames': frames,
            'message': 'Uploading video...',
            'timestamp': time.time()
        }
        
        # Start processing in a background thread
        threading.Thread(
            target=process_video_task,
            args=(file_path, task_id, frames),
            daemon=True
        ).start()
        
        return jsonify({'task_id': task_id, 'status': 'queued'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/audio', methods=['POST'])
def upload_audio():
    """Handle audio file upload and start processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_audio_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: mp3, wav, ogg, flac, m4a'}), 400
    
    # Generate a task ID and save the file
    task_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
    
    try:
        file.save(file_path)
        
        # Initialize task
        tasks[task_id] = {
            'id': task_id,
            'filename': filename,
            'type': 'audio',
            'status': 'queued',
            'message': 'Uploading audio...',
            'timestamp': time.time()
        }
        
        # Start processing in a background thread
        threading.Thread(
            target=process_audio_task,
            args=(file_path, task_id),
            daemon=True
        ).start()
        
        return jsonify({'task_id': task_id, 'status': 'queued'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/task/<task_id>', methods=['GET'])
def task_status(task_id):
    """Check the status of a processing task"""
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(tasks[task_id])

# Task cleanup - remove old tasks periodically
def cleanup_tasks():
    while True:
        current_time = time.time()
        to_delete = []
        
        for task_id, task in tasks.items():
            # Remove tasks older than 1 hour
            if current_time - task.get('timestamp', 0) > 3600:
                to_delete.append(task_id)
        
        for task_id in to_delete:
            # Remove associated file if it exists
            file_path = tasks[task_id].get('file_path')
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            
            # Remove task
            del tasks[task_id]
        
        time.sleep(300)  # Check every 5 minutes

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_tasks)
cleanup_thread.daemon = True
cleanup_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) 