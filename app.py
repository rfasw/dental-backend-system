from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import numpy as np
import base64
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os
import logging
from datetime import datetime
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Constants - UPDATED MODEL PATH
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'final_dental_model.h5')
CLASS_NAMES = ['Healthy Teeth', 'Unhealthy Teeth']
HEALTH_TIPS = {
    'Healthy Teeth': [
        "Continue brushing twice daily with fluoride toothpaste",
        "Floss at least once per day",
        "Schedule regular dental check-ups",
        "Limit sugary snacks and drinks"
    ],
    'Unhealthy Teeth': [
        "Schedule a dentist appointment immediately",
        "Increase brushing to 2-3 times daily",
        "Use antiseptic mouthwash",
        "Avoid acidic foods/drinks"
    ]
}

# Load the trained teeth health model
def load_dental_model():
    """Load the dental health classification model with proper error handling"""
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Verify model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")
            
        # Verify file is not empty
        if os.path.getsize(MODEL_PATH) == 0:
            raise ValueError("Model file is empty")
        
        # Load the model
        model = load_model(MODEL_PATH)
        logger.info("Dental health model loaded successfully")
        return model
        
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: {str(e)}")
        logger.critical("Shutting down due to model loading failure")
        # Graceful shutdown for production environments
        if 'gunicorn' in os.environ.get('SERVER_SOFTWARE', ''):
            os._exit(1)
        return None

# Load model at startup
model = load_dental_model()

def enhance_image_quality(img):
    """Improve image quality using CLAHE contrast enhancement"""
    try:
        img_array = np.array(img)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        
        # Merge channels and convert back to RGB
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced_rgb)
    except Exception as e:
        logger.error(f"Image enhancement error: {str(e)}")
        return img

def preprocess_image(image_data):
    """Process base64 image data from camera"""
    try:
        # Extract base64 string
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Apply computer vision enhancements
        img = enhance_image_quality(img)
        
        # Standard preprocessing
        img = img.resize((224, 224)).convert('RGB')
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """HTTP endpoint for teeth health prediction"""
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500

        if 'image' not in request.json:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400

        img_array = preprocess_image(request.json['image'])
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        result = CLASS_NAMES[1] if confidence > 0.5 else CLASS_NAMES[0]
        
        response = {
            'result': result,
            'confidence': confidence,
            'health_tips': HEALTH_TIPS[result],
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    socketio.emit('connection_response', {'data': 'Connected successfully'}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_stream')
def handle_start_stream():
    """Handle new stream request from client"""
    logger.info(f"Client {request.sid} started stream")
    socketio.emit('stream_started', {'status': 'ready'}, room=request.sid)

@socketio.on('stop_stream')
def handle_stop_stream():
    """Handle stream termination request"""
    logger.info(f"Client {request.sid} stopped stream")
    socketio.emit('stream_stopped', {'status': 'inactive'}, room=request.sid)

@socketio.on('image_frame')
def handle_image_frame(data):
    """Process image frame from client"""
    try:
        if model is None:
            socketio.emit('classification_error', {'error': 'Model not loaded'}, room=request.sid)
            return

        if 'image' not in data:
            socketio.emit('classification_error', {'error': 'No image data provided'}, room=request.sid)
            return

        # Process the image
        img_array = preprocess_image(data['image'])
        
        # Make prediction
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        result = CLASS_NAMES[1] if confidence > 0.5 else CLASS_NAMES[0]
        
        # Prepare response matching frontend expectations
        if result == 'Healthy Teeth':
            response = {
                'prediction': 'Healthy',
                'confidence': f"{(1 - confidence) * 100:.1f}%",
                'status': 'normal',
                'message': 'No dental issues detected',
                'tips': HEALTH_TIPS['Healthy Teeth']
            }
        else:
            response = {
                'prediction': 'Unhealthy',
                'confidence': f"{(confidence * 100):.1f}%",
                'status': 'warning',
                'condition': 'Potential Dental Issue',
                'message': 'Potential dental issue detected. Please consult with a dentist.',
                'tips': HEALTH_TIPS['Unhealthy Teeth']
            }
        
        socketio.emit('classification_result', response, room=request.sid)
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        socketio.emit('classification_error', {'error': str(e)}, room=request.sid)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with model status"""
    status_code = 200 if model else 503
    return jsonify({
        'status': 'up' if model else 'down',
        'model_loaded': bool(model),
        'timestamp': datetime.now().isoformat()
    }), status_code

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)