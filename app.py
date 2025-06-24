from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import numpy as np
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image
import io
import os
import logging
from datetime import datetime
import cv2
import tensorflow as tf
import h5py
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Constants
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

# Recursive function to clean model config
def clean_model_config(config, keys_to_remove=['batch_shape']):
    """Recursively remove problematic keys from model configuration"""
    if isinstance(config, dict):
        for key in keys_to_remove:
            if key in config:
                del config[key]
        for k, v in list(config.items()):
            config[k] = clean_model_config(v, keys_to_remove)
    elif isinstance(config, list):
        return [clean_model_config(item, keys_to_remove) for item in config]
    return config

# Robust model loading with compatibility fixes
def load_dental_model():
    """Load model with multiple fallback strategies"""
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        
        # Create model directory if needed
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Verify model exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")
        
        # Attempt 1: Standard load
        try:
            model = load_model(MODEL_PATH, compile=False)
            logger.info("Model loaded successfully with standard method")
            return model
        except Exception as e:
            logger.warning(f"Standard load failed: {str(e)}")
        
        # Attempt 2: Load with config cleanup
        try:
            with h5py.File(MODEL_PATH, 'r') as f:
                model_config = f.attrs.get('model_config')
                if model_config is None:
                    raise ValueError("Model config not found")
                
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                
                config_json = json.loads(model_config)
                cleaned_config = clean_model_config(config_json)
                
                model = tf.keras.models.model_from_config(
                    cleaned_config, 
                    custom_objects={}
                )
                model.load_weights(MODEL_PATH)
                logger.info("Model loaded with config cleanup method")
                return model
        except Exception as e:
            logger.warning(f"Config cleanup load failed: {str(e)}")
        
        # Attempt 3: Load only weights (requires knowing architecture)
        try:
            # Placeholder - you would need to define your model architecture here
            # For example:
            # from tensorflow.keras.applications import MobileNetV2
            # base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False)
            # x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            # output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            # model = tf.keras.Model(inputs=base_model.input, outputs=output)
            # model.load_weights(MODEL_PATH)
            raise NotImplementedError("Architecture-specific loading not implemented")
        except Exception as e:
            logger.warning(f"Weights-only load failed: {str(e)}")
        
        # All methods failed
        raise RuntimeError("All loading methods failed")
    
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: {str(e)}")
        
        # Try legacy format as last resort
        legacy_path = os.path.join(MODEL_DIR, 'final_dental_model.keras')
        if os.path.exists(legacy_path):
            try:
                logger.warning("Attempting to load legacy .keras format")
                model = load_model(legacy_path, compile=False)
                logger.warning("Legacy model loaded successfully")
                return model
            except Exception as legacy_e:
                logger.critical(f"Legacy load failed: {str(legacy_e)}")
        
        logger.critical("Shutting down due to model loading failure")
        if 'gunicorn' in os.environ.get('SERVER_SOFTWARE', ''):
            os._exit(1)
        return None

# Load model at startup
model = load_dental_model()

# Image processing functions remain the same
def enhance_image_quality(img):
    try:
        img_array = np.array(img)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced_rgb)
    except Exception as e:
        logger.error(f"Image enhancement error: {str(e)}")
        return img

def preprocess_image(image_data):
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        img = enhance_image_quality(img)
        img = img.resize((224, 224)).convert('RGB')
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise

# Routes and WebSocket handlers remain the same
@app.route('/predict', methods=['POST'])
def predict():
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

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    socketio.emit('connection_response', {'data': 'Connected successfully'}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_stream')
def handle_start_stream():
    logger.info(f"Client {request.sid} started stream")
    socketio.emit('stream_started', {'status': 'ready'}, room=request.sid)

@socketio.on('stop_stream')
def handle_stop_stream():
    logger.info(f"Client {request.sid} stopped stream")
    socketio.emit('stream_stopped', {'status': 'inactive'}, room=request.sid)

@socketio.on('image_frame')
def handle_image_frame(data):
    try:
        if model is None:
            socketio.emit('classification_error', {'error': 'Model not loaded'}, room=request.sid)
            return

        if 'image' not in data:
            socketio.emit('classification_error', {'error': 'No image data provided'}, room=request.sid)
            return

        img_array = preprocess_image(data['image'])
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        result = CLASS_NAMES[1] if confidence > 0.5 else CLASS_NAMES[0]
        
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
    status_code = 200 if model else 503
    return jsonify({
        'status': 'up' if model else 'down',
        'model_loaded': bool(model),
        'tensorflow_version': tf.__version__,
        'timestamp': datetime.now().isoformat()
    }), status_code

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)