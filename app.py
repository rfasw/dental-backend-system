import os
import logging
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from PIL import Image
import io
import cv2
from datetime import datetime
import tensorflow as tf

# Fix NumPy compatibility before importing TensorFlow
np.set_printoptions(suppress=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# Model loading with compatibility workaround
def load_dental_model():
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")
        
        # Workaround for TensorFlow 2.x compatibility
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.critical(f"Model loading failed: {str(e)}")
        
        # Fallback to simplified model architecture
        try:
            logger.warning("Attempting fallback model loading")
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.load_weights(MODEL_PATH)
            logger.warning("Fallback model loaded successfully")
            return model
        except Exception as fallback_e:
            logger.critical(f"Fallback model failed: {str(fallback_e)}")
            return None

# Load model at startup
model = load_dental_model()

# Image processing functions
def enhance_image_quality(img):
    try:
        img_array = np.array(img)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)
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

# Routes and WebSocket handlers
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

@app.route('/health', methods=['GET'])
def health_check():
    status_code = 200 if model else 503
    return jsonify({
        'status': 'up' if model else 'down',
        'model_loaded': bool(model),
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__,
        'timestamp': datetime.now().isoformat()
    }), status_code

# SocketIO handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

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

def create_app():
    return app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port)