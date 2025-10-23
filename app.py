from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import logging
import requests
import time
import io
import base64
from werkzeug.utils import secure_filename
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model URLs
MODEL_FILES = [
    {
        'url': 'https://github.com/isini312/fruit/raw/main/best_model.keras',
        'filename': 'best_model.keras'
    },
    {
        'url': 'https://github.com/isini312/fruit/raw/main/fruitnet_final_model.keras',
        'filename': 'fruitnet_final_model.keras'
    }
]

# Try to import dependencies
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
    logger.info("✅ TensorFlow imported successfully")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    logger.error(f"❌ TensorFlow import failed: {e}")

try:
    import cv2
    OPENCV_AVAILABLE = True
    logger.info("✅ OpenCV imported successfully")
except ImportError as e:
    OPENCV_AVAILABLE = False
    logger.error(f"❌ OpenCV import failed: {e}")

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
    logger.info("✅ PIL imported successfully")
except ImportError as e:
    PILLOW_AVAILABLE = False
    logger.error(f"❌ PIL import failed: {e}")

# Global variables
predictor = None
initialization_in_progress = True
initialization_error = None

def download_model(url, filename):
    """Download model file from URL"""
    try:
        logger.info(f"📥 Starting download: {filename}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        if downloaded_size % (5 * 1024 * 1024) == 0:  # Log every 5MB
                            logger.info(f"📦 {filename}: {progress:.1f}% ({downloaded_size/(1024*1024):.1f} MB)")
        
        file_size = os.path.getsize(filename)
        logger.info(f"✅ Download complete: {filename} ({file_size/(1024*1024):.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {filename} - {e}")
        return False

class SimpleFruitPredictor:
    def __init__(self):
        self.IMG_SIZE = (224, 224)
        
        # Class names
        self.fruit_classes = [
            'Bad Quality_Fruits_Apple_Bad', 'Bad Quality_Fruits_Banana_Bad', 'Bad Quality_Fruits_Guava_Bad',
            'Bad Quality_Fruits_Lime_Bad', 'Bad Quality_Fruits_Orange_Bad', 'Bad Quality_Fruits_Pomegranate_Bad',
            'Good Quality_Fruits_Apple_Good', 'Good Quality_Fruits_Banana_Good', 'Good Quality_Fruits_Guava_Good',
            'Good Quality_Fruits_Lime_Good', 'Good Quality_Fruits_Orange_Good', 'Good Quality_Fruits_Pomegranate_Good',
            'Mixed Qualit_Fruits_Apple', 'Mixed Qualit_Fruits_Banana', 'Mixed Qualit_Fruits_Guava',
            'Mixed Qualit_Fruits_Lemon', 'Mixed Qualit_Fruits_Orange', 'Mixed Qualit_Fruits_Pomegranate'
        ]
        
        self.fruit_name_mapping = {
            'Apple_Bad': 'Apple', 'Apple_Good': 'Apple', 'Apple': 'Apple',
            'Banana_Bad': 'Banana', 'Banana_Good': 'Banana', 'Banana': 'Banana',
            'Guava_Bad': 'Guava', 'Guava_Good': 'Guava', 'Guava': 'Guava',
            'Lime_Bad': 'Lime', 'Lime_Good': 'Lime', 'Lemon': 'Lemon',
            'Orange_Bad': 'Orange', 'Orange_Good': 'Orange', 'Orange': 'Orange',
            'Pomegranate_Bad': 'Pomegranate', 'Pomegranate_Good': 'Pomegranate', 'Pomegranate': 'Pomegranate'
        }
        
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the first available model"""
        for model_info in MODEL_FILES:
            filename = model_info['filename']
            if os.path.exists(filename) and os.path.getsize(filename) > 1024 * 1024:
                try:
                    logger.info(f"🔄 Loading model: {filename}")
                    self.model = load_model(filename)
                    logger.info(f"✅ Model loaded: {filename}")
                    return
                except Exception as e:
                    logger.error(f"❌ Failed to load {filename}: {e}")
        
        raise Exception("No valid model found")
    
    def preprocess_image(self, image_path):
        """Preprocess image"""
        if OPENCV_AVAILABLE:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.array(Image.open(image_path).convert('RGB'))
        
        image = cv2.resize(image, self.IMG_SIZE) if OPENCV_AVAILABLE else np.array(Image.fromarray(image).resize(self.IMG_SIZE))
        image = image / 255.0
        return np.expand_dims(image, axis=0)
    
    def parse_prediction(self, predicted_class):
        """Parse prediction result"""
        if 'Bad Quality_Fruits' in predicted_class:
            quality = "Bad Quality"
            fruit = predicted_class.replace('Bad Quality_Fruits_', '').replace('_Bad', '')
        elif 'Good Quality_Fruits' in predicted_class:
            quality = "Good Quality"
            fruit = predicted_class.replace('Good Quality_Fruits_', '').replace('_Good', '')
        elif 'Mixed Qualit_Fruits' in predicted_class:
            quality = "Mixed Quality"
            fruit = predicted_class.replace('Mixed Qualit_Fruits_', '')
        else:
            quality = "Unknown"
            fruit = predicted_class
        
        fruit = self.fruit_name_mapping.get(fruit, fruit)
        return quality, fruit
    
    def predict(self, image_path):
        """Make prediction"""
        try:
            processed_image = self.preprocess_image(image_path)
            predictions = self.model.predict(processed_image, verbose=0)[0]
            predicted_class_idx = np.argmax(predictions)
            confidence = np.max(predictions)
            predicted_class = self.fruit_classes[predicted_class_idx]
            quality, fruit = self.parse_prediction(predicted_class)
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_predictions = []
            for idx in top_3_indices:
                pred_class = self.fruit_classes[idx]
                pred_quality, pred_fruit = self.parse_prediction(pred_class)
                top_predictions.append({
                    'fruit': pred_fruit,
                    'quality': pred_quality,
                    'confidence': float(predictions[idx])
                })
            
            return {
                'success': True,
                'prediction': {
                    'fruit': fruit,
                    'quality': quality,
                    'confidence': float(confidence),
                    'full_class': predicted_class
                },
                'top_predictions': top_predictions
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}

def initialize_app():
    """Initialize the application in background"""
    global predictor, initialization_in_progress, initialization_error
    
    try:
        logger.info("🚀 Starting application initialization...")
        
        # Download models
        for model_info in MODEL_FILES:
            filename = model_info['filename']
            url = model_info['url']
            
            if not os.path.exists(filename) or os.path.getsize(filename) < 1024 * 1024:
                logger.info(f"📥 Downloading {filename}...")
                if not download_model(url, filename):
                    logger.error(f"❌ Failed to download {filename}")
        
        # Initialize predictor
        if TENSORFLOW_AVAILABLE:
            predictor = SimpleFruitPredictor()
            logger.info("✅ Application initialized successfully!")
        else:
            raise Exception("TensorFlow not available")
            
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        initialization_error = str(e)
    finally:
        initialization_in_progress = False

# Start initialization in background thread
init_thread = threading.Thread(target=initialize_app, daemon=True)
init_thread.start()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    status = 'initializing' if initialization_in_progress else 'ready' if predictor else 'error'
    return jsonify({
        'message': 'Fruit Quality Prediction API',
        'status': status,
        'initialization_in_progress': initialization_in_progress,
        'initialization_error': initialization_error,
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Upload image for prediction'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    if initialization_in_progress:
        return jsonify({
            'status': 'initializing',
            'message': 'Application is starting up, please wait...',
            'timestamp': time.time()
        })
    elif predictor:
        return jsonify({
            'status': 'healthy',
            'message': 'Application is ready',
            'timestamp': time.time()
        })
    else:
        return jsonify({
            'status': 'error',
            'error': initialization_error,
            'timestamp': time.time()
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    if initialization_in_progress:
        return jsonify({
            'error': 'Application is still starting up. Please try again in a few moments.',
            'status': 'initializing'
        }), 503
    
    if not predictor:
        return jsonify({
            'error': f'Application failed to initialize: {initialization_error}',
            'status': 'error'
        }), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result = predictor.predict(filepath)
            
            if os.path.exists(filepath):
                os.remove(filepath)
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 500
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': f'Server error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    if initialization_in_progress:
        return jsonify({
            'error': 'Application is still starting up. Please try again in a few moments.',
            'status': 'initializing'
        }), 503
    
    if not predictor:
        return jsonify({
            'error': f'Application failed to initialize: {initialization_error}',
            'status': 'error'
        }), 500
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image']
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
        
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        result = predictor.predict(temp_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Base64 processing error: {e}")
        return jsonify({'error': f'Base64 processing error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("=" * 60)
    print("🚀 Fruit Quality Prediction API Starting...")
    print("=" * 60)
    print("📝 The application will be ready once models are downloaded.")
    print("🌐 Check /health endpoint for status")
    print(f"🔧 Port: {port}")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=port)
