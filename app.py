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
        'url': 'https://raw.githubusercontent.com/isini312/fruit/main/best_model.keras',
        'filename': 'best_model.keras'
    },
    {
        'url': 'https://raw.githubusercontent.com/isini312/fruit/main/fruitnet_final_model.keras',
        'filename': 'fruitnet_final_model.keras'
    }
]

# Try to import dependencies with fallbacks
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

def download_model(url, filename):
    """Download model file from URL"""
    try:
        logger.info(f"📥 Downloading {filename} from {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0 and downloaded_size % (1024 * 1024) == 0:
                        progress = (downloaded_size / total_size) * 100
                        logger.info(f"📦 Downloading {filename}: {progress:.1f}%")
        
        file_size = os.path.getsize(filename)
        logger.info(f"✅ Downloaded {filename} ({file_size / (1024*1024):.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {filename}: {e}")
        return False

def download_all_models():
    """Download all required model files"""
    logger.info("🔄 Checking for model files...")
    
    models_downloaded = 0
    for model_info in MODEL_FILES:
        filename = model_info['filename']
        url = model_info['url']
        
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            logger.info(f"✅ {filename} already exists ({file_size / (1024*1024):.2f} MB)")
            models_downloaded += 1
        else:
            logger.info(f"🔄 Downloading {filename}...")
            if download_model(url, filename):
                models_downloaded += 1
            else:
                logger.error(f"❌ Failed to download {filename}")
    
    return models_downloaded == len(MODEL_FILES)

def preprocess_image_pil(image_data):
    """Preprocess image using PIL (fallback when OpenCV is not available)"""
    try:
        # Convert to RGB if needed
        if image_data.mode != 'RGB':
            image_data = image_data.convert('RGB')
        
        # Resize image
        img_resized = image_data.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(img_resized) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"PIL preprocessing error: {e}")
        raise

def preprocess_image_cv2(image_path):
    """Preprocess image using OpenCV"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (224, 224))
        
        # Normalize pixel values
        image = image / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        logger.error(f"OpenCV preprocessing error: {e}")
        raise

class FruitQualityPredictor:
    def __init__(self):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available")
            
        self.IMG_SIZE = (224, 224)
        
        # Class names based on dataset structure
        self.fruit_classes = [
            # Bad Quality Fruits
            'Bad Quality_Fruits_Apple_Bad',
            'Bad Quality_Fruits_Banana_Bad', 
            'Bad Quality_Fruits_Guava_Bad',
            'Bad Quality_Fruits_Lime_Bad',
            'Bad Quality_Fruits_Orange_Bad',
            'Bad Quality_Fruits_Pomegranate_Bad',
            
            # Good Quality Fruits
            'Good Quality_Fruits_Apple_Good',
            'Good Quality_Fruits_Banana_Good',
            'Good Quality_Fruits_Guava_Good',
            'Good Quality_Fruits_Lime_Good',
            'Good Quality_Fruits_Orange_Good',
            'Good Quality_Fruits_Pomegranate_Good',
            
            # Mixed Quality Fruits
            'Mixed Qualit_Fruits_Apple',
            'Mixed Qualit_Fruits_Banana',
            'Mixed Qualit_Fruits_Guava',
            'Mixed Qualit_Fruits_Lemon',
            'Mixed Qualit_Fruits_Orange',
            'Mixed Qualit_Fruits_Pomegranate'
        ]
        
        # Fruit name mapping
        self.fruit_name_mapping = {
            'Apple_Bad': 'Apple', 'Apple_Good': 'Apple', 'Apple': 'Apple',
            'Banana_Bad': 'Banana', 'Banana_Good': 'Banana', 'Banana': 'Banana',
            'Guava_Bad': 'Guava', 'Guava_Good': 'Guava', 'Guava': 'Guava',
            'Lime_Bad': 'Lime', 'Lime_Good': 'Lime', 'Lemon': 'Lemon',
            'Orange_Bad': 'Orange', 'Orange_Good': 'Orange', 'Orange': 'Orange',
            'Pomegranate_Bad': 'Pomegranate', 'Pomegranate_Good': 'Pomegranate', 'Pomegranate': 'Pomegranate'
        }
        
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all available models"""
        try:
            model_files = {}
            
            # Check for model files
            for model_info in MODEL_FILES:
                filename = model_info['filename']
                if os.path.exists(filename):
                    model_name = filename.replace('.keras', '')
                    model_files[model_name] = filename
                    logger.info(f"✅ Found model: {filename}")
                else:
                    logger.warning(f"⚠️ Model file not found: {filename}")
            
            if not model_files:
                raise Exception("No model files found. Please download models first.")
            
            logger.info("🔄 Loading models...")
            for model_name, model_path in model_files.items():
                logger.info(f"   Loading {model_name} from {model_path}")
                try:
                    self.models[model_name] = load_model(model_path)
                    logger.info(f"   ✅ {model_name} loaded successfully")
                except Exception as e:
                    logger.error(f"   ❌ Failed to load {model_name}: {e}")
                    raise
            
            logger.info(f"🎯 All models loaded successfully! {len(self.models)} models ready.")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def parse_prediction(self, predicted_class):
        """Parse prediction into fruit and quality"""
        try:
            if 'Bad Quality_Fruits' in predicted_class:
                quality = "Bad Quality"
                fruit_part = predicted_class.replace('Bad Quality_Fruits_', '')
                fruit = self.fruit_name_mapping.get(fruit_part, fruit_part.replace('_Bad', ''))
            elif 'Good Quality_Fruits' in predicted_class:
                quality = "Good Quality"
                fruit_part = predicted_class.replace('Good Quality_Fruits_', '')
                fruit = self.fruit_name_mapping.get(fruit_part, fruit_part.replace('_Good', ''))
            elif 'Mixed Qualit_Fruits' in predicted_class:
                quality = "Mixed Quality"
                fruit_part = predicted_class.replace('Mixed Qualit_Fruits_', '')
                fruit = self.fruit_name_mapping.get(fruit_part, fruit_part)
            else:
                quality = "Unknown"
                fruit = predicted_class
            
            return quality, fruit
        except Exception as e:
            logger.error(f"Error parsing prediction: {e}")
            return "Unknown", predicted_class
    
    def predict_quality(self, image_path):
        """Predict fruit quality from image"""
        try:
            # Preprocess image based on available libraries
            if OPENCV_AVAILABLE:
                processed_image = preprocess_image_cv2(image_path)
            elif PILLOW_AVAILABLE:
                image = Image.open(image_path)
                processed_image = preprocess_image_pil(image)
            else:
                return {"error": "No image processing library available"}
            
            # Get predictions from all models
            all_predictions = []
            model_results = {}
            
            for model_name, model in self.models.items():
                try:
                    predictions = model.predict(processed_image, verbose=0)[0]
                    predicted_class_idx = np.argmax(predictions)
                    confidence = np.max(predictions)
                    predicted_class = self.fruit_classes[predicted_class_idx]
                    quality, fruit = self.parse_prediction(predicted_class)
                    
                    model_results[model_name] = {
                        'fruit': fruit,
                        'quality': quality,
                        'confidence': float(confidence),
                        'full_class': predicted_class
                    }
                    all_predictions.append(predictions)
                    logger.info(f"✅ {model_name} prediction: {fruit} - {quality} ({confidence:.2%})")
                except Exception as e:
                    logger.error(f"❌ {model_name} prediction failed: {e}")
                    continue
            
            if not all_predictions:
                return {"error": "All model predictions failed"}
            
            # Ensemble prediction
            ensemble_predictions = np.mean(all_predictions, axis=0)
            ensemble_class_idx = np.argmax(ensemble_predictions)
            ensemble_confidence = np.max(ensemble_predictions)
            ensemble_class = self.fruit_classes[ensemble_class_idx]
            ensemble_quality, ensemble_fruit = self.parse_prediction(ensemble_class)
            
            # Get top 3 predictions
            top_3_indices = np.argsort(ensemble_predictions)[-3:][::-1]
            top_3_predictions = []
            
            for idx in top_3_indices:
                pred_class = self.fruit_classes[idx]
                pred_quality, pred_fruit = self.parse_prediction(pred_class)
                top_3_predictions.append({
                    'fruit': pred_fruit,
                    'quality': pred_quality,
                    'confidence': float(ensemble_predictions[idx])
                })
            
            # Check model agreement
            individual_predictions = [result['fruit'] for result in model_results.values()]
            agreement_count = len([p for p in individual_predictions if p == ensemble_fruit])
            
            return {
                'success': True,
                'prediction': {
                    'fruit': ensemble_fruit,
                    'quality': ensemble_quality,
                    'confidence': float(ensemble_confidence),
                    'full_class': ensemble_class
                },
                'top_predictions': top_3_predictions,
                'model_info': {
                    'models_used': list(self.models.keys()),
                    'models_agree': agreement_count == len(self.models),
                    'agreement_count': agreement_count,
                    'total_models': len(self.models),
                    'image_processor': 'opencv' if OPENCV_AVAILABLE else 'pil'
                },
                'individual_results': model_results
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the predictor with model downloading"""
    global predictor
    
    if not TENSORFLOW_AVAILABLE:
        logger.error("❌ TensorFlow not available")
        return False
    
    try:
        logger.info("🚀 Initializing Fruit Quality Predictor...")
        
        # Download models first
        logger.info("📥 Downloading model files...")
        if not download_all_models():
            logger.error("❌ Failed to download all models")
            return False
        
        # Initialize predictor
        logger.info("🔄 Loading models into memory...")
        predictor = FruitQualityPredictor()
        logger.info("✅ Fruit Quality Predictor initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictor: {e}")
        return False

# Initialize on startup
logger.info("🔧 Starting application initialization...")
initialize_predictor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return jsonify({
        'message': 'Fruit Quality Prediction API',
        'status': 'running' if predictor else 'error',
        'tensorflow_available': TENSORFLOW_AVAILABLE,
        'opencv_available': OPENCV_AVAILABLE,
        'pillow_available': PILLOW_AVAILABLE,
        'predictor_initialized': predictor is not None,
        'models_loaded': len(predictor.models) if predictor else 0,
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Upload image for prediction',
            '/predict_base64': 'POST - Send base64 image',
            '/health': 'GET - API health check',
            '/reload': 'POST - Reload models'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if predictor else 'error',
        'tensorflow_available': TENSORFLOW_AVAILABLE,
        'opencv_available': OPENCV_AVAILABLE,
        'pillow_available': PILLOW_AVAILABLE,
        'predictor_initialized': predictor is not None,
        'models_loaded': len(predictor.models) if predictor else 0,
        'models': list(predictor.models.keys()) if predictor else [],
        'timestamp': time.time()
    })

@app.route('/reload', methods=['POST'])
def reload_models():
    """Reload models (useful for updating models without restarting)"""
    global predictor
    try:
        logger.info("🔄 Reloading models...")
        predictor = None
        
        # Re-download and re-initialize
        success = initialize_predictor()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Models reloaded successfully',
                'models_loaded': len(predictor.models)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to reload models'
            }), 500
            
    except Exception as e:
        logger.error(f"Reload error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({
            'error': 'Predictor not initialized',
            'tensorflow_available': TENSORFLOW_AVAILABLE
        }), 500
    
    # Check if image file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Secure filename and save
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            result = predictor.predict_quality(filepath)
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 500
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': f'Server error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg, bmp'}), 400

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Alternative endpoint for base64 encoded images"""
    if predictor is None:
        return jsonify({
            'error': 'Predictor not initialized',
            'tensorflow_available': TENSORFLOW_AVAILABLE
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Save temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        # Make prediction
        result = predictor.predict_quality(temp_path)
        
        # Clean up
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
    print("🚀 Starting Fruit Quality Prediction API")
    print("=" * 60)
    print(f"✅ TensorFlow Available: {TENSORFLOW_AVAILABLE}")
    print(f"✅ OpenCV Available: {OPENCV_AVAILABLE}")
    print(f"✅ PIL Available: {PILLOW_AVAILABLE}")
    print(f"✅ Predictor Initialized: {predictor is not None}")
    
    if predictor:
        print(f"✅ Models Loaded: {len(predictor.models)}")
        for model_name in predictor.models.keys():
            print(f"   - {model_name}")
    else:
        print("❌ Predictor not initialized - check logs for errors")
    
    print("\n📝 API Endpoints:")
    print("   GET  /               - API information")
    print("   GET  /health         - Health check")
    print("   POST /predict        - Upload image file")
    print("   POST /predict_base64 - Send base64 image")
    print("   POST /reload         - Reload models")
    print(f"\n🌐 Server running on port: {port}")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=port)
