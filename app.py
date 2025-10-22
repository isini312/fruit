import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import requests
from PIL import Image
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class FruitQualityPredictor:
    def __init__(self):
        self.IMG_SIZE = (100, 100)
        self.fruit_classes = [
            'Bad Quality_Fruits_Apple_Bad', 'Bad Quality_Fruits_Banana_Bad', 
            'Bad Quality_Fruits_Guava_Bad', 'Bad Quality_Fruits_Lime_Bad',
            'Bad Quality_Fruits_Orange_Bad', 'Bad Quality_Fruits_Pomegranate_Bad',
            'Good Quality_Fruits_Apple_Good', 'Good Quality_Fruits_Banana_Good',
            'Good Quality_Fruits_Guava_Good', 'Good Quality_Fruits_Lime_Good',
            'Good Quality_Fruits_Orange_Good', 'Good Quality_Fruits_Pomegranate_Good',
            'Mixed Qualit_Fruits_Apple', 'Mixed Qualit_Fruits_Banana',
            'Mixed Qualit_Fruits_Guava', 'Mixed Qualit_Fruits_Lemon',
            'Mixed Qualit_Fruits_Orange', 'Mixed Qualit_Fruits_Pomegranate'
        ]
        
        self.fruit_name_mapping = {
            'Apple_Bad': 'Apple', 'Apple_Good': 'Apple', 'Apple': 'Apple',
            'Banana_Bad': 'Banana', 'Banana_Good': 'Banana', 'Banana': 'Banana',
            'Guava_Bad': 'Guava', 'Guava_Good': 'Guava', 'Guava': 'Guava',
            'Lime_Bad': 'Lime', 'Lime_Good': 'Lime', 'Lemon': 'Lemon',
            'Orange_Bad': 'Orange', 'Orange_Good': 'Orange', 'Orange': 'Orange',
            'Pomegranate_Bad': 'Pomegranate', 'Pomegranate_Good': 'Pomegranate', 
            'Pomegranate': 'Pomegranate'
        }
        
        self.model = None
        self.load_model()
    
    def download_from_github(self, url, filename):
        """Download file from GitHub raw URL"""
        try:
            logger.info(f"📥 Downloading {filename} from GitHub...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✅ Successfully downloaded {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            return False
    
    def load_model(self):
        """Load the trained model from GitHub"""
        model_files = [
            {
                'url': 'https://raw.githubusercontent.com/isini312/fruit/main/best_model.keras',
                'filename': 'best_model.keras'
            },
            {
                'url': 'https://raw.githubusercontent.com/isini312/fruit/main/fruitnet_final_model.keras', 
                'filename': 'fruitnet_final_model.keras'
            }
        ]
        
        logger.info("🔄 Attempting to load model...")
        
        # Try each model file
        for model_info in model_files:
            filename = model_info['filename']
            url = model_info['url']
            
            # Download if file doesn't exist
            if not os.path.exists(filename):
                logger.info(f"📁 {filename} not found, downloading...")
                if not self.download_from_github(url, filename):
                    continue
            
            # Try to load the model
            try:
                logger.info(f"🔄 Loading model from: {filename}")
                self.model = load_model(filename)
                logger.info("✅ Model loaded successfully!")
                return
                
            except Exception as e:
                logger.error(f"❌ Error loading {filename}: {e}")
                # Remove corrupted file and try next one
                if os.path.exists(filename):
                    os.remove(filename)
                continue
        
        logger.error("❌ All model loading attempts failed")
    
    def preprocess_image(self, image):
        """Preprocess the image for prediction"""
        image = image.resize(self.IMG_SIZE)
        image_array = np.array(image)
        
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]
        
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    
    def parse_prediction(self, predicted_class):
        """Parse the predicted class into fruit type and quality"""
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
    
    def predict_image(self, image_file):
        """Predict fruit quality from image"""
        if self.model is None:
            return {"error": "Model not loaded. Please check server logs."}
        
        try:
            # Read image file
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess and predict
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Get results
            predicted_class = self.fruit_classes[predicted_class_idx]
            quality, fruit = self.parse_prediction(predicted_class)
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = []
            
            for idx in top_3_indices:
                pred_class = self.fruit_classes[idx]
                pred_quality, pred_fruit = self.parse_prediction(pred_class)
                top_3_predictions.append({
                    'fruit': pred_fruit,
                    'quality': pred_quality,
                    'confidence': float(predictions[0][idx])
                })
            
            return {
                'success': True,
                'fruit': fruit,
                'quality': quality,
                'confidence': float(confidence),
                'top_predictions': top_3_predictions
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

# Initialize predictor
predictor = FruitQualityPredictor()

@app.route('/')
def home():
    return jsonify({
        "message": "Fruit Quality Prediction API",
        "status": "running",
        "model_loaded": predictor.model is not None
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor.model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Check file type
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp'}
        if not ('.' in image_file.filename and 
                image_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({"error": "Invalid file type"}), 400
        
        # Make prediction
        result = predictor.predict_image(image_file)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Server error in /predict: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
