import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
import gdown

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
    
    def load_model(self):
        """Load the trained model with fallback options"""
        model_path = 'fruitnet_final_model.keras'
        
        # If model doesn't exist locally, download it
        if not os.path.exists(model_path):
            print("📥 Model not found locally. Downloading...")
            self.download_model()
        
        try:
            self.model = load_model(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def download_model(self):
        """Download model from cloud storage"""
        try:
            # Replace with your actual Google Drive file ID
            file_id = '1YOUR_FILE_ID_HERE'  # ⚠️ Replace this!
            url = f'https://drive.google.com/uc?id={file_id}'
            output = 'fruitnet_final_model.keras'
            
            gdown.download(url, output, quiet=False)
            print("✅ Model downloaded successfully!")
        except Exception as e:
            print(f"❌ Download failed: {e}")
    
    def preprocess_image(self, image):
        """Preprocess the image for prediction"""
        image = cv2.resize(image, self.IMG_SIZE)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    
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
            return "Unknown", predicted_class
    
    def predict_image(self, image_file):
        """Predict fruit quality from image"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Read image file
            image_bytes = image_file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"error": "Could not decode image"}
            
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
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# This is important for Railway
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
