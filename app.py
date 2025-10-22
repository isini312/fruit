import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
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
        
        # Create fruit name mapping for consistent display
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
        """Load model from local files with multiple fallbacks"""
        model_paths = [
            '/fruitnet_final_model.keras',
            '/best_model.keras',
            'fruitnet_final_model.keras',
            'best_model.keras',
            '/app/fruitnet_final_model.keras',  # Railway path
            '/app/best_model.keras',            # Railway path
        ]
        
        logger.info("🔄 Attempting to load model...")
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    logger.info(f"📁 Found model at: {model_path}")
                    self.model = load_model(model_path)
                    logger.info("✅ Model loaded successfully!")
                    logger.info(f"📊 Model summary: Input shape - {self.model.input_shape}, Output shape - {self.model.output_shape}")
                    return
                except Exception as e:
                    logger.error(f"❌ Error loading {model_path}: {e}")
                    continue
        
        logger.error("❌ No model files could be loaded from any path")
        logger.info("📋 Checked paths: " + ", ".join(model_paths))
    
    def preprocess_image(self, image):
        """Preprocess the image for prediction"""
        try:
            # Resize image to match model input size
            image = image.resize(self.IMG_SIZE)
            image_array = np.array(image)
            
            # If image has alpha channel, remove it
            if image_array.shape[-1] == 4:
                image_array = image_array[:, :, :3]
            
            # Normalize pixel values to [0, 1]
            image_array = image_array / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"❌ Image preprocessing failed: {e}")
            raise e
    
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
            logger.error(f"❌ Error parsing prediction class '{predicted_class}': {e}")
            return "Unknown", predicted_class
    
    def predict_image(self, image_file):
        """Predict fruit quality from image"""
        if self.model is None:
            return {"error": "Model not loaded. Please check server logs."}
        
        try:
            # Read and validate image file
            image_bytes = image_file.read()
            if len(image_bytes) == 0:
                return {"error": "Empty image file"}
            
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"📐 Image size: {image.size}, Mode: {image.mode}")
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            logger.info(f"🔧 Processed image shape: {processed_image.shape}")
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            logger.info(f"🎯 Raw prediction scores: {predictions[0]}")
            logger.info(f"📈 Predicted class index: {predicted_class_idx}, Confidence: {confidence:.4f}")
            
            # Validate prediction index
            if predicted_class_idx >= len(self.fruit_classes):
                return {"error": f"Invalid prediction index: {predicted_class_idx}"}
            
            # Get results
            predicted_class = self.fruit_classes[predicted_class_idx]
            quality, fruit = self.parse_prediction(predicted_class)
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = []
            
            for idx in top_3_indices:
                if idx < len(self.fruit_classes):
                    pred_class = self.fruit_classes[idx]
                    pred_quality, pred_fruit = self.parse_prediction(pred_class)
                    top_3_predictions.append({
                        'fruit': pred_fruit,
                        'quality': pred_quality,
                        'confidence': float(predictions[0][idx]),
                        'class_index': int(idx)
                    })
            
            # Calculate prediction quality metrics
            prediction_metrics = {
                'confidence_level': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.5 else 'Low',
                'top_3_agreement': len(set([p['fruit'] for p in top_3_predictions])) == 1
            }
            
            return {
                'success': True,
                'fruit': fruit,
                'quality': quality,
                'confidence': float(confidence),
                'full_class': predicted_class,
                'class_index': int(predicted_class_idx),
                'top_predictions': top_3_predictions,
                'prediction_metrics': prediction_metrics,
                'timestamp': np.datetime64('now').astype(str)
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

# Initialize predictor
predictor = FruitQualityPredictor()

@app.route('/')
def home():
    """Root endpoint with API information"""
    return jsonify({
        "message": "Fruit Quality Prediction API",
        "status": "running",
        "model_loaded": predictor.model is not None,
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)",
            "info": "/info (GET)"
        },
        "supported_fruits": ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate", "Lemon"],
        "quality_types": ["Good Quality", "Bad Quality", "Mixed Quality"]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "total_classes": len(predictor.fruit_classes) if predictor.model else 0
    })

@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    if predictor.model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_loaded": True,
        "input_shape": predictor.model.input_shape,
        "output_shape": predictor.model.output_shape,
        "total_classes": len(predictor.fruit_classes),
        "image_size": predictor.IMG_SIZE,
        "available_classes": predictor.fruit_classes
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        
        # Check if file is selected
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Check file type
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
        file_extension = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
        
        if file_extension not in allowed_extensions:
            return jsonify({
                "error": f"Invalid file type: {file_extension}. Supported types: {', '.join(allowed_extensions)}"
            }), 400
        
        # Check file size (limit to 10MB)
        image_file.seek(0, os.SEEK_END)
        file_size = image_file.tell()
        image_file.seek(0)  # Reset file pointer
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            return jsonify({"error": "File too large. Maximum size is 10MB"}), 400
        
        if file_size == 0:
            return jsonify({"error": "Empty file"}), 400
        
        logger.info(f"📨 Received prediction request: {image_file.filename} ({file_size} bytes)")
        
        # Make prediction
        result = predictor.predict_image(image_file)
        
        if 'error' in result:
            logger.error(f"❌ Prediction error: {result['error']}")
            return jsonify(result), 500
        
        logger.info(f"✅ Prediction successful: {result['fruit']} - {result['quality']} ({result['confidence']:.2%})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Server error in /predict: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting Fruit Quality API on port {port}")
    logger.info(f"📊 Model loaded: {predictor.model is not None}")
    app.run(host='0.0.0.0', port=port, debug=False)
