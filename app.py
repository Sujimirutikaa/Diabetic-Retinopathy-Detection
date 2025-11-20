from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Disease stage mapping
STAGE_MAPPING = {
    0: "No DR (Healthy)",
    1: "Mild DR",
    2: "Moderate DR", 
    3: "Severe DR",
    4: "Proliferative DR"
}

# Global model variable
model = None

def create_model(input_shape=(224, 224, 3), num_classes=5):
    """Create a simple CNN model for diabetic retinopathy detection"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for prediction"""
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def load_pretrained_model():
    """Load a pre-trained model if available"""
    global model
    
    if os.path.exists('diabetic_retinopathy_model.h5'):
        try:
            model = tf.keras.models.load_model('diabetic_retinopathy_model.h5')
            print("Pre-trained model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = create_model()
            print("Created new model instead.")
    else:
        # Create a new model for demo
        model = create_model()
        print("Created new model. Train it with your APTOS data for better results.")

@app.route('/')
def home():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict diabetic retinopathy stage"""
    global model
    
    if model is None:
        load_pretrained_model()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess image
            processed_image = preprocess_image(filepath)
            
            # Make prediction
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get stage name
            stage_name = STAGE_MAPPING[predicted_class]
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'prediction': stage_name,
                'confidence': round(confidence * 100, 2),
                'stage': int(predicted_class),
                'all_probabilities': {
                    STAGE_MAPPING[i]: round(float(predictions[0][i]) * 100, 2) 
                    for i in range(len(predictions[0]))
                }
            })
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'error': 'Invalid file format. Please upload JPG, JPEG, or PNG files.'})

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Load or create model on startup
    load_pretrained_model()
    print("Starting Flask application...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)