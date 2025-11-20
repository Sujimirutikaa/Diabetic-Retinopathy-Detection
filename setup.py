import os
import subprocess
import sys

def create_project_structure():
    """Create the required project directory structure"""
    
    directories = [
        'templates',
        'uploads',
        'organized_data',
        'APTOS_2019_dataset',
        'APTOS_2019_dataset/train_images',
        'APTOS_2019_dataset/test_images'
    ]
    
    print("📁 Creating project directory structure...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def install_requirements():
    """Install required Python packages"""
    
    print("\n📦 Installing Python packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    
    return True

def check_gpu_support():
    """Check if GPU support is available"""
    
    print("\n🔧 Checking GPU support...")
    
    try:
        import tensorflow as tf
        
        print(f"TensorFlow version: {tf.__version__}")
        
        if tf.test.is_gpu_available():
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print(f"✅ GPU support available! Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("⚠️  GPU support not available. Training will use CPU.")
            print("   For faster training, consider installing GPU-enabled TensorFlow.")
    
    except ImportError:
        print("❌ TensorFlow not installed properly")
        return False
    
    return True

def create_sample_files():
    """Create sample configuration files"""
    
    print("\n📄 Creating configuration files...")
    
    # Create a simple config file
    config_content = """# Diabetic Retinopathy Detection Configuration

# Model settings
MODEL_TYPE = 'simple'  # 'simple' or 'efficientnet'
INPUT_SIZE = (224, 224, 3)
NUM_CLASSES = 5

# Training settings
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Data paths
APTOS_CSV = 'APTOS_2019_dataset/train.csv'
APTOS_IMAGES = 'APTOS_2019_dataset/train_images/'
ORGANIZED_DATA = 'organized_data/'
MODEL_SAVE_PATH = 'diabetic_retinopathy_model.h5'

# Flask settings
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
"""
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print("✅ Created: config.py")

def create_run_script():
    """Create a simple run script"""
    
    run_script_content = """#!/usr/bin/env python3
\"\"\"
Simple script to run the Diabetic Retinopathy Detection application
\"\"\"

import os
import sys

def main():
    print("🏥 Diabetic Retinopathy Detection System")
    print("=" * 40)
    
    while True:
        print("\\nChoose an option:")
        print("1. Prepare APTOS dataset")
        print("2. Train model")
        print("3. Run Flask web application")
        print("4. Exit")
        
        choice = input("\\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\\n📊 Preparing APTOS dataset...")
            os.system('python data_preparation.py')
            
        elif choice == '2':
            print("\\n🚀 Training model...")
            os.system('python model_training.py')
            
        elif choice == '3':
            print("\\n🌐 Starting Flask application...")
            os.system('python app.py')
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
"""
    
    with open('run.py', 'w') as f:
        f.write(run_script_content)
    
    print("✅ Created: run.py")

def print_instructions():
    """Print setup instructions"""
    
    instructions = """
🎉 Setup completed successfully!

📋 Next Steps:

1. Download APTOS 2019 Dataset:
   - Go to: https://www.kaggle.com/competitions/aptos2019-blindness-detection
   - Download 'train.csv' and 'train_images.zip'
   - Extract files to 'APTOS_2019_dataset/' folder

2. Organize your dataset:
   python data_preparation.py

3. Train your model:
   python model_training.py

4. Run the web application:
   python app.py
   
   Then open: http://localhost:5000

🚀 Quick Start (Alternative):
   python run.py

📁 Project Structure:
   diabetic_retinopathy_detection/
   ├── app.py                    # Flask web application
   ├── data_preparation.py       # Dataset organization
   ├── model_training.py         # Model training
   ├── requirements.txt          # Dependencies
   ├── config.py                 # Configuration
   ├── run.py                    # Quick run script
   ├── templates/
   │   └── index.html           # Web interface
   ├── uploads/                 # Temporary uploads
   ├── organized_data/          # Organized training data
   └── APTOS_2019_dataset/      # Original dataset

💡 Tips:
   - For GPU training, install tensorflow-gpu
   - Adjust batch_size in config.py if you get memory errors
   - Use 'efficientnet' model type for better accuracy (requires more resources)

📚 Documentation:
   - Check individual Python files for detailed usage
   - Modify config.py to customize settings
   
🆘 Need help? Check the comments in each Python file!
"""
    
    print(instructions)

def main():
    """Main setup function"""
    
    print("🏥 Diabetic Retinopathy Detection - Project Setup")
    print("=" * 50)
    
    # Create project structure
    create_project_structure()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Check GPU support
    check_gpu_support()
    
    # Create sample files
    create_sample_files()
    create_run_script()
    
    # Print instructions
    print_instructions()

if __name__ == "__main__":
    main()