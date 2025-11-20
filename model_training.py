import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def create_simple_cnn_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Create a simple CNN model for diabetic retinopathy detection
    
    Args:
        input_shape (tuple): Input image shape
        num_classes (int): Number of classes (5 for DR stages)
    
    Returns:
        keras.Model: Compiled model
    """
    
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
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Create an EfficientNet-based model for better performance
    
    Args:
        input_shape (tuple): Input image shape
        num_classes (int): Number of classes
    
    Returns:
        keras.Model: Compiled model
    """
    
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_data_generators(data_dir, batch_size=32, validation_split=0.2):
    """
    Create data generators for training and validation
    
    Args:
        data_dir (str): Directory containing organized training data
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data to use for validation
    
    Returns:
        tuple: (train_generator, validation_generator)
    """
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )
    
    # Validation data generator
    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def train_diabetic_retinopathy_model(
    data_dir,
    model_type='simple',
    model_save_path='diabetic_retinopathy_model.h5',
    epochs=20,
    batch_size=32,
    validation_split=0.2
):
    """
    Train a CNN model for diabetic retinopathy detection
    
    Args:
        data_dir (str): Directory containing organized training data
        model_type (str): 'simple' or 'efficientnet'
        model_save_path (str): Path to save trained model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data for validation
    
    Returns:
        tuple: (model, history)
    """
    
    print(f"🚀 Starting training with {model_type} model...")
    print(f"Data directory: {data_dir}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Create model
    if model_type == 'efficientnet':
        model = create_efficientnet_model()
        print("📱 Using EfficientNetB0 model")
    else:
        model = create_simple_cnn_model()
        print("🔧 Using simple CNN model")
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Create data generators
    train_generator, validation_generator = create_data_generators(
        data_dir, batch_size, validation_split
    )
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\n🏃 Starting training...")
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ Training completed!")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Model saved to: {model_save_path}")
    
    return model, history

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history
    
    Args:
        history: Training history object
        save_path (str): Path to save the plot
    """
    
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate (if available)
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], marker='o')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
    else:
        plt.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                ha='center', va='center', transform=plt.gca().transAxes)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training history plot saved to: {save_path}")

def evaluate_model(model, data_dir, batch_size=32):
    """
    Evaluate the trained model
    
    Args:
        model: Trained Keras model
        data_dir (str): Directory containing test/validation data
        batch_size (int): Batch size for evaluation
    """
    
    print("📊 Evaluating model...")
    
    # Create test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )
    
    # Predict
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(
        true_classes, 
        predicted_classes, 
        target_names=class_labels
    ))
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_classes, predicted_classes)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main training function
    """
    
    print("🏥 Diabetic Retinopathy Detection - Model Training")
    print("=" * 50)
    
    # Configuration
    DATA_DIR = 'organized_data/'
    MODEL_TYPE = 'simple'  # 'simple' or 'efficientnet'
    EPOCHS = 20
    BATCH_SIZE = 32
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        print("Please run data_preparation.py first to organize your APTOS dataset.")
        return
    
    # Get user preferences
    print(f"Current configuration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Model type: {MODEL_TYPE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    proceed = input("\nProceed with training? (y/n): ").lower().strip() == 'y'
    if not proceed:
        print("Training cancelled.")
        return
    
    try:
        # Train model
        model, history = train_diabetic_retinopathy_model(
            data_dir=DATA_DIR,
            model_type=MODEL_TYPE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate model
        evaluate_model(model, DATA_DIR, BATCH_SIZE)
        
        print("\n🎉 Training completed successfully!")
        print("You can now use the trained model in your Flask application.")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()