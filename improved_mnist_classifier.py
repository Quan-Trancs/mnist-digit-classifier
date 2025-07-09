import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
from datetime import datetime

class MNISTClassifier:
    def __init__(self, config=None):
        """Initialize the MNIST classifier with configuration."""
        self.config = config or self.get_default_config()
        self.model = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def get_default_config(self):
        """Default configuration for the model."""
        return {
            'img_height': 28,
            'img_width': 28,
            'num_classes': 10,
            'batch_size': 32,
            'epochs': 50,
            'validation_split': 0.2,
            'learning_rate': 0.001,
            'use_data_augmentation': True,
            'use_dropout': True,
            'model_save_path': 'models/mnist_cnn_model.h5',
            'history_save_path': 'models/training_history.json'
        }
    
    def load_data(self):
        """Load and preprocess MNIST data."""
        print("Loading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Normalize pixel values
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Reshape for CNN (add channel dimension)
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        
        # Convert labels to categorical
        y_train = to_categorical(y_train, 10)  # MNIST has 10 classes
        y_test = to_categorical(y_test, 10)    # MNIST has 10 classes
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_cnn_model(self):
        """Create a CNN model optimized for MNIST."""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25) if self.config['use_dropout'] else layers.Layer(),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25) if self.config['use_dropout'] else layers.Layer(),
            
            # Third Convolutional Block (optional, only if space allows)
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),  # Use global pooling instead of max pooling
            layers.Dropout(0.25) if self.config['use_dropout'] else layers.Layer(),
            
            # Dense Layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5) if self.config['use_dropout'] else layers.Layer(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5) if self.config['use_dropout'] else layers.Layer(),
            layers.Dense(10, activation='softmax')  # MNIST has 10 classes
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_callbacks(self):
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Add model checkpoint if save path is specified
        if self.config['model_save_path']:
            os.makedirs(os.path.dirname(self.config['model_save_path']), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    self.config['model_save_path'],
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        return callbacks
    
    def get_data_generator(self):
        """Get data generator for augmentation."""
        if not self.config['use_data_augmentation']:
            return None
            
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )
        
        return datagen
    
    def train(self):
        """Train the model."""
        if self.model is None:
            self.create_cnn_model()
        
        print("Model Summary:")
        self.model.summary()
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Get data generator
        datagen = self.get_data_generator()
        
        print(f"\nTraining for {self.config['epochs']} epochs...")
        
        if datagen:
            # Train with data augmentation
            self.history = self.model.fit(
                datagen.flow(self.X_train, self.y_train, batch_size=self.config['batch_size']),
                steps_per_epoch=len(self.X_train) // self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_data=(self.X_test, self.y_test),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Train without data augmentation
            self.history = self.model.fit(
                self.X_train, self.y_train,
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_split=self.config['validation_split'],
                callbacks=callbacks,
                verbose=1
            )
        
        return self.history
    
    def evaluate(self):
        """Evaluate the model on test set."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        print("\nEvaluating model on test set...")
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy
    
    def predict(self, image):
        """Predict digit from image."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Preprocess image
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            image = image.reshape(1, 28, 28, 1)
        
        # Normalize
        image = image.astype('float32') / 255.0
        
        # Predict
        prediction = self.model.predict(image, verbose=0)
        predicted_digit = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]
        
        return predicted_digit, confidence, prediction[0]
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path=None):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        save_path = path or self.config['model_save_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, path):
        """Load a trained model."""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
    
    def save_history(self, path=None):
        """Save training history."""
        if self.history is None:
            print("No training history to save.")
            return
        
        save_path = path or self.config['history_save_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, value in self.history.history.items():
            history_dict[key] = [float(x) for x in value]
        
        with open(save_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved to {save_path}")

def main():
    """Main function to run the MNIST classifier."""
    # Configuration
    config = {
        'batch_size': 64,
        'epochs': 30,
        'learning_rate': 0.001,
        'use_data_augmentation': True,
        'use_dropout': True,
        'model_save_path': 'models/mnist_cnn_model.h5',
        'history_save_path': 'models/training_history.json'
    }
    
    # Initialize classifier
    classifier = MNISTClassifier(config)
    
    # Load data
    classifier.load_data()
    
    # Train model
    history = classifier.train()
    
    # Evaluate model
    test_loss, test_accuracy = classifier.evaluate()
    
    # Plot training history
    classifier.plot_training_history()
    
    # Save model and history
    classifier.save_model()
    classifier.save_history()
    
    print(f"\n🎉 Training completed!")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main() 