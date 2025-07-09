import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import Hyperband, BayesianOptimization
import os
import json
from datetime import datetime

class ImprovedMNISTTuner:
    def __init__(self, tuner_type='hyperband'):
        """
        Initialize the improved MNIST tuner.
        
        Args:
            tuner_type: 'hyperband' or 'bayesian'
        """
        self.tuner_type = tuner_type
        self.tuner = None
        self.best_model = None
        self.best_hps = None
        
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
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_cnn_model(self, hp):
        """
        Build a CNN model with hyperparameters to tune.
        
        Args:
            hp: HyperParameters object from Keras Tuner
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(28, 28, 1)))
        
        # First convolutional block
        conv1_filters = hp.Int('conv1_filters', min_value=16, max_value=64, step=16)
        model.add(layers.Conv2D(conv1_filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        
        # Add second conv layer in first block (optional)
        if hp.Boolean('conv1_double'):
            model.add(layers.Conv2D(conv1_filters, (3, 3), activation='relu', padding='same'))
            model.add(layers.BatchNormalization())
        
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Second convolutional block
        conv2_filters = hp.Int('conv2_filters', min_value=32, max_value=128, step=32)
        model.add(layers.Conv2D(conv2_filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        
        if hp.Boolean('conv2_double'):
            model.add(layers.Conv2D(conv2_filters, (3, 3), activation='relu', padding='same'))
            model.add(layers.BatchNormalization())
        
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Third convolutional block (optional)
        if hp.Boolean('use_conv3'):
            conv3_filters = hp.Int('conv3_filters', min_value=64, max_value=256, step=64)
            model.add(layers.Conv2D(conv3_filters, (3, 3), activation='relu', padding='same'))
            model.add(layers.BatchNormalization())
            
            if hp.Boolean('conv3_double'):
                model.add(layers.Conv2D(conv3_filters, (3, 3), activation='relu', padding='same'))
                model.add(layers.BatchNormalization())
            
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(hp.Float('dropout3', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Flatten
        model.add(layers.Flatten())
        
        # Dense layers
        dense1_units = hp.Int('dense1_units', min_value=128, max_value=512, step=128)
        model.add(layers.Dense(dense1_units, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(hp.Float('dropout_dense1', min_value=0.2, max_value=0.6, step=0.1)))
        
        # Second dense layer (optional)
        if hp.Boolean('use_dense2'):
            dense2_units = hp.Int('dense2_units', min_value=64, max_value=256, step=64)
            model.add(layers.Dense(dense2_units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(hp.Float('dropout_dense2', min_value=0.2, max_value=0.6, step=0.1)))
        
        # Output layer
        model.add(layers.Dense(10, activation='softmax'))
        
        # Compile model
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_tuner(self, max_epochs=20, factor=3):
        """
        Create the hyperparameter tuner.
        
        Args:
            max_epochs: Maximum epochs for each trial
            factor: Factor for Hyperband (ignored for Bayesian)
        """
        if self.tuner_type == 'hyperband':
            self.tuner = Hyperband(
                self.build_cnn_model,
                objective='val_accuracy',
                max_epochs=max_epochs,
                factor=factor,
                directory='improved_tuning',
                project_name='mnist_cnn_tuner',
                overwrite=True
            )
        elif self.tuner_type == 'bayesian':
            self.tuner = BayesianOptimization(
                self.build_cnn_model,
                objective='val_accuracy',
                max_trials=50,
                directory='improved_tuning',
                project_name='mnist_cnn_bayesian',
                overwrite=True
            )
        else:
            raise ValueError("tuner_type must be 'hyperband' or 'bayesian'")
        
        return self.tuner
    
    def get_callbacks(self):
        """Get training callbacks."""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            )
        ]
    
    def search(self, max_trials=None):
        """
        Perform hyperparameter search.
        
        Args:
            max_trials: Maximum number of trials (for Bayesian optimization)
        """
        if self.tuner is None:
            self.create_tuner()
        
        print(f"Starting {self.tuner_type} hyperparameter search...")
        
        if max_trials and self.tuner_type == 'bayesian':
            self.tuner.search(
                self.X_train, self.y_train,
                epochs=20,
                validation_split=0.2,
                callbacks=self.get_callbacks(),
                max_trials=max_trials,
                verbose=1
            )
        else:
            self.tuner.search(
                self.X_train, self.y_train,
                epochs=20,
                validation_split=0.2,
                callbacks=self.get_callbacks(),
                verbose=1
            )
        
        # Get best model and hyperparameters
        self.best_model = self.tuner.get_best_models(num_models=1)[0]
        self.best_hps = self.tuner.get_best_hyperparameters(1)[0]
        
        return self.best_model, self.best_hps
    
    def print_best_hyperparameters(self):
        """Print the best hyperparameters found."""
        if self.best_hps is None:
            print("No hyperparameters found. Run search() first.")
            return
        
        print("\n" + "="*50)
        print("🎯 BEST HYPERPARAMETERS FOUND")
        print("="*50)
        
        # Convolutional layers
        print(f"📊 Convolutional Layers:")
        print(f"   Conv1 filters: {self.best_hps.get('conv1_filters')}")
        print(f"   Conv1 double: {self.best_hps.get('conv1_double')}")
        print(f"   Conv2 filters: {self.best_hps.get('conv2_filters')}")
        print(f"   Conv2 double: {self.best_hps.get('conv2_double')}")
        print(f"   Use Conv3: {self.best_hps.get('use_conv3')}")
        
        if self.best_hps.get('use_conv3'):
            print(f"   Conv3 filters: {self.best_hps.get('conv3_filters')}")
            print(f"   Conv3 double: {self.best_hps.get('conv3_double')}")
        
        # Dropout rates
        print(f"\n🔒 Dropout Rates:")
        print(f"   Dropout1: {self.best_hps.get('dropout1'):.2f}")
        print(f"   Dropout2: {self.best_hps.get('dropout2'):.2f}")
        if self.best_hps.get('use_conv3'):
            print(f"   Dropout3: {self.best_hps.get('dropout3'):.2f}")
        
        # Dense layers
        print(f"\n🧠 Dense Layers:")
        print(f"   Dense1 units: {self.best_hps.get('dense1_units')}")
        print(f"   Use Dense2: {self.best_hps.get('use_dense2')}")
        if self.best_hps.get('use_dense2'):
            print(f"   Dense2 units: {self.best_hps.get('dense2_units')}")
        
        # Training parameters
        print(f"\n⚙️ Training Parameters:")
        print(f"   Learning rate: {self.best_hps.get('learning_rate'):.6f}")
        
        print("="*50)
    
    def evaluate_best_model(self):
        """Evaluate the best model on test set."""
        if self.best_model is None:
            print("No best model found. Run search() first.")
            return None, None
        
        print("\nEvaluating best model on test set...")
        test_loss, test_accuracy = self.best_model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print(f"🎯 Best Model Performance:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy
    
    def save_best_model(self, path='models/best_mnist_cnn_model.h5'):
        """Save the best model."""
        if self.best_model is None:
            print("No best model to save.")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.best_model.save(path)
        print(f"Best model saved to {path}")
    
    def save_hyperparameters(self, path='models/best_hyperparameters.json'):
        """Save the best hyperparameters."""
        if self.best_hps is None:
            print("No hyperparameters to save.")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert hyperparameters to dictionary
        hps_dict = {}
        for param in self.best_hps.values:
            hps_dict[param] = self.best_hps.get(param)
        
        with open(path, 'w') as f:
            json.dump(hps_dict, f, indent=2)
        
        print(f"Best hyperparameters saved to {path}")
    
    def retrain_best_model(self, epochs=30, batch_size=64):
        """
        Retrain the best model with more epochs.
        
        Args:
            epochs: Number of epochs for retraining
            batch_size: Batch size for training
        """
        if self.best_model is None:
            print("No best model found. Run search() first.")
            return None
        
        print(f"\nRetraining best model for {epochs} epochs...")
        
        # Create a fresh model with best hyperparameters
        best_model = self.build_cnn_model(self.best_hps)
        
        # Train with more epochs
        history = best_model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = best_model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"\n🎉 Retrained Model Performance:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        
        return best_model, history

def main():
    """Main function to run hyperparameter tuning."""
    # Initialize tuner
    tuner = ImprovedMNISTTuner(tuner_type='hyperband')  # or 'bayesian'
    
    # Load data
    tuner.load_data()
    
    # Create tuner
    tuner.create_tuner(max_epochs=20, factor=3)
    
    # Perform search
    best_model, best_hps = tuner.search()
    
    # Print results
    tuner.print_best_hyperparameters()
    
    # Evaluate best model
    test_loss, test_accuracy = tuner.evaluate_best_model()
    
    # Save results
    tuner.save_best_model()
    tuner.save_hyperparameters()
    
    # Optional: Retrain with more epochs
    print("\n" + "="*50)
    print("🔄 RETRAINING WITH MORE EPOCHS")
    print("="*50)
    retrained_model, history = tuner.retrain_best_model(epochs=50, batch_size=64)
    
    if retrained_model:
        # Save retrained model
        retrained_model.save('models/retrained_best_model.h5')
        print("Retrained model saved to models/retrained_best_model.h5")

if __name__ == "__main__":
    main() 