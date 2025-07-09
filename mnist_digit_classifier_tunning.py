# type: ignore
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras_tuner.tuners import Hyperband

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model builder for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Dense(
        units=hp.Int('units_input', min_value=32, max_value=512, step=32),
        activation='relu',
        input_shape=(784,)
    ))
    
    for i in range(hp.Int('num_hidden_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation='relu'
        ))

    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize tuner
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=15,
    factor=3,
    directory='mnist_tuning',
    project_name='digit_classifier'
)

# Early stopping
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Perform search
tuner.search(X_train, y_train, epochs=15, validation_split=0.2, callbacks=[stop_early])

# Best model
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(1)[0]

# Output best parameters
print(f"""
✅ Best hyperparameters found:
- Input layer units: {best_hps.get('units_input')}
- Hidden layers: {best_hps.get('num_hidden_layers')}
- Units per hidden layer: {[best_hps.get(f'units_{i}') for i in range(best_hps.get('num_hidden_layers'))]}
- Learning rate: {best_hps.get('learning_rate')}
""")

# Retrain best model (optional)
history = best_model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=128)

# Evaluate
loss, acc = best_model.evaluate(X_test, y_test)
print(f"🧪 Test accuracy: {acc:.4f}")
