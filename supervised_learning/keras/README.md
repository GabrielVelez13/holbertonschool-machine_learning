## Introduction
This project demonstrates how to use Keras for building, training, evaluating, and predicting with neural network models.


## Usage

### Training a Model
```python
from train import train_model
import tensorflow.keras as K

# Define your model
model = K.Sequential([...])

# Train the model
history = train_model(model, X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_valid, Y_valid), early_stopping=True, patience=2, learning_rate_decay=True, alpha=0.01, decay_rate=1, verbose=True, shuffle=True)
```

### Testing a Model
```python
from test import test_model

# Test the model
loss, accuracy = test_model(model, X_test, Y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
```

### Making Predictions
```python
from predict import predict

# Make predictions
predictions = predict(model, X_new)
print(predictions)
```
