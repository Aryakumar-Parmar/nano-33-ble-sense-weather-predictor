# ------------------------- Libraries -------------------------------------------------------------------------------------------------------
import pandas as pd                                   # For reading CSV and data manipulation
import numpy as np                                    # For numerical operations
from sklearn.preprocessing import MinMaxScaler        # To normalize data to [0,1]
from sklearn.model_selection import train_test_split  # To split data into train/test sets
import tensorflow as tf                               # TensorFlow core
from tensorflow.keras import Sequential, Input        # For building Keras sequential models
from tensorflow.keras.layers import Dense             # Fully connected (Dense) layer
from sklearn.metrics import mean_squared_error        # For RMSE calculation

# ------------------------- 1. Load CSV ----------------------------------------------------------------------------------------------------
df = pd.read_csv("A:\project\weather\data - Copy.csv")     # Load weather CSV
data = df[['pressure', 'temperature', 'humidity']].values  # Extract relevant features

# ------------------------- 2. Normalize ---------------------------------------------------------------------------------------------------
scaler = MinMaxScaler()                       # Initialize MinMaxScaler to scale features to [0,1]
data_scaled = scaler.fit_transform(data)      # Fit scaler to data and transform

# ------------------------- 3. Create windows and delta targets ---------------------------------------------------------------------------
window_size = 12                               # Number of past readings to use as input
pred_size = 12                                 # Number of future readings to predict
X, y = [], []                                  # Lists to hold input/output samples
testfrom = 1170                                # Index from where to test prediction

# Create input/output windows
for i in range(len(data_scaled) - window_size - pred_size + 1):
    X_window = data_scaled[i : i+window_size].flatten()                                             # Flatten past window to 1D array
    Y_window = data_scaled[i+window_size : i+window_size+pred_size] - data_scaled[i+window_size-1]  # Predict deltas
    X.append(X_window)                                                                              # Add input to X
    y.append(Y_window.flatten())                                                                    # Add output (delta) to y

X = np.array(X)   # Convert list to NumPy array
y = np.array(y)   # Convert list to NumPy array

print("Input shape:", X.shape)     # Print input array shape
print("Output shape:", y.shape)    # Print output array shape

# ------------------------- 4. Train/Test split ---------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)  # Split 80/20

# ------------------------- 5. Build model -------------------------------------------------------------------------------------------------
model = Sequential([
    Input(shape=(window_size*3,)),  # Input layer: flattened 12x3 window
    Dense(64, activation='relu'),   # Dense layer with ReLU activation
    Dense(64, activation='relu'),   # Another Dense layer with ReLU
    Dense(pred_size*3)               # Output layer: predict deltas for next 12 readings
])

# ------------------------- Weighted Loss --------------------------------------------------------------------------------------------------
def weighted_mse(y_true, y_pred):
    """
    Custom loss to emphasize temperature prediction errors more.
    """
    weights = tf.constant([1.0, 3.0, 1.0]*pred_size, dtype=tf.float32)   # Weight T three times more
    return tf.reduce_mean(tf.square((y_true - y_pred) * weights))        # Weighted mean squared error

model.compile(optimizer='adam', loss=weighted_mse)  # Compile model with Adam optimizer

# ------------------------- 6. Train model -------------------------------------------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,                 # Number of times the entire training dataset is passed through the model
    batch_size=16,             # Number of samples processed before updating the model weights
    validation_split=0.2,      # Fraction of training data used to evaluate performance after each epoch
    verbose=1                  # Controls the amount of output shown during training
)

# ------------------------- 7. Predict function -------------------------------------------------------------------------------------------
def predict_from_raw(raw_window):
    """
    Predict next 12 readings given raw 12 past readings.
    """
    raw_window = np.array(raw_window)
    if raw_window.shape != (window_size, 3):
        raise ValueError(f"Expected shape ({window_size}, 3)")

    scaled = scaler.transform(raw_window)          # Normalize input using fitted scaler
    inp = scaled.flatten().reshape(1, -1)          # Flatten and reshape for model input

    pred_delta = model.predict(inp).reshape(pred_size, 3)  # Predict deltas for next 12 readings
    last_scaled = scaled[-1]                                # Last scaled value in input window
    pred_scaled = last_scaled + pred_delta                  # Add delta to last input to get scaled predictions
    pred_real = scaler.inverse_transform(pred_scaled)       # Convert back to original units
    return pred_real

# ------------------------- 8. Test example ------------------------------------------------------------------------------------------------
# Select input window and true next 12 readings
X_window = data[testfrom:testfrom+window_size]                  # Input: last 12 readings
y_true_window = data[testfrom+window_size:testfrom+window_size+pred_size]  # True next 12 readings

y_pred = predict_from_raw(X_window)                             # Get predictions from model

# ------------------------- 9. Compute RMSE -----------------------------------------------------------------------------------------------
rmse_total = np.sqrt(mean_squared_error(y_true_window.flatten(), y_pred.flatten()))  # Overall RMSE
print(f"\nOverall RMSE: {rmse_total:.3f}")

feature_names = ["Pressure", "Temperature", "Humidity"]
for i, name in enumerate(feature_names):
    rmse = np.sqrt(mean_squared_error(y_true_window[:, i], y_pred[:, i]))  # RMSE per feature
    print(f"{name} RMSE: {rmse:.3f}")

print(f"Predicted values:\n", y_pred)  # Print predicted next 12 readings

# ------------------------- 10. Save Keras model -------------------------------------------------------------------------------------------
model.save("weather_model.h5")  # Save trained model in .h5 format

# ------------------------- 11. Convert to TensorFlow Lite --------------------------------------------------------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)  # Convert Keras model to TFLite
tflite_model = converter.convert()

with open("weather_model.tflite", "wb") as f:                  # Save .tflite model to file
    f.write(tflite_model)

# ------------------------- 12. Convert .tflite → C header file for Arduino ---------------------------------------------------------------
with open("weather_model.tflite", "rb") as f:
    tflite_bytes = f.read()

with open("weather_model.h", "w") as f:
    f.write("unsigned char weather_model_tflite[] = {")        # Start C array
    for i, b in enumerate(tflite_bytes):
        if i % 12 == 0:
            f.write("\n ")
        f.write(f"0x{b:02x}, ")                               # Write each byte in hex
    f.write("\n};\n")
    f.write(f"unsigned int weather_model_tflite_len = {len(tflite_bytes)};\n")  # Store length

# ------------------------- 13. Print scaler info ------------------------------------------------------------------------------------------
print("Min values:", scaler.data_min_)    # Minimums used for normalization
print("Max values:", scaler.data_max_)    # Maximums used for normalization
