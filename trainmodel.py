import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

# --------------------
# 1. Load CSV
# --------------------
df = pd.read_csv("A:\project\weather\data - Copy.csv")
data = df[['pressure', 'temperature', 'humidity']].values

# --------------------
# 2. Normalize
# --------------------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# --------------------
# 3. Create windows and delta targets
# --------------------
window_size = 12
pred_size = 12
X, y = [], []
testfrom=1170

for i in range(len(data_scaled) - window_size - pred_size + 1):
    X_window = data_scaled[i : i+window_size].flatten()
    Y_window = data_scaled[i+window_size : i+window_size+pred_size] - data_scaled[i+window_size-1]  # predict deltas
    X.append(X_window)
    y.append(Y_window.flatten())

X = np.array(X)
y = np.array(y)

print("Input shape:", X.shape)
print("Output shape:", y.shape)

# --------------------
# 4. Train/Test split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# --------------------
# 5. Build model
# --------------------
model = Sequential([
    Input(shape=(window_size*3,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(pred_size*3)  # predict deltas
])

# Weighted MSE to emphasize temperature
def weighted_mse(y_true, y_pred):
    weights = tf.constant([1.0, 3.0, 1.0]*pred_size, dtype=tf.float32)
    return tf.reduce_mean(tf.square((y_true - y_pred) * weights))

model.compile(optimizer='adam', loss=weighted_mse)

# --------------------
# 6. Train
# --------------------
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# --------------------
# 7. Predict function (accept raw 12 readings)
# --------------------
def predict_from_raw(raw_window):
    raw_window = np.array(raw_window)
    if raw_window.shape != (window_size, 3):
        raise ValueError(f"Expected shape ({window_size}, 3)")

    scaled = scaler.transform(raw_window)
    inp = scaled.flatten().reshape(1, -1)

    # predict delta
    pred_delta = model.predict(inp).reshape(pred_size, 3)

    # convert delta → absolute
    last_scaled = scaled[-1]
    pred_scaled = last_scaled + pred_delta
    pred_real = scaler.inverse_transform(pred_scaled)
    return pred_real

# --------------------
# 8. Test example
# --------------------
raw_text_input = """
96.45,29.82,73.00
96.45,29.81,73.00
96.45,29.81,79.40
96.45,29.81,73.00
96.45,29.79,72.80
96.45,29.79,72.80
96.45,29.81,73.00
96.45,29.82,72.80
96.45,29.84,73.00
96.45,29.84,73.00
96.45,29.81,73.00
96.45,29.83,73.00
"""

raw_text_next = """
96.45,29.82,73.00
96.45,29.81,73.00
96.45,29.81,79.40
96.45,29.81,73.00
96.45,29.79,72.80
96.45,29.79,72.80
96.45,29.81,73.00
96.45,29.82,72.80
96.45,29.84,73.00
96.45,29.84,73.00
96.45,29.81,73.00
96.45,29.83,73.00

"""

# Convert to numpy
#X_window = np.array([[float(x) for x in line.split(",")] for line in raw_text_input.strip().split("\n")])
#y_true_window = np.array([[float(x) for x in line.split(",")] for line in raw_text_next.strip().split("\n")])

# Get input & next 12 values as arrays
X_window = data[testfrom:testfrom+window_size]        
y_true_window = data[testfrom+window_size:testfrom+window_size+pred_size]  

# Predict
y_pred = predict_from_raw(X_window)

# RMSE
rmse_total = np.sqrt(mean_squared_error(y_true_window.flatten(), y_pred.flatten()))
print(f"\nOverall RMSE: {rmse_total:.3f}")

feature_names = ["Pressure", "Temperature", "Humidity"]
for i, name in enumerate(feature_names):
    rmse = np.sqrt(mean_squared_error(y_true_window[:, i], y_pred[:, i]))
    print(f"{name} RMSE: {rmse:.3f}")

print(f"predicted values:\n",y_pred)

# Save Keras model
model.save("weather_model.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save .tflite file
with open("weather_model.tflite", "wb") as f:
    f.write(tflite_model)

# Convert .tflite → .h file for Arduino
with open("weather_model.tflite", "rb") as f:
    tflite_bytes = f.read()

with open("weather_model.h", "w") as f:
    f.write("unsigned char weather_model_tflite[] = {")
    for i, b in enumerate(tflite_bytes):
        if i % 12 == 0:
            f.write("\n ")
        f.write(f"0x{b:02x}, ")
    f.write("\n};\n")
    f.write(f"unsigned int weather_model_tflite_len = {len(tflite_bytes)};\n")

print(scaler.data_min_)
print(scaler.data_max_)

