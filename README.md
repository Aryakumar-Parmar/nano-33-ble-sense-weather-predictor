# 🌤️ nano 33 ble sense weather predictor

A simple weather prediction project using the **Arduino Nano 33 BLE Sense Lite** and a **DHT11** sensor.  
It collects real-time **temperature, humidity, and pressure** data for a **hour** and predicts **next hours** upcoming values using a trained model.

---

## 🧠 Overview

This project demonstrates how to:
- Collect live sensor data via Arduino.
- Send it to a Python script for CSV logging.
- Train a lightweight model to predict short-term weather changes.

If you’re using the **regular Nano 33 BLE** (non-Lite), you can skip the external humidity sensor —  
it already includes a built-in one.

---

## ⚙️ Hardware Setup

| Component | Purpose | Notes |
|------------|----------|-------|
| **Arduino Nano 33 BLE Sense Lite** | Core MCU | Used for data collection |
| **DHT11 Sensor** | Humidity & temperature | Replace with DHT22 / BME280 for higher accuracy |
| **Barometric Pressure Sensor (optional)** | Pressure readings | Any I²C sensor like BMP280 works |

⚠️ Important: The Lite version doesn’t have a humidity sensor bult-in,use DHT11 or better.
It is important to use good sensor for any actual use

remember to put nano 33 to programming mode before uploading

conncetion are shown in .ino file

---

## 💻 Software Components

| File | Description |
|------|-------------|
| `requirements.txt` | Python libraries required for the project. |
| `install_req.bat` | Batch file to install the Python libraries. |
| `collectdata.py` | Reads data from Arduino Nano and saves to `data.csv` (no Arduino IDE needed). |
| `weather_collect.ino` | Arduino sketch to collect training data. |
| `data.csv` | Logged data file created automatically by `collectdata.py`. |
| `data-Copy.csv` *(optional)* | Backup copy of `data.csv`. Only one app can use `data.csv` at a time. |
| `trainmodel.py` | Script to train and test the prediction model. |
| `model_test.ino` | Arduino sketch to test the trained model before deployment. |
| `weather_predict.ino` | Arduino sketch to predict sensor values using the trained model. |

---

## How It Works

1. **Data Collection**:  
   Arduino collects pressure, temperature, and humidity samples. Data is logged via Python.  

2. **Preprocessing**:  
   - Min-Max normalization  
   - Sliding windows of 12 samples (`window_size`)  
   - Predicting deltas for next 12 samples (`pred_size`)  

3. **Model**:  
   - Dense neural network with 2 hidden layers (64 neurons each, ReLU activation)  
   - Weighted MSE loss to emphasize temperature prediction  

4. **Prediction**:  
   - Model predicts deltas, converts them to absolute values using the last reading, and denormalizes using the `MinMaxScaler`.  

5. **Arduino Deployment**:  
   - Model converted to `.tflite` → `.h`  
   - Loaded in Arduino for real-time predictions  

---

## How to Use

1. **Install all required libraries** for both Python and Arduino IDE.

2. **Upload** `weather_collect.ino` to your Arduino Nano 33 BLE Sense.

3. **Run** `collectdata.py` (ensure the Serial Monitor is closed).  
   Repeat twice if it does not work the first time.

4. **Collect data** for at least 3–4 days.  
   For long-term prediction, extend data collection to 1–2 weeks per season.

5. **Run** `trainmodel.py` to train and test the model.

6. **Copy the min and max values** from the Python scaler and use them in the Arduino sketch  
   to scale sensor data in the same range as used for training.
   
8. **Place** the `weather_model.h` file and the min/max arrays into the Arduino project folder you want to use.

   **It’s best to export the compiled binary** (Sketch → Export Compiled Binary)  
   **so it doesn't compile every time. Note: compilation may take some time.**

9. **(Optional)** Test the model with model_test.ino in a controlled environment and verify your preprocessing steps.

10. **Upload** `weather_predict.ino` to your Arduino for real-time predictions.


---

## Requirements

- **Python 3.x**  
- **Arduino IDE**  
- Python Libraries:
  - pandas
  - numpy
  - scikit-learn
  - tensorflow  

---

## Notes

- Only one application should use `data.csv` at a time; use `data-Copy.csv` as backup if needed.  
- Weighted MSE loss emphasizes temperature prediction accuracy.  
- RMSE for each feature is printed after training.  

