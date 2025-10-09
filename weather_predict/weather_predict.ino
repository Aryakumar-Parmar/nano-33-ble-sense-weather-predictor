// -------------------------  Hardware headers ------------------------------------------------------------------------------------------------
#include <Arduino.h>                                          // Core Arduino functions and definitions (Serial, pinMode, digitalWrite, etc.)
#include <Arduino_LPS22HB.h>                                  // Library for LPS22HB Barometer/Temperature sensor
#include "dht.h"                                              // Library for DHT11 humidity/temperature sensor
#include "weather_model.h"                                    // TensorFlow Lite model header (compiled tflite model)
#include "mbed.h"                                             // MBED OS header (provides sleep, thread, and timing utilities)
// -------------------------  TensorFlow Lite Micro headers -----------------------------------------------------------------------------------
#include "TensorFlowLite.h"                                   // Core TFLite Micro definitions
#include "tensorflow/lite/micro/micro_interpreter.h"          // Interpreter to run TFLite Micro models
#include "tensorflow/lite/micro/micro_error_reporter.h"       // For reporting errors from TFLite Micro
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"  // Registers operations for the model
#include "tensorflow/lite/schema/schema_generated.h"          // TFLite model schema (generated from .tflite file)
#include "tensorflow/lite/version.h"                          // TensorFlow Lite version info

// ------------------------- Sensor & Pin Definitions -----------------------------------------------------------------------------------------
#define dht_apin A6                                           // Analog pin connected to DHT11 data line
dht DHT;                                                      // Instantiate DHT11 object
#define averageOver 5                                         // Number of readings to average for smoother sensor output
//#define DHTPIN 2                                              // Digital pin number for DHT11 (optional, if using digital read)
#define DHTTYPE DHT11                                         // DHT sensor type (DHT11)

// ------------------------- Global Variables ------------------------------------------------------------------------------------------------
float pressure, temperature, humidity;                        // Variables to store raw sensor readings
float minVals[3] = {96.18, 28.72, 55.4};                      // Minimum expected values for Pressure, Temperature, Humidity for processing same as we did in training
float maxVals[3] = {97.11, 36.03, 144.8};                     // Maximum expected values for Pressure, Temperature, Humidity for processing same as we did in training
float p_pred,t_pred,h_pred;                                   // Variables to store predicted sensor readings
unsigned long startTime;                                      // Variable to store timestamp for timing loops or delays
// ------------------------- TensorFlow Lite Error Handling ----------------------------------------------------------------------------------
tflite::ErrorReporter* error_reporter = nullptr;              // Pointer for error reporting
tflite::MicroErrorReporter micro_error_reporter;              // TFLite Micro error reporter instance

// ------------------------- TensorFlow Lite Interpreter -------------------------------------------------------------------------------------
//const tflite::Model* model = nullptr;                         // Pointer to TFLite model, will be assigned later
tflite::MicroInterpreter* interpreter = nullptr;              // Pointer to TFLite Micro interpreter
tflite::MicroMutableOpResolver<5> resolver;                   // Resolver to register up to 5 operations (fully connected, relu, etc.)

// ------------------------- Tensor Arena ----------------------------------------------------------------------------------------------------
constexpr int kTensorArenaSize = 20 * 1024;                   // 20 KB arena for tensor computations (RAM reserved for model)
static uint8_t tensor_arena[kTensorArenaSize];                // Memory buffer used by TFLite Micro for tensors

// ------------------------- Input/Output Tensors --------------------------------------------------------------------------------------------
TfLiteTensor* input;                                          // Pointer to input tensor of the model
TfLiteTensor* output;                                         // Pointer to output tensor of the model

// ------------------------- Value Processing Function ---------------------------------------------------------------------------------------
// Function to normalize and clamp sensor readings to the range 0-1 for TFLite model input
// This is required because the TFLite model was trained on normalized data
// n=0 for Pressure
// n=1 for Temperature
// n=2 for Humidity
float processValue(float val, int n) 
{
  // ------------------------- Clamping ------------------------------------------------------------------------------------------------------
  // Ensure the sensor value does not go below the minimum expected value
  if(val < minVals[n]) val = minVals[n];

  // Ensure the sensor value does not exceed the maximum expected value
  if(val > maxVals[n]) val = maxVals[n];

  // ------------------------- Scaling -------------------------------------------------------------------------------------------------------
  // Convert the clamped value to a 0-1 range based on the min/max values
  return (val - minVals[n]) / (maxVals[n] - minVals[n]);
}
// ------------------------- Sensor Reading & Averaging Function -----------------------------------------------------------------------------
// Function to read sensor values multiple times, average them, and store in global variables
void readValues() 
{
    // ------------------------- Initialize variables ----------------------------------------------------------------------------------------
    // Reset the variables before accumulating readings
    pressure = 0;      
    temperature = 0;   
    humidity = 0;      

    // ------------------------- Take Multiple Readings --------------------------------------------------------------------------------------
    // Loop to read each sensor multiple times for averaging
    for(int count = 0; count < averageOver; count++) 
    {
        // Read pressure from LPS22HB barometer and add to accumulator
        pressure += BARO.readPressure();

        // Read temperature from LPS22HB barometer and add to accumulator
        temperature += BARO.readTemperature();

        // Read humidity from DHT11 sensor
        DHT.read11(dht_apin);
        humidity += DHT.humidity;

        // ------------------------- Delay Between Readings -----------------------------------------------------------------------------
        // Wait 2 seconds between readings to allow sensors to stabilize
        // thread_sleep_for() is used instead of delay() for better timing with MBED
        thread_sleep_for(2000UL); 
    }

    // ------------------------- Compute Averages ---------------------------------------------------------------------------------------
    // Divide the accumulated sums by the number of readings to get average
    pressure    /= averageOver;
    temperature /= averageOver;
    humidity    /= averageOver;
}

// ------------------------- Arduino Setup Function ------------------------------------------------------------------------------------------
void setup() 
{
    // ------------------------- Initialize Serial Communication -----------------------------------------------------------------------------
    Serial.begin(9600);           // Start serial communication at 9600 baud
    while (!Serial);              // Wait for the Serial port to be ready (necessary for some boards)

    // ------------------------- Initialize Sensors ------------------------------------------------------------------------------------------
    BARO.begin();                 // Initialize LPS22HB barometer/temperature sensor
    // DHT11 does not require explicit begin() in this library

    // ------------------------- Setup TensorFlow Lite Error Reporting -----------------------------------------------------------------------
    error_reporter = &micro_error_reporter; // Assign global error reporter to TFLite Micro error instance

    // ------------------------- Load TFLite Model -------------------------------------------------------------------------------------------
    // Load the compiled TensorFlow Lite model from header file
    const tflite::Model* model  = tflite::GetModel(weather_model_tflite);

    // ------------------------- Setup Operation Resolver ------------------------------------------------------------------------------------
    // Resolver maps TFLite operations to their implementations in memory
    resolver.AddFullyConnected(); // Register FullyConnected (Dense) layer
    resolver.AddRelu();           // Register ReLU activation

    // ------------------------- Initialize TFLite Micro Interpreter -------------------------------------------------------------------------
    static tflite::MicroInterpreter static_interpreter(
        model,                // Pointer to the TFLite model
        resolver,             // Registered operations
        tensor_arena,         // Memory buffer for tensor calculations
        kTensorArenaSize,     // Size of tensor arena
        error_reporter,       // Error reporting instance
        nullptr               // Optional profiler (not used here)
    );

    interpreter = &static_interpreter; // Assign the global interpreter pointer

    // ------------------------- Allocate Memory for Tensors ---------------------------------------------------------------------------------
    if (interpreter->AllocateTensors() != kTfLiteOk) { // Allocate memory for model tensors
        Serial.println("AllocateTensors() failed");     // Print error if allocation fails
        while (1);                                     // Stop execution if tensor allocation fails
    }

    // ------------------------- Assign Input and Output Tensors -----------------------------------------------------------------------------
    input = interpreter->input(0);   // Pointer to the model's input tensor
    output = interpreter->output(0); // Pointer to the model's output tensor

    // ------------------------- Startup Messages --------------------------------------------------------------------------------------------
    Serial.println("Model loaded!");                        // Inform user that model is ready
    Serial.println("Pressure, Temperature, Humidity");      // Header for sensor output
}

void loop()
{
    // -----------------------------Loop to collect 12 sets of sensor readings (1 per 5 minutes)----------------------------------------------
    for(int i = 0; i < 12; i++)
    {
        startTime = millis();                 // Store current timestamp for timing delays
        readValues();                         // Read and average Pressure, Temperature, Humidity from sensors
        // ------------------------- Print Raw Sensor Values ---------------------------------------------------------------------------------
        Serial.print(i);
        Serial.print(") raw P: ");
        Serial.print(pressure, 2);            // Print pressure with 2 decimal places
        Serial.print(", T: ");
        Serial.print(temperature, 2);         // Print temperature with 2 decimal places
        Serial.print(", H: ");
        Serial.println(humidity, 2);          // Print humidity with 2 decimal places

        // ------------------------- Process Sensor Values -----------------------------
        pressure    = processValue(pressure, 0);      // Normalize pressure (0-1) based on min/max
        temperature = processValue(temperature, 1);   // Normalize temperature
        humidity    = processValue(humidity, 2);      // Normalize humidity

        // ------------------------- Assign Values to Model Input ---------------------
        input->data.f[i*3 + 0] = pressure;           // Input tensor: pressure
        input->data.f[i*3 + 1] = temperature;        // Input tensor: temperature
        input->data.f[i*3 + 2] = humidity;           // Input tensor: humidity

        // ------------------------- Wait for 5 Minutes Before Next Reading ----------
        while(millis() - startTime <= 5UL * 60UL * 1000UL) // 5 minutes delay
        {  
            thread_sleep_for(1000UL);                 // Sleep 1 second to reduce CPU usage
        }
    }

    // ------------------------- Store Last Scaled Values ----------------------------
    // Save last normalized readings for prediction as in training we made it predict(deltas) change in sensor values
    float lastScaled[3] = {input->data.f[33], input->data.f[34], input->data.f[35]}; 

    // ------------------------- Invoke TFLite Model --------------------------------
    // In simple terms run the model
    if (interpreter->Invoke() != kTfLiteOk) 
    {
        Serial.println("Invoke failed!");           // Print error if model invocation fails
        return;
    }
    // ------------------------- Convert Model Output Back to Real Values -----------
    // Output tensor contains deltas for next 12 readings
    Serial.println("Predicted next hour:");
    for(int t = 0; t < 12; t++)
    {
        // Add predicted delta to last known normalized value
        p_pred = lastScaled[0] + output->data.f[t*3 + 0];  
        t_pred = lastScaled[1] + output->data.f[t*3 + 1];
        h_pred = lastScaled[2] + output->data.f[t*3 + 2];

        // Convert normalized values back to original units
        p_pred = p_pred * (maxVals[0] - minVals[0]) + minVals[0]; 
        t_pred = t_pred * (maxVals[1] - minVals[1]) + minVals[1];
        h_pred = h_pred * (maxVals[2] - minVals[2]) + minVals[2];

        // Print predicted values
        Serial.print(t);
        Serial.print(") predicted P: ");
        Serial.print(p_pred, 2);
        Serial.print(", T: ");
        Serial.print(t_pred, 2);
        Serial.print(", H: ");
        Serial.println(h_pred, 2);
    }
    Serial.println("-------------1 hour completed------------");  // End of one prediction cycle
}


