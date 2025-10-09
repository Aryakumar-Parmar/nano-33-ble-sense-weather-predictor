// ------------------------- Hardware headers ------------------------------------------------------------------------------------------------
#include <Arduino.h>                                         // Core Arduino functions (Serial, pinMode, digitalWrite, etc.)
#include <Arduino_LPS22HB.h>                                 // Library for LPS22HB Barometer/Temperature sensor
#include "dht.h"                                             // Library for DHT11 humidity/temperature sensor
#include "weather_model.h"                                   // TensorFlow Lite model header (.h file generated from .tflite)
// ------------------------- TensorFlow Lite Micro headers -----------------------------------------------------------------------------------
#include "TensorFlowLite.h"                                  // Core TFLite Micro definitions
#include "tensorflow/lite/micro/micro_interpreter.h"         // Interpreter to run TFLite Micro models
#include "tensorflow/lite/micro/micro_error_reporter.h"      // For reporting errors from TFLite Micro
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" // Registers operations for the model
#include "tensorflow/lite/schema/schema_generated.h"         // TFLite model schema (generated from .tflite file)
#include "tensorflow/lite/version.h"                         // TensorFlow Lite version info

// ------------------------- Sensor & Pin Definitions ----------------------------------------------------------------------------------------
#define dht_apin A6                                         // Analog pin connected to DHT11 data line
dht DHT;                                                    // Instantiate DHT11 object
#define averageOver 5                                       // Number of readings to average for smoother sensor output
#define DHTTYPE DHT11                                       // DHT sensor type (DHT11)

// ------------------------- Global Variables ------------------------------------------------------------------------------------------------
float pressure, temperature, humidity;                     // Variables to store raw sensor readings
float minVals[3] = {96.18, 28.72, 55.4};                   // Minimum expected values for Pressure, Temperature, Humidity
float maxVals[3] = {97.11, 36.03, 144.8};                  // Maximum expected values for Pressure, Temperature, Humidity
float p_pred, t_pred, h_pred;                              // Variables to store predicted sensor readings

// ------------------------- TensorFlow Lite Error Handling ----------------------------------------------------------------------------------
tflite::ErrorReporter* error_reporter = nullptr;           // Pointer for error reporting
tflite::MicroErrorReporter micro_error_reporter;           // TFLite Micro error reporter instance

// ------------------------- TensorFlow Lite Interpreter -------------------------------------------------------------------------------------
const tflite::Model* model = nullptr;                      // Pointer to TFLite model, assigned in setup
tflite::MicroInterpreter* interpreter = nullptr;           // Pointer to TFLite Micro interpreter
tflite::MicroMutableOpResolver<5> resolver;                // Resolver to register up to 5 operations (FullyConnected, ReLU, etc.)

// ------------------------- Tensor Arena ---------------------------------------------------------------------------------------------------
constexpr int kTensorArenaSize = 20 * 1024;                // 20 KB arena for tensor computations (RAM reserved for model)
static uint8_t tensor_arena[kTensorArenaSize];             // Memory buffer used by TFLite Micro for tensors

TfLiteTensor* input;                                       // Pointer to input tensor of the model
TfLiteTensor* output;                                      // Pointer to output tensor of the model

// ------------------------- Functions ------------------------------------------------------------------------------------------------------
float processValue(float val, int n) 
{
  // Clamp value between min and max
  if(val < minVals[n]) val = minVals[n];
  if(val > maxVals[n]) val = maxVals[n];
  
  // Scale value to range 0-1
  return (val - minVals[n]) / (maxVals[n] - minVals[n]);
}

// ------------------------- Raw Input Data (example 24 readings) ----------------------------------------------------------------------------
float raw_input[24][3] = {
    {96.87f, 33.39f, 68.4f}, {96.87f, 33.41f, 68.0f}, {96.87f, 33.40f, 68.2f},
    {96.87f, 33.45f, 68.4f}, {96.88f, 33.39f, 68.2f}, {96.87f, 33.43f, 68.2f},
    {96.87f, 32.44f, 68.2f}, {96.87f, 32.00f, 68.8f}, {96.88f, 31.95f, 69.6f},
    {96.89f, 31.91f, 69.8f}, {96.90f, 31.86f, 69.6f}, {96.91f, 31.88f, 69.2f},
    {96.92f, 31.82f, 70.0f}, {96.92f, 31.71f, 70.0f}, {96.92f, 31.63f, 70.6f},
    {96.91f, 31.57f, 70.8f}, {96.89f, 31.61f, 71.0f}, {96.90f, 31.54f, 72.0f},
    {96.90f, 31.59f, 72.0f}, {96.90f, 31.55f, 72.0f}, {96.88f, 31.58f, 72.2f},
    {96.88f, 31.56f, 72.6f}, {96.88f, 31.51f, 73.0f}, {96.86f, 31.59f, 72.8f}
};

// ------------------------- Arduino Setup --------------------------------------------------------------------------------------------------
void setup() {
  Serial.begin(9600);                   // Start serial communication
  while (!Serial);                      // Wait for serial to be ready

  BARO.begin();                         // Initialize barometer sensor
  error_reporter = &micro_error_reporter; // Assign error reporter

  // Load TFLite model from header
  const tflite::Model* model = tflite::GetModel(weather_model_tflite);

  // Setup operation resolver
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddFullyConnected();          // Register FullyConnected (Dense) layer
  resolver.AddRelu();                    // Register ReLU activation

  // Setup interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, resolver,
    tensor_arena, kTensorArenaSize,
    error_reporter, nullptr            // Optional profiler not used
  );
  interpreter = &static_interpreter;     // Assign global interpreter pointer

  // Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) 
  {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  input = interpreter->input(0);         // Get input tensor pointer
  output = interpreter->output(0);       // Get output tensor pointer

  Serial.println("Model loaded!");
  Serial.println("Pressure, Temperature, Humidity");

  // ------------------------- Fill input tensor with first 12 raw readings ------------------------
  for(int i=0; i<12; i++) 
  {
    pressure = raw_input[i][0];          // Get pressure
    temperature = raw_input[i][1];       // Get temperature
    humidity = raw_input[i][2];          // Get humidity

    Serial.print(", raw P: ");
    Serial.print(pressure, 2);
    Serial.print(", T: ");
    Serial.print(temperature, 2);
    Serial.print(", H: ");
    Serial.println(humidity, 2);

    // Process and scale values
    pressure    = processValue(pressure, 0);    
    temperature = processValue(temperature, 1);    
    humidity    = processValue(humidity, 2); 

    // Assign to model input tensor
    input->data.f[i*3 + 0] = pressure;
    input->data.f[i*3 + 1] = temperature;
    input->data.f[i*3 + 2] = humidity;
  }

  // ------------------------- Invoke model and predict next 12 readings --------------------------------
  float lastScaled[3] = {input->data.f[33], input->data.f[34], input->data.f[35]}; // Last scaled values for delta
  if (interpreter->Invoke() != kTfLiteOk) 
  {
    Serial.println("Invoke failed!"); // Check if model ran successfully
    return;
  }

  // Print predicted next hour
  Serial.println("Predicted next hour:");
  for(int t = 0; t < 12; t++)
  {
    // Add predicted delta to last scaled value
    p_pred = lastScaled[0] + output->data.f[t*3 + 0];
    t_pred = lastScaled[1] + output->data.f[t*3 + 1];
    h_pred = lastScaled[2] + output->data.f[t*3 + 2];

    // Convert back to original units
    p_pred = p_pred * (maxVals[0] - minVals[0]) + minVals[0];
    t_pred = t_pred * (maxVals[1] - minVals[1]) + minVals[1];
    h_pred = h_pred * (maxVals[2] - minVals[2]) + minVals[2];

    Serial.print(t);
    Serial.print(") predicted P: ");
    Serial.print(p_pred, 2);
    Serial.print(", T: ");
    Serial.print(t_pred, 2);
    Serial.print(", H: ");
    Serial.println(h_pred, 2);
  }

  Serial.println("-------------1 hour completed------------");
}

// ------------------------- Arduino Loop ---------------------------------------------------------------------------------------------------
void loop(){}
