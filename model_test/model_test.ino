#include <Arduino.h>
#include <Arduino_LPS22HB.h>      // Barometer/Temperature sensor
#include "dht.h"                   // DHT11 library
#include "weather_model.h"         // TensorFlow Lite model
// TensorFlow Lite Micro headers
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


#define dht_apin A6                // Analog pin for DHT11
dht DHT;
#define averageOver 5              // Average over 5 readings
#define DHTPIN 2
#define DHTTYPE DHT11
float pressure, temperature, humidity;

// Error reporter
tflite::ErrorReporter* error_reporter = nullptr;
tflite::MicroErrorReporter micro_error_reporter;

// TFLM objects
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::MicroMutableOpResolver<5> resolver;

constexpr int kTensorArenaSize = 20 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

TfLiteTensor* input;
TfLiteTensor* output;

// Min/max arrays for easy indexing
float minVals[3] = {96.18, 28.72, 55.4};
float maxVals[3] = {97.11, 36.03, 144.8};

// Raw window array (12 readings × 3 features)

    float p_pred;
    float t_pred;
    float h_pred;

float processValue(float val, int n) 
{
  // Clamp
  if(val < minVals[n]) val = minVals[n];
  if(val > maxVals[n]) val = maxVals[n];
  
  // Scale to 0-1
  return (val - minVals[n]) / (maxVals[n] - minVals[n]);
}
void readValues() 
{
    pressure = 0;
    temperature = 0;
    humidity = 0;
    for(int count = 0; count < averageOver; count++) 
    {
      pressure    += BARO.readPressure();
      temperature += BARO.readTemperature();
      DHT.read11(dht_apin);
      humidity    += DHT.humidity;
      delay(2000); // wait between readings
    }
    pressure    /= averageOver;
    temperature /= averageOver;
    humidity    /= averageOver;
}
#include <iostream>
using namespace std;

 
float raw_input[24][3] = {
    {96.87f, 33.39f, 68.4f},
    {96.87f, 33.41f, 68.0f},
    {96.87f, 33.40f, 68.2f},
    {96.87f, 33.45f, 68.4f},
    {96.88f, 33.39f, 68.2f},
    {96.87f, 33.43f, 68.2f},
    {96.87f, 32.44f, 68.2f},
    {96.87f, 32.00f, 68.8f},
    {96.88f, 31.95f, 69.6f},
    {96.89f, 31.91f, 69.8f},
    {96.90f, 31.86f, 69.6f},
    {96.91f, 31.88f, 69.2f},// 1hour
    {96.92f, 31.82f, 70.0f},
    {96.92f, 31.71f, 70.0f},
    {96.92f, 31.63f, 70.6f},
    {96.91f, 31.57f, 70.8f},
    {96.89f, 31.45f, 71.0f},
    {96.90f, 31.44f, 72.0f},
    {96.90f, 31.40f, 72.0f},
    {96.90f, 31.38f, 72.0f},
    {96.88f, 31.28f, 72.2f},
    {96.88f, 31.30f, 72.6f},
    {96.88f, 31.23f, 73.0f},
    {96.86f, 31.16f, 72.8f}
};

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize sensors
  BARO.begin();
  // Setup error reporter
  error_reporter = &micro_error_reporter;
  // Load model
  const tflite::Model* model = tflite::GetModel(weather_model_tflite);
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddFullyConnected();
  resolver.AddRelu();
  //resolver.AddQuantize();
  //resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
    model, resolver,
    tensor_arena, kTensorArenaSize,
    error_reporter, nullptr  // profiler is optional
    );

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) 
  {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded!");
  Serial.println("Pressure, Temperature, Humidity");

    for(int i=0;i<12;i++)// input loop
  {
    //read sensor values
    pressure = raw_input[i][0];
    temperature = raw_input[i][1];
    humidity = raw_input[i][2];
    Serial.print(", raw P: ");
    Serial.print(pressure, 2);
    Serial.print(", T: ");
    Serial.print(temperature, 2);
    Serial.print(", H: ");
    Serial.println(humidity, 2);
    // processing the values
    pressure=processValue(pressure,0);    
    temperature=processValue(temperature,1);    
    humidity=processValue(humidity,2); 
    //giving input to model 
    input->data.f[i*3+0] = pressure;
    input->data.f[i*3+1] = temperature;
    input->data.f[i*3+2] = humidity;
  }
    // Invoke model
  float lastScaled[3] = {input->data.f[33], input->data.f[34], input->data.f[35]};
  if (interpreter->Invoke() != kTfLiteOk) 
  {
    Serial.println("Invoke failed!");
    return;
  }

  // Convert output back to real values
  // output->data.f contains deltas for next 12 readings

  Serial.println("Predicted next hour:");
  for(int t = 0; t < 12; t++)
  {
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

void loop()
{

}

