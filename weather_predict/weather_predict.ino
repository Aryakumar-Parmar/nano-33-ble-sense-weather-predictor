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
}
void loop()
{
  for(int i=0;i<12;i++)
  {
    //read sensor values
    readValues();   
    // printing values
    Serial.print(i);
    Serial.print(") raw P: ");
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
    
    delay(5*60*1E3);
  }
  float lastScaled[3] = {input->data.f[33], input->data.f[34], input->data.f[35]};
  // Invoke model
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

