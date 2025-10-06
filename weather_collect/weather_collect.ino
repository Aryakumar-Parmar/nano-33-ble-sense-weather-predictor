#include <Arduino.h>
#include <Arduino_LPS22HB.h>  // Barometer/Temperature sensor
#include "dht.h"               // DHT11 library

#define DHTPIN A6
#define DHTTYPE DHT11
#define AVERAGE_OVER 5

dht DHT;
float pressure, temperature, humidity;

void readSensors() 
{
  pressure = 0;
  temperature = 0;
  humidity = 0;

  for (int i = 0; i < AVERAGE_OVER; i++) {
    pressure    += BARO.readPressure();
    temperature += BARO.readTemperature();
    DHT.read11(DHTPIN);
    humidity    += DHT.humidity;
    delay(2000); // small delay between averages
  }

  pressure    /= AVERAGE_OVER;
  temperature /= AVERAGE_OVER;
  humidity    /= AVERAGE_OVER;
}

void setup() {
  Serial.begin(9600);
  while(!Serial);

  if(!BARO.begin()) {
    Serial.println("Barometer initialization failed!");
    while(1);
  }
  Serial.println("Pressure,Temperature,Humidity"); // CSV header
}

void loop() {
  readSensors();
  Serial.print(pressure, 2); Serial.print(",");
  Serial.print(temperature, 2); Serial.print(",");
  Serial.println(humidity, 2);

  delay(5*60*1E3);
}
