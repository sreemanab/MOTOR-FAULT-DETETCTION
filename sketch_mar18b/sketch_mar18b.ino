#include <Wire.h>

#define ADXL345_ADDR 0x53

float x, y, z;

void setup() {
  Serial.begin(115200);
  Wire.begin(21,22);

  // Wake up ADXL345
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(0x2D);
  Wire.write(8);
  Wire.endTransmission();

  Serial.println("ADXL345 Ready");
}

void loop() {

  int16_t rawX, rawY, rawZ;

  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(0x32);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_ADDR,6,true);

  rawX = Wire.read() | Wire.read() << 8;
  rawY = Wire.read() | Wire.read() << 8;
  rawZ = Wire.read() | Wire.read() << 8;

  x = rawX * 0.004;
  y = rawY * 0.004;
  z = rawZ * 0.004;

  float rms = sqrt((x*x + y*y + z*z)/3);
  float mean = (x+y+z)/3;
  float peak = max(max(abs(x),abs(y)),abs(z));
  float std = sqrt(((x-mean)*(x-mean)+(y-mean)*(y-mean)+(z-mean)*(z-mean))/3);

  Serial.print("RMS:");
  Serial.print(rms);
  Serial.print(",");

  Serial.print("MEAN:");
  Serial.print(mean);
  Serial.print(",");

  Serial.print("STD:");
  Serial.print(std);
  Serial.print(",");

  Serial.print("PEAK:");
  Serial.println(peak);

  delay(200);
}