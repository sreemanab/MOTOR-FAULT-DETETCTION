from machine import I2C, Pin
import time

i2c = I2C(0, scl=Pin(22), sda=Pin(21))

while True:
    devices = i2c.scan()
    print(devices)
    time.sleep(2) 