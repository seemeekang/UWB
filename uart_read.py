# Example code for UART reading

import serial

ser = serial.Serial(
    port='COM5',\
    baudrate=115200,\
    parity=serial.PARITY_NONE,\
    stopbits=serial.STOPBITS_ONE,\
    bytesize=serial.EIGHTBITS,\
    timeout=None)

print("connected to: " + ser.portstr)

# Read data on the COM port for each line (12-bytes) and decode the bytes that you want to convert to a string
data = ser.readline(12).decode('utf-8')
print(data)
