import serial

ser = serial.Serial('COM3', 9600)  # Replace COM3 with your Nano's port
with open("a:/project/weather/data.csv", "a") as f:
    while True:
        line = ser.readline().decode().strip()
        if not line:
            continue
        # Split by commas and format like Arduino (2 decimal places)
        values = [f"{float(v):.2f}" for v in line.split(',')]
        f.write(",".join(values) + "\n")
        print(",".join(values))
