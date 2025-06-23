import serial
import matplotlib.pyplot as plt

ser = serial.Serial('COM15', 9600)  # Replace 'COM3' with your Arduino port
eeg_data = []

plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()

while True:
    data = ser.readline().strip()
    try:
        eeg_value = int(data)
        eeg_data.append(eeg_value)
        if len(eeg_data) > 100:  # Keep last 100 samples
            eeg_data.pop(0)
        ax.clear()
        ax.plot(eeg_data)
        plt.pause(0.01)
    except:
        continue
