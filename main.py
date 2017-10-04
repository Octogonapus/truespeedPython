import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

demaS = 0
demaB = 0
demaLastS = 0
demaLastB = 0

alpha = 0.19
beta = 0.041

def demaInit():
    global demaS
    global demaB
    global demaLastS
    global demaLastB
    demaS = 0
    demaB = 0
    demaLastS = 0
    demaLastB = 0

def demaFilter(val):
    global demaS
    global demaB
    global demaLastS
    global demaLastB
    demaS = (alpha * val) + ((1 - alpha) * (demaLastS + demaLastB))
    demaB = (beta * (demaS - demaLastS)) + ((1 - beta) * demaLastB)
    demaLastS = demaS
    demaLastB = demaB
    return demaS + demaB

def findPowerForRPM(desiredRPMVal, measuredRPM, motorPower):
    idx = (np.abs(measuredRPM - desiredRPMVal)).argmin();
    return motorPower[idx]

motorPower = [int(round(a)) for a in np.linspace(0,127,1031)]
rpm = np.genfromtxt("rpm.csv", delimiter=",")
demaInit()
rpmFilt = [demaFilter(a) for a in rpm]

optimalRPM = np.linspace(0, 83, 1031)
remappedPower = [findPowerForRPM(a, rpmFilt, motorPower) for a in optimalRPM]

demaInit()
filtPower = [demaFilter(a) for a in medfilt(remappedPower, 3)]

filtPowerSmooth = filtPower.copy()
for i in range(0, len(filtPowerSmooth), 3):
   slice_from_index = i
   slice_to_index = slice_from_index + 3
   avg = np.mean(filtPowerSmooth[slice_from_index:slice_to_index])
   filtPowerSmooth[slice_from_index] = avg;
   if slice_from_index + 1 < len(filtPowerSmooth):
       filtPowerSmooth[slice_from_index + 1] = avg;
       if slice_from_index + 2 < len(filtPowerSmooth):
           filtPowerSmooth[slice_from_index + 2] = avg;

#plt.plot(rpm, label="Unfiltered RPM")
plt.plot(rpmFilt, label="Filtered RPM")
#plt.plot(optimalRPM, label="Optimal RPM")
#plt.plot(motorPower, label="Motor Power")
#plt.plot(remappedPower, label="Remapped Power")
#plt.plot(filtPower, label="Filtered Power")
plt.plot(filtPowerSmooth, label="Filtered Power Smooth")
plt.legend()
plt.show()
