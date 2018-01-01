import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import csv

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

def fifthOrder(rpm):
    return rpm * 0.5 + rpm * 0.25 + rpm * 0.125 + rpm * 0.0625 + rpm * 0.03125

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

def getXHat():
    raw_rpm = np.genfromtxt("rpm.csv", delimiter=",")
    prev_est = np.genfromtxt("xhat_out.csv", delimiter=",")
#    z = [x[0] + x[1] for x in zip(raw_rpm, prev_est)]
    z = np.genfromtxt("rpm.csv", delimiter=",")
    n_iter = len(z)
    sz = (n_iter,) # size of array
    #x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
    #z = np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)
    
    Q = 1e-5 # process variance
    
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    R = 0.17**2 # estimate of measurement variance, change to see effect
    
    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0
    
    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1] + (prev_est[k]-prev_est[k-1])
        Pminus[k] = P[k-1]+Q
    
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
    np.savetxt("xhat_out.csv", xhat, delimiter=",")
    return xhat;

#plt.plot(rpm, label="Unfiltered RPM")
plt.plot(rpmFilt, label="Filtered RPM")
plt.plot(getXHat(), label="xhat")
#plt.plot([fifthOrder(x) for x in rpmFilt], label="5th order RPM")
#plt.plot(optimalRPM, label="Optimal RPM")
#plt.plot(motorPower, label="Motor Power")
plt.plot(remappedPower, label="Remapped Power")
#plt.plot(filtPower, label="Filtered Power")
#plt.plot(filtPowerSmooth, label="Filtered Power Smooth")
plt.legend()
plt.show()
