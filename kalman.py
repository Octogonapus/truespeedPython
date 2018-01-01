import numpy as np
import matplotlib.pyplot as plt
import csv

plt.rcParams['figure.figsize'] = (10, 8)

# intial parameters
optimalRPM = np.linspace(0, 83, 1031)
motorPower = np.linspace(0, 127, 1031)
rpm = []
with open('rpm.csv', 'r') as file:
    re = csv.reader(file, delimiter=',')
    for row in re:
        rpm = row

z = np.genfromtxt("rpm.csv", delimiter=",")

print(np.var(z[900:]))

def func1():
    #z = np.genfromtxt("rpm.csv", delimiter=",")
    n_iter = len(z)
    sz = (n_iter,) # size of array
    #x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
    #z = np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)
    
    Q = 0.2 # process variance
    
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    R = 110.37 #0.17**2 # estimate of measurement variance, change to see effect
    
    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0
    
    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q
    
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
    return xhat

def func2(prev_est):
    #z = np.genfromtxt("rpm.csv", delimiter=",")
    n_iter = len(z)
    sz = (n_iter,) # size of array
    #x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
    #z = np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)
    
    Q = 1e-4 # process variance
    
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    R = 0.2**2 # estimate of measurement variance, change to see effect
    
    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0
    
    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1] + (prev_est[k] - prev_est[k-1])
        Pminus[k] = P[k-1]+Q
    
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
    return xhat

plt.figure()
plt.plot(z,'k+',label='noisy measurements')
plt.plot(func1(),'b-',label='a posteri estimate')
plt.plot(func2(func1()),'r-',label='a posteri estimate with previous knowledge')
#plt.plot(motorPower,label='optimal rpm')
#plt.axhline(x,color='g',label='truth value')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Voltage')

#with open('xhat.csv', 'w') as file:
#    wr = csv.writer(file, delimiter=',')
#    wr.writerow(xhat)

#plt.figure()
#valid_iter = range(1,n_iter) # Pminus not valid at step 0
#plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
#plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
#plt.xlabel('Iteration')
#plt.ylabel('$(Voltage)^2$')
#plt.setp(plt.gca(),'ylim',[0,.01])
#plt.show()
