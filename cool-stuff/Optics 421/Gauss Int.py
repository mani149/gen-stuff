"""
Problem 2 (Assignment #10)
Gaussian Integration Adapted from Far Field Intensity code

Immanuel Schmidt

"""
import math
import cmath
import scipy.integrate
from scipy.integrate import quad
import scipy
import matplotlib.pyplot as plt
import numpy as np

#in micrometers
a = 1.0
sig = .2
N = 200                     # of different integrations

#Let xeq be the equilibrium distance such that xeq = (x - x0)
def PSF(x0, x, sigma):
    return (1 / (2*cmath.pi*(sigma*sigma) )**.5 )*math.exp( -((x-x0)*(x-x0)) / (2*(sigma*sigma) )  )

dx = 1/N
I=np.zeros(N,np.float64)
X=np.zeros(N,np.float64)
X[0] = -a/2
for i in range(1,N):
#integrations
    I[i] = quad(PSF, -a/2, a/2, args=(X[i], sig))[0]
    X[i] = X[i-1] + dx


plt.plot(X/1000.0,I/max(I))
#plt.plot(YY/1000.0,I1/max(I1),'r')
plt.title('Fresnel and Far Field')
plt.xlabel('Y [millimeters]')
plt.ylabel('I/Imax')
plt.show()