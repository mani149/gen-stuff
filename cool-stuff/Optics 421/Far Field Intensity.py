"""
Problem 2 (Assignment #9)
Double Slit diffraction Adapted from Original Single Slit code

Original: Jaan Mannik
Adapted: Immanuel Schmidt

"""
import cmath
import scipy.integrate
from scipy.integrate import quad
import scipy
import matplotlib.pyplot as plt
import numpy as np

wavelen = 0.5               #in micrometers
a = 5.0                     #slit width in micrometers
d = 15.0                      #distance between slits
k = 2.0*cmath.pi/wavelen      #magnitude of the wavevector
XX=1e6                      #Distance to screen in micrometers
N = 200                     # of different integrations
dtheta=(4/N)*(wavelen/a)

def Fresnel(y,Y,X,k):
    return np.cos(k*(Y-y)*(Y-y)/(2*X))

def FarField(y,Y,X,k):
    return np.cos(k*Y*y/X)

YY=np.zeros(N,np.float64)
E=np.zeros(N,np.float64)
I=np.zeros(N,np.float64)
E1=np.zeros(N,np.float64)
I1=np.zeros(N,np.float64)
Eanal=np.zeros(N,np.float64)
Ianal=np.zeros(N,np.float64)

for i in range(1,N):
#integrations
    YY[i] = (i-1)*dtheta*XX
    E[i] = quad(Fresnel,(-a - d/2.0), (-d/2.0), args=(YY[i],XX,k))[0] + quad(Fresnel,(d/2.0), (a + d/2.0), args=(YY[i],XX,k))[0]
    I[i] = E[i]*E[i]
    E1[i] = quad(FarField,(-a - d/2.0), (-d/2.0), args=(YY[i],XX,k))[0] + quad(FarField,(d/2.0), (a + d/2.0), args=(YY[i],XX,k))[0]
    I1[i] = E1[i]*E1[i]

# Eanal[i]=sinc(YY[i]*k*a/(2.0*XX*pi))
# Matlab defines sinc with factor pi
# sinc(x)=sin(pi*x)/(pi*x) unlike in the notes
# Ianal[i]=Eanal[i]*Eanal[i]

plt.plot(YY/1000.0,I/max(I))
plt.plot(YY/1000.0,I1/max(I1),'r')
plt.title('Fresnel and Far Field')
plt.xlabel('Y [millimeters]')
plt.ylabel('I/Imax')
plt.show()