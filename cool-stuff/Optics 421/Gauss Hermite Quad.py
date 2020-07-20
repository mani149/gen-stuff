from scipy.special import roots_hermite
import scipy
import numpy


#from numpy.polynomial.hermite import hermgauss

degree = int(input("Enter polynomial order: "))
order = 2*degree - 1
Sum = 0
i = 0
[points, weights] = roots_hermite(order, False)
print(weights)


while(i < len(points)):
    Sum += weights[i]*((points[i])**degree)
    i = i + 1
print("Integral = ", Sum )


