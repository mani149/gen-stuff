# Lab 4, Using forward finite difference to approximate the derivative at given x

import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

# Approximation Function
def approx(x):
    return math.sin(x)

def realderivative(x):
    return math.cos(x)

# Computes derivative of function using Forward Finite Difference
def derivative(func, n, xi, difftype):
    h = .5 * .5 * .5 * .5 * .5
    k = 5
    fdarray = []
    harray = []
    karray = []
    errarray = []

    # Loops through smaller h and computes derivative
    while k <= n:

        if difftype == 'F':
            fd = (func(xi + h) - func(xi)) / h
        else:
            fd = (func(xi + h) - func(xi - h)) / (2 * h)
        # error checking
        fdarray.append(fd)
        harray.append(h)
        karray.append(k)
        errarray.append(realderivative(xi) - fd)

        #print(k)
        h *= .5
        k += 1

        #print(fd)
    return k - 1, h / .5, fd, fdarray, harray, karray, errarray

difftype = input("(F)FD of (C)FD?: ")
iter = int(input("Input a number of iterations (n): "))
aval = float(input("Input an approximation point (x): "))

output = derivative(approx, iter, aval, difftype)
error = realderivative(aval) - output[2]

print("k = ", output[0], "\nh = ", output[1], "\nError = ", error)

# error checking
"""
pdf = matplotlib.backends.backend_pdf.PdfPages("out.pdf")

x = []
y = []
for i in range(len(output[4]) - 1):
    x.append(math.log(output[4][i]))



fig1 = plt.plot(output[5], output[6])
fig2 = plt.plot(output[4], output[6])
fig3 = plt.plot(x)
pdf.savefig(fig1)
pdf.savefig(fig2)
pdf.savefig(fig3)
pdf.close()
"""