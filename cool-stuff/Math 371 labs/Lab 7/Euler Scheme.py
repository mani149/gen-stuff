# Using Basic Euler Scheme of the form Y_n+1 = Y_n + dt * f(t_n, Y_n)
import numpy

t0 = float(input("Input initial time t0: "))
tend = float(input("Input end time tend: "))
y0 = float(input("Input initial value y0: "))
Nsteps = int(input("Input number of steps N: "))

# Debug on y' = 2t, y(0) = -1
def FCN(t, y):
    #return 2*t
    return -t/y

def yEXACT(t):
    #return t*t - 1
    return numpy.sqrt((-t*t) + 1)

# Time step h (dt), initial step n is updated in loop
h = (tend - t0)/Nsteps
Y_n = y0
t_n = t0
yEXACTn = yEXACT(t_n)
error = abs(Y_n - yEXACT(t_n))

#format
sp = 10*" "
print('\n' "t_n" + 25*" " + "Y_n" + 2*sp + "yEXACTn" + 2*sp + "ERRn" '\n')

for n in range(Nsteps + 1):
    print(str(t_n) + (25 - len(str(t_n))) * " " + str(Y_n) + (25 - (len(str(Y_n)))) * " " + str(yEXACTn) + (
            25 - len(str(yEXACTn))) * " " + str(error))

    t_n += h
    Y_n += h * FCN(t_n, Y_n)
    yEXACTn = yEXACT(t_n)
    error = abs(Y_n - yEXACT(t_n))