import cmath

sum = 0
a = int(input("Input a value: "))
b = int(input("Input b value: "))
M = int(input("Input number of sub-intervals M: "))

def subf(x):
    y = cmath.log(x)
    #debugging functions
    #y = 2*x - 1
    #y = 3*(1 - x*x)
    return y

#Midpt difference
dx = (b - a)/M
evalpt = dx/2

while evalpt < b:
    sum += subf(evalpt)*dx
    evalpt += dx

print("M: ", M)
print("MR: ", sum.real)
print("Error: ", -1 - sum.real)

