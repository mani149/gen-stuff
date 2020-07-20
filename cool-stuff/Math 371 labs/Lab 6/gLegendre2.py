import cmath

a = int(input("Input a value: "))
b = float(input("Input b value: "))

#Parameterizing
def x(a, b, t):
    return ((a + b)/2) + ((b - a)*t/2)

#Actual Function
def subf(x):
    return cmath.log(x)

L = ((subf(x(a, b, (-1/(cmath.sqrt(3))))) + subf(x(a, b, (1/(cmath.sqrt(3))))))*(b - a)/2).real

print("L: ", L)
print("Error: ", -1 - L)