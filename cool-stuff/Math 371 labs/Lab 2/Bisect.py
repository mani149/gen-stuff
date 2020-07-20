"""
Author: Immanuel Schmidt
Date: 9/13/19
Bisection code for Math 371 Lab 2

Finds the roots of a function by continuously slicing
a given interval into 2 and checking for a change in sign,
and repeats in slicing the interval in which a sign change
was found for n iterations.

F(x) = x^3 + 2x^2 + 10x - 20
"""
import math

"""
Defining functions within the code
check_input calls F(x) and asks the user for necessary values and checks them
"""
def check_input(T):
    while T:
        try:
            a = float(input("Input 'a' value: "))
            b = float(input("Input 'b' value: "))
            TOL = float(input("Input tolerance: "))
            maxIT = int(input("Input maximum iterations: "))

            if FCN(a) * FCN(b) > 0:
                print("Given interval has either 0, or an even number of roots. Please specify an interval with "
                      "exactly 1 root.")
                exit()

            break

        except ValueError:
            print("Entered value was not a valid number.")
            continue
    return a, b, TOL, maxIT

"""Defining F(x) as its own function"""
def FCN(x):
    #y = 1.3 - x #debug test
    #y = x * (x * (x + 2) + 10) - 20
    y = x - math.tan(x)
    #y_prime = 1 - (1/(math.cos(x) * math.cos(x)))
    return y

""""Actual code starts"""
iterations = 0
UI = check_input(True) #User Input
a, b, maxIT = UI[0], UI[1], UI[3]
residual = abs(FCN(b))
error = abs(b - a)

#Bad formatting
print("     n     xn                   Fn                      ERRn ")
printf = (5 * " ") + str(iterations) + ((6 - len(str(iterations))) * " ") + str(b) + ((21 - len(str(b))) * " ") \
         + str(residual) + ((24 - len(str(residual))) * " ") + str(error)
print(printf)

#print("     {0}     {1}                  {2}                    {3} ".format(iterations, b, residual, error))

while (UI[2] < error or UI[2] < residual) and iterations < maxIT:
    iterations += 1
    r = (a + b)/2

    if FCN(a) * FCN(r) < 0:
        b = r
    else:
        a = r

    error /= 2
    residual = abs(FCN(b))

    #Extremely painful formatting
    printf = (5 * " ") + str(iterations) + ((6 - len(str(iterations))) * " ") + str(b) + ((21 - len(str(b))) * " ") \
             + str(residual) + ((24 - len(str(residual))) * " ") + str(error)
    print(printf)

if iterations == maxIT:
    print("Max iterations reached. The root exists {0} <= x <= {1}, with error +- {2}".format(a, b, error))
else:
    print("\nDONE: root = {0}, residual = {1}, in {2} iterations".format(b, residual, iterations))
