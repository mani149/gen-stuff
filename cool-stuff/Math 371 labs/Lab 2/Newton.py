"""
Author: Immanuel Schmidt
Date: 9/13/19
Newton-Raphson Method code for Math 371 Lab2

F(x) = x^3 + 2x^2 + 10x - 20
F'(x) = 3x^2 + 4x + 10

G(x) = x - tan(x)       near x = 99 (radians)
G'(x) = 1 - sec^2(x) => -tan^2(x)

"""
import math
"""Calculates specified output given a function and its derivative"""
def FCN(x, outp):
#    y = x * (x * (x + 2) + 10) - 20
#    y_prime = x * ((3 * x) + 4) + 10
    y = x - math.tan(x)
    y_prime = 1 - (1 / (math.cos(x) * math.cos(x)))

    if outp == 'yp':
        return y_prime
    else:
        return y


"""Asks user for x0, TOL, maxIT and checks the input"""
def check_input(T):
    while T:
        try:
            x0 = float(input("Input x0 (root guess): "))

            y = FCN(x0, 'y')
            yp = FCN(x0, 'yp')
            TOL = float(input("Input tolerance: "))
            maxIT = int(input("Input maximum iterations: "))
            break

        except (ValueError, ZeroDivisionError):
            print("Entered value was not a valid number.")
            continue
    return x0, y, TOL, maxIT, (abs(y/yp))

"""Actual code"""
iterations = 0
UI = check_input(True) #User Input
x, residual, TOL, maxIT, error = UI[0], abs(UI[1]), UI[2], UI[3], UI[4]

#Bad formatting
print("     n     xn                      Fn                      ERRn ")
printf = (5 * " ") + str(iterations) + ((6 - len(str(iterations))) * " ") + str(x) + ((24 - len(str(x))) * " ") \
         + str(residual) + ((24 - len(str(residual))) * " ") + str(error)
print(printf)

while (TOL < error or TOL < residual) and iterations < maxIT:
    iterations += 1
    dx = (FCN(x, 'y')/FCN(x, 'yp'))
    x -= dx

    error = abs(dx)
    residual = abs    (FCN(x, 'y'))

    #Extremely painful formatting
    printf = (5 * " ") + str(iterations) + ((6 - len(str(iterations))) * " ") + str(x) + ((24 - len(str(x))) * " ") \
             + str(residual) + ((24 - len(str(residual))) * " ") + str(error)
    print(printf)

if iterations == maxIT:
    print("Max iterations reached. There is either no root, or the initial guess was not close"
          " enough to the real zero.")
else:
    print("\nDONE: root = {0}, residual = {1}, in {2} iterations".format(x, residual, iterations))