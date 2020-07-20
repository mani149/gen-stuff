import cmath

a1 = 0.853553390593
a2 = 0.146446609407
t1 = 0.585786437627
t2 = 3.414213562373

def subf(t):
    #x(t) = e^-t,
    #dx  = -e^-t dt
    return -t

L = (a1*subf(t1) + a2*subf(t2))
print("L: ", L.real)
print("Error: ", -1 - L.real)