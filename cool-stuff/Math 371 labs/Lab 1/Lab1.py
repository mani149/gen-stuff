"""
E is sum
(A) ln(1-x) = -E[(x^k)/k], -1<=x<1
(B) ln[(1+x)/(1-x)] = 2E[(x^(2k-1))/(2k-1)], -1<x<1
for ln(1.9)
"""
import math

partsum = 0
series = input("Which series do you want to use, [A] or [B]?")
val = input("What value for 'n' do you want?")
n = int(val)

if series == 'A':
    x = -0.9
    def psum(i):
        return -(x ** (i + 1)) / (i + 1)

elif series == 'B':
    x = 9 / 29
    def psum(i):
        return 2 * (x ** ((2 * (i + 1)) - 1)) / ((2 * (i + 1)) - 1)

for i in range(n):
    partsum = partsum + psum(i)

error = math.log(1.9) - partsum
print("When N is {0}, the partial sum is {1}, and the error is {2}".format(n, partsum, error))