#!/usr/bin/env python
#
# This program takes a file with a random sequence and calculates mean, variance and plots the cumulative distribution
#
# Author: Subhasis Ray, Sept 2008
from math import *
import sys
if len(sys.argv) < 2:
	print "usage:", sys.argv[0], "<file>\n calculate the mean and variance of the numbers in file and plot the cumulative distribution. <file> should have one number on each line."
nrng = open(sys.argv[1], "r")
values = []
mysum = 0.0
for line in nrng:
    if len(line.strip()) > 0:
	    values.append(float(line))
	    mysum += values[-1]
mymean = mysum/len(values)
myvar = sum((value - mymean) ** 2 for value in values)
myvar /= len(values)
print "mean:", mymean, "variance:", myvar
values.sort()
from pylab import *
xx = arange(mymean - 3 * myvar, mymean + 3 * myvar, myvar/10)
yy = []

ii = 0
for x_val in xx:
    count = 0
    while ii < len(values) and x_val >= values[ii]:
        count += 1
        ii += 1
    yy.append(count*1.0/len(values))

plot(xx,yy)
show()
