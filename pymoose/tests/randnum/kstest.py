#!/usr/bin/env python
#
# This is a simple implementation of KS-test.

from math import *
from numpy import *

# Values taken from Knuth, TAOCP II: 3.3.1, Table 2
test_table = {1: [0.01000, 0.0500, 0.2500, 0.5000, 0.7500, 0.9500, 0.9900],
              2: [0.01400, 0.06749, 0.2929, 0.5176, 0.7071, 1.0980, 1.2728],
              5: [0.02152, 0.09471, 0.3249, 0.5242, 0.7674, 1.1392, 1.4024],
              10: [0.02912, 0.1147, 0.3297, 0.5426, 0.7845, 1.1658, 1.444],
              20: [0.03807, 0.1298, 0.3461, 0.5547, 0.7975, 1.1839, 1.4698],
              30: [0.04354, 0.1351, 0.3509, 0.5605, 0.8036, 1.1916, 1.4801]}
p_list = [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0] # percentage points the table entries correspond to

def ks_distribution(xx, nn):
    """Calculate P(Knn+ <= xx). See Knuth TAOCP Vol II for details."""
    if nn < 30:
        print "!! Larger sample size is recommended."
    return (1 - exp(-2.0*xx*xx)*(1-2.0*xx/(3.0*sqrt(1.0 * nn))))

def ks_test(rand_num_list, distr_fn):
    """Execute a ks test on the given list of random numbers and tests if they have the distribution defined by distr_fn.

	parameters: 
	rand_num_list - list containing the random sequence to be tested.
	distr_fn - a function that calculates the distribution function for this sequence. TODO: allow another sample list to check if they are from same distribution.

	Note that according to theory, KS test requires that the distribution be continuous"""
    
    result = True
    nn = len(rand_num_list)
    inp_list = array(rand_num_list)
    inp_list.sort()
    distr_list =  map(distr_fn, inp_list)
    sample_distr = arange(nn+1) * 1.0/nn
    k_plus = sqrt(nn) * max(sample_distr[1:] - distr_list)
    k_minus = sqrt(nn) * max(distr_list - sample_distr[:nn])
    p_k_plus = ks_distribution(k_plus, nn)
    if p_k_plus < 0.05 or p_k_plus > 0.95:
        print "ERROR: outside 5%-95% range. The P( K", nn, "+ <=", k_plus, ") is", p_k_plus
        result = False
    p_k_minus = ks_distribution(k_minus, nn)
    if p_k_minus < 0.05 or p_k_minus > 0.95:
        print "ERROR: outside 5%-95% range. The P( K", nn, "- <=", k_minus, ") is", p_k_minus
        result = False
    return result
    
def test_ks_distribution():
    for key in test_table.keys():
        values = test_table[key]
        for ii in range(len(p_list)):
            print "... Testing n =", key,
            value = ks_distribution(values[ii], key)
            print ", expected =", p_list[ii]/100.0, ", calculated =", value
            if (fabs( value - p_list[ii]/100.0) <= 0.005):
                print "... OK"
            else:
                print "FAILED"
        
if __name__ == "__main__":
    test_ks_distribution()
    
