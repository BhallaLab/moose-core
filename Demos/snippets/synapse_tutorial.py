# -*- coding: utf-8 -*-
"""This is a tutorial based on an example Upi suggested. The code is
exported from an ipython notebook and the comments present the
markdown version of teh tutorial text."""


import moose



size = 1024     # number of IntFire objects in a vec
delayMin = 0    
delayMax = 4
Vmax = 1.0
thresh = 0.8
refractoryPeriod = 0.4
connectionProbability = 0.1
weightMax = 0.5



# The above sets the constants we shall use in this example. Now we create a vector of IntFire elements of size `size`.



net = moose.IntFire('/network', size)



# This creates a `vec` of `IntFire`  elements of size 1024 and returns the first `element`, i.e. "/network[0]".



net == moose.element('/network[0]')



# You can access the underlying vector of elements using the `vec` field on any element. This is very useful for vectorized field access:



net.vec.bufferTime = [2 * delayMax] * size



# The right part of the assigment creates a Python list of length `size` with each element set to `2 * delayMax`, which is 8.0. You can index into the `vec` to access individual elements' field:



print net.vec[1].bufferTime



# `IntFire` class has an `ElementField` called `synapse`. It is just like a `vec` above in terms of field access, but by default its size is 0.



print len(net.synapse)



# To actually create synapses, you can explicitly assign the `num` field of this, or set the `numSynapses` field of the `IntFire` element. There are some functions which can implicitly set the size of the `ElementField`.



net.numSynapses = 3
print len(net.synapse)



net.synapse.num = 4
print len(net.synapse)



# Now you can index into `net.synapse` as if it was an array.



print 'Before:', net.synapse[0].delay
net.synapse[0].delay = 1.0
print 'After:', net.synapse[0].delay



# You could do the same vectorized assignment as with `vec` directly:



net.synapse.weight = [0.2] * len(net.synapse)
print net.synapse.weight



# You can create the synapses and assign the weights and delays using loops:



import random # We need this for random number generation
from numpy import random as nprand
for item in net.vec:
    neuron = moose.element(item) # You have to convert each entry of a `vec` into corresponding IntFire element in order to access class-specific fields
    neuron.synapse.num = random.randint(1,10) # create synapse fields with random size between 1 and 10, end points included
    # Below is one (inefficient) way of setting the individual weights of the elements in 'synapse'
    for ii in range(len(neuron.synapse)):
        neuron.synapse[ii].weight = random.random() * weightMax
    # This is a more efficient way - rhs of `=` is list comprehension in Python and rather fast
    neuron.synapse.delay = [delayMin + random.random() * delayMax for ii in range(len(neuron.synapse))]
    # An even faster way will be to use numpy.random.rand(size) which produces array of random numbers uniformly distributed between 0 and 1
    neuron.synapse.delay = delayMin + nprand.rand(len(neuron.synapse)) * delayMax
    



# Now display the results, we use slice notation on `vec` to show the values of delay and weight for the first 5 elements in `/network`



for item in net.vec[:5]:
    neuron = moose.element(item)
    print 'Delays for synapses on ', neuron.path, ':', neuron.synapse.delay
    print 'Weights for synapses on ', neuron.path, ':', neuron.synapse.weight




