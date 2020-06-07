# Modified from moose-example/synapse_tutorial.py ---

import moose
import random  
import numpy as np

def test_synapse():
    """
    In this example we walk through creation of a vector of IntFire
    elements and setting up synaptic connection between them. Synapse on
    IntFire elements is an example of ElementField - elements that do not
    exist on their own, but only as part of another element. This example
    also illustrates various operations on `vec` objects and
    ElementFields.
    """

    size = 1024  # number of IntFire objects in a vec
    delayMin = 0
    delayMax = 4
    thresh = 0.8
    weightMax = 0.5

    # The above sets the constants we shall use in this example. Now we create
    # a vector of IntFire elements of size `size`.
    net = moose.IntFire("/network", size)

    # This creates a `vec` of `IntFire`  elements of size 1024 and returns the
    # first `element`, i.e. "/network[0]".
    net = moose.element("/network[0]")

    # You need now to provide synaptic input to the network
    synh = moose.SimpleSynHandler("/network/synh", size)

    # These need to be connected to the nodes in the network
    moose.connect(synh, "activationOut", net, "activation", "OneToOne")

    # You can access the underlying vector of elements using the `vec` field on
    # any element. This is very useful for vectorized field access:
    _vm = [thresh / 2.0] * size
    net.vec.Vm = _vm
    assert np.allclose(net.vec.Vm, _vm)

    # The right part of the assigment creates a Python list of length `size`
    # with each element set to `thresh/2.0`, which is 0.4. You can index into
    # the `vec` to access individual elements' field:

    print(net.vec[1].Vm)
    assert net.vec[1].Vm == 0.4

    # `SimpleSynHandler` class has an `ElementField` called `synapse`. It is
    # just like a `vec` above in terms of field access, but by default its size
    # is 0.

    assert len(synh.synapse) == 0
    print(len(synh.synapse))

    # To actually create synapses, you can explicitly assign the `num` field of
    # this, or set the `numSynapses` field of the `IntFire` element. There are
    # some functions which can implicitly set the size of the `ElementField`.

    synh.numSynapses = 3
    assert len(synh.synapse) == 3
    print(len(synh.synapse))

    synh.numSynapses = 4
    print(synh.synapse, 111)
    assert len(synh.synapse) == 4, (4, len(synh.synapse))
    print(len(synh.synapse))

    # Now you can index into `net.synapse` as if it was an array.
    print("Before:", synh.synapse[0].delay)
    assert synh.synapse[0].delay == 0.0
    synh.synapse[0].delay = 1.0
    assert synh.synapse[0].delay == 1.0
    print("After:", synh.synapse[0].delay)

    # You could do the same vectorized assignment as with `vec` directly:

    syns = synh.synapse.vec
    syns.weight = [0.2] * len(syns)
    assert np.allclose(syns.weight, 0.2), syns.weight
    print(syns.weight)

    # You can create the synapses and assign the weights and delays using loops:
    for syn in synh.vec:
        syn.numSynapses = random.randint(1, 10) 

        # create synapse fields with random size between 1 and 10, end points
        # included. Below is one (inefficient) way of setting the individual weights of
        # the elements in 'synapse'
        syns  = syn.synapse
        for ii in range(len(syns)):
            syns[ii].weight = random.random() * weightMax

        # This is a more efficient way - rhs of `=` is list comprehension in
        # Python and rather fast.
        syns.delay = [
            delayMin + random.random() * delayMax for ii in range(len(syn.synapse))
        ]

        # An even faster way will be to use numpy.random.rand(size) which
        # produces array of random numbers uniformly distributed between 0 and
        # 1
        syns.delay = delayMin + np.random.rand(len(syn.synapse)) * delayMax

    # Now display the results, we use slice notation on `vec` to show the
    # values of delay and weight for the first 5 elements in `/network`
    for syn in synh.vec[:5]:
        print("Delays for synapses on ", syn.path, ":", syn.synapse.vec.delay)
        print("Weights for synapses on ", syn.path, ":", syn.synapse.vec.weight)

if __name__ == "__main__":
    test_synapse()
