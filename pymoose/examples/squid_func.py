def squid_compt(path, length, diameter):
    if not instanceof(path, string):
        print "Error: path must be string"
        exit(0)
    else if not instanceof(length, double):
        print "Error: length must be a numebr"
        exit(0)
    else if not instanceof(diameter, double):
        print "Error: diameter must be a number"

    cmpt = moose.Compartment(path)
    cmpt.length, cmpt.diameter, cmpt.angle  = length, diameter, 90
    # TODO: complete
# cmpt.getContext().do_deep_copy(PyMooseBase.pathToId(
    
