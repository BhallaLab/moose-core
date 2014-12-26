#######################################################################
# This model replicates the rxd Neuron demonstration of a propagating
# wave model. It gets the same results but the model setup and numerics
# are not yet streamlined.
# Ported by Upi Bhalla, Nov 2014.
#######################################################################

import numpy
import pylab
import moose

# define the geometry
compt = moose.CylMesh( '/cylinder' )
compt.r0 = compt.r1 = 1
compt.x1 = 1000
compt.diffLength = 1
assert( compt.numDiffCompts == 1000 )

#define the molecule. Its geometry is defined by its parent volume, cylinder
c = moose.Pool( '/cylinder/pool' )
c.diffConst = 1 # define diffusion constant
# There is an implicit reaction substrate/product. MOOSE makes it explicit.
buf = moose.BufPool( '/cylinder/buf' )
buf.nInit = 1

# The reaction is something entirely peculiar, not a chemical thing.
reaction = moose.Reac( '/cylinder/reac' )
reaction.Kb = 0

# so here we set up a function calculation to do the same thing.
func = moose.Function( '/cylinder/reac/func' )
func.expr = "(0.3 - x0) * (1 - x0)"
func.x.num = 1 #specify number of input variables.

#Connect the reaction to the pools
moose.connect( reaction, 'sub', c, 'reac' )
moose.connect( reaction, 'prd', buf, 'reac' )

#Connect the function to the reaction
moose.connect( func, 'valueOut', reaction, 'setNumKf' )

#Connect the molecules to the func
moose.connect( c, 'nOut', func.x[0], 'input' )

#Set up solvers

ksolve = moose.Ksolve( '/cylinder/ksolve' )
dsolve = moose.Dsolve( '/cylinder/dsolve' )
stoich = moose.Stoich( '/cylinder/stoich' )
stoich.compartment = compt
stoich.ksolve = ksolve
stoich.dsolve = dsolve
stoich.path = '/cylinder/##'
for i in range( 10, 18 ):
    moose.setClock( i, 0.1 )
#initialize
c.vec.nInit = [ (x < 200) for x in range( 1000 ) ]
# Run and plot it.
moose.reinit()
updateDt = 200
runtime = 600
for t in range( 0, runtime+1, updateDt ):
    plt = pylab.plot( c.vec.n, label='t = '+str(t) )
    moose.start( updateDt )
pylab.xlim( 150, 400 )
pylab.ylim( 0, 1.05 )
pylab.legend()
pylab.show()

