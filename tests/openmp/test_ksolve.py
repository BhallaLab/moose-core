import moose
import moose.utils as mu
import numpy as np
import matplotlib.pyplot as plt
import time

#to know the path from where the files are taken
print( '[INFO] Using moose from %s' % moose.__file__ )


#Tables for storing the concentration of each species
tables_ = { }


#The following function creates a compartment which can have voxels and in each voxel computations can be done
def create_cylinder( name ):
#creating a model, just a model, created a file path
    moose.Neutral( '/model' )
#Making the model cylindrical, setting the properties of the model...like here its a cylinder
    c = moose.CylMesh( '/model/%s' % name )
#Setting the length of the cyliinder
    c.x1 = 1e-3
#Setting the number of voxels
    c.diffLength = c.x1 / 10
#radius of the cylinder at both the faces, in here its kept same at both the faces
    c.r0 = c.r1 = 1e-3
    print( 'Volume of cylinder %s is %g' % (c.name,  c.volume) )
    return c

#In the following function we are defining which solvee to use
def setup_solver( c ):
#Its important step
    s = moose.Stoich( '/model/stoich')
#we declared k to be denoting ksolve
    k = moose.Ksolve( '/model/ksolve' )
#we declared d to be dsolve
    d = moose.Dsolve( '/model/dsolve' )
    # s.dsolve =  d
#solver for the s is ksolve
    s.ksolve = k
#On which compartment does s has to work
    s.compartment = c
#Where is the compartment and there we add the path of the c compartment
    s.path = '%s/##' % c.path 


#Defining wha should be happening in the compartment and therefore in the voxels, the reactions are defined 
#There are different species in a pool
def make_model( ):
    c = create_cylinder( 'a' )
#it stores the pool species
    pools = { }
    for x in [ 'A', 'C' ]:
#storing the species and concentrations
        p = moose.Pool( '/model/a/%s' % x )
#for getting the concentrations of the species
        t = moose.Table2( '/model/a/tab%s' %  x)
#connecting the pool and the table for transferring concentrations
        moose.connect( t, 'requestOut', p, 'getN' )
#storing species names and concentrations
        pools[ x ] = p
        tables_[ x ] = t

#setting up the initial concentrations of the reactants
    pools[ 'A' ].concInit = 1.0 
    # pools[ 'A' ].diffConst = 1.0 
    pools[ 'C' ].concInit = 0.1 

    # A + 2B <-- kf, kb --> C + D 
#defining the reaction in the comapartment(a)
    r1 = moose.Reac( '/model/a/r1' )
#connect is used here for connecting the species the different species in a reaction, like tis reaction consisits of 'A' as the substrate and 'C ' as the product, see the syntax, it also describes which reaction they are part of!! 
    moose.connect( r1, 'sub', pools['A'], 'reac' )
    # moose.connect( r1, 'sub', pools['B'], 'reac' )
    # moose.connect( r1, 'sub', pools['B'], 'reac' )
    moose.connect( r1, 'prd', pools['C'], 'reac' )
    # moose.connect( r1, 'prd', pools['D'], 'reac' )
#setting the reaction forward and backward rate constansts
    r1.Kf = 1
    r1.Kb = 0.1
#The solver for c has been defined using the setup_solver function
    setup_solver( c )
    moose.reinit( )   # Build the system
    moose.start( 1 ) # Run the system
    #mu.plot_records( tables_, subplot = True, outfile = 'result.png' )
    k = moose.element( '/model/ksolve' )
#printing the number of species
    print( k.numPools )


def main( ):
    make_model( )

#running the function!
if __name__ == '__main__':
    start = time.clock()
    main()
    print(time.clock()-start)
