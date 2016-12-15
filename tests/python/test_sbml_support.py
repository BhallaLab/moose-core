import moose
import sys
import os

script_dir = os.path.dirname( os.path.realpath( __file__) )

print( "Using moose from: %s" % moose.__file__ )

def main():
    """ This example illustrates loading, running of an SBML model defined in XML format.\n
	The model 00001-sbml-l3v1.xml is taken from l3v1 SBML testcase.\n
	Plots are setup.\n
	Model is run for 20sec.\n
	As a general rule we created model under '/path/model' and plots under '/path/graphs'.\n
    """

    mfile =  os.path.join( script_dir, 'chem_models/acc27.g')
    runtime = 20.0
    writefile =  os.path.join( script_dir, 'chem_models/acc27.xml')    
    
    #Load model to moose and write to SBML
    moose.loadModel(mfile, '/acc27')
    writeerror, message, sbmlId = moose.readSBML('/acc27',writefile)
    if writeerror == -2:
        print ( "Could not save the Model" )
    elif writeerror == -1:
        print ( "\n This model is not valid SBML Model, failed in the consistency check ")
    elif writeerror == 0:
        print ("Could not save the Model ")
    elif writeerror == 1:
        print ( "Model is loaded using \'loadModel\' function to moose and using \'moose.SBML.mooseWriteSBML\' converted to SBML. \n Ran for 20 Sec" )
        # Reset and Run
        moose.reinit()
        moose.start(runtime)
    

def displayPlots():
    import matplotlib
    import numpy as np
    import matplotlib.pyplot as plt
    import pylab

    # Display all plots.
    for x in moose.wildcardFind( '/sbml/graphs/#[TYPE=Table2]' ):
        t = np.arange( 0, x.vector.size, 1 ) #sec
        plt.plot( t, x.vector, label=x.name )
    
    pylab.legend()
    pylab.show()
    quit()

if __name__=='__main__':
    main()
    displayPlots()
