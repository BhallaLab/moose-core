print """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Model: TimeTable object connected to a single compartment via a SynChan.
       The TimeTable object reads spike-times from a text file, and emits
       spikes at the specified times.
Plots: - Vm of postsynaptic compartment
       - Gk of synaptic channel (embedded in compartment)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import moose
context = moose.PyMooseBase.getContext()

################################################################################
## MODEL CONSTRUCTION
################################################################################
SIMDT = 50e-6
IODT = 50e-6
SIMLENGTH = 60e-3

##=====================================
##  Create single-compartment cell with a synaptic channel
##=====================================
cc = moose.Compartment( "/cc" )
cc.Ra = 1.0e10
cc.Rm = 1.0e7
cc.Cm = 1.0e-10
cc.initVm = -0.07
cc.Em = -0.07

cc.syn = moose.SynChan( cc.path + "/syn" )
cc.syn.Ek = 0.0
cc.syn.gmax = 1.0e-9
cc.syn.tau1 = 1.0e-3
cc.syn.tau2 = 2.0e-3
cc.connect( "channel", cc.syn, "channel" )

##=====================================
##  Create TimeTable object
##=====================================
tt = moose.TimeTable( "/tt" )

# Method is 4 by default, but setting it anyway.
tt.method = 4
tt.filename = "spikes.txt"

tt.connect( "event", cc.syn, "synapse" )


################################################################################
## PLOTTING
################################################################################
data = moose.Neutral( "/data" )

data.Vm = moose.Table( data.path + "/Vm" )
data.Vm.xdivs = int( SIMLENGTH / IODT )
data.Vm.xmin = 0 
data.Vm.xmax = SIMLENGTH
data.Vm.step_mode = 3

data.Gk = moose.Table( data.path + "/Gk" )
data.Gk.xdivs = int( SIMLENGTH / IODT )
data.Gk.xmin = 0 
data.Gk.xmax = SIMLENGTH
data.Gk.step_mode = 3

##=====================================
##  Record from compartment and channel
##=====================================
cc.connect( "Vm", data.Vm, "inputRequest" )
cc.syn.connect( "Gk", data.Gk, "inputRequest" )


################################################################################
## SIMULATION CONTROL
################################################################################

##=====================================
##  Clocks
##=====================================
context.setClock(0, SIMDT, 0)
context.setClock(1, SIMDT, 0)
context.setClock(2, IODT, 0)

context.useClock( 2, "/data/#[TYPE=Table]" )

##=====================================
##  Simulation
##=====================================
context.reset()
context.step( SIMLENGTH )


################################################################################
##  Write Plots
################################################################################
outdir = "output/"
extension = ".moose.plot"

filename = outdir + "Vm" + extension
outfile = open( filename, "w" )
outfile.write( "/newplot\n" )
outfile.write( "/plotname Vm\n" )
for value in data.Vm:
    outfile.write( str( value ) + "\n" )
outfile.close()

filename = outdir + "Gk" + extension
outfile = open( filename, "w" )
outfile.write( "/newplot\n" )
outfile.write( "/plotname Gk\n" )
for value in data.Gk:
    outfile.write( str( value ) + "\n" )
outfile.close()


print """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plots written to *.moose.plot. Reference curves from GENESIS are in files named
*.genesis.plot.

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
