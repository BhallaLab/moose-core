print """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Model: 2 PulseGens (running in free run mode) connected to 2 AscFile objects.
       Both AscFiles record both PulseGens, but the columns are swapped.

Type "help AscFile -full" and "help PulseGen -full" to see documentation on
these objects.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import moose
context = moose.PyMooseBase.getContext()

################################################################################
## MODEL CONSTRUCTION
################################################################################
SIMDT = 50e-6
IODT = 50e-6
SIMLENGTH = 100e-3

##=====================================
##  Create PulseGen objects
##=====================================
p1 = moose.PulseGen( "/p1" )
p1.baselevel = 0.0
p1.level1 = 1.0
p1.width1 = 10e-3
p1.delay1 = 10e-3
p1.trig_mode = 0		/* free run */

p2 = moose.PulseGen( "/p2" )
p2.baselevel = 4.0
p2.level1 = 2.0
p2.width1 = 2e-3
p2.delay1 = 1e-3
p2.level2 = 5.0
p2.width2 = 5e-3
p2.delay2 = 5e-3
p2.trig_mode = 0	/* free run */


##=====================================
##  Create AscFile objects
##=====================================
outdir = "output"
extension = ".moose.plot"

a1 = moose.AscFile( "/a1" )
a1.filename = outdir + "/a1" + extension
a1.append = 0

a2 = moose.AscFile( "/a2" )
a2.filename = outdir + "/a2" + extension
a2.append = 0

##=====================================
##  Connect PulseGens with AscFiles
##=====================================
p1.connect( "output", a1, "save" )
p2.connect( "output", a1, "save" )

p2.connect( "output", a2, "save" )
p1.connect( "output", a2, "save" )


################################################################################
## SIMULATION CONTROL
################################################################################

##=====================================
##  Clocks
##=====================================
context.setClock(0, SIMDT, 0)
context.setClock(1, SIMDT, 0)
context.setClock(2, IODT, 0)

context.useClock( 2, "/data/#[TYPE=AscFile]" )

##=====================================
##  Simulation
##=====================================
context.reset()
context.step( SIMLENGTH )

print """
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Hack: Calling the "close" function for AscFile:s explicitly to flush the file.
Should not be required, really. Currently a bug in MOOSE does not finalize
objects at exit.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
#~ call /a1 close
#~ call /a2 close

print """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plots written to *.moose.plot. Reference curves from GENESIS are in files named
*.genesis.plot.

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
