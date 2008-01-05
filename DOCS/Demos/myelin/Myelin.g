// moose

include defaults.g
include chan.g

/********************************************************************
**                                                                 **
**                       Simulation parameters                     **
**                                                                 **
********************************************************************/
float inj = 1.0e-10
float dt = 50e-6
float iodt = 50e-6
float runtime = 0.25

/********************************************************************
**                                                                 **
**                       Model construction                        **
**                                                                 **
********************************************************************/
ce /library
make_Na_mit_usb
make_K_mit_usb
ce /

readcell myelin2.p /axon

/********************************************************************
**                                                                 **
**                       File I/0                                  **
**                                                                 **
********************************************************************/
create Neutral /output
create Table /output/out0
create Table /output/outx

addmsg /output/out0/inputRequest /axon/soma/Vm
addmsg /output/outx/inputRequest /axon/n99/i20/Vm

setfield /output/##[TYPE=Table] stepmode 3
useclock /output/##[TYPE=Table] 1

/********************************************************************
**                                                                 **
**                       Simulation control                        **
**                                                                 **
********************************************************************/

/* Set up the clocks that we are going to use */
setclock 0 {dt} 0
setclock 1 {iodt} 1

/* Set the stimulus conditions */
setfield /axon/soma inject {inj}

/* Run the simulation */
reset
step {runtime} -time

/* Write plots */
setfield /output/out0 print "axon.out0"
setfield /output/outx print "axon.outx"

quit
