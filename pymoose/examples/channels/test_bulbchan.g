
/*******************************************************************
 * File:            test_bulbchan.g
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray dot subhasis at gmail dot com
 * Created:         2008-10-23 14:32:18
 ********************************************************************/
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
include compatibility.g
include bulbchan.g

float SIMDT = 50e-6
float IODT = 50e-6
float SIMLENGTH = 0.05
float INJECT = 5e-10
float EREST_ACT = -0.065


function test_K_mit_usb
    create neutral /test_K_mit_usb
    create compartment /test_K_mit_usb/comp

    // the values are set from that calculated in the python version
    setfield ^ \
        dia 19e-6 \
        len 28e-6 \
        Ra 49377.7108762 \
        Rm 1196653707.46 \
        Cm 1.67132729171e-11 \
        Em {EREST_ACT}  \
        initVm {EREST_ACT} \
        inject 0.0

    ce /test_K_mit_usb/comp
    make_K_mit_usb
   
    // the values are set from that calculated in the python version
    setfield K_mit_usb Gbar 2.00559275005e-06 Ek -0.07
    ce /test_K_mit_usb
    addmsg comp/K_mit_usb/channel comp/channel

    // Create a pulsegen object to give injection current
    create pulsegen /test_K_mit_usb/pulse
    setfield ^ level1 {INJECT} delay1 0.01 width1 0.01
    addmsg /test_K_mit_usb/pulse /test_K_mit_usb/comp INJECT output

    // Keep all the tables for recording data inside one container
    create neutral /test_K_mit_usb/data
    // Table to record channel current
    create table /test_K_mit_usb/data/K_mit_usb_Ik
    setfield ^ stepmode 3
    addmsg /test_K_mit_usb/data/K_mit_usb_Ik/inputRequest /test_K_mit_usb/comp/K_mit_usb/Ik
    // Table to record injection current
    create table /test_K_mit_usb/data/inject
    setfield ^ stepmode 3
    addmsg /test_K_mit_usb/data/inject/inputRequest /test_K_mit_usb/pulse/output
    // Table to record membrane potential
    create table /test_K_mit_usb/data/vm 
    setfield ^ step_mode 3
    addmsg /test_K_mit_usb/comp /test_K_mit_usb/data/vm INPUT Vm

    // setup the scheduling
    setclock 0 {SIMDT}
    setclock 1 {SIMDT}
    setclock 2 {IODT}

    useclock /test_K_mit_usb/data/# 2
//  the following useclock calls are unnecessary as default scheduling takes care of them
//     useclock /test_K_mit_usb/comp,/test_K_mit_usb/comp/## 0
//     useclock /test_K_mit_usb/pulse 0

    reset

    step {SIMLENGTH} -time
    setfield /test_K_mit_usb/data/K_mit_usb_Ik print "K_mit_usb_Ik.moose.plot" 
    setfield /test_K_mit_usb/data/inject print "K_mit_usb_inject.moose.plot"
    setfield /test_K_mit_usb/data/vm print "K_mit_usb_Vm.moose.plot"

    //-----------------------------
    // These are for diagonostics
    //-----------------------------
    setfield /test_K_mit_usb/comp/K_mit_usb/xGate/A print "K_XA.moose.plot"
    setfield /test_K_mit_usb/comp/K_mit_usb/xGate/B print "K_XB.moose.plot"
    setfield /test_K_mit_usb/comp/K_mit_usb/yGate/A print "K_YA.moose.plot"
    setfield /test_K_mit_usb/comp/K_mit_usb/yGate/B print "K_YB.moose.plot"
end

///////////////
// Main
///////////////
test_K_mit_usb
