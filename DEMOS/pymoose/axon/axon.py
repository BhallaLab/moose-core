#!/usr/bin/env python

#/*******************************************************************
# * File:            axon.py
# * Description:     PyMOOSE version of DEMOS/axon/Axon.g
# *                  usage: python axon.py
# * Author:          Subhasis Ray
# * E-mail:          ray dot subhasis at gmail dot com
# * Created:         2008-09-30 17:53:34
# ********************************************************************/
#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/
import sys
sys.path.append("../channels")
sys.path.append("../..")
from bulbchan import *


print """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This demo loads a 501 compartment model of a linear excitable neuron. A square
wave pulse of current injection is given to the first compartment, and activity
is recorded from equally spaced compartments. The plots of membrane potential
from these compartments show propagation of action potentials along the length
of the axon.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

USE_SOLVER = True

#////////////////////////////////////////////////////////////////////////////////
#// MODEL CONSTRUCTION
#////////////////////////////////////////////////////////////////////////////////
SIMDT = 50e-6
IODT = 50e-6
SIMLENGTH = 0.05
INJECT = 5e-10
EREST_ACT = -0.065

context = moose.PyMooseBase.getContext()


context.setCwe("/library")
print "Making channels"
make_Na_mit_usb()
print "Making K_mit_usb"
make_K_mit_usb()

for child in moose.Neutral("/library").children():
    print "Objects in library:", child.path

context.setCwe("/")

#//=====================================
#//  Create cells
#//=====================================
context.readCell("axon.p", "/axon")


#////////////////////////////////////////////////////////////////////////////////
#// PLOTTING
#////////////////////////////////////////////////////////////////////////////////
plots = moose.Neutral("/plots")
vm0Table = moose.Table("/plots/Vm0")
vm0Table.xdivs = int(SIMLENGTH / IODT)
vm0Table.xmin = 0 
vm0Table.xmax = SIMLENGTH
vm0Table.stepMode = 3
soma = moose.Compartment("/axon/soma")
#addmsg /axon/soma /plots/Vm0 INPUT Vm
vm0Table.connect("inputRequest", soma, "Vm")
#create table /plots/Vm100
vm100Table = moose.Table("/plots/Vm100")
vm100Table.xdivs = int(SIMLENGTH / IODT)
vm100Table.xmin = 0 
vm100Table.xmax = SIMLENGTH
vm100Table.stepMode = 3
comp100 = moose.Compartment("/axon/c100")
#addmsg /axon/soma /plots/Vm0 INPUT Vm
vm100Table.connect("inputRequest", comp100, "Vm")

# create table /plots/Vm200
# call /plots/Vm200 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
# setfield /plots/Vm200 step_mode 3
# addmsg /axon/c200 /plots/Vm200 INPUT Vm
vm200Table = moose.Table("/plots/Vm200")
vm200Table.xdivs = int(SIMLENGTH / IODT)
vm200Table.xmin = 0 
vm200Table.xmax = SIMLENGTH
vm200Table.stepMode = 3
comp200 = moose.Compartment("/axon/c200")
vm200Table.connect("inputRequest", comp200, "Vm")

# create table /plots/Vm300
# call /plots/Vm300 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
# setfield /plots/Vm300 step_mode 3
# addmsg /axon/c300 /plots/Vm300 INPUT Vm
vm300Table = moose.Table("/plots/Vm300")
vm300Table.xdivs = int(SIMLENGTH / IODT)
vm300Table.xmin = 0 
vm300Table.xmax = SIMLENGTH
vm300Table.stepMode = 3
comp300 = moose.Compartment("/axon/c300")
vm300Table.connect("inputRequest", comp300, "Vm")

# create table /plots/Vm400
# call /plots/Vm400 TABCREATE {SIMLENGTH / IODT} 0 {SIMLENGTH}
# setfield /plots/Vm400 step_mode 3
# addmsg /axon/c400 /plots/Vm400 INPUT Vm
vm400Table = moose.Table("/plots/Vm400")
vm400Table.xdivs = int(SIMLENGTH / IODT)
vm400Table.xmin = 0 
vm400Table.xmax = SIMLENGTH
vm400Table.stepMode = 3
comp400 = moose.Compartment("/axon/c400")
vm400Table.connect("inputRequest", comp400, "Vm")


#////////////////////////////////////////////////////////////////////////////////
#// SIMULATION CONTROL
#////////////////////////////////////////////////////////////////////////////////

#//=====================================
#//  Stimulus
#//=====================================
#// Varying current injection
# create table /inject
# call /inject TABCREATE 100 0 {SIMLENGTH}
# setfield /inject step_mode 2
# setfield /inject stepsize 0
# addmsg /inject /axon/soma INJECT output
injectTable = moose.Table("/plots/Inject")
injectTable.xdivs = 100
injectTable.xmin = 0 
injectTable.xmax = SIMLENGTH
injectTable.stepMode = 2
injectTable.stepSize = 0
injectTable.connect("outputSrc", soma, "injectMsg")

current = INJECT 

# Injection current flips between 0.0 and {INJECT} at regular intervals
for i in range(101):
    if i % 20 == 0:
        current = INJECT - current
	injectTable[i] = current


#//=====================================
#//  Clocks
#//=====================================
#setclock 0 {SIMDT}
#setclock 1 {SIMDT}
#setclock 2 {IODT}
context.setClock(0, SIMDT, 0)
context.setClock(1, SIMDT, 0)
context.setClock(2, IODT, 0)
# useclock /inject 0
# useclock /plots/#[TYPE=table] 2
injectTable.useClock(0)
context.useClock(2, "/plots/#[TYPE=table]")

# //=====================================
# //  Solvers
# //=====================================
# // In Genesis, an hsolve object needs to be created.
# //
# // In Moose, hsolve is enabled by default. If USE_SOLVER is 0, we disable it by
# // switching to the Exponential Euler method.

# if ( USE_SOLVER )
# 	if ( GENESIS )
# 		create hsolve /axon/solve
# 		setfield /axon/solve \
# 			path /axon/##[TYPE=symcompartment],/axon/##[TYPE=compartment] \
# 			chanmode 1
# 		call /axon/solve SETUP
# 		setmethod 11
# 	end
# else
# 	if ( MOOSE )
# 		setfield /axon method "ee"
# 	end
# end
axon = moose.Cell("/axon")
if not USE_SOLVER:
    axon.method = "ee"
# //=====================================
# //  Simulation
# //=====================================
# reset
context.reset()
# step {SIMLENGTH} -time
context.step(SIMLENGTH)

# ////////////////////////////////////////////////////////////////////////////////
# //  Write Plots
# ////////////////////////////////////////////////////////////////////////////////
extension = ".moose.plot"
filename = "axon0" + extension
outfile = open(filename, "w")
outfile.write("/newplot\n")
outfile.write("/plotname Vm(0)\n")
for value in vm0Table:
    outfile.write(str(value)+"\n")
outfile.close()


filename = "axon100" + extension
outfile = open(filename, "w")
outfile.write("/newplot\n")
outfile.write("/plotname Vm(100)\n")
for value in vm100Table:
    outfile.write(str(value)+"\n")
outfile.close()

# filename = "axon200" @ {extension}
# openfile {filename} w
# writefile {filename} "/newplot"
# writefile {filename} "/plotname Vm(200)"
# closefile {filename}
# tab2file {filename} /plots/Vm200 table
filename = "axon200" + extension
outfile = open(filename, "w")
outfile.write("/newplot\n")
outfile.write("/plotname Vm(200)\n")
for value in vm200Table:
    outfile.write(str(value)+"\n")
outfile.close()

# filename = "axon300" @ {extension}
# openfile {filename} w
# writefile {filename} "/newplot"
# writefile {filename} "/plotname Vm(300)"
# closefile {filename}
# tab2file {filename} /plots/Vm300 table
filename = "axon300" + extension
outfile = open(filename, "w")
outfile.write("/newplot\n")
outfile.write("/plotname Vm(300)\n")
for value in vm300Table:
    outfile.write(str(value)+"\n")
outfile.close()

# filename = "axon400" @ {extension}
# openfile {filename} w
# writefile {filename} "/newplot"
# writefile {filename} "/plotname Vm(400)"
# closefile {filename}
# tab2file {filename} /plots/Vm400 table
filename = "axon400" + extension
outfile = open(filename, "w")
outfile.write("/newplot\n")
outfile.write("/plotname Vm(400)\n")
for value in vm400Table:
    outfile.write(str(value)+"\n")
outfile.close()

print """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plots written to axon*.plot. Each plot is a trace of membrane potential at a
different point along the axon.

If you have gnuplot, run 'gnuplot plot.gnuplot' to view the graphs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
