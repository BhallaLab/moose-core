dt = 50e-6    # Should be same as 'IODT' from the GENESIS script.

set datafile comment '/'

#===============================================================================
# NMDA Gk: Gk blocked by MgBlock
#===============================================================================
set title 'NMDA Gk blocked by MgBlock.'
set xlabel 'Time (s)'
set ylabel 'Gk (S)'

p \
 'output/mgblock.Gk.1.genesis.plot' every ::::10000 u ($0)*dt:1 with line title 'Genesis: -70mV', \
 'output/mgblock.Gk.2.genesis.plot' every ::::10000 u ($0)*dt:1 with line title 'Genesis: -50mV', \
 'output/mgblock.Gk.3.genesis.plot' every ::::10000 u ($0)*dt:1 with line title 'Genesis: -30mV', \
 'output/mgblock.Gk.4.genesis.plot' every ::::10000 u ($0)*dt:1 with line title 'Genesis: -10mV', \
 'output/mgblock.Gk.5.genesis.plot' every ::::10000 u ($0)*dt:1 with line title 'Genesis: 10mV', \
 'output/mgblock.Gk.6.genesis.plot' every ::::10000 u ($0)*dt:1 with line title 'Genesis: 30mV', \
 'output/mgblock.Gk.7.genesis.plot' every ::::10000 u ($0)*dt:1 with line title 'Genesis: 50mV', \
 'output/mgblock.Gk.1.moose.plot' every 100::::10000 u (($0)*100)*dt:1 with points title 'Moose: -70mV', \
 'output/mgblock.Gk.2.moose.plot' every 100::::10000 u (($0)*100)*dt:1 with points title 'Moose: -50mV', \
 'output/mgblock.Gk.3.moose.plot' every 100::::10000 u (($0)*100)*dt:1 with points title 'Moose: -30mV', \
 'output/mgblock.Gk.4.moose.plot' every 100::::10000 u (($0)*100)*dt:1 with points title 'Moose: -10mV', \
 'output/mgblock.Gk.5.moose.plot' every 100::::10000 u (($0)*100)*dt:1 with points title 'Moose: 10mV', \
 'output/mgblock.Gk.6.moose.plot' every 100::::10000 u (($0)*100)*dt:1 with points title 'Moose: 30mV', \
 'output/mgblock.Gk.7.moose.plot' every 100::::10000 u (($0)*100)*dt:1 with points title 'Moose: 50mV'

pause mouse key "Any key to continue.\n"

set term png
set output 'output/mgblock.Gk.png'
replot
set output
set term pop

#===============================================================================
# Non-blocked Gk of synapse.
#===============================================================================
set title 'Unblocked NMDA Gk (before blocking by MgBlock).'
set xlabel 'Time (s)'
set ylabel 'Gk (S)'

p \
 'output/syn.Gk.1.genesis.plot' every ::::10000 u ($0)*dt:1 with line title 'Genesis: -70mV', \
 'output/syn.Gk.7.genesis.plot' every ::::10000 u ($0)*dt:1 with line title 'Genesis: 50mV', \
 'output/syn.Gk.1.moose.plot' every 100::::10000 u (($0)*100)*dt:1 with points title 'Moose: -70mV', \
 'output/syn.Gk.7.moose.plot' every 100::::10000 u (($0)*100)*dt:1 with points title 'Moose: 50mV'

pause mouse key "Any key to continue.\n"

set term png
set output 'output/syn.Gk.png'
replot
set output
set term pop

#===============================================================================
# Vm of postsynaptic compartment.
#===============================================================================
set title 'Vm of postsynaptic compartment. Should be flat if voltage-clamped.'
set xlabel 'Time (s)'
set ylabel 'Vm (mV)'

p \
 'output/c2.Vm.1.genesis.plot' u ($0)*dt:1 with line title 'Genesis: -70mV', \
 'output/c2.Vm.2.genesis.plot' u ($0)*dt:1 with line title 'Genesis: -50mV', \
 'output/c2.Vm.3.genesis.plot' u ($0)*dt:1 with line title 'Genesis: -30mV', \
 'output/c2.Vm.4.genesis.plot' u ($0)*dt:1 with line title 'Genesis: -10mV', \
 'output/c2.Vm.5.genesis.plot' u ($0)*dt:1 with line title 'Genesis: 10mV', \
 'output/c2.Vm.6.genesis.plot' u ($0)*dt:1 with line title 'Genesis: 30mV', \
 'output/c2.Vm.7.genesis.plot' u ($0)*dt:1 with line title 'Genesis: 50mV', \
 'output/c2.Vm.1.moose.plot' every 1000 u ($0)*1000*dt:1 with points title 'Moose: -70mV', \
 'output/c2.Vm.2.moose.plot' every 1000 u ($0)*1000*dt:1 with points title 'Moose: -50mV', \
 'output/c2.Vm.3.moose.plot' every 1000 u ($0)*1000*dt:1 with points title 'Moose: -30mV', \
 'output/c2.Vm.4.moose.plot' every 1000 u ($0)*1000*dt:1 with points title 'Moose: -10mV', \
 'output/c2.Vm.5.moose.plot' every 1000 u ($0)*1000*dt:1 with points title 'Moose: 10mV', \
 'output/c2.Vm.6.moose.plot' every 1000 u ($0)*1000*dt:1 with points title 'Moose: 30mV', \
 'output/c2.Vm.7.moose.plot' every 1000 u ($0)*1000*dt:1 with points title 'Moose: 50mV'

pause mouse key "Any key to continue.\n"

set term png
set output 'output/c2.Vm.png'
replot
set output
set term pop
