dt = 50e-6    # Should be same as 'IODT' from the GENESIS script.

set datafile comment '/'

#===============================================================================
# NMDA Gk: Gk blocked by MgBlock
#===============================================================================
set title 'NMDA Gk blocked by MgBlock.'
set xlabel 'Time (s)'
set ylabel 'Gk (S)'

p \
 'output/mgblock.Gk.-70.genesis.plot' every ::400::800 u (($0)+400)*dt:1 with line title 'Genesis: -70mV', \
 'output/mgblock.Gk.-50.genesis.plot' every ::400::800 u (($0)+400)*dt:1 with line title 'Genesis: -50mV', \
 'output/mgblock.Gk.-30.genesis.plot' every ::400::800 u (($0)+400)*dt:1 with line title 'Genesis: -30mV', \
 'output/mgblock.Gk.-10.genesis.plot' every ::400::800 u (($0)+400)*dt:1 with line title 'Genesis: -10mV', \
 'output/mgblock.Gk.10.genesis.plot' every ::400::800 u (($0)+400)*dt:1 with line title 'Genesis: 10mV', \
 'output/mgblock.Gk.30.genesis.plot' every ::400::800 u (($0)+400)*dt:1 with line title 'Genesis: 30mV', \
 'output/mgblock.Gk.50.genesis.plot' every ::400::800 u (($0)+400)*dt:1 with line title 'Genesis: 50mV', \
 'output/mgblock.Gk.-70.moose.plot' every 5::401::800 u (($0)*5+400)*dt:1 with points title 'Moose: -70mV', \
 'output/mgblock.Gk.-50.moose.plot' every 5::401::800 u (($0)*5+400)*dt:1 with points title 'Moose: -50mV', \
 'output/mgblock.Gk.-30.moose.plot' every 5::401::800 u (($0)*5+400)*dt:1 with points title 'Moose: -30mV', \
 'output/mgblock.Gk.-10.moose.plot' every 5::401::800 u (($0)*5+400)*dt:1 with points title 'Moose: -10mV', \
 'output/mgblock.Gk.10.moose.plot' every 5::401::800 u (($0)*5+400)*dt:1 with points title 'Moose: 10mV', \
 'output/mgblock.Gk.30.moose.plot' every 5::401::800 u (($0)*5+400)*dt:1 with points title 'Moose: 30mV', \
 'output/mgblock.Gk.50.moose.plot' every 5::401::800 u (($0)*5+400)*dt:1 with points title 'Moose: 50mV'

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
 'output/syn.Gk.-70.genesis.plot' every ::400::800 u (($0)+400)*dt:1 with line title 'Genesis: -70mV', \
 'output/syn.Gk.50.genesis.plot' every ::400::800 u (($0)+400)*dt:1 with line title 'Genesis: 50mV', \
 'output/syn.Gk.-70.moose.plot' every 5::401::800 u (($0)*5+400)*dt:1 with points title 'Moose: -70mV', \
 'output/syn.Gk.50.moose.plot' every 5::401::800 u (($0)*5+400)*dt:1 with points title 'Moose: 50mV'

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
 'output/c2.Vm.-70.genesis.plot' u ($0)*dt:1 with line title 'Genesis: -70mV', \
 'output/c2.Vm.-50.genesis.plot' u ($0)*dt:1 with line title 'Genesis: -50mV', \
 'output/c2.Vm.-30.genesis.plot' u ($0)*dt:1 with line title 'Genesis: -30mV', \
 'output/c2.Vm.-10.genesis.plot' u ($0)*dt:1 with line title 'Genesis: -10mV', \
 'output/c2.Vm.10.genesis.plot' u ($0)*dt:1 with line title 'Genesis: 10mV', \
 'output/c2.Vm.30.genesis.plot' u ($0)*dt:1 with line title 'Genesis: 30mV', \
 'output/c2.Vm.50.genesis.plot' u ($0)*dt:1 with line title 'Genesis: 50mV', \
 'output/c2.Vm.-70.moose.plot' every 50::1 u ($0)*50*dt:1 with points title 'Moose: -70mV', \
 'output/c2.Vm.-50.moose.plot' every 50::1 u ($0)*50*dt:1 with points title 'Moose: -50mV', \
 'output/c2.Vm.-30.moose.plot' every 50::1 u ($0)*50*dt:1 with points title 'Moose: -30mV', \
 'output/c2.Vm.-10.moose.plot' every 50::1 u ($0)*50*dt:1 with points title 'Moose: -10mV', \
 'output/c2.Vm.10.moose.plot' every 50::1 u ($0)*50*dt:1 with points title 'Moose: 10mV', \
 'output/c2.Vm.30.moose.plot' every 50::1 u ($0)*50*dt:1 with points title 'Moose: 30mV', \
 'output/c2.Vm.50.moose.plot' every 50::1 u ($0)*50*dt:1 with points title 'Moose: 50mV'

pause mouse key "Any key to continue.\n"

set term png
set output 'output/c2.Vm.png'
replot
set output
set term pop
