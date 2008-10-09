# Gnuplot script
# First run 'moose Network.g' to generate *.plot files

set datafile commentschars '/#'
set xlabel 'Step # [dt = 100e-6 s]'    # This is the plot dt
set ylabel 'Vm (V)'

#
# Output cell number 0
#
set title '10x10 network of neurons: Output cell #1'
plot \
	'network.moose.plot' every :::0::0 with line title 'MOOSE', \
	'network.genesis.plot' every :::0:1499:0 with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'V0.png'
replot
set output
set term x11

print "Plot image written to V0.png.\n"

#
# Output cell number 1
#
set title '10x10 network of neurons: Output cell #1'
plot \
	'network.moose.plot' every :::1::1 with line title 'MOOSE', \
	'network.genesis.plot' every :::1:1499:1 with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 2
#
set title '10x10 network of neurons: Output cell #2'
plot \
	'network.moose.plot' every :::2::2 with line title 'MOOSE', \
	'network.genesis.plot' every :::2:1499:2 with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 3
#
set title '10x10 network of neurons: Output cell #3'
plot \
	'network.moose.plot' every :::3::3 with line title 'MOOSE', \
	'network.genesis.plot' every :::3:1499:3 with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 4
#
set title '10x10 network of neurons: Output cell #4'
plot \
	'network.moose.plot' every :::4::4 with line title 'MOOSE', \
	'network.genesis.plot' every :::4:1499:4 with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 5
#
set title '10x10 network of neurons: Output cell #5'
plot \
	'network.moose.plot' every :::5::5 with line title 'MOOSE', \
	'network.genesis.plot' every :::5:1499:5 with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 6
#
set title '10x10 network of neurons: Output cell #6'
plot \
	'network.moose.plot' every :::6::6 with line title 'MOOSE', \
	'network.genesis.plot' every :::6:1499:6 with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 7
#
set title '10x10 network of neurons: Output cell #7'
plot \
	'network.moose.plot' every :::7::7 with line title 'MOOSE', \
	'network.genesis.plot' every :::7:1499:7 with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 8
#
set title '10x10 network of neurons: Output cell #8'
plot \
	'network.moose.plot' every :::8::8 with line title 'MOOSE', \
	'network.genesis.plot' every :::8:1499:8 with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 9
#
set title '10x10 network of neurons: Output cell #9'
plot \
	'network.moose.plot' every :::9::9 with line title 'MOOSE', \
	'network.genesis.plot' every :::9:1499:9 with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'V9.png'
replot
set output
set term x11

print "Plot image written to V9.png.\n"
