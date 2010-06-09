# Gnuplot script
# First run 'moose Network.g' to generate *.plot files

set datafile commentschars '/#'
set xlabel 'Step # [dt = 100e-6 s]'    # This is the plot dt
set ylabel 'Vm (V)'

#
# Input cell number 0
#
i = 0
set title '10x10 network of neurons: Input cell #1'
plot \
	'network.moose.plot' every :::i::i with line title 'MOOSE', \
	'network.genesis.plot' every :::i:1499:i with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'I0.png'
replot
set output
set term pop

print "Plot image written to I0.png.\n"

#
# Output cell number 0
#
i = i + 1
set title '10x10 network of neurons: Output cell #1'
plot \
	'network.moose.plot' every :::i::i with line title 'MOOSE', \
	'network.genesis.plot' every :::i:1499:i with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'V0.png'
replot
set output
set term pop

print "Plot image written to V0.png.\n"

#
# Output cell number 1
#
i = i + 1
set title '10x10 network of neurons: Output cell #1'
plot \
	'network.moose.plot' every :::i::i with line title 'MOOSE', \
	'network.genesis.plot' every :::i:1499:i with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 2
#
i = i + 1
set title '10x10 network of neurons: Output cell #2'
plot \
	'network.moose.plot' every :::i::i with line title 'MOOSE', \
	'network.genesis.plot' every :::i:1499:i with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 3
#
i = i + 1
set title '10x10 network of neurons: Output cell #3'
plot \
	'network.moose.plot' every :::i::i with line title 'MOOSE', \
	'network.genesis.plot' every :::i:1499:i with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 4
#
i = i + 1
set title '10x10 network of neurons: Output cell #4'
plot \
	'network.moose.plot' every :::i::i with line title 'MOOSE', \
	'network.genesis.plot' every :::i:1499:i with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 5
#
i = i + 1
set title '10x10 network of neurons: Output cell #5'
plot \
	'network.moose.plot' every :::i::i with line title 'MOOSE', \
	'network.genesis.plot' every :::i:1499:i with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 6
#
i = i + 1
set title '10x10 network of neurons: Output cell #6'
plot \
	'network.moose.plot' every :::i::i with line title 'MOOSE', \
	'network.genesis.plot' every :::i:1499:i with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 7
#
i = i + 1
set title '10x10 network of neurons: Output cell #7'
plot \
	'network.moose.plot' every :::i::i with line title 'MOOSE', \
	'network.genesis.plot' every :::i:1499:i with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 8
#
i = i + 1
set title '10x10 network of neurons: Output cell #8'
plot \
	'network.moose.plot' every :::i::i with line title 'MOOSE', \
	'network.genesis.plot' every :::i:1499:i with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

#
# Output cell number 9
#
i = i + 1
set title '10x10 network of neurons: Output cell #9'
plot \
	'network.moose.plot' every :::i::i with line title 'MOOSE', \
	'network.genesis.plot' every :::i:1499:i with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'V9.png'
replot
set output
set term pop

print "Plot image written to V9.png.\n"
