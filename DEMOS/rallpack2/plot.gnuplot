# Gnuplot script
# First run 'moose Rallpack2.g' to generate *.plot files

set datafile comments '/#'
set title 'Rallpack 2 (Branched passive cable)'
set xlabel 'Step # [dt = 50e-6 s]'
set ylabel 'Vm (V)'

#
# Root compartment
#
plot \
	'branch-0.moose.plot' with line title 'Compt 1 (MOOSE)', \
	'branch-0.genesis.plot' with line title 'Compt 1 (GENESIS)', \
	'branch-0.analytical.plot' using 2 with line title 'Compt 1 (Analytical)'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'branch-0.png'
replot
set output
set term pop

print "Plot image written to branch-0.png.\n"

#
# Leaf compartment
#
plot \
	'branch-x.moose.plot' with line title 'Compt 1023 (MOOSE)', \
	'branch-x.genesis.plot' with line title 'Compt 1023 (GENESIS)', \
	'branch-x.analytical.plot' using 2 with line title 'Compt 1023 (Analytical)'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'branch-x.png'
replot
set output
set term pop

print "Plot image written to branch-x.png.\n"
