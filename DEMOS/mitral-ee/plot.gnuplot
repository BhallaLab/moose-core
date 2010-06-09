# Gnuplot script
# First run 'moose Mitral.g' to generate *.plot files

set datafile commentschars '/#'
set title 'Mitral cell model.'
set xlabel 'Step # [dt = 1.0e-4 s]'    # This is the plot dt
set ylabel 'Vm (V)'

plot \
	'mitral.moose.plot' with line title 'MOOSE', \
	'mitral.genesis.plot' with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'mitral.png'
replot
set output
set term pop

print "Plot image written to mitral.png.\n"
