# Gnuplot script to draw plot on screen, and write image to file

set datafile comments '/#'
set title 'Rallpack 2: Inject #1023, Record #1023'
set xlabel 'Step # [dt = 50e-6 s]'
set ylabel 'Vm (V)'

p \
	'sim_branch.x.genesis' every ::1 t 'Genesis', \
	'sim_branch.x.moose-new' w l t 'Moose - new solver', \
	'sim_branch.x.moose-old' w l t 'Moose - old solver'

pause 1

set term png
set output 'c1023.png'
replot
set output
set term x11
