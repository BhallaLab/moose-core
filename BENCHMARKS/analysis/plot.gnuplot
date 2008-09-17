set title 'Linear Cable: Run-time v/s no. of compartments'
set xlabel 'Number of compartments'
set ylabel 'Time to run (seconds)'
set logscale x
p \
	'rall1-size' u 2:3 ev ::4 w lp lw 2 ps 2 t 'GENESIS',	\
	'rall1-size' u 2:4 ev ::4 w lp lw 2 ps 2 t 'MOOSE'
set term png
set output 'rall1-size.png'
replot
set output
set term x11

set logscale x
p \
	'rall1-size' u 2:($3/$2) ev ::4 w lp lw 2 ps 2 t 'GENESIS',	\
	'rall1-size' u 2:($4/$2) ev ::4 w lp lw 2 ps 2 t 'MOOSE'

set title 'Branching Cable: Run-time v/s no. of compartments'
set xlabel 'Number of compartments'
set ylabel 'Time to run (seconds)'
set logscale x
p \
	'rall2-size' u 3:4 w lp lw 2 ps 2 t 'GENESIS',	\
	'rall2-size' u 3:5 w lp lw 2 ps 2 t 'MOOSE'
set term png
set output 'rall2-size.png'
replot
set output
set term x11

set logscale x
p \
	'rall2-size' u 3:($4/$3) w lp lw 2 ps 2 t 'GENESIS',	\
	'rall2-size' u 3:($5/$3) w lp lw 2 ps 2 t 'MOOSE'

unset logscale x
p \
	'rall2-size' u 2:4 w lp lw 2 ps 2 t 'GENESIS',	\
	'rall2-size' u 2:5 w lp lw 2 ps 2 t 'MOOSE'

set title 'CA3 cell model by Traub: Run-time v/s no. of cells'
set xlabel 'Number of cells'
set ylabel 'Time to run (seconds)'
set logscale x
p \
	'traub91-ncell' u 2:3 w lp lw 2 ps 2 t 'GENESIS',	\
	'traub91-ncell' u 2:4 w lp lw 2 ps 2 t 'MOOSE'
set term png
set output 'traub91-ncell.png'
replot
set output
set term x11

set logscale x
p \
	'traub91-ncell' u 2:($3/$2) w lp lw 2 ps 2 t 'GENESIS',	\
	'traub91-ncell' u 2:($4/$2) w lp lw 2 ps 2 t 'MOOSE'

set title 'Myelinated axon: Run-time v/s no. of cells'
set xlabel 'Number of cells'
set ylabel 'Time to run (seconds)'
set logscale x
p \
	'myelin-ncell' u 2:3 w lp lw 2 ps 2 t 'GENESIS',	\
	'myelin-ncell' u 2:4 w lp lw 2 ps 2 t 'MOOSE'
set term png
set output 'myelin-ncell.png'
replot
set output
set term x11

set title 'Myelinated axon: Normalized run-time v/s no. of cells'
set xlabel 'Number of cells'
set ylabel 'Time to run / Number of cells (seconds)'
set logscale x
set yrange[0:4]
p \
	'myelin-ncell' u 2:($3/$2) w lp lw 2 ps 2 t 'GENESIS',	\
	'myelin-ncell' u 2:($4/$2) w lp lw 2 ps 2 t 'MOOSE'
set term png
set output 'myelin-ncell-normalized.png'
replot
set output
set term x11
set autoscale

set title 'Linear Cable: Run-time v/s no. of cells (No. of compartments: 10, 100, 1000)'
set xlabel 'Number of cells'
set ylabel 'Time to run (seconds)'
set logscale x
p \
	'rall1-ncell-size' u 2:3 w lp lw 2 ps 2 t 'GENESIS',	\
	'rall1-ncell-size' u 2:4 w lp lw 2 ps 2 t 'MOOSE'
set term png
set output 'rall1-size-ncell.png'
replot
set output
set term x11

set logscale x
p \
	'rall1-ncell-size' u 2:($3/$2) w lp lw 2 ps 2 t 'GENESIS',	\
	'rall1-ncell-size' u 2:($4/$2) w lp lw 2 ps 2 t 'MOOSE'

set logscale x
p \
	'rall1-ncell-size-10' u 2:($3/10) w lp lw 2 ps 2 t 'GENESIS 10',	\
	'rall1-ncell-size-10' u 2:($4/10) w lp lw 2 ps 2 t 'MOOSE 10',	\
	'rall1-ncell-size-100' u 2:($3/100) w lp lw 2 ps 2 t 'GENESIS 100',	\
	'rall1-ncell-size-100' u 2:($4/100) w lp lw 2 ps 2 t 'MOOSE 100',	\
	'rall1-ncell-size-1000' u 2:($3/1000) w lp lw 2 ps 2 t 'GENESIS 1000',	\
	'rall1-ncell-size-1000' u 2:($4/1000) w lp lw 2 ps 2 t 'MOOSE 1000'

set logscale x
p \
	'rall1-ncell-size-10' u 2:($4/10) w lp lw 2 ps 2 t 'MOOSE 10',	\
	'rall1-ncell-size-100' u 2:($4/100) w lp lw 2 ps 2 t 'MOOSE 100',	\
	'rall1-ncell-size-1000' u 2:($4/1000) w lp lw 2 ps 2 t 'MOOSE 1000'

set logscale x
p \
	'rall1-ncell-size-10' u 2:($3/10) w lp lw 2 ps 2 t 'GENESIS 10',	\
	'rall1-ncell-size-100' u 2:($3/100) w lp lw 2 ps 2 t 'GENESIS 100',	\
	'rall1-ncell-size-1000' u 2:($3/1000) w lp lw 2 ps 2 t 'GENESIS 1000'

set logscale x
p \
	'rall1-ncell-size-10' u 2:($3/($2)/10) w lp lw 2 ps 2 t 'GENESIS 10',	\
	'rall1-ncell-size-10' u 2:($4/($2)/10) w lp lw 2 ps 2 t 'MOOSE 10',	\
	'rall1-ncell-size-100' u 2:($3/($2)/100) w lp lw 2 ps 2 t 'GENESIS 100',	\
	'rall1-ncell-size-100' u 2:($4/($2)/100) w lp lw 2 ps 2 t 'MOOSE 100',	\
	'rall1-ncell-size-1000' u 2:($3/($2)/1000) w lp lw 2 ps 2 t 'GENESIS 1000',	\
	'rall1-ncell-size-1000' u 2:($4/($2)/1000) w lp lw 2 ps 2 t 'MOOSE 1000'

set logscale x
p \
	'rall1-ncell-size-10' u 2:($4/($2)/10) w lp lw 2 ps 2 t 'MOOSE 10',	\
	'rall1-ncell-size-100' u 2:($4/($2)/100) w lp lw 2 ps 2 t 'MOOSE 100',	\
	'rall1-ncell-size-1000' u 2:($4/($2)/1000) w lp lw 2 ps 2 t 'MOOSE 1000'

set logscale x
p \
	'rall1-ncell-size-10' u 2:($3/($2)/10) w lp lw 2 ps 2 t 'GENESIS 10',	\
	'rall1-ncell-size-100' u 2:($3/($2)/100) w lp lw 2 ps 2 t 'GENESIS 100',	\
	'rall1-ncell-size-1000' u 2:($3/($2)/1000) w lp lw 2 ps 2 t 'GENESIS 1000'

set title 'Branching Cable: Run-time v/s no. of cells (No. of compartments: 31, 127, 511)'
set xlabel 'Number of cells'
set ylabel 'Time to run (seconds)'
set logscale x
p \
	'rall2-ncell-size-small' u 2:3 w lp lw 2 ps 2 t 'GENESIS',	\
	'rall2-ncell-size-small' u 2:4 w lp lw 2 ps 2 t 'MOOSE'
set term png
set output 'rall2-size-ncell.png'
replot
set output
set term x11
