# Gnuplot script
# First run 'moose Kholodenko.g' to generate *.plot files

set datafile comments '/#'
set title 'Concentration vs time'
set xlabel 'Step # [dt = 0.1 s]'    # This is the plot dt
set ylabel 'Concentration'

set title 'Species:MAPK'
plot \
	'reference.plot' every :::0::0 w l title 'GENESIS', \
	'test.plot' every :::3::3 w l title 'MOOSE'
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'MAPK.png'
	replot
	set output
	set term pop
	print "Plot image written to MAPK.png.\n"

set title 'Species:MKK'
plot \
	'reference.plot' every :::1::1 w l title 'GENESIS', \
	'test.plot' every :::5::5 w l title 'MOOSE'
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'MKK.png'
	replot
	set output
	set term pop
	print "Plot image written to MKK.png.\n"

set title 'Species:MAPK-PP'
plot \
	'reference.plot' every :::2::2 w l title 'GENESIS', \
	'test.plot' every :::0::0 w l title 'MOOSE'
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'MAPK-PP.png'
	replot
	set output
	set term pop
	print "Plot image written to MAPK-PP.png.\n"

set title 'Species:Ras-MKKKK'
plot \
	'reference.plot' every :::3::3 w l title 'GENESIS', \
	'test.plot' every :::2::2 w l title 'MOOSE'
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'Ras-MKKKK.png'
	replot
	set output
	set term pop
	print "Plot image written to Ras-MKKKK.png.\n"

set title 'Species:MAPK'
plot \
	'reference.plot' every :::4::4 w l title 'GENESIS', \
	'test.plot' every :::1::1 w l title 'MOOSE'
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'MAPK.png'
	replot
	set output
	set term pop
	print "Plot image written to MAPK.png.\n"
	
set title 'Species:MKKK'
plot \	
	'reference.plot' every :::5::5 w l title 'GENESIS', \
	'test.plot' every :::4::4 w l title 'MOOSE'
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'MKKK.png'
	replot
	set output
	set term pop
	print "Plot image written to MKKK.png.\n"

