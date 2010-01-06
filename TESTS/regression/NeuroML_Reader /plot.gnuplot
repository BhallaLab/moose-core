set datafile commentschars '/#'

set title 'soma'
p \
	'moose.plot' every :::0::0  title 'soma(nml)' w l, \
	'Vm.genesis.plot' every :::0::0 title 'soma(genesis)' w l, \
	'Vm.moose.plot' every :::0::0 title 'soma(moose)' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'Vm.png'
	replot
	set output
	set term x11
	print "Plot image written to Vm.png.\n"


