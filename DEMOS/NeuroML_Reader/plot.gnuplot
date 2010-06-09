set datafile commentschars '/#'

set title 'soma'
p \
	'moose.plot' every :::0::0  title 'soma(neuroml)' w l, \
	'Vm.genesis.plot' every :::0::0 title 'soma(genesis)' w l, \
	'Vm.moose.plot' every :::0::0 title 'soma(moose)' w l
	set term png
	set output 'Vm.png'
	replot
	set output
	set term pop
	print "Plot image written to Vm.png.\n"


