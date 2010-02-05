set datafile commentschars '/#'

p \
	'Ca.moose.plot' every :::0::0 w l, \
	'Ca.soma.plot' every :::0::0  w l
	set term png
	set output 'ca.png'
	replot
	set output
	set term x11


p \
	'Vm.moose.plot' every :::0::0 w l, \
	'Vm.soma.plot' every :::0::0  w l
	set term png
	set output 'vm.png'
	replot
	set output
	set term x11

