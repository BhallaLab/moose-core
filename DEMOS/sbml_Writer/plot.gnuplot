set datafile commentschars '/#'

set title 'Species:PDK1'
p \
	'moose.plot' every :::0::0  title 'moose' w l, \
	'copasi.plot' every :::0::0 title 'copasi' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'PDK1.png'
	replot
	set output
	set term pop
	print "Plot image written to PDK1.png.\n"

set title 'Species:PP2A'
p \
	'moose.plot' every :::2::2  title 'moose' w l, \
	'copasi.plot' every :::1::1 title 'copasi' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'PP2A.png'
	replot
	set output
	set term pop
	print "Plot image written to PP2A.png.\n"

set title 'Species:Rheb_minus_GTP'
p \
	'moose.plot' every :::6::6  title 'moose' w l, \
	'copasi.plot' every :::2::2 title 'copasi' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'Rheb_minus_GTP.png'
	replot
	set output
	set term pop
	print "Plot image written to Rheb_minus_GTP.png.\n"

set title 'Species:TOR_minus_clx'
p \
	'moose.plot' every :::7::7  title 'moose' w l, \
	'copasi.plot' every :::3::3 title 'copasi' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'TOR_minus_clx.png'
	replot
	set output
	set term pop
	print "Plot image written to TOR_minus_clx.png.\n"

set title 'Species:TOR_Rheb_minus_GTP_clx'
p \
	'moose.plot' every :::8::8  title 'moose' w l, \
	'copasi.plot' every :::4::4 title 'copasi' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'TOR_Rheb_minus_GTP_clx.png'
	replot
	set output
	set term pop
	print "Plot image written to TOR_Rheb_minus_GTP_clx.png.\n"

set title 'Species:S6K_star_'
p \
	'moose.plot' every :::10::10  title 'S6K_star_(moose)' w l, \
	'copasi.plot' every :::5::5 title 'S6K_star_(copasi)' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'S6K_star_.png'
	replot
	set output
	set term pop
	print "Plot image written to S6K_star_.png.\n"

set title 'Species:S6K'
p \
	'moose.plot' every :::11::11  title 'S6K(moose)' w l, \
	'copasi.plot' every :::6::6 title 'S6K(copasi)' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'S6K.png'
	replot
	set output
	set term pop
	print "Plot image written to S6K.png.\n"

set title 'Species:S6K_thr_minus_412'
p \
	'moose.plot' every :::12::12  title 'S6K_thr_minus_412(moose)' w l, \
	'copasi.plot' every :::7::7 title 'S6K_thr_minus_412(copasi)' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'S6K_thr_minus_412.png'
	replot
	set output
	set term pop
	print "Plot image written to S6K_thr_minus_412.png.\n"

set title 'Species:40S_inact'
p \
	'moose.plot' every :::14::14  title '40S_inact(moose)' w l, \
	'copasi.plot' every :::8::8 title '40S_inact(copasi)' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output '40S_inact.png'
	replot
	set output
	set term pop
	print "Plot image written to 40S_inact.png.\n"

set title 'Species:S6K_tot'
p \
	'moose.plot' every :::15::15  title 'S6K_tot(moose)' w l, \
	'copasi.plot' every :::9::9 title 'S6K_tot(copasi)' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'S6K_tot.png'
	replot
	set output
	set term pop
	print "Plot image written to S6K_tot.png.\n"

set title 'Species:_40S'
p \
	'moose.plot' every :::16::16  title '_40S(moose)' w l, \
	'copasi.plot' every :::10::10 title '_40S(copasi)' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output '_40S.png'
	replot
	set output
	set term pop
	print "Plot image written to _40S.png.\n"

set title 'Species:S6K_thr_minus_252'
p \
	'moose.plot' every :::17::17  title 'S6K_thr_minus_252(moose)' w l, \
	'copasi.plot' every :::11::11 title 'S6K_thr_minus_252(copasi)' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'S6K_thr_minus_252.png'
	replot
	set output
	set term pop
	print "Plot image written to S6K_thr_minus_252.png.\n"

set title 'Species:MAPK_star'
p \
	'moose.plot' every :::19::19  title 'MAPK_star(moose)' w l, \
	'copasi.plot' every :::12::12 title 'MAPK_star(copasi)' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'MAPK_star.png'
	replot
	set output
	set term pop
	print "Plot image written to MAPK_star.png.\n"

set title 'Species:_40S_basal'
p \
	'moose.plot' every :::21::21  title '_40S_basal(moose)' w l, \
	'copasi.plot' every :::13::13 title '_40S_basal(copasi)' w l 
	pause mouse key "Any key to continue.\n"
	set term png
	set output '_40S_basal.png'
	replot
	set output
	set term pop
	print "Plot image written to _40S_basal.png.\n"


