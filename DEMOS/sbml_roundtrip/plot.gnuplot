set datafile commentschars '/#'

set title 'Species:TOR_minus_clx'
p \
	'moose1.plot' every :::0::0  title 'moose1' w l, \
	'moose2.plot' every :::0::0  title 'moose2' , \
	'copasi.plot' every :::0::0 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'TOR_minus_clx.png'
	replot
	set output
	set term pop
	print "Plot image written to TOR_minus_clx.png.\n"

set title 'Species:S6K_star_'
p \
	'moose1.plot' every :::1::1  title 'moose1' w l, \
	'moose2.plot' every :::1::1  title 'moose2' , \
	'copasi.plot' every :::1::1 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'S6K_star_.png'
	replot
	set output
	set term pop
	print "Plot image written to S6K_star_.png.\n"

set title 'Species:S6K'
p \
	'moose1.plot' every :::2::2  title 'moose1' w l, \
	'moose2.plot' every :::2::2  title 'moose2' , \
	'copasi.plot' every :::2::2 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'S6K.png'
	replot
	set output
	set term pop
	print "Plot image written to S6K.png.\n"

set title 'Species:S6K_thr_minus_412'
p \
	'moose1.plot' every :::3::3  title 'moose1' w l, \
	'moose2.plot' every :::3::3  title 'moose2' , \
	'copasi.plot' every :::3::3 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'S6K_thr_minus_412.png'
	replot
	set output
	set term pop
	print "Plot image written to S6K_thr_minus_412.png.\n"

set title 'Species:_40S_inact_'
p \
	'moose1.plot' every :::4::4  title 'moose1' w l, \
	'moose2.plot' every :::4::4  title 'moose2' , \
	'copasi.plot' every :::4::4 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output '_40S_inact_.png'
	replot
	set output
	set term pop
	print "Plot image written to _40S_inact_.png.\n"

set title 'Species:PDK1'
p \
	'moose1.plot' every :::5::5  title 'moose1' w l, \
	'moose2.plot' every :::5::5  title 'moose2' , \
	'copasi.plot' every :::5::5 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'PDK1.png'
	replot
	set output
	set term pop
	print "Plot image written to PDK1.png.\n"

set title 'Species:PP2A'
p \
	'moose1.plot' every :::6::6  title 'moose1' w l, \
	'moose2.plot' every :::6::6  title 'moose2' , \
	'copasi.plot' every :::6::6 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'PP2A.png'
	replot
	set output
	set term pop
	print "Plot image written to PP2A.png.\n"

set title 'Species:Rheb_minus_GTP'
p \
	'moose1.plot' every :::7::7  title 'moose1' w l, \
	'moose2.plot' every :::7::7  title 'moose2' , \
	'copasi.plot' every :::7::7 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'Rheb_minus_GTP.png'
	replot
	set output
	set term pop
	print "Plot image written to Rheb_minus_GTP.png.\n"

set title 'Species:TOR_Rheb_minus_GTP_clx'
p \
	'moose1.plot' every :::8::8  title 'moose1' w l, \
	'moose2.plot' every :::8::8  title 'moose2' , \
	'copasi.plot' every :::8::8 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'TOR_Rheb_minus_GTP_clx.png'
	replot
	set output
	set term pop
	print "Plot image written to TOR_Rheb_minus_GTP_clx.png.\n"

set title 'Species:S6K_tot'
p \
	'moose1.plot' every :::9::9  title 'moose1' w l, \
	'moose2.plot' every :::9::9  title 'moose2' , \
	'copasi.plot' every :::9::9 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'S6K_tot.png'
	replot
	set output
	set term pop
	print "Plot image written to S6K_tot.png.\n"

set title 'Species:_40S'
p \
	'moose1.plot' every :::10::10  title 'moose1' w l, \
	'moose2.plot' every :::10::10  title 'moose2' , \
	'copasi.plot' every :::10::10 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output '_40S.png'
	replot
	set output
	set term pop
	print "Plot image written to _40S.png.\n"

set title 'Species:S6K_thr_minus_252'
p \
	'moose1.plot' every :::11::11  title 'moose1' w l, \
	'moose2.plot' every :::11::11  title 'moose2' , \
	'copasi.plot' every :::11::11 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'S6K_thr_minus_252.png'
	replot
	set output
	set term pop
	print "Plot image written to S6K_thr_minus_252.png.\n"

set title 'Species:MAPK_star'
p \
	'moose1.plot' every :::12::12  title 'moose1' w l, \
	'moose2.plot' every :::12::12  title 'moose2' , \
	'copasi.plot' every :::12::12 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output 'MAPK_star.png'
	replot
	set output
	set term pop
	print "Plot image written to MAPK_star.png.\n"

set title 'Species:_40S_basal'
p \
	'moose1.plot' every :::13::13  title 'moose1' w l, \
	'moose2.plot' every :::13::13  title 'moose2' , \
	'copasi.plot' every :::13::13 title 'copasi' w l
	pause mouse key "Any key to continue.\n"
	set term png
	set output '_40S_basal.png'
	replot
	set output
	set term pop
	print "Plot image written to _40S_basal.png.\n"


