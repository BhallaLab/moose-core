include plotUtil.g

str infile = "acc88.xml"
str outfile = "moose.plot"
int plot_all = 1
str plot_some = " "   

str target = "/kinetics"

int USE_SOLVER = 0
float SIMLENGTH = 1000
float SIMDT = 0.001
float IODT = 10.0
int IOCLOCK = 2

setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {IODT}

include acc88.g
writeSBML { infile } { target }

init_plots { SIMLENGTH } { IOCLOCK } { IODT }

str mol
if ( plot_all )
	foreach mol ( { el { target }/##[TYPE=kpool] } )
		echo "Adding plot: "{ mol }
		add_plot { mol } Co { outfile }
	end
else
	foreach mol ( { arglist { plot_some } } )
		echo "Adding plot: "{ mol }
		add_plot { mol } Co { outfile }
	end
end

if ( ! USE_SOLVER )
	setfield { target } method "ee"
end
reset
step {SIMLENGTH} -time

/* Clear the output file */
openfile { outfile } w
closefile { outfile }

save_plots
quit

