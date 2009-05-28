include plotUtil.g

str infile = "acc71.xml"
str outfile1 = "moose1.plot"
str outfile2 = "moose2.plot"
int plot_all = 1
str plot_some = " "   

str target = "/kinetics"

int USE_SOLVER = 0
float SIMLENGTH = 100
float SIMDT = 0.01
float IODT = 1.0
int IOCLOCK = 2

setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {IODT}

readSBML { infile } { target }
writeSBML acc71_dup.xml { target }

init_plots { SIMLENGTH } { IOCLOCK } { IODT }

str mol
if ( plot_all )
	foreach mol ( { el { target }/##[TYPE=kpool] } )
		echo "Adding plot: "{ mol }
		add_plot { mol } Co { outfile1 }
	end
else
	foreach mol ( { arglist { plot_some } } )
		echo "Adding plot: "{ mol }
		add_plot { mol } Co { outfile1 }
	end
end

if ( ! USE_SOLVER )
	setfield { target } method "ee"
end
reset
step {SIMLENGTH} -time

/* Clear the output file */
openfile { outfile1 } w
closefile { outfile1 }
save_plots

readSBML acc71_dup.xml { target }

init_plots { SIMLENGTH } { IOCLOCK } { IODT }

str mol
if ( plot_all )
	foreach mol ( { el { target }/##[TYPE=kpool] } )
		echo "Adding plot: "{ mol }
		add_plot { mol } Co { outfile2 }
	end
else
	foreach mol ( { arglist { plot_some } } )
		echo "Adding plot: "{ mol }
		add_plot { mol } Co { outfile2 }
	end
end

if ( ! USE_SOLVER )
	setfield { target } method "ee"
end
reset
step {SIMLENGTH} -time


/* Clear the output file */
openfile { outfile2 } w
closefile { outfile2 }

save_plots

quit

