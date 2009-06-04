include sbml_Read_Write/plotUtil.g

str infile1 = "sbml_Read_Write/acc88.xml"
str outfile1 = "moose.plot"
str infile2 = "acc88_dup.xml"
str outfile2 = "test.plot"
int plot_all = 0
str plot_some = "/kinetics/kinetics_602_0_/_40S_639_0_  /kinetics/kinetics_602_0_/S6K_tot_638_0_ /kinetics/kinetics_602_0_/MAPK_star__643_0_ /kinetics/kinetics_602_0_/S6K_thr_minus_252_640_0_ /kinetics/S6Kinase_630_0_/S6K_thr_minus_412_633_0_ /kinetics/S6Kinase_630_0_/S6K_star__631_0_ /kinetics/kinetics_602_0_/TOR_Rheb_minus_GTP_clx_627_0_ /kinetics/kinetics_602_0_/Rheb_minus_GTP_623_0_"   

str target = "/kinetics"

int USE_SOLVER = 0
float SIMLENGTH = 1000
float SIMDT = 0.001
float IODT = 10.0
int IOCLOCK = 2

setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {IODT}

readSBML { infile1 } { target }
writeSBML { infile2 } { target }

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

readSBML { infile2 } { target }

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

