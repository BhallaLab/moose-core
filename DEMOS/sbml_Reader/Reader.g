include plotUtil.g

str infile = "acc88.xml"
str outfile = "moose.plot"
int plot_all = 0
str plot_some = "/kinetics/mTOR_model_624_0_/TOR_minus_clx_626_0_ /kinetics/S6Kinase_630_0_/S6K_star__631_0_ /kinetics/S6Kinase_630_0_/S6K_632_0_ /kinetics/S6Kinase_630_0_/S6K_thr_minus_412_633_0_ /kinetics/S6Kinase_630_0_/_40S_inact_637_0_ /kinetics/kinetics_602_0_/PDK1_613_0_ /kinetics/kinetics_602_0_/PP2A_616_0_ /kinetics/kinetics_602_0_/Rheb_minus_GTP_623_0_  /kinetics/kinetics_602_0_/TOR_Rheb_minus_GTP_clx_627_0_ /kinetics/kinetics_602_0_/S6K_tot_638_0_ /kinetics/kinetics_602_0_/_40S_639_0_ /kinetics/kinetics_602_0_/S6K_thr_minus_252_640_0_  /kinetics/kinetics_602_0_/MAPK_star__643_0_ /kinetics/kinetics_602_0_/_40S_basal_647_0_"

str target = "/kinetics"

int USE_SOLVER = 1
float SIMLENGTH = 1000
float SIMDT = 0.001
float IODT = 10.0
int IOCLOCK = 2

setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {IODT}

readSBML { infile } { target }

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
//step 10000

/* Clear the output file */
openfile { outfile } w
closefile { outfile }

save_plots
quit

