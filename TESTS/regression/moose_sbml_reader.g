include sbml_Reader/plotUtil.g

str infile = "sbml_Reader/acc88.xml"
str outfile = "test.plot"
int plot_all = 0
str plot_some = "/kinetics/kinetics_602_0_/_40S_639_0_  /kinetics/kinetics_602_0_/S6K_tot_638_0_ /kinetics/kinetics_602_0_/MAPK_star__643_0_ /kinetics/kinetics_602_0_/S6K_thr_minus_252_640_0_ /kinetics/S6Kinase_630_0_/S6K_thr_minus_412_633_0_ /kinetics/S6Kinase_630_0_/S6K_star__631_0_ /kinetics/kinetics_602_0_/TOR_Rheb_minus_GTP_clx_627_0_ /kinetics/kinetics_602_0_/Rheb_minus_GTP_623_0_"
/* str plot_some = "/kinetics/CaMKII_PSD_678_0_/actCaMKII_minus_PSD_680_0_ /kinetics/CaMKII_PSD_678_0_/tot_CaMKII_PSD_681_0_  /kinetics/kinetics_602_0_/act_CaMKII_cyt_660_0_  /kinetics/kinetics_602_0_/tot_CaMKII_cyt_659_0_ /kinetics/CaMKII_636_0_/CaMK_minus_thr306_644_0_ /kinetics/CaMKII_PSD_678_0_/CaMKII_star__star__star__minus_PSD_692_0_ /kinetics/CaMKII_PSD_678_0_/CaMKII_minus_thr306_minus_PSD_693_0_ /kinetics/CaMKII_PSD_678_0_/tot_minus_CaM_minus_CaMKII_minus_PSD_694_0_"
float SIMLENGTH = 3000
float SIMDT = 0.001
float IODT = 1.0 */
str target = "/kinetics"

int USE_SOLVER = 0
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

/* Clear the output file */
openfile { outfile } w
closefile { outfile }
save_plots
quit

