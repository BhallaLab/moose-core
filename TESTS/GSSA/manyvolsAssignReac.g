//moose
include assignReac.g

float vol
foreach vol ( 1.6667e-15 1.6667e-16 1.6667e-17 1.6667e-18 1.6667e-19 1.6667e-20 1.6667e-21 )
	setfield /kinetics volume {vol}
	reset
	step 100 -t
	setfield /kinetics/A conc 1
	step 100 -t
	do_save_all_plots ar_{vol}.plot
end
