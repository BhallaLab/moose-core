//moose
include assignReac.g // This has both A and be at zero

float vol = 1.6667e-17
setfield /kinetics volume {vol}
reset
step 100 -t
setfield /kinetics/A conc 1
step 100 -t
setfield /kinetics/A slave_enable 4 concInit 2
step 100 -t
setfield /kinetics/A  concInit 1
step 100 -t
setfield /kinetics/A slave_enable 0 conc 0
step 100 -t

do_save_all_plots db_{vol}.plot
