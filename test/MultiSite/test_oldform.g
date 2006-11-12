//genesis
include MultiSite_oldform.g
SIMDT = 0.001

reset

setfield /kinetics/Ca n 0
step 100 -t

setfield /kinetics/Ca n { {getfield /kinetics/Ca n} + 1 }
step 100 -t

setfield /kinetics/Ca n { {getfield /kinetics/Ca n} + 1 }
step 100 -t

do_save_all_plots oldform.plot
