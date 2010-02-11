include plotUtil.g

str infile = "PurkinjeCell.morph.xml"
//~ str infile = "GranuleCell.morph.xml"
str outfile = "moose.plot"
str target = "/cell"

int USE_SOLVER = 1
float SIMDT = 100e-6
float IODT = 100e-6
float VIZDT = 2e-4
float SIMLENGTH = 1.00
int IOCLOCK = 2

setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {IODT}
setclock 3 {VIZDT}

readNeuroML { infile } { target }
setfield { target }/Soma_0 inject 10.0e-10

init_plots { SIMLENGTH } { IOCLOCK } { IODT }

add_plot  { target }/Soma_0 Vm { outfile }  
if ( ! USE_SOLVER )
	setfield { target } method "ee"
end

//=====================================
//  Vis object
//=====================================
create GLcell /gl0
setfield /gl0 vizpath /cell
setfield /gl0 port 9999
setfield /gl0 host localhost
setfield /gl0 attribute Vm
setfield /gl0 threshold 0.0015
setfield /gl0 sync off

useclock /gl0 3

reset
step {SIMLENGTH} -time

/* Clear the output file */
openfile { outfile } w
closefile { outfile }

save_plots
quit
