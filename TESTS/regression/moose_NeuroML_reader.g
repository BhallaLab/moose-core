include NeuroML_Reader/plotUtil.g

str infile = "PurkinjeCell.morph.xml"
str outfile = "test.plot"
str target = "/cell"

int USE_SOLVER = 1
float SIMDT = 10e-6
float IODT = 10e-6
float SIMLENGTH = 0.10
int IOCLOCK = 2

setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {IODT}

readNeuroML { infile } { target }

setfield { target }/Soma_0 inject 10.0e-10

init_plots { SIMLENGTH } { IOCLOCK } { IODT }

add_plot  { target }/Soma_0 Vm { outfile }  

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

